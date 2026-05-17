"""Qualitative spot-check: top-shifted CrowS-Pairs examples per model.

For one model, run both the baseline and the routed + per-domain-alpha
intervention (same hooks as crowspairs_routed_eval.py). For every row,
compute

  gap_base   = logprob(sent_more) - logprob(sent_less)            baseline
  gap_routed = logprob(sent_more) - logprob(sent_less)   under routed hooks
  shift      = gap_routed - gap_base

If the row is `stereo_antistereo == "stereo"` then sent_more carries the
stereotype, so a *negative* shift pushes the model away from the
stereotype. If `anti`, sent_less carries the stereotype and a *positive*
shift is the debiasing direction.

Output: top-N rows by |shift|, emitted as markdown with:
  - domain, stereo-direction
  - sent_more / sent_less
  - gap_base, gap_routed, shift
  - direction verdict ("toward anti-stereo" / "toward stereo" / null)
  - whether the sign of gap flipped

Usage:
  python -u experiments/qualitative_spotcheck.py --model phi3 \
      --alpha_json helix_usage_validated/per_domain_alpha_phi3_snr0.08.json \
      --top_n 50

Outputs:
  helix_usage_validated/qualitative_spotcheck_{model}.md
  helix_usage_validated/qualitative_spotcheck_{model}.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

warnings.filterwarnings("ignore")

OUT_DIR = Path("helix_usage_validated")
CROWS_CSV = Path("data/crows_pairs_anonymized.csv")

PRONOUNS = {"he", "she", "her", "his", "him", "hers", "himself", "herself",
            "He", "She", "His", "Her", "Him", "Hers"}

CROWS_DOMAINS = {
    "gender": "gender",
    "race-color": "race",
    "religion": "religion",
    "socioeconomic": "profession",
}

MODEL_SPECS = {
    "gpt2":  ("gpt2",                              "gpt2"),
    "gemma": ("google/gemma-2-2b",                 "gemma"),
    "phi3":  ("microsoft/Phi-3-mini-4k-instruct",  "phi3"),
    "llama": ("meta-llama/Llama-3.2-3B",           "llama"),
}

DEFAULT_ALPHA_JSON = {
    "gpt2":  "per_domain_alpha_gpt2.json",
    "phi3":  "per_domain_alpha_phi3_snr0.08.json",
    "gemma": "per_domain_alpha_gemma.json",
    "llama": "per_domain_alpha_llama_snr0.10.json",
}


def load_scan_rows(tag):
    path = OUT_DIR / f"stereoset_scan_{tag}.jsonl"
    return [json.loads(line) for line in path.open()]


def pick_heads_per_domain(scan_rows, K=4):
    """Same selection policy as crowspairs_routed_eval.py."""
    by_head_domain = defaultdict(lambda: Counter())
    for r in scan_rows:
        if r["passed"]:
            by_head_domain[(r["L"], r["H"])][r["domain"]] += 1

    routed = {}
    for domain in ["gender", "race", "profession", "religion"]:
        scored = []
        for (L, H), c in by_head_domain.items():
            n = c.get(domain, 0)
            if n > 0:
                scored.append(((L, H), n))
        scored.sort(key=lambda x: x[1], reverse=True)
        chosen_heads = [lh for lh, _ in scored[:K]]
        fb = False
        if len(chosen_heads) < K:
            fb = True
            best_by_head = {}
            for r in scan_rows:
                if r["domain"] != domain:
                    continue
                lh = (r["L"], r["H"])
                if (lh not in best_by_head
                        or r["amp_slope"] > best_by_head[lh]["amp_slope"]):
                    best_by_head[lh] = r
            ranked = sorted(best_by_head.values(),
                            key=lambda r: r["amp_slope"], reverse=True)
            for r in ranked:
                lh = (r["L"], r["H"])
                if lh in chosen_heads:
                    continue
                chosen_heads.append(lh)
                if len(chosen_heads) >= K:
                    break

        best_per_head = []
        for (L, H) in chosen_heads:
            cands = [r for r in scan_rows
                     if r["L"] == L and r["H"] == H
                     and r["domain"] == domain]
            if not cands:
                cands = [r for r in scan_rows
                         if r["L"] == L and r["H"] == H]
            non_pron = [r for r in cands
                        if r["stereo"] not in PRONOUNS
                        and r["anti"] not in PRONOUNS]
            pool = non_pron or cands
            pool.sort(key=lambda r: r["amp_slope"], reverse=True)
            best_per_head.append(pool[0])
        routed[domain] = dict(heads=best_per_head, fallback=fb)
    return routed


def build_domain_hooks(model, heads, alpha, device, dtype):
    hooks = []
    for h in heads:
        L, H = h["L"], h["H"]
        d1, d2 = h["d1"], h["d2"]
        phi = h["phi_hi"]
        theta = math.radians(phi + 180.0)

        W_V = model.W_V[L, H].detach().float().cpu()
        W_O = model.W_O[L, H].detach().float().cpu()
        _U, S, Vt = torch.linalg.svd(W_V @ W_O, full_matrices=False)
        u1 = Vt[d1, :].to(device).to(dtype)
        u2 = Vt[d2, :].to(device).to(dtype)
        s1, s2 = float(S[d1]), float(S[d2])
        vec = (alpha * s1 * float(math.cos(theta))) * u1 \
            + (alpha * s2 * float(math.sin(theta))) * u2
        hook_name = f"blocks.{L}.hook_resid_pre"

        def _hook(r, hook, _v=vec):
            r[0, :, :] = r[0, :, :] + _v
            return r

        hooks.append((hook_name, _hook))
    return hooks


def sentence_logprob(model, text, fwd_hooks):
    tokens = model.to_tokens(text)
    with torch.no_grad():
        if fwd_hooks is None:
            logits = model(tokens)[0]
        else:
            logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)[0]
    log_probs = torch.log_softmax(logits[:-1].float(), dim=-1)
    targets = tokens[0, 1:]
    lp = log_probs[torch.arange(targets.shape[0]), targets]
    return float(lp.sum())


def verdict_label(row, shift, gap_base, gap_routed):
    """Human-readable debiasing verdict for one row."""
    stereo_is_more = row.get("stereo_antistereo", "stereo") == "stereo"
    base_pref_stereo = (gap_base > 0) == stereo_is_more
    routed_pref_stereo = (gap_routed > 0) == stereo_is_more
    if base_pref_stereo and not routed_pref_stereo:
        return "FLIP -> anti-stereo"
    if not base_pref_stereo and routed_pref_stereo:
        return "FLIP -> stereo (regression)"
    if base_pref_stereo and routed_pref_stereo:
        stronger = abs(gap_routed) > abs(gap_base)
        return ("stereo stronger" if stronger
                else "stereo weaker (partial debias)")
    return ("anti-stereo stronger" if abs(gap_routed) > abs(gap_base)
            else "anti-stereo weaker (partial regression)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_SPECS.keys()), required=True)
    ap.add_argument("--alpha_json", default="")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--top_n", type=int, default=50,
                    help="how many examples to emit to markdown")
    ap.add_argument("--limit", type=int, default=0,
                    help="if >0, process only first --limit rows (debug)")
    args = ap.parse_args()

    hf_name, scan_tag = MODEL_SPECS[args.model]
    alpha_json = args.alpha_json or str(OUT_DIR / DEFAULT_ALPHA_JSON[args.model])

    t0 = time.time()
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    dtype = torch.bfloat16
    print(f"[{time.time()-t0:6.1f}s] Loading {hf_name} on {device}", flush=True)
    model = HookedTransformer.from_pretrained(hf_name, device=device,
                                              dtype=dtype)

    print(f"[{time.time()-t0:6.1f}s] Calib alphas from {alpha_json}",
          flush=True)
    calib = json.loads(Path(alpha_json).read_text())
    per_domain_alpha = {d: v["alpha"] for d, v in calib["by_domain"].items()}
    print(f"[{time.time()-t0:6.1f}s]   {per_domain_alpha}", flush=True)

    scan_rows = load_scan_rows(scan_tag)
    routed = pick_heads_per_domain(scan_rows, K=args.K)
    hooks_by_domain = {}
    for d, info in routed.items():
        hooks_by_domain[d] = build_domain_hooks(
            model, info["heads"], per_domain_alpha[d], device, dtype)

    keep = set(CROWS_DOMAINS.keys())
    rows_all = []
    with CROWS_CSV.open() as f:
        for row in csv.DictReader(f):
            if row["bias_type"] in keep:
                rows_all.append(row)
    if args.limit > 0:
        rows_all = rows_all[: args.limit]
    print(f"[{time.time()-t0:6.1f}s] {len(rows_all)} CrowS-Pairs rows",
          flush=True)

    records = []
    for i, ex in enumerate(rows_all):
        domain = CROWS_DOMAINS[ex["bias_type"]]
        try:
            lp_more_base = sentence_logprob(model, ex["sent_more"], None)
            lp_less_base = sentence_logprob(model, ex["sent_less"], None)
            hooks = hooks_by_domain[domain]
            lp_more_rt = sentence_logprob(model, ex["sent_more"], hooks)
            lp_less_rt = sentence_logprob(model, ex["sent_less"], hooks)
        except Exception as exc:
            print(f"  skip row {i}: {exc}", flush=True)
            continue
        gap_base = lp_more_base - lp_less_base
        gap_routed = lp_more_rt - lp_less_rt
        shift = gap_routed - gap_base
        stereo_is_more = ex.get("stereo_antistereo", "stereo") == "stereo"
        debias_shift = -shift if stereo_is_more else shift  # >0 = away-from-stereo
        verdict = verdict_label(ex, shift, gap_base, gap_routed)
        records.append(dict(
            idx=i, domain=domain,
            stereo_antistereo=ex.get("stereo_antistereo", "stereo"),
            sent_more=ex["sent_more"], sent_less=ex["sent_less"],
            gap_base=gap_base, gap_routed=gap_routed,
            shift=shift, debias_shift=debias_shift,
            verdict=verdict,
        ))
        if (i + 1) % 100 == 0:
            print(f"[{time.time()-t0:6.1f}s]   scored {i+1}/{len(rows_all)}",
                  flush=True)

    OUT_DIR.mkdir(exist_ok=True)
    csv_cols = ["idx", "domain", "stereo_antistereo",
                "sent_more", "sent_less",
                "gap_base", "gap_routed", "shift", "debias_shift", "verdict"]
    csv_path = OUT_DIR / f"qualitative_spotcheck_{args.model}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        for r in records:
            w.writerow({k: r[k] for k in csv_cols})
    print(f"[{time.time()-t0:6.1f}s] wrote {csv_path}", flush=True)

    records_sorted = sorted(records, key=lambda r: abs(r["debias_shift"]),
                            reverse=True)
    top = records_sorted[: args.top_n]

    counts = Counter(r["verdict"] for r in records)
    overall_mean_shift = sum(r["debias_shift"] for r in records) / max(1, len(records))

    md_lines = [
        f"# Qualitative spot-check: {hf_name}",
        f"Alpha source: `{alpha_json}`",
        f"Per-domain alphas: `{per_domain_alpha}`",
        f"Rows scored: {len(records)} "
        f"(mean debias shift = {overall_mean_shift:+.3f} nats)",
        "",
        "## Verdict distribution",
        "",
        "| verdict | count |",
        "|---|---|",
    ]
    for v, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        md_lines.append(f"| {v} | {n} |")
    md_lines += ["",
                 f"## Top {len(top)} by |debias_shift|",
                 "",
                 "`debias_shift > 0` = intervention pushes *away* from "
                 "stereotype on this row (the desired direction). "
                 "`gap = logp(sent_more) − logp(sent_less)`.",
                 ""]

    for r in top:
        star = "" if r["debias_shift"] > 0 else " [REGRESSION]"
        md_lines += [
            f"### #{r['idx']} — {r['domain']} / {r['stereo_antistereo']}"
            f"{star}",
            "",
            f"- `gap_base   = {r['gap_base']:+.3f}`",
            f"- `gap_routed = {r['gap_routed']:+.3f}`",
            f"- `shift      = {r['shift']:+.3f}`",
            f"- `debias_shift = {r['debias_shift']:+.3f}`",
            f"- verdict: **{r['verdict']}**",
            "",
            f"- sent_more: {r['sent_more']}",
            f"- sent_less: {r['sent_less']}",
            "",
        ]

    md_path = OUT_DIR / f"qualitative_spotcheck_{args.model}.md"
    md_path.write_text("\n".join(md_lines))
    print(f"[{time.time()-t0:6.1f}s] wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
