"""Routed-ensemble CrowS-Pairs eval.

Differs from crowspairs_eval.py: instead of one global K=4 ensemble injected
on every domain, we pick heads PER domain from the scan JSONL.

For each bias domain d in {gender, race, profession, religion}:
  - rank heads (L,H) by n_passes_in_d (strict passes from two-pillar test)
  - keep top-Kd heads; per head, pick the plane/probe with best amp_slope
    within that domain
  - build hooks only from those heads and inject them only on rows where
    bias_type == d

This makes the compass domain-aware: a gender-target row is perturbed by
gender-selective heads, not by profession-selective heads.

Usage:
  python -u experiments/crowspairs_routed_eval.py --model gpt2  --alphas 0.75,1.5
  python -u experiments/crowspairs_routed_eval.py --model phi3  --alphas 5.0,10.0
  python -u experiments/crowspairs_routed_eval.py --model llama --alphas 10.0,20.0

Outputs:
  helix_usage_validated/crowspairs_routed_{tag}.csv
  helix_usage_validated/crowspairs_routed_{tag}.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
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

# crows_pairs bias_type -> StereoSet domain label (what the scan uses)
CROWS_DOMAINS = {
    "gender": "gender",
    "race-color": "race",
    "religion": "religion",
    "socioeconomic": "profession",
}

MODEL_SPECS = {
    "gpt2":  ("gpt2",                        "gpt2"),
    "gemma": ("google/gemma-2-2b",           "gemma"),
    "phi3":  ("microsoft/Phi-3-mini-4k-instruct", "phi3"),
    "llama": ("meta-llama/Llama-3.2-3B",     "llama"),
}


def load_scan_rows(tag):
    path = OUT_DIR / f"stereoset_scan_{tag}.jsonl"
    return [json.loads(l) for l in path.open()]


def pick_heads_per_domain(scan_rows, K=4, seed=0):
    """For each domain, return list of (L,H,d1,d2,phi,amp_slope) for top-K
    heads ranked by domain-specific pass count, tie-broken by amp_slope.

    Falls back (within the domain) to top amp_slope if <K heads passed.

    Non-zero `seed` randomizes tie-breaking: among heads with the same
    pass count we sample K from all ties uniformly (instead of taking the
    first K in dict-iteration order). Same for probe tie-breaks among
    probes with equal amp_slope (to within 1e-6). Lets us report std over
    seeds for the head-selection source of nondeterminism.
    """
    rng = random.Random(seed)
    by_head_domain = defaultdict(lambda: Counter())  # (L,H) -> Counter(domain -> passes)
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
        # Group by pass count desc, shuffle within ties (stable under seed=0)
        buckets = defaultdict(list)
        for lh, n in scored:
            buckets[n].append(lh)
        ordered = []
        for n in sorted(buckets.keys(), reverse=True):
            ties = buckets[n][:]
            if seed != 0:
                rng.shuffle(ties)
            ordered.extend(ties)
        chosen_heads = ordered[:K]
        fb = False
        if len(chosen_heads) < K:
            fb = True
            best_by_head = {}
            for r in scan_rows:
                if r["domain"] != domain:
                    continue
                lh = (r["L"], r["H"])
                if lh not in best_by_head or r["amp_slope"] > best_by_head[lh]["amp_slope"]:
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
            if seed != 0 and len(pool) > 1:
                top = pool[0]["amp_slope"]
                ties = [r for r in pool if abs(r["amp_slope"] - top) < 1e-6]
                best_per_head.append(rng.choice(ties))
            else:
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


def sentence_logprob(model, text: str, fwd_hooks):
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


def compute_ss(model, rows, fwd_hooks):
    pro = total = skipped = 0
    for ex in rows:
        try:
            lp_more = sentence_logprob(model, ex["sent_more"], fwd_hooks)
            lp_less = sentence_logprob(model, ex["sent_less"], fwd_hooks)
        except Exception:
            skipped += 1
            continue
        direction = ex.get("stereo_antistereo", "stereo")
        stereo_is_more = direction == "stereo"
        if stereo_is_more:
            if lp_more > lp_less:
                pro += 1
        else:
            if lp_less > lp_more:
                pro += 1
        total += 1
    ss = 100.0 * pro / max(1, total)
    return dict(ss=ss, n=total, pro=pro, skipped=skipped)


CSV_COLS = ["condition", "alpha", "domain", "ss", "n", "skipped"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_SPECS.keys()), required=True)
    ap.add_argument("--alphas", default="")
    ap.add_argument("--alpha_json",
                    help="Path to per_domain_alpha_{tag}.json. If set, "
                         "uses per-domain calibrated alphas instead of "
                         "the --alphas grid (grid is ignored).")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--sanity_n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0,
                    help="Non-zero randomizes tie-breaking in head & probe "
                         "selection. 0 = deterministic paper default.")
    args = ap.parse_args()

    hf_name, scan_tag = MODEL_SPECS[args.model]
    default_alphas = {
        "gpt2":  "0.75,1.5",
        "gemma": "5.0,10.0",
        "phi3":  "5.0,10.0",
        "llama": "10.0,20.0",
    }
    alphas = [float(a) for a in (args.alphas or default_alphas[args.model])
              .split(",") if a.strip()]
    per_domain_alpha = None
    if args.alpha_json:
        data = json.loads(Path(args.alpha_json).read_text())
        per_domain_alpha = {d: v["alpha"] for d, v in data["by_domain"].items()}
        print(f"[calib] per-domain alphas from {args.alpha_json}: "
              f"{per_domain_alpha}", flush=True)

    t0 = time.time()
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    dtype = torch.bfloat16
    print(f"[{time.time()-t0:6.1f}s] Loading {hf_name} ...", flush=True)
    model = HookedTransformer.from_pretrained(hf_name, device=device,
                                              dtype=dtype)

    scan_rows = load_scan_rows(scan_tag)
    routed = pick_heads_per_domain(scan_rows, K=args.K, seed=args.seed)
    print(f"[{time.time()-t0:6.1f}s] Routed ensembles (K={args.K} per domain):",
          flush=True)
    for d in ["gender", "race", "profession", "religion"]:
        fb_tag = " [FALLBACK]" if routed[d]["fallback"] else ""
        print(f"  -- {d}{fb_tag}", flush=True)
        for h in routed[d]["heads"]:
            print(f"      L{h['L']:>2}H{h['H']:<2} SV({h['d1']},{h['d2']}) "
                  f"phi={h['phi_hi']:+.1f} slope={h['amp_slope']:.3f} "
                  f"probe='{h['stereo']}/{h['anti']}' "
                  f"passed={h['passed']}", flush=True)

    print(f"[{time.time()-t0:6.1f}s] Loading CrowS-Pairs ...", flush=True)
    keep = set(CROWS_DOMAINS.keys())
    rows_all = []
    with CROWS_CSV.open() as f:
        for row in csv.DictReader(f):
            if row["bias_type"] in keep:
                rows_all.append(row)
    if args.sanity_n > 0:
        rows_all = rows_all[: args.sanity_n]
    print(f"[{time.time()-t0:6.1f}s]   {len(rows_all)} rows", flush=True)

    by_domain = defaultdict(list)
    for r in rows_all:
        by_domain[CROWS_DOMAINS[r["bias_type"]]].append(r)
    for d, rows in by_domain.items():
        print(f"    {d:12s} n={len(rows)}", flush=True)

    runs = []
    print(f"[{time.time()-t0:6.1f}s] baseline ...", flush=True)
    for d, rows in by_domain.items():
        m = compute_ss(model, rows, None)
        runs.append(dict(condition="baseline", alpha=0.0, domain=d,
                         ss=m["ss"], n=m["n"], skipped=m["skipped"]))
        print(f"    baseline  {d:12s}  SS={m['ss']:6.2f}  n={m['n']}",
              flush=True)
    m = compute_ss(model, rows_all, None)
    runs.append(dict(condition="baseline", alpha=0.0, domain="overall",
                     ss=m["ss"], n=m["n"], skipped=m["skipped"]))
    print(f"    baseline  overall       SS={m['ss']:6.2f}  n={m['n']}",
          flush=True)

    alpha_conditions = []
    if per_domain_alpha is not None:
        alpha_conditions.append(("routed_calib", per_domain_alpha))
    else:
        for a in alphas:
            alpha_conditions.append((f"routed_K{args.K}",
                                     {d: a for d in by_domain}))

    for cond_name, a_by_domain in alpha_conditions:
        # a_by_domain may have a single value per domain (for grid runs this
        # is just a repeated scalar). Pick one representative for logging.
        a_log = (list(a_by_domain.values())[0]
                 if len(set(round(v, 6) for v in a_by_domain.values())) == 1
                 else -1.0)
        print(f"[{time.time()-t0:6.1f}s] {cond_name} alpha={a_by_domain} ...",
              flush=True)
        per_domain_metrics = {}
        for d, rows in by_domain.items():
            a_d = a_by_domain[d]
            hooks = build_domain_hooks(model, routed[d]["heads"],
                                       a_d, device, dtype)
            m = compute_ss(model, rows, hooks)
            per_domain_metrics[d] = m
            runs.append(dict(condition=cond_name, alpha=a_d, domain=d,
                             ss=m["ss"], n=m["n"], skipped=m["skipped"]))
            print(f"    {cond_name} a={a_d:.3f}  {d:12s}  "
                  f"SS={m['ss']:6.2f}  n={m['n']}  skipped={m['skipped']}",
                  flush=True)
        total_n = sum(m["n"] for m in per_domain_metrics.values())
        total_pro = sum(m["pro"] for m in per_domain_metrics.values())
        overall_ss = 100.0 * total_pro / max(1, total_n)
        runs.append(dict(condition=cond_name,
                         alpha=a_log, domain="overall",
                         ss=overall_ss, n=total_n, skipped=0))
        print(f"    {cond_name} overall  SS={overall_ss:6.2f}  "
              f"n={total_n}", flush=True)

    OUT_DIR.mkdir(exist_ok=True)
    suffix = "_calib" if per_domain_alpha is not None else ""
    if args.seed != 0:
        suffix = f"{suffix}_seed{args.seed}"
    csv_path = OUT_DIR / f"crowspairs_routed_{args.model}{suffix}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for r in runs:
            w.writerow({k: r.get(k, "") for k in CSV_COLS})
    header_alpha = (f"per-domain (calibrated): {per_domain_alpha}"
                    if per_domain_alpha is not None
                    else f"grid={alphas}")
    lines = [
        f"Model: {hf_name}",
        f"Routed ensemble: K={args.K} per domain  alphas={header_alpha}",
        "Per-domain heads (best-probe within that domain):",
    ]
    for d in ["gender", "race", "profession", "religion"]:
        fb_tag = " [FALLBACK]" if routed[d]["fallback"] else ""
        lines.append(f"  {d}:{fb_tag}")
        for h in routed[d]["heads"]:
            lines.append(
                f"    L{h['L']:>2}H{h['H']:<2} SV({h['d1']},{h['d2']}) "
                f"phi={h['phi_hi']:+.1f}  slope={h['amp_slope']:.3f}  "
                f"passed={h['passed']}")
    lines += ["",
              f"{'cond':<22} {'alpha':>6} {'domain':<12} "
              f"{'SS':>7} {'n':>6}",
              "-" * 60]
    for r in runs:
        lines.append(
            f"{r['condition']:<22} {r['alpha']:>6.2f} {r['domain']:<12} "
            f"{r['ss']:>7.2f} {r['n']:>6}")
    txt_path = OUT_DIR / f"crowspairs_routed_{args.model}{suffix}.txt"
    txt_path.write_text("\n".join(lines))
    print(f"[{time.time()-t0:6.1f}s] wrote {csv_path}", flush=True)
    print(f"[{time.time()-t0:6.1f}s] wrote {txt_path}", flush=True)


if __name__ == "__main__":
    main()
