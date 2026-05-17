"""Transfer eval: CrowS-Pairs under compass ensemble injection.

Same ensemble heads we selected on StereoSet scans -> evaluated zero-shot
on CrowS-Pairs. If gains hold, the ensemble is not StereoSet-specific.

Metric: stereotype score = 100 * P(sent_more > sent_less).
  50 = unbiased, >50 = biased toward more-stereotypical sentence.

For each model we run:
  - baseline (no hooks)
  - ensemble at a small grid of alphas (default = best-alpha-from-StereoSet
    and 0.5x best-alpha)

Usage:
  python -u experiments/crowspairs_eval.py --model gpt2        --alphas 0.75,1.5
  python -u experiments/crowspairs_eval.py --model gemma       --alphas 5.0,10.0
  python -u experiments/crowspairs_eval.py --model phi3        --alphas 5.0,10.0
  python -u experiments/crowspairs_eval.py --model llama       --alphas 10.0,20.0

Outputs (one per --model):
  helix_usage_validated/crowspairs_{tag}.csv
  helix_usage_validated/crowspairs_{tag}.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
import warnings
from collections import Counter
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

CROWS_CSV = Path("data/crows_pairs_anonymized.csv")

warnings.filterwarnings("ignore")

OUT_DIR = Path("helix_usage_validated")
SCAN_DIR = OUT_DIR

PRONOUNS = {"he", "she", "her", "his", "him", "hers", "himself", "herself",
            "He", "She", "His", "Her", "Him", "Hers"}

# keep StereoSet->CrowS-Pairs domain mapping stable across the study.
# CrowS-Pairs bias_type categories we keep; mapping to StereoSet equivalents:
CROWS_DOMAINS = {
    "gender": "gender",
    "race-color": "race",
    "religion": "religion",
    "socioeconomic": "profession",  # closest analogue
}

MODEL_SPECS = {
    # tag -> (hf_name, scan_jsonl_tag, primary_head_LHplane_phi_from_stereoset)
    "gpt2":  ("gpt2",                        "gpt2"),
    "gemma": ("google/gemma-2-2b",           "gemma"),
    "phi3":  ("microsoft/Phi-3-mini-4k-instruct", "phi3"),
    "llama": ("meta-llama/Llama-3.2-3B",     "llama"),
}


def load_scan_rows(scan_tag: str):
    path = SCAN_DIR / f"stereoset_scan_{scan_tag}.jsonl"
    return [json.loads(l) for l in path.open()]


def pick_ensemble(scan_rows, K=4):
    """Same selection policy used in stereoset_ensemble_eval_*.py."""
    passing = Counter()
    for r in scan_rows:
        if r["passed"]:
            passing[(r["L"], r["H"])] += 1

    chosen = [lh for lh, _ in passing.most_common(K)]
    fallback_used = False
    if len(chosen) < K:
        fallback_used = True
        best_by_head = {}
        for r in scan_rows:
            lh = (r["L"], r["H"])
            if lh not in best_by_head or r["amp_slope"] > best_by_head[lh]["amp_slope"]:
                best_by_head[lh] = r
        ranked = sorted(best_by_head.values(),
                        key=lambda r: r["amp_slope"], reverse=True)
        for r in ranked:
            lh = (r["L"], r["H"])
            if lh in chosen:
                continue
            chosen.append(lh)
            if len(chosen) >= K:
                break

    best = []
    for (L, H) in chosen:
        cands = [r for r in scan_rows if r["L"] == L and r["H"] == H]
        non_pron = [r for r in cands
                    if r["stereo"] not in PRONOUNS
                    and r["anti"] not in PRONOUNS]
        pool = non_pron or cands
        pool.sort(key=lambda r: r["amp_slope"], reverse=True)
        best.append(pool[0])
    return best, fallback_used


def build_ensemble_hooks(model, ensemble, alpha, device, dtype):
    hooks = []
    for h in ensemble:
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


def compute_stereotype_score(model, rows, fwd_hooks):
    """CrowS-Pairs stereotype score via joint log-likelihood of each sentence.

    more_stereo = direction == 'stereo' -> sent_more is the stereotyping sentence
                  (count P(sent_more)>P(sent_less) as 'biased toward stereo')
    More-stereo direction is encoded in 'direction' == 'stereo'/'antistereo'.
    """
    pro_stereo = total = 0
    skipped = 0
    for ex in rows:
        try:
            lp_more = sentence_logprob(model, ex["sent_more"], fwd_hooks)
            lp_less = sentence_logprob(model, ex["sent_less"], fwd_hooks)
        except Exception:
            skipped += 1
            continue
        # if direction == 'stereo' then sent_more IS the stereotyping one;
        # if 'antistereo' then sent_less is the stereotyping one.
        direction = ex.get("stereo_antistereo", ex.get("direction", "stereo"))
        if direction == "stereo":
            stereo_is_more = True
        else:
            stereo_is_more = False
        if stereo_is_more:
            if lp_more > lp_less:
                pro_stereo += 1
        else:
            if lp_less > lp_more:
                pro_stereo += 1
        total += 1
    ss = 100.0 * pro_stereo / max(1, total)
    return dict(ss=ss, n=total, pro=pro_stereo, skipped=skipped)


CSV_COLS = ["condition", "alpha", "domain", "ss", "n", "skipped"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_SPECS.keys()), required=True)
    ap.add_argument("--alphas", default="")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--sanity_n", type=int, default=0)
    args = ap.parse_args()

    hf_name, scan_tag = MODEL_SPECS[args.model]
    default_alphas = {
        "gpt2":  "0.75,1.5",
        "gemma": "5.0,10.0",
        "phi3":  "5.0,10.0",
        "llama": "10.0,20.0",
    }
    alphas_str = args.alphas or default_alphas[args.model]
    alphas = [float(a) for a in alphas_str.split(",") if a.strip()]

    t0 = time.time()
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    dtype = torch.bfloat16
    print(f"[{time.time()-t0:6.1f}s] Loading {hf_name} on {device} ...",
          flush=True)
    model = HookedTransformer.from_pretrained(hf_name, device=device,
                                              dtype=dtype)

    scan_rows = load_scan_rows(scan_tag)
    ensemble, fallback = pick_ensemble(scan_rows, K=args.K)
    tag = " [FALLBACK: fewer than K strict passes]" if fallback else ""
    print(f"[{time.time()-t0:6.1f}s] Ensemble (K={args.K}):{tag}", flush=True)
    for h in ensemble:
        print(f"   L{h['L']:>2}H{h['H']:<2} SV({h['d1']},{h['d2']}) "
              f"phi={h['phi_hi']:+.1f} slope={h['amp_slope']:.3f} "
              f"passed={h['passed']}", flush=True)

    print(f"[{time.time()-t0:6.1f}s] Loading CrowS-Pairs from "
          f"{CROWS_CSV} ...", flush=True)
    if not CROWS_CSV.exists():
        raise FileNotFoundError(
            f"{CROWS_CSV} missing. Run: curl -sL "
            "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/"
            f"data/crows_pairs_anonymized.csv -o {CROWS_CSV}")
    keep_types = set(CROWS_DOMAINS.keys())
    rows_all = []
    with CROWS_CSV.open() as f:
        for row in csv.DictReader(f):
            if row["bias_type"] in keep_types:
                rows_all.append(row)
    print(f"[{time.time()-t0:6.1f}s]   {len(rows_all)} rows after filter "
          f"(types={sorted(keep_types)})", flush=True)
    if args.sanity_n > 0:
        rows_all = rows_all[: args.sanity_n]
        print(f"[{time.time()-t0:6.1f}s]   [sanity] -> {len(rows_all)} rows",
              flush=True)

    # group by StereoSet-equivalent domain label
    by_domain = {}
    for r in rows_all:
        dom = CROWS_DOMAINS[r["bias_type"]]
        by_domain.setdefault(dom, []).append(r)
    domains_present = list(by_domain.keys())
    print(f"[{time.time()-t0:6.1f}s]   domains={domains_present}", flush=True)
    for d in domains_present:
        print(f"      {d:12s} n={len(by_domain[d])}", flush=True)

    runs = []

    print(f"[{time.time()-t0:6.1f}s] baseline ...", flush=True)
    for d in domains_present:
        m = compute_stereotype_score(model, by_domain[d], None)
        runs.append(dict(condition="baseline", alpha=0.0, domain=d,
                         ss=m["ss"], n=m["n"], skipped=m["skipped"]))
        print(f"    baseline  {d:12s}  SS={m['ss']:6.2f}  n={m['n']}  "
              f"skipped={m['skipped']}", flush=True)
    # overall
    m = compute_stereotype_score(model, rows_all, None)
    runs.append(dict(condition="baseline", alpha=0.0, domain="overall",
                     ss=m["ss"], n=m["n"], skipped=m["skipped"]))
    print(f"    baseline  overall       SS={m['ss']:6.2f}  n={m['n']}",
          flush=True)

    for a in alphas:
        hooks = build_ensemble_hooks(model, ensemble, a, device, dtype)
        print(f"[{time.time()-t0:6.1f}s] ensemble K={args.K} alpha={a} ...",
              flush=True)
        for d in domains_present:
            m = compute_stereotype_score(model, by_domain[d], hooks)
            runs.append(dict(condition=f"ensemble_K{args.K}", alpha=a,
                             domain=d, ss=m["ss"], n=m["n"],
                             skipped=m["skipped"]))
            print(f"    ens a={a}  {d:12s}  SS={m['ss']:6.2f}  "
                  f"n={m['n']}  skipped={m['skipped']}", flush=True)
        m = compute_stereotype_score(model, rows_all, hooks)
        runs.append(dict(condition=f"ensemble_K{args.K}", alpha=a,
                         domain="overall", ss=m["ss"], n=m["n"],
                         skipped=m["skipped"]))
        print(f"    ens a={a}  overall       SS={m['ss']:6.2f}  n={m['n']}",
              flush=True)

    OUT_DIR.mkdir(exist_ok=True)
    out_tag = args.model
    csv_path = OUT_DIR / f"crowspairs_{out_tag}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for r in runs:
            w.writerow({k: r.get(k, "") for k in CSV_COLS})
    lines = [
        f"Model: {hf_name}",
        f"Ensemble: K={args.K}  alphas={alphas}"
        + (" [FALLBACK]" if fallback else ""),
        "Ensemble heads:",
    ]
    for h in ensemble:
        lines.append(f"   L{h['L']:>2}H{h['H']:<2} SV({h['d1']},{h['d2']}) "
                     f"phi={h['phi_hi']:+.1f}  slope={h['amp_slope']:.3f}  "
                     f"passed={h['passed']}")
    lines += [
        "",
        f"{'cond':<22} {'alpha':>6} {'domain':<12} "
        f"{'SS':>7} {'n':>6} {'skip':>5}",
        "-" * 60,
    ]
    for r in runs:
        lines.append(
            f"{r['condition']:<22} {r['alpha']:>6.2f} {r['domain']:<12} "
            f"{r['ss']:>7.2f} {r['n']:>6} {r['skipped']:>5}")
    (OUT_DIR / f"crowspairs_{out_tag}.txt").write_text("\n".join(lines))
    print(f"[{time.time()-t0:6.1f}s] wrote {csv_path}", flush=True)
    print(f"[{time.time()-t0:6.1f}s] wrote "
          f"{OUT_DIR / f'crowspairs_{out_tag}.txt'}", flush=True)


if __name__ == "__main__":
    main()
