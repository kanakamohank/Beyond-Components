"""Per-domain alpha calibration for routed ensembles.

For each (model, domain), we use the scan-selected K=4 heads and choose
an alpha_d so that the mean per-head SNR in the ensemble matches a fixed
target budget (default 0.20, which is roughly where GPT-2 sits at its
working alpha=1.5 in the paper).

SNR per head, for our compass injection:
  v_h = alpha * sigma_{d1} * cos(theta) * u1
      + alpha * sigma_{d2} * sin(theta) * u2     (u1, u2 orthonormal)
so ||v_h|| = alpha * sqrt(sigma_{d1}^2 cos^2 theta + sigma_{d2}^2 sin^2 theta)
and SNR_h = ||v_h|| / ||resid_pre_{L_h}||

Mean SNR over the K heads = alpha * (1/K) * sum_h s_h  where
  s_h = sqrt(...) / ||resid_pre_{L_h}||
Solving:  alpha_d = target / mean_h(s_h)

We use a short neutral prompt to measure ||resid_pre||, consistent with
introspect_compass_scale.py.

Writes:
  helix_usage_validated/per_domain_alpha_{tag}.json
  -- { "target_snr": ..., "by_domain": { "gender": {"alpha": ..., "mean_snr_at_alpha1": ..., "heads": [ ... ] }, ... } }

Usage:
  python -u experiments/calibrate_per_domain_alpha.py --model gpt2
  python -u experiments/calibrate_per_domain_alpha.py --model phi3
  python -u experiments/calibrate_per_domain_alpha.py --model llama
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

warnings.filterwarnings("ignore")

OUT_DIR = Path("helix_usage_validated")
PROMPT = "The doctor said that"

MODEL_SPECS = {
    "gpt2":  ("gpt2",                        "gpt2"),
    "gemma": ("google/gemma-2-2b",           "gemma"),
    "phi3":  ("microsoft/Phi-3-mini-4k-instruct", "phi3"),
    "llama": ("meta-llama/Llama-3.2-3B",     "llama"),
}


PRONOUNS = {"he", "she", "her", "his", "him", "hers", "himself", "herself",
            "He", "She", "His", "Her", "Him", "Hers"}


def load_scan_rows(tag):
    return [json.loads(l) for l in
            (OUT_DIR / f"stereoset_scan_{tag}.jsonl").open()]


def pick_heads_per_domain(scan_rows, K=4):
    """Same policy as crowspairs_routed_eval.py."""
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
        fb = len(chosen_heads) < K
        if fb:
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
            best_per_head.append(pool[0])
        routed[domain] = dict(heads=best_per_head, fallback=fb)
    return routed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_SPECS.keys()), required=True)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--target_snr", type=float, default=0.20)
    args = ap.parse_args()

    hf_name, scan_tag = MODEL_SPECS[args.model]
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    dtype = torch.bfloat16
    print(f"Loading {hf_name} on {device} ...", flush=True)
    model = HookedTransformer.from_pretrained(hf_name, device=device,
                                              dtype=dtype)

    scan_rows = load_scan_rows(scan_tag)
    routed = pick_heads_per_domain(scan_rows, K=args.K)

    # cache residuals for one neutral prompt
    tokens = model.to_tokens(PROMPT)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    resid_norm_by_layer = {}
    for L in range(model.cfg.n_layers):
        r = cache[f"blocks.{L}.hook_resid_pre"][0, -1, :].float().cpu()
        resid_norm_by_layer[L] = float(r.norm())

    out = {"model": hf_name, "target_snr": args.target_snr,
           "prompt_for_norm": PROMPT, "by_domain": {}}

    print(f"\nTarget SNR: {args.target_snr}\n", flush=True)
    for domain, d in routed.items():
        heads = d["heads"]
        per_head_snr_at_alpha1 = []  # s_h in the formula
        detail = []
        for h in heads:
            L, H = h["L"], h["H"]
            d1, d2 = h["d1"], h["d2"]
            phi = h["phi_hi"]
            theta = math.radians(phi + 180.0)

            W_V = model.W_V[L, H].detach().float().cpu()
            W_O = model.W_O[L, H].detach().float().cpu()
            _U, S, _Vt = torch.linalg.svd(W_V @ W_O, full_matrices=False)
            s1 = float(S[d1])
            s2 = float(S[d2])
            vec_norm_at_alpha1 = math.sqrt(
                (s1 * math.cos(theta)) ** 2 + (s2 * math.sin(theta)) ** 2)
            rn = resid_norm_by_layer[L]
            snr_a1 = vec_norm_at_alpha1 / rn
            per_head_snr_at_alpha1.append(snr_a1)
            detail.append(dict(L=L, H=H, d1=d1, d2=d2, phi=phi,
                               sigma_d1=s1, sigma_d2=s2,
                               resid_norm=rn,
                               snr_at_alpha1=snr_a1,
                               passed=h["passed"]))

        mean_s = sum(per_head_snr_at_alpha1) / max(1, len(per_head_snr_at_alpha1))
        alpha_d = args.target_snr / mean_s if mean_s > 0 else float("nan")
        out["by_domain"][domain] = dict(
            alpha=alpha_d,
            target_snr=args.target_snr,
            mean_snr_at_alpha1=mean_s,
            fallback=d["fallback"],
            heads=detail,
        )
        fb = " [FALLBACK]" if d["fallback"] else ""
        print(f"  {domain:12s}{fb}", flush=True)
        for h in detail:
            print(f"      L{h['L']:>2}H{h['H']:<2} "
                  f"sigma_d1={h['sigma_d1']:.3f} sigma_d2={h['sigma_d2']:.3f} "
                  f"||resid||={h['resid_norm']:.2f} "
                  f"snr@a=1={h['snr_at_alpha1']:.4f}",
                  flush=True)
        print(f"      -> mean_snr_at_alpha1={mean_s:.4f}  "
              f"alpha_d={alpha_d:.3f}", flush=True)

    path = OUT_DIR / f"per_domain_alpha_{args.model}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}", flush=True)

    print("\nSummary (alpha per domain):", flush=True)
    for domain, v in out["by_domain"].items():
        print(f"  {domain:12s}  alpha = {v['alpha']:6.2f}  "
              f"(mean_SNR@a=1={v['mean_snr_at_alpha1']:.4f})",
              flush=True)


if __name__ == "__main__":
    main()
