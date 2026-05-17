"""Per-domain α sweep targeting CrowS-Pairs SS = 50 (matched-strength
variant of routed+calib). Answers the reviewer question: if you dial
the intervention strength to match the SS=50 neutral line instead of
the SNR-matched calibration, how does the compass compare to INLP /
SentenceDebias?

For each (model, domain) in {phi3, gemma} x {gender, race}, we:
  1. Load the calibrated per-domain α_d from per_domain_alpha_*.json.
  2. Run CrowS-Pairs (domain-restricted) at several multipliers
     M * α_d for M in MULTIPLIERS.
  3. Pick the M* that minimizes |SS - 50|, report both SS and M*.

Usage
-----
  python -u experiments/target_ss50_sweep.py --model phi3
  python -u experiments/target_ss50_sweep.py --model gemma

Outputs:
  helix_usage_validated/target_ss50_{model}.csv
  helix_usage_validated/target_ss50_{model}.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

# Ensure we can import sibling script and resolve OUT_DIR relative to repo root
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))
os.chdir(_REPO)

from crowspairs_routed_eval import (  # noqa: E402
    CROWS_DOMAINS, CROWS_CSV, MODEL_SPECS, OUT_DIR,
    build_domain_hooks, compute_ss, load_scan_rows,
    pick_heads_per_domain,
)

warnings.filterwarnings("ignore")

DEFAULT_ALPHA_JSON = {
    "gpt2":  "per_domain_alpha_gpt2.json",
    "phi3":  "per_domain_alpha_phi3_snr0.08.json",
    "gemma": "per_domain_alpha_gemma.json",
    "llama": "per_domain_alpha_llama_snr0.10.json",
}

MULTIPLIERS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    choices=list(MODEL_SPECS.keys()), required=True)
    ap.add_argument("--alpha_json", default="")
    ap.add_argument("--domains",
                    default="gender,race",
                    help="comma-separated subset")
    ap.add_argument("--K", type=int, default=4)
    args = ap.parse_args()

    hf_name, scan_tag = MODEL_SPECS[args.model]
    alpha_json = args.alpha_json or str(
        OUT_DIR / DEFAULT_ALPHA_JSON[args.model])
    target_domains = [d.strip() for d in args.domains.split(",") if d.strip()]

    t0 = time.time()
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    dtype = torch.bfloat16
    print(f"[{time.time()-t0:6.1f}s] Loading {hf_name} on {device}",
          flush=True)
    model = HookedTransformer.from_pretrained(hf_name, device=device,
                                              dtype=dtype)

    calib = json.loads(Path(alpha_json).read_text())
    per_domain_alpha = {d: v["alpha"] for d, v in calib["by_domain"].items()}
    print(f"[{time.time()-t0:6.1f}s] calib alphas = {per_domain_alpha}",
          flush=True)

    scan_rows = load_scan_rows(scan_tag)
    routed = pick_heads_per_domain(scan_rows, K=args.K)

    keep = set(CROWS_DOMAINS.keys())
    by_domain = defaultdict(list)
    with CROWS_CSV.open() as f:
        for row in csv.DictReader(f):
            if row["bias_type"] in keep:
                by_domain[CROWS_DOMAINS[row["bias_type"]]].append(row)

    runs = []
    for d in target_domains:
        rows = by_domain[d]
        print(f"[{time.time()-t0:6.1f}s] === domain={d} n={len(rows)} ===",
              flush=True)

        # Include baseline once (M=0.0 == no hooks).
        base = compute_ss(model, rows, None)
        runs.append(dict(domain=d, multiplier=0.0, alpha=0.0,
                         ss=base["ss"], n=base["n"],
                         dist50=abs(base["ss"] - 50.0)))
        print(f"    baseline          SS={base['ss']:6.2f}  "
              f"|Δ50|={abs(base['ss']-50):.2f}", flush=True)

        for m in MULTIPLIERS:
            if m == 0.0:
                continue
            a = m * per_domain_alpha[d]
            hooks = build_domain_hooks(model, routed[d]["heads"],
                                       a, device, dtype)
            met = compute_ss(model, rows, hooks)
            dist = abs(met["ss"] - 50.0)
            runs.append(dict(domain=d, multiplier=m, alpha=a,
                             ss=met["ss"], n=met["n"], dist50=dist))
            print(f"    M={m:4.2f}  a={a:7.3f}  SS={met['ss']:6.2f}  "
                  f"|Δ50|={dist:.2f}", flush=True)

    OUT_DIR.mkdir(exist_ok=True)
    csv_path = OUT_DIR / f"target_ss50_{args.model}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["domain", "multiplier", "alpha",
                                          "ss", "n", "dist50"])
        w.writeheader()
        for r in runs:
            w.writerow(r)

    # Pick best multiplier per domain.
    best_by_dom = {}
    for r in runs:
        d = r["domain"]
        if d not in best_by_dom or r["dist50"] < best_by_dom[d]["dist50"]:
            best_by_dom[d] = r

    lines = [
        f"Model: {hf_name}",
        f"Calib α source: {alpha_json}",
        f"Per-domain α (M=1.0 reference): {per_domain_alpha}",
        "",
        f"{'domain':<12} {'mult':>6} {'alpha':>8} {'SS':>7} "
        f"{'|Δ50|':>7} {'n':>5}",
        "-" * 50,
    ]
    for r in runs:
        lines.append(f"{r['domain']:<12} {r['multiplier']:>6.2f} "
                     f"{r['alpha']:>8.3f} {r['ss']:>7.2f} "
                     f"{r['dist50']:>7.2f} {r['n']:>5}")
    lines += ["", "Best multiplier per domain:"]
    for d, r in best_by_dom.items():
        lines.append(f"  {d}: M*={r['multiplier']:.2f}  α*={r['alpha']:.3f}  "
                     f"SS*={r['ss']:.2f}  |Δ50|={r['dist50']:.2f}")
    txt_path = OUT_DIR / f"target_ss50_{args.model}.txt"
    txt_path.write_text("\n".join(lines))
    print(f"[{time.time()-t0:6.1f}s] wrote {csv_path}  {txt_path}",
          flush=True)


if __name__ == "__main__":
    main()
