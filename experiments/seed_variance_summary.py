"""Aggregate seed-variance runs of crowspairs_routed_eval.py into a
mean ± std table, per (model, condition, domain).

Reads:
  helix_usage_validated/crowspairs_routed_{model}_calib.csv         (seed 0)
  helix_usage_validated/crowspairs_routed_{model}_calib_seed{i}.csv (seeds 1..3)

Writes:
  helix_usage_validated/seed_variance_summary.csv
  helix_usage_validated/seed_variance_summary.md
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

OUT_DIR = Path("helix_usage_validated")
MODELS = ["phi3", "gemma"]
SEEDS = [0, 1, 2, 3]
DOMAINS = ["gender", "race", "profession", "religion", "overall"]


SEED0_FILENAME = {
    "phi3":  "crowspairs_routed_phi3_snr0.08.csv",
    "gemma": "crowspairs_routed_gemma_calib.csv",
    "llama": "crowspairs_routed_llama_snr0.10.csv",
    "gpt2":  "crowspairs_routed_gpt2_calib.csv",
}


def load_seed(model, seed):
    if seed == 0:
        name = SEED0_FILENAME[model]
    else:
        name = f"crowspairs_routed_{model}_calib_seed{seed}.csv"
    path = OUT_DIR / name
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def mean_std(xs):
    n = len(xs)
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / max(1, n - 1)
    return m, math.sqrt(v)


def main():
    # (model, condition, domain) -> list of (seed, ss)
    bucket = defaultdict(list)
    for m in MODELS:
        for s in SEEDS:
            try:
                rows = load_seed(m, s)
            except FileNotFoundError:
                print(f"  missing seed {s} for {m}")
                continue
            for r in rows:
                bucket[(m, r["condition"], r["domain"])].append(
                    (s, float(r["ss"])))

    out = []
    for (m, cond, dom), pairs in bucket.items():
        vals = [v for _, v in pairs]
        mean, std = mean_std(vals)
        out.append(dict(
            model=m, condition=cond, domain=dom,
            n_seeds=len(vals),
            seed0=next((v for s, v in pairs if s == 0), float("nan")),
            mean=mean, std=std,
            min=min(vals), max=max(vals),
        ))

    out.sort(key=lambda r: (r["model"], r["condition"], DOMAINS.index(r["domain"])
                            if r["domain"] in DOMAINS else 99))

    csv_cols = ["model", "condition", "domain", "n_seeds",
                "seed0", "mean", "std", "min", "max"]
    csv_path = OUT_DIR / "seed_variance_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        for r in out:
            w.writerow(r)
    print(f"wrote {csv_path}  ({len(out)} rows)")

    md = [
        "# Seed variance: CrowS-Pairs SS under routed + per-domain α",
        "",
        "Head selection and probe tie-breaks are randomized by `--seed`; "
        "everything else is deterministic. Four seeds per model: 0 (paper "
        "default), 1, 2, 3. `mean ± std` is over these four runs. Column "
        "`seed0` is the paper-default deterministic value for reference.",
        "",
    ]
    for m in MODELS:
        md.append(f"## {m}")
        md.append("")
        md.append(
            "| condition | domain | seed0 | mean ± std | min | max |"
        )
        md.append("|---|---|---|---|---|---|")
        for r in out:
            if r["model"] != m:
                continue
            md.append(
                f"| {r['condition']} | {r['domain']} | "
                f"{r['seed0']:.2f} | "
                f"{r['mean']:.2f} ± {r['std']:.2f} | "
                f"{r['min']:.2f} | {r['max']:.2f} |"
            )
        md.append("")
    md_path = OUT_DIR / "seed_variance_summary.md"
    md_path.write_text("\n".join(md))
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()