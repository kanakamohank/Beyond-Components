"""Bootstrap 95% CIs + Wilson binomial CIs for every bias/honesty CSV
in helix_usage_validated/.

Post-processing only, no model forward passes.

For each row we infer the underlying success count:
  CrowS-Pairs  : k = round(ss/100 * n)   (metric col = "ss")
  TruthfulQA   : k = correct              (metric col = "mc1_acc", exact k)
  StereoSet CSVs with LMS/SS/ICAT: bootstraps SS only (LMS/ICAT need per-row data)

Then, for each row:
  - empirical bootstrap with B=10000 resamples of n Bernoulli trials with
    success prob k/n (equivalent to resampling the underlying 0/1 vector)
  - Wilson score interval (closed-form, n > 30 safe)

Aggregated outputs:
  helix_usage_validated/bootstrap_ci_summary.csv
  helix_usage_validated/bootstrap_ci_summary.md

The script is idempotent and only reads CSVs; existing eval files are
untouched.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

OUT_DIR = Path("helix_usage_validated")
B = 10000
SEED = 0
ALPHA = 0.05  # 95% CI

METRIC_COLS = {
    # CSV filename pattern -> (metric_col, max_val, metric_is_percent,
    #                          exact_k_col_or_None)
    "crowspairs": ("ss", 100.0, True, None),
    "truthfulqa": ("mc1_acc", 100.0, True, "correct"),
    "stereoset_ensemble": ("SS", 100.0, True, None),
    "stereoset_gpt2":   ("SS", 100.0, True, None),
    "stereoset_phi3":   ("SS", 100.0, True, None),
    "stereoset_llama":  ("SS", 100.0, True, None),
    "stereoset_gemma":  ("SS", 100.0, True, None),
    "winogender":       ("stereo_corr", 1.0, False, None),  # bounded in [-1,1]
}


def classify(path):
    name = path.name.lower()
    if name.startswith("crowspairs"):
        return METRIC_COLS["crowspairs"]
    if name.startswith("truthfulqa"):
        return METRIC_COLS["truthfulqa"]
    if name.startswith("stereoset_ensemble"):
        return METRIC_COLS["stereoset_ensemble"]
    for key in ("stereoset_gpt2", "stereoset_phi3", "stereoset_llama",
                "stereoset_gemma"):
        if name.startswith(key):
            return METRIC_COLS[key]
    if name.startswith("winogender"):
        return METRIC_COLS["winogender"]
    return None


def wilson_ci(k, n, alpha=ALPHA):
    if n <= 0:
        return (float("nan"), float("nan"))
    z = 1.959963984540054  # norminv(1 - alpha/2) for alpha=0.05
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def empirical_bootstrap(k, n, B=B, seed=SEED, alpha=ALPHA):
    if n <= 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    p = k / n
    samples = rng.binomial(n, p, size=B) / n
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return (lo, hi)


def float_or_none(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def process_csv(path):
    spec = classify(path)
    if spec is None:
        return []
    metric_col, max_val, is_percent, exact_k_col = spec
    rows_out = []
    with path.open() as f:
        r = csv.DictReader(f)
        if metric_col not in (r.fieldnames or []):
            return []
        has_exact = exact_k_col and exact_k_col in r.fieldnames
        for row in r:
            m = float_or_none(row.get(metric_col))
            n = float_or_none(row.get("n"))
            if m is None or n is None or n <= 0:
                continue
            n = int(n)
            if has_exact:
                k = int(float(row[exact_k_col]))
            else:
                if is_percent:
                    p = m / 100.0
                else:
                    p = (m + 1.0) / 2.0
                k = int(round(p * n))
                k = max(0, min(n, k))
            wlo, whi = wilson_ci(k, n)
            blo, bhi = empirical_bootstrap(k, n)
            if is_percent:
                value_disp = m
                wlo_disp, whi_disp = 100 * wlo, 100 * whi
                blo_disp, bhi_disp = 100 * blo, 100 * bhi
            else:
                # winogender-style bounded [-1, 1]; map CI back from [0,1]
                value_disp = m
                wlo_disp, whi_disp = 2 * wlo - 1, 2 * whi - 1
                blo_disp, bhi_disp = 2 * blo - 1, 2 * bhi - 1
            rows_out.append(dict(
                file=path.name,
                condition=row.get("condition", ""),
                alpha=row.get("alpha", ""),
                domain=row.get("domain", ""),
                metric=metric_col,
                value=value_disp,
                n=n,
                k_inferred=k,
                wilson_lo=wlo_disp,
                wilson_hi=whi_disp,
                boot_lo=blo_disp,
                boot_hi=bhi_disp,
                half_width=(bhi_disp - blo_disp) / 2,
            ))
    return rows_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*.csv",
                    help="Glob inside helix_usage_validated/ to include.")
    ap.add_argument("--out_csv", default="bootstrap_ci_summary.csv")
    ap.add_argument("--out_md", default="bootstrap_ci_summary.md")
    args = ap.parse_args()

    paths = sorted(OUT_DIR.glob(args.glob))
    all_rows = []
    for p in paths:
        try:
            rs = process_csv(p)
        except Exception as exc:
            print(f"  skip {p.name}: {exc}")
            continue
        if rs:
            print(f"  {p.name}: {len(rs)} rows")
            all_rows.extend(rs)

    if not all_rows:
        print("No matching rows found.")
        return

    csv_path = OUT_DIR / args.out_csv
    cols = ["file", "condition", "alpha", "domain", "metric",
            "value", "n", "k_inferred",
            "wilson_lo", "wilson_hi", "boot_lo", "boot_hi", "half_width"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            w.writerow({c: r[c] for c in cols})
    print(f"wrote {csv_path}  ({len(all_rows)} rows)")

    md_path = OUT_DIR / args.out_md
    by_file = {}
    for r in all_rows:
        by_file.setdefault(r["file"], []).append(r)
    lines = [f"# Bootstrap 95% CIs ({B} reps) + Wilson CIs",
             "",
             "All intervals are 95% two-sided. `boot_*` = empirical bootstrap, "
             "`wilson_*` = Wilson score (closed-form). Values in the same "
             "percent / [-1,1] units as the source CSV.",
             ""]
    for fname in sorted(by_file.keys()):
        rows = by_file[fname]
        metric = rows[0]["metric"]
        lines.append(f"## {fname}  (metric={metric})")
        lines.append("")
        lines.append(f"| cond | alpha | domain | value | n | 95% boot CI "
                     f"| 95% Wilson CI | ±half |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in rows:
            lines.append(
                f"| {r['condition']} | {r['alpha']} | {r['domain']} | "
                f"{r['value']:.2f} | {r['n']} | "
                f"[{r['boot_lo']:.2f}, {r['boot_hi']:.2f}] | "
                f"[{r['wilson_lo']:.2f}, {r['wilson_hi']:.2f}] | "
                f"{r['half_width']:.2f} |"
            )
        lines.append("")
    md_path.write_text("\n".join(lines))
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
