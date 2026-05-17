"""Per-domain flip-ratio summary for the qualitative spot-check CSVs.

Reads helix_usage_validated/qualitative_spotcheck_{model}.csv for all four
models and, for every (model, domain) cell, reports:

    n                     rows scored in that cell
    mean_debias_shift     nats; >0 = intervention moves mass away from
                          the stereotype on average
    flip_anti             FLIP -> anti-stereo count
    flip_stereo           FLIP -> stereo (regression) count
    flip_anti_rate        flip_anti  / n
    flip_stereo_rate      flip_stereo / n
    net_flip              flip_anti - flip_stereo
    flip_ratio            flip_anti / max(1, flip_stereo)
    chi2                  chi-square on {flip_anti, flip_stereo} vs 50/50
    chi2_p                two-sided p

If one domain shows disproportionately many FLIP -> stereo events (or a
tiny flip ratio), the intervention is likely catching a generic lexical
direction, not the stereotype-specific signal. That is the main
methodological risk we want to surface.

Outputs:
    helix_usage_validated/flip_ratio_by_domain.csv
    helix_usage_validated/flip_ratio_by_domain.md
"""
from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

OUT_DIR = Path("helix_usage_validated")
MODELS = ["gpt2", "phi3", "gemma", "llama"]
DOMAINS = ["gender", "race", "profession", "religion"]


def chi2_one_df(k1, k2):
    """Two-cell chi-square against 50/50 null, df=1.

    Returns (chi2, p_two_sided). Uses survival function of chi2(1) via the
    erfc trick: p = erfc(sqrt(chi2/2)).
    """
    n = k1 + k2
    if n == 0:
        return (0.0, 1.0)
    expected = n / 2.0
    chi2 = ((k1 - expected) ** 2 + (k2 - expected) ** 2) / expected
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return (chi2, p)


def main():
    rows_by_cell = defaultdict(list)
    for m in MODELS:
        path = OUT_DIR / f"qualitative_spotcheck_{m}.csv"
        if not path.exists():
            print(f"  missing: {path}")
            continue
        with path.open() as f:
            for r in csv.DictReader(f):
                rows_by_cell[(m, r["domain"])].append(r)

    out_rows = []
    for m in MODELS:
        for d in DOMAINS:
            cell = rows_by_cell.get((m, d), [])
            n = len(cell)
            if n == 0:
                continue
            verdicts = Counter(r["verdict"] for r in cell)
            flip_anti = verdicts.get("FLIP -> anti-stereo", 0)
            flip_stereo = verdicts.get("FLIP -> stereo (regression)", 0)
            debias = [float(r["debias_shift"]) for r in cell]
            mean_debias = sum(debias) / n
            chi2, p = chi2_one_df(flip_anti, flip_stereo)
            out_rows.append(dict(
                model=m, domain=d, n=n,
                mean_debias_shift=mean_debias,
                flip_anti=flip_anti,
                flip_stereo=flip_stereo,
                flip_anti_rate=flip_anti / n,
                flip_stereo_rate=flip_stereo / n,
                net_flip=flip_anti - flip_stereo,
                flip_ratio=(flip_anti / flip_stereo) if flip_stereo > 0
                           else float("inf") if flip_anti > 0 else 0.0,
                chi2=chi2,
                chi2_p=p,
            ))

    csv_cols = ["model", "domain", "n", "mean_debias_shift",
                "flip_anti", "flip_stereo",
                "flip_anti_rate", "flip_stereo_rate",
                "net_flip", "flip_ratio", "chi2", "chi2_p"]
    csv_path = OUT_DIR / "flip_ratio_by_domain.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"wrote {csv_path}  ({len(out_rows)} rows)")

    md_lines = [
        "# Flip-ratio summary per (model, domain)",
        "",
        "`mean_debias_shift` is in nats and measures `logprob(anti-stereo) "
        "- logprob(stereo)` shift under the routed + per-domain-alpha "
        "intervention relative to baseline. Positive values mean the "
        "intervention on average pushes probability mass off the "
        "stereotype on that row.",
        "",
        "`flip_anti` / `flip_stereo` count rows whose preferred sentence "
        "*changed sign* under the intervention. `chi2 / chi2_p` tests "
        "the null that flips are 50/50 (df=1, asymptotic).",
        "",
    ]
    for m in MODELS:
        cells = [r for r in out_rows if r["model"] == m]
        if not cells:
            continue
        md_lines.append(f"## {m}")
        md_lines.append("")
        md_lines.append(
            "| domain | n | mean Δ (nats) | flip→anti | flip→stereo "
            "| flip→anti rate | flip→stereo rate | net flip | "
            "flip ratio | χ²(1) | p |"
        )
        md_lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in cells:
            fr = (f"{r['flip_ratio']:.2f}" if math.isfinite(r["flip_ratio"])
                  else "∞")
            md_lines.append(
                f"| {r['domain']} | {r['n']} | "
                f"{r['mean_debias_shift']:+.3f} | "
                f"{r['flip_anti']} | {r['flip_stereo']} | "
                f"{r['flip_anti_rate']:.3f} | "
                f"{r['flip_stereo_rate']:.3f} | "
                f"{r['net_flip']:+d} | {fr} | "
                f"{r['chi2']:.2f} | {r['chi2_p']:.3g} |"
            )
        md_lines.append("")

    md_lines += [
        "## Interpretation guide",
        "",
        "- `net_flip > 0` and `flip_ratio > 1` at p < 0.05 = intervention "
        "is directionally debiasing on that (model, domain) cell.",
        "- A single domain with `flip_ratio < 1` while other domains are "
        "positive suggests the intervention is catching a non-stereotype "
        "lexical confound in that domain (e.g. the Phi-3 profession "
        "top-regressions were driven by name swaps, not poor/rich).",
        "- `mean_debias_shift` aggregates *all* row-level shifts including "
        "non-flips, so it can be positive even when flip counts are "
        "balanced; disagreement between mean shift and flip ratio is "
        "itself informative.",
    ]
    md_path = OUT_DIR / "flip_ratio_by_domain.md"
    md_path.write_text("\n".join(md_lines))
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
