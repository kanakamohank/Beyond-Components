"""Offline domain-routing analysis of StereoSet scans.

Reads helix_usage_validated/stereoset_scan_{tag}.jsonl and answers:

  Do the heads that passed the two-pillar test cluster by bias domain,
  or do they pass broadly across all four domains?

Outputs per model:
  helix_usage_validated/head_domain_heatmap_{tag}.png
  stdout summary: top-K heads, per-domain pass counts, specialization index

Also writes:
  helix_usage_validated/head_domain_routing.md -- cross-model table + verdict

Specialization index per head:
    normalize domain pass-count vector to sum 1, take max entry.
    1.0 = entirely one domain;  0.25 = perfectly spread across 4 domains.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("helix_usage_validated")
MODELS = ["gpt2", "gemma", "phi3", "llama"]
DOMAINS = ["gender", "race", "profession", "religion"]


def load_scan(tag):
    path = OUT_DIR / f"stereoset_scan_{tag}.jsonl"
    return [json.loads(l) for l in path.open()]


def head_domain_matrix(rows, top_k_heads=12):
    """Return (heads sorted by total passes, matrix of passes[h, domain])."""
    passes = defaultdict(lambda: Counter())  # (L,H) -> Counter(domain -> n)
    for r in rows:
        if r["passed"]:
            passes[(r["L"], r["H"])][r["domain"]] += 1
    totals = {lh: sum(c.values()) for lh, c in passes.items()}
    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:top_k_heads]
    heads = [lh for lh, _ in ranked]
    mat = np.zeros((len(heads), len(DOMAINS)), dtype=int)
    for i, lh in enumerate(heads):
        for j, d in enumerate(DOMAINS):
            mat[i, j] = passes[lh].get(d, 0)
    return heads, mat


def specialization_index(row):
    """row: 1D array of per-domain passes for one head."""
    s = row.sum()
    if s == 0:
        return float("nan")
    p = row / s
    return float(p.max())  # 1.0 = fully specialized, 0.25 = perfectly spread


def plot_heatmap(heads, mat, model_tag, title):
    if len(heads) == 0:
        print(f"  [{model_tag}] no passing heads; skipping heatmap")
        return
    fig, ax = plt.subplots(figsize=(4.5, 0.38 * len(heads) + 1.2))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(DOMAINS)))
    ax.set_xticklabels(DOMAINS, rotation=30, ha="right")
    ax.set_yticks(range(len(heads)))
    ax.set_yticklabels([f"L{L}H{H}" for L, H in heads])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center",
                        color="white" if v < mat.max() * 0.6 else "black",
                        fontsize=8)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, label="# passes")
    fig.tight_layout()
    out = OUT_DIR / f"head_domain_heatmap_{model_tag}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  [{model_tag}] wrote {out}")


def summarize_model(tag):
    rows = load_scan(tag)
    passing_rows = [r for r in rows if r["passed"]]
    print(f"\n=== {tag} ===")
    print(f"  total rows: {len(rows)}   passing (probe,plane) tests: "
          f"{len(passing_rows)}")

    heads, mat = head_domain_matrix(rows, top_k_heads=12)
    n_unique_heads = len(heads)
    print(f"  unique heads with >=1 pass: "
          f"{len({(r['L'], r['H']) for r in passing_rows})}  "
          f"(showing top {n_unique_heads})")

    if n_unique_heads == 0:
        return dict(tag=tag, n_heads=0, mean_spec=float("nan"),
                    heads=[], mat=mat.tolist())

    print(f"  {'head':>8}  {'gend':>4} {'race':>4} {'prof':>4} {'reli':>4}  "
          f"{'tot':>4} {'spec':>5}")
    specs = []
    for i, (L, H) in enumerate(heads):
        row = mat[i]
        spec = specialization_index(row)
        specs.append(spec)
        print(f"  L{L:>2}H{H:<2}     "
              f"{row[0]:>4} {row[1]:>4} {row[2]:>4} {row[3]:>4}  "
              f"{row.sum():>4} {spec:>5.2f}")

    mean_spec = float(np.nanmean(specs))
    # also the top-4 (which is what the ensemble actually uses)
    top4_spec = float(np.nanmean(specs[:4]))
    print(f"  mean specialization (top-{n_unique_heads}): {mean_spec:.2f}")
    print(f"  mean specialization (top-4 = ensemble):    {top4_spec:.2f}")
    verdict = "specialized" if top4_spec >= 0.55 else \
              "generic"     if top4_spec <= 0.35 else \
              "mixed"
    print(f"  -> top-4 verdict: {verdict}  "
          f"(>=0.55 routed, <=0.35 generic, else mixed)")

    plot_heatmap(heads, mat, tag,
                 f"{tag}: head x domain passes (top-{n_unique_heads})")
    return dict(tag=tag, n_heads=n_unique_heads,
                mean_spec=mean_spec, top4_spec=top4_spec,
                verdict=verdict,
                heads=[(L, H) for L, H in heads],
                mat=mat.tolist())


def main():
    OUT_DIR.mkdir(exist_ok=True)
    results = [summarize_model(tag) for tag in MODELS]

    md = ["# Head x domain routing analysis (StereoSet scans)",
          "",
          "Specialization index per head = max over domains of the "
          "normalized pass-count vector.",
          "  1.00 -> fully one-domain;  0.25 -> perfectly spread across "
          "4 domains.",
          "",
          "| Model | Unique heads (>=1 pass) | top-4 ensemble mean spec "
          "| Verdict |",
          "|---|---|---|---|"]
    for r in results:
        if r["n_heads"] == 0:
            md.append(f"| {r['tag']} | 0 | n/a | no passes |")
            continue
        md.append(f"| {r['tag']} | {r['n_heads']} | "
                  f"{r.get('top4_spec', float('nan')):.2f} | "
                  f"{r.get('verdict','?')} |")
    md += ["",
           "## Interpretation",
           "- **specialized (>= 0.55)**: heads pass mostly on one domain. "
           "A **routed ensemble** (pick heads per incoming bias_type) "
           "should reduce overshoot.",
           "- **generic (<= 0.35)**: heads pass across many domains. "
           "The ensemble is fine; overshoot is an alpha problem and "
           "per-domain alpha (with dev/test split) is the fix.",
           "- **mixed**: both effects matter; per-domain routing *and* "
           "per-domain alpha likely help.",
           "",
           "Per-model top-12 tables are in the stdout of this script; "
           "heatmaps are saved as head_domain_heatmap_{tag}.png."]
    (OUT_DIR / "head_domain_routing.md").write_text("\n".join(md))
    print("\nwrote", OUT_DIR / "head_domain_routing.md")


if __name__ == "__main__":
    main()
