"""Build the two bespoke poster figures that aren't already in
`figures/` or `helix_usage_validated/`.

Run from repo root:
    .venv/bin/python paper_compass/poster/make_poster_figs.py

Outputs (paths relative to repo root):
    paper_compass/poster/images/pipeline_schematic.png
    paper_compass/poster/images/crowspairs_compare.png
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "images"
OUT.mkdir(parents=True, exist_ok=True)

ACCENT_A = "#2E5EAA"
ACCENT_B = "#C0392B"
MUTED = "#888888"
LIGHT = "#E4E4E4"


def pipeline_schematic():
    """Four-stage horizontal pipeline with arrows and captions."""
    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    boxes = [
        ("1.  OV-SVD", "per head",
         r"$W_{OV}=W_V W_O$" + "\n" + r"$=U\,\Sigma\,V^{\!\top}$"),
        ("2.  $\\alpha$-sweep", "+ 9 null tests",
         "discover (u_i, u_j)\nwhere LD swings\nsinusoidally"),
        ("3.  Route", "by pass-count",
         "K=4 heads\nper domain"),
        ("4.  Calibrate", "SNR-matched",
         r"$\alpha_d = \tau / \overline{\mathrm{SNR}}_d$"),
    ]

    box_w, box_h = 2.15, 2.0
    gap = 0.2
    y = 0.5
    positions = []
    for i, (title, sub, body) in enumerate(boxes):
        x = i * (box_w + gap) + 0.1
        positions.append(x + box_w / 2)
        rect = patches.FancyBboxPatch(
            (x, y), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.2, edgecolor=ACCENT_A,
            facecolor="white")
        ax.add_patch(rect)
        ax.text(x + box_w / 2, y + box_h - 0.28,
                title, ha="center", va="top",
                fontsize=13, fontweight="bold", color=ACCENT_A)
        ax.text(x + box_w / 2, y + box_h - 0.65,
                sub, ha="center", va="top",
                fontsize=9, color=MUTED, style="italic")
        ax.text(x + box_w / 2, y + box_h - 1.0,
                body, ha="center", va="top",
                fontsize=10.5, color="#222")

    for xa, xb in zip(positions[:-1], positions[1:]):
        ax.annotate("", xy=(xb - box_w / 2 - 0.02, y + box_h / 2),
                    xytext=(xa + box_w / 2 + 0.02, y + box_h / 2),
                    arrowprops=dict(arrowstyle="-|>",
                                    color=ACCENT_B, lw=2.5,
                                    shrinkA=0, shrinkB=0))

    fig.savefig(OUT / "pipeline_schematic.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'pipeline_schematic.png'}")


def crowspairs_compare():
    """Grouped bar chart: |SS-50| baseline vs INLP vs SentDebias vs
    routed+calib vs routed+matched, on gender+race, 4 models."""
    csv_path = Path("helix_usage_validated/debias_method_comparison.csv")
    rows = list(csv.DictReader(csv_path.open()))
    models = ["gpt2", "phi3", "gemma", "llama"]
    domains = ["gender", "race"]
    methods = [
        ("base_dist50",    "baseline",       "#BBBBBB"),
        ("inlp_dist50",    "INLP",           "#888888"),
        ("sent_dist50",    "SentDebias",     "#555555"),
        ("routed_dist50",  "routed+calib",   ACCENT_A),
        ("matched_dist50", "routed+matched", ACCENT_B),
    ]

    def getv(mdl, dom, col):
        for r in rows:
            if r["model"] == mdl and r["domain"] == dom:
                v = r[col]
                return float(v) if v not in ("", "nan") else np.nan
        return np.nan

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), sharey=True)
    w = 0.16
    x = np.arange(len(models))
    offsets = np.linspace(-2, 2, len(methods)) * w

    for ax, dom in zip(axes, domains):
        for (col, label, c), off in zip(methods, offsets):
            vals = [getv(m, dom, col) for m in models]
            ax.bar(x + off, vals, w, label=label, color=c,
                   edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in models])
        ax.set_title(f"CrowS-Pairs {dom}  —  |SS − 50|  (↓ better)",
                     fontsize=11)
        ax.set_ylim(0, 20)
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("distance from SS = 50")
    axes[0].legend(loc="upper left", frameon=False,
                   fontsize=9, ncol=1)

    fig.suptitle(
        "Routed + matched-strength compass beats INLP on 7/8 cells "
        "and SentenceDebias on 6/8",
        fontsize=11, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "crowspairs_compare.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'crowspairs_compare.png'}")


if __name__ == "__main__":
    pipeline_schematic()
    crowspairs_compare()
