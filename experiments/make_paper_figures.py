"""Build all paper figures from existing CSV/PNG outputs.

Reads from:
  helix_usage_validated/winogender_{gpt2,phi3,gemma}_sweep.csv
  helix_usage_validated/stereoset_{gpt2,phi3,gemma}_*.csv
  helix_usage_validated/{gpt2,phi3,gemma}_scan_gender*.png  (re-tiled)

Outputs:
  paper_compass/figures/fig_alpha_sweep.pdf        (main F1)
  paper_compass/figures/fig_scan_concentration.pdf (main F2)
  paper_compass/figures/fig_downstream_summary.pdf (main F3)
  paper_compass/figures/fig_head_specificity.pdf   (appendix A1)
  paper_compass/figures/fig_stereoset_domains.pdf  (appendix A2)

Usage:
  .venv/bin/python experiments/make_paper_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.image import imread

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C_BASELINE = "#555555"
C_COMPASS = "#0b7a75"
C_ACTADD = "#d97706"
C_COMPASS2 = "#84cc16"

DATA = Path("helix_usage_validated")
OUT = Path("paper_compass/figures")
OUT.mkdir(parents=True, exist_ok=True)


MODELS = [
    dict(key="gpt2", title="GPT-2 (124M) — L10 H9", compass_alpha_sweet=1.5),
    dict(key="phi3", title="Phi-3 (3.8B) — L28 H1", compass_alpha_sweet=1.5),
    dict(key="gemma", title="Gemma-2 (2B) — L21 H4", compass_alpha_sweet=1.5),
    dict(key="llama", title="Llama-3.2 (3B) — L26 H14 SV2", compass_alpha_sweet=1.5),
]


def load_sweep(model_key):
    return pd.read_csv(DATA / f"winogender_{model_key}_sweep.csv")


def load_stereoset(model_key):
    """Map key → actual filename."""
    fname = {
        "gpt2": "stereoset_gpt2_l10h9.csv",
        "phi3": "stereoset_phi3_l28h1.csv",
        "gemma": "stereoset_gemma_l21h4.csv",
        "llama": "stereoset_llama_l26h14.csv",
    }[model_key]
    return pd.read_csv(DATA / fname)


# ------------------------------------------------------------------
#  F1 — alpha sweep Pareto (main)
# ------------------------------------------------------------------
def fig_alpha_sweep():
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4), sharey=False)

    for ax, m in zip(axes, MODELS):
        df = load_sweep(m["key"])
        primary = df[
            (df["condition"] == "compass")
            & (df["layer"] == df[df["condition"] == "compass"]["layer"].iloc[0])
            & (df["svd"] == df[df["condition"] == "compass"]["svd"].iloc[0])
            & (df["head"] == df[df["condition"] == "compass"]["head"].iloc[0])
        ].sort_values("alpha")

        base = df[df["condition"] == "baseline"].iloc[0]
        aa = df[(df["condition"] == "actadd")
                & (df["layer"] == primary["layer"].iloc[0])].iloc[0]

        alphas_x = list(primary["alpha"])
        corr = list(primary["stereo_corr"])
        delta = list(primary["stereo_delta"])

        ax.plot([0] + alphas_x, [base["stereo_corr"]] + corr,
                marker="o", color=C_COMPASS, linewidth=2,
                label="compass · stereo_corr")
        ax.plot([0] + alphas_x, [base["stereo_delta"]] + delta,
                marker="s", color=C_COMPASS2, linewidth=1.5,
                linestyle="--", label="compass · stereo_delta")

        ax.scatter([1.0], [aa["stereo_corr"]], color=C_ACTADD,
                   marker="D", s=60, zorder=5, label="ActAdd (α=1.0)")
        ax.scatter([1.0], [aa["stereo_delta"]], color=C_ACTADD,
                   marker="*", s=80, zorder=5)

        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax.set_xlabel(r"compass strength $\alpha$")
        if ax is axes[0]:
            ax.set_ylabel("bias metric (↓ better)")
        ax.set_title(m["title"])
        ax.grid(True, alpha=0.3)

        # Twin axis for PPL Δ
        ax2 = ax.twinx()
        ppl = list(primary["ppl_delta"])
        ax2.fill_between([0] + alphas_x,
                         [0] + [0] * len(ppl),
                         [0] + ppl,
                         color=C_COMPASS, alpha=0.10)
        ax2.plot([0] + alphas_x, [0] + ppl,
                 color=C_COMPASS, alpha=0.35, linewidth=1,
                 label="compass · PPL Δ")
        ax2.scatter([1.0], [aa["ppl_delta"]], color=C_ACTADD,
                    marker="x", s=40, zorder=5)
        ax2.set_ylabel("PPL Δ vs baseline", color="#888", fontsize=8)
        ax2.tick_params(axis="y", colors="#888")
        ax2.set_ylim(bottom=-0.2)

        if ax is axes[0]:
            ax.legend(loc="upper right", framealpha=0.9)

    fig.suptitle(
        "WinoGender bias vs compass strength: compass Pareto-dominates ActAdd across 4 models",
        y=1.05, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_alpha_sweep.pdf")
    plt.close(fig)
    print(f"  wrote {OUT / 'fig_alpha_sweep.pdf'}")


# ------------------------------------------------------------------
#  F2 — scan concentration (reuse existing heatmaps)
# ------------------------------------------------------------------
def fig_scan_concentration():
    heatmaps = [
        ("gpt2", "GPT-2: 16/864 heads·planes pass (1.9%)"),
        ("phi3", "Phi-3: 14/1344 heads·planes pass (1.0%)"),
        ("gemma", "Gemma-2: 3/240 heads·planes pass (1.2%)"),
        ("llama", "Llama-3.2-3B: 9/4032 heads·planes pass (0.2%)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    for ax, (key, title) in zip(axes, heatmaps):
        img_path = DATA / f"{key}_scan_gender_heatmap.png"
        if not img_path.exists():
            ax.text(0.5, 0.5, f"missing\n{img_path.name}",
                    ha="center", va="center")
        else:
            ax.imshow(imread(img_path))
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(
        "Compass scan: heads × SV-planes that pass both amplitude and phase tests",
        y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_scan_concentration.pdf")
    plt.close(fig)
    print(f"  wrote {OUT / 'fig_scan_concentration.pdf'}")


# ------------------------------------------------------------------
#  F3 — downstream summary (main)
# ------------------------------------------------------------------
def fig_downstream_summary():
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.5), sharey="row")

    # Row 0: WinoGender stereo_corr
    for j, m in enumerate(MODELS):
        df = load_sweep(m["key"])
        base = df[df["condition"] == "baseline"].iloc[0]
        L_pri = df[df["condition"] == "compass"].iloc[0]["layer"]
        compass_15 = df[
            (df["condition"] == "compass") & (df["alpha"] == 1.5)
            & (df["layer"] == L_pri)
        ].iloc[0]
        actadd = df[(df["condition"] == "actadd") & (df["layer"] == L_pri)].iloc[0]

        ax = axes[0, j]
        vals = [base["stereo_corr"], compass_15["stereo_corr"],
                actadd["stereo_corr"]]
        colors = [C_BASELINE, C_COMPASS, C_ACTADD]
        labels = ["baseline", "compass α=1.5", "ActAdd α=1.0"]
        bars = ax.bar(labels, vals, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        for b, v, pd_v in zip(bars, vals,
                              [0.0, compass_15["ppl_delta"], actadd["ppl_delta"]]):
            ax.text(b.get_x() + b.get_width() / 2,
                    v + (0.03 if v >= 0 else -0.05),
                    f"{v:+.2f}\nΔPPL {pd_v:+.2f}",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
        ax.set_title(m["title"], fontsize=10)
        if j == 0:
            ax.set_ylabel("WinoGender stereo_corr\n(↓ better, 0 = unbiased)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(-0.25, 0.85)

    # Row 1: StereoSet Gender SS
    for j, m in enumerate(MODELS):
        df = load_stereoset(m["key"])
        gender = df[df["domain"] == "gender"]
        base = gender[gender["condition"] == "baseline"].iloc[0]
        c15 = gender[(gender["condition"] == "compass") & (gender["alpha"] == 1.5)].iloc[0]
        aa = gender[gender["condition"] == "actadd"].iloc[0]

        ax = axes[1, j]
        vals = [base["ss"], c15["ss"], aa["ss"]]
        bars = ax.bar(["baseline", "compass α=1.5", "ActAdd α=1.0"], vals,
                      color=[C_BASELINE, C_COMPASS, C_ACTADD])
        ax.axhline(50, color="black", linewidth=0.8, linestyle=":",
                   label="unbiased (50)")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.5,
                    f"{v:.1f}", ha="center", fontsize=8)
        if j == 0:
            ax.set_ylabel("StereoSet Gender SS\n(50 = unbiased)")
        ax.set_ylim(40, 85)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Downstream bias reduction: compass on WinoGender (token-level) and StereoSet (sentence-level)",
        y=1.00, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_downstream_summary.pdf")
    plt.close(fig)
    print(f"  wrote {OUT / 'fig_downstream_summary.pdf'}")


# ------------------------------------------------------------------
#  A1 — head specificity (appendix)
# ------------------------------------------------------------------
def fig_head_specificity():
    """Per model, show stereo_corr drop for primary vs secondary compass
    heads at α=1.0.  Only 2 models have multi-head data (GPT-2, Phi-3)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)

    for ax, m in zip(axes, [MODELS[0], MODELS[1]]):
        df = load_sweep(m["key"])
        base = df[df["condition"] == "baseline"].iloc[0]
        heads = df[
            (df["condition"] == "compass") & (df["alpha"] == 1.0)
        ].copy()
        heads = heads.drop_duplicates(subset=["layer", "head", "svd"])
        heads = heads.sort_values("stereo_corr")
        labels = [f"L{int(r['layer'])}H{int(r['head'])}\nSV{int(r['svd'])}"
                  for _, r in heads.iterrows()]
        vals = heads["stereo_corr"].values
        # baseline marker
        colors = []
        for _, r in heads.iterrows():
            # First (primary) head is the lowest stereo_corr → best debiaser
            colors.append(C_COMPASS)
        # Mark primary in darker green; others in lighter
        for i, c in enumerate(colors):
            if i == 0:
                colors[i] = C_COMPASS
            else:
                colors[i] = C_COMPASS2

        bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(base["stereo_corr"], color=C_BASELINE,
                   linestyle="--", label=f"baseline ({base['stereo_corr']:.2f})")
        ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01,
                    f"{v:+.2f}", ha="center", fontsize=8)
        ax.set_title(m["title"].split(" — ")[0])
        if ax is axes[0]:
            ax.set_ylabel("WinoGender stereo_corr at α=1.0")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Head specificity: only the primary compass head debiases (secondary compasses are compasses but not readouts)",
        y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_head_specificity.pdf")
    plt.close(fig)
    print(f"  wrote {OUT / 'fig_head_specificity.pdf'}")


# ------------------------------------------------------------------
#  A2 — StereoSet cross-domain (appendix)
# ------------------------------------------------------------------
def fig_stereoset_domains():
    domains = ["gender", "race", "profession"]
    fig, axes = plt.subplots(3, 4, figsize=(14, 8), sharey="row")

    for i, dom in enumerate(domains):
        for j, m in enumerate(MODELS):
            ax = axes[i, j]
            df = load_stereoset(m["key"])
            d = df[df["domain"] == dom]
            base = d[d["condition"] == "baseline"].iloc[0]
            c15 = d[(d["condition"] == "compass") & (d["alpha"] == 1.5)].iloc[0]
            aa = d[d["condition"] == "actadd"].iloc[0]

            labels = ["baseline", "compass α=1.5", "ActAdd α=1.0"]
            vals = [base["ss"], c15["ss"], aa["ss"]]
            colors = [C_BASELINE, C_COMPASS, C_ACTADD]

            bars = ax.bar(labels, vals, color=colors)
            ax.axhline(50, color="black", linewidth=0.8, linestyle=":")
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.4,
                        f"{v:.1f}", ha="center", fontsize=7)

            if i == 0:
                ax.set_title(m["title"].split(" — ")[0], fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{dom} SS")
            ax.set_ylim(45, 82)
            ax.grid(True, axis="y", alpha=0.3)
            ax.tick_params(axis="x", labelsize=7)

    fig.suptitle(
        "StereoSet across 3 domains: compass is gender-specific (works where designed, not a universal bias knob)",
        y=1.01, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_stereoset_domains.pdf")
    plt.close(fig)
    print(f"  wrote {OUT / 'fig_stereoset_domains.pdf'}")


def main():
    print("Building figures...")
    fig_alpha_sweep()
    fig_scan_concentration()
    fig_downstream_summary()
    fig_head_specificity()
    fig_stereoset_domains()
    print("Done.")


if __name__ == "__main__":
    main()
