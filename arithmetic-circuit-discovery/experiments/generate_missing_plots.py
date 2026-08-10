#!/usr/bin/env python3
"""
Generate three missing paper plots from existing experimental data:
  1. Layer scan curves (full-patch transfer across layers, 3 models)
  2. Neuron-level Fourier analysis (MLP neuron purity/frequency distributions)
  3. Ablation curves (multi-layer frequency ablation, cumulative)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PLOT_DIR = Path("mathematical_toolkit_results/paper_plots")
PLOT_DIR.mkdir(exist_ok=True)

# ============================================================
# Color scheme
# ============================================================
MODEL_COLORS = {
    "Gemma 2B": "#F44336",
    "Phi-3 Mini": "#FF9800",
    "LLaMA 3.2-3B": "#2196F3",
}
FREQ_COLORS = {1: "#2196F3", 2: "#FF9800", 3: "#4CAF50", 4: "#9C27B0", 5: "#F44336"}
FREQ_LABELS = {1: "k=1 (ordinal)", 2: "k=2 (mod-5)", 3: "k=3", 4: "k=4", 5: "k=5 (parity)"}


# ============================================================
# PLOT 1: Layer Scan Curves (from log-extracted data)
# ============================================================
def plot_layer_scan_curves():
    """Full-patch transfer across layers for all 3 models."""

    # Gemma 2B: 26 layers (from gemma2b_layer_scan.log)
    gemma_layers = list(range(26))
    gemma_transfer = [
        5.2, 5.6, 5.6, 4.8, 4.8, 4.8, 5.2, 5.6, 5.6, 5.6,
        5.6, 5.9, 5.9, 5.6, 5.9, 7.4, 7.8, 7.8, 65.2, 65.6,
        72.6, 100.0, 100.0, 100.0, 100.0, 100.0,
    ]

    # Phi-3 Mini: 32 layers (from phi3_layer_scan_unembed.txt)
    phi3_layers = list(range(32))
    phi3_transfer = [
        1.1, 1.5, 1.5, 1.5, 1.5, 1.9, 0.7, 0.7, 0.7, 0.7,
        1.1, 0.7, 1.1, 0.7, 1.1, 0.7, 1.1, 1.1, 2.2, 3.0,
        26.3, 75.9, 75.9, 74.8, 75.6, 77.0, 77.4, 78.1, 100.0, 100.0,
        100.0, 100.0,
    ]

    # LLaMA 3.2-3B: 28 layers (from arithmetic_scan_llama3b_direct_mps.log)
    llama_layers = list(range(28))
    llama_transfer = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 15.9, 15.9, 98.5, 99.3, 98.9, 98.9,
        99.6, 99.6, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
    ]

    # Normalize to fractional depth for cross-model comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    # Panel A: Absolute layer numbers
    ax = axes[0]
    for name, layers, transfer in [
        ("Gemma 2B", gemma_layers, gemma_transfer),
        ("Phi-3 Mini", phi3_layers, phi3_transfer),
        ("LLaMA 3.2-3B", llama_layers, llama_transfer),
    ]:
        ax.plot(layers, transfer, "o-", color=MODEL_COLORS[name], label=name,
                linewidth=2, markersize=3, alpha=0.9)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Full-Patch Transfer (%)", fontsize=12)
    ax.set_title("(a) Absolute Layer Index", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(-5, 110)

    # Panel B: Normalized depth (0–1)
    ax = axes[1]
    for name, layers, transfer in [
        ("Gemma 2B", gemma_layers, gemma_transfer),
        ("Phi-3 Mini", phi3_layers, phi3_transfer),
        ("LLaMA 3.2-3B", llama_layers, llama_transfer),
    ]:
        n = len(layers)
        norm_depth = [l / (n - 1) for l in layers]
        ax.plot(norm_depth, transfer, "o-", color=MODEL_COLORS[name], label=name,
                linewidth=2, markersize=3, alpha=0.9)

    ax.set_xlabel("Normalized Depth (layer / total)", fontsize=12)
    ax.set_title("(b) Normalized Depth", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(-5, 110)

    # Annotate phase transitions
    ax_a = axes[0]
    ax_a.annotate("L18: 65%", xy=(18, 65.2), fontsize=8, color=MODEL_COLORS["Gemma 2B"],
                  xytext=(12, 78), arrowprops=dict(arrowstyle="->", color=MODEL_COLORS["Gemma 2B"], lw=1))
    ax_a.annotate("L21: 76%", xy=(21, 75.9), fontsize=8, color=MODEL_COLORS["Phi-3 Mini"],
                  xytext=(25, 60), arrowprops=dict(arrowstyle="->", color=MODEL_COLORS["Phi-3 Mini"], lw=1))
    ax_a.annotate("L16: 99%", xy=(16, 98.5), fontsize=8, color=MODEL_COLORS["LLaMA 3.2-3B"],
                  xytext=(8, 95), arrowprops=dict(arrowstyle="->", color=MODEL_COLORS["LLaMA 3.2-3B"], lw=1))

    fig.suptitle("Arithmetic Information Across Layers — Full-Patch Transfer",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = PLOT_DIR / "layer_scan_curves.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ============================================================
# PLOT 2: Neuron-level Fourier Analysis
# ============================================================
def plot_neuron_level_analysis():
    """MLP neuron frequency distribution and power concentration from neuron_trig logs."""

    # Data extracted from neuron_trig_*.log files
    models = {
        "Gemma 2B\n(L19, 9216 neurons)": {
            "d_mlp": 9216,
            "active": 9212,
            "high_purity": 809,  # >80%
            "moderate": 4694,    # 50-80%
            "freq_dist": {1: 4023, 2: 445, 3: 103, 4: 61, 5: 871},
            # Top 3 neurons: power, purity, freq
            "top_neurons": [
                (4838, 0.1973, 88.4, 1),
                (8205, 0.1534, 94.2, 5),
                (6745, 0.1429, 93.7, 5),
            ],
            "resid_high_purity": 179,
            "resid_moderate": 1223,
            "resid_d": 2304,
            "resid_freq": {1: 757, 2: 79, 3: 34, 4: 6, 5: 526},
        },
        "Phi-3 Mini\n(L26, 8192 neurons)": {
            "d_mlp": 8192,
            "active": 8192,
            "high_purity": 743,
            "moderate": 4145,
            "freq_dist": {1: 3492, 2: 501, 3: 92, 4: 54, 5: 749},
            "top_neurons": [
                (1034, 9.6309, 76.8, 1),
            ],
            "resid_high_purity": 275,
            "resid_moderate": 1666,
            "resid_d": 3072,
            "resid_freq": {1: 1320, 2: 197, 3: 13, 4: 5, 5: 406},
        },
        "LLaMA 3.2-3B\n(L20, 8192 neurons)": {
            "d_mlp": 8192,
            "active": 8192,
            "high_purity": 350,
            "moderate": 3679,
            "freq_dist": {1: 2157, 2: 969, 3: 116, 4: 85, 5: 702},
            "top_neurons": [],
            "resid_high_purity": 91,
            "resid_moderate": 1336,
            "resid_d": 3072,
            "resid_freq": {1: 657, 2: 411, 3: 104, 4: 39, 5: 216},
        },
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    model_names = list(models.keys())
    bar_colors = [MODEL_COLORS["Gemma 2B"], MODEL_COLORS["Phi-3 Mini"], MODEL_COLORS["LLaMA 3.2-3B"]]

    # Row 1: MLP neuron frequency distribution (stacked bar showing purity categories)
    for i, (name, data) in enumerate(models.items()):
        ax = axes[0, i]
        freqs = [1, 2, 3, 4, 5]
        counts = [data["freq_dist"][k] for k in freqs]
        colors = [FREQ_COLORS[k] for k in freqs]
        labels = [FREQ_LABELS[k] for k in freqs]

        bars = ax.bar(range(5), counts, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(5))
        ax.set_xticklabels([f"k={k}" for k in freqs], fontsize=10)
        ax.set_ylabel("# MLP Neurons (purity > 50%)" if i == 0 else "", fontsize=10)
        ax.set_title(name, fontsize=11, fontweight="bold", color=bar_colors[i])

        # Annotate bars with counts
        for bar, count in zip(bars, counts):
            if count > 50:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                        str(count), ha="center", va="bottom", fontsize=8, fontweight="bold")

        # Add purity stats
        total_pure = data["high_purity"] + data["moderate"]
        frac = total_pure / data["d_mlp"] * 100
        ax.text(0.98, 0.95,
                f"High purity (>80%): {data['high_purity']}\n"
                f"Moderate (50-80%): {data['moderate']}\n"
                f"Total tuned: {frac:.0f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Row 2: Residual stream dimension frequency distribution
    for i, (name, data) in enumerate(models.items()):
        ax = axes[1, i]
        freqs = [1, 2, 3, 4, 5]
        counts = [data["resid_freq"][k] for k in freqs]
        colors = [FREQ_COLORS[k] for k in freqs]

        bars = ax.bar(range(5), counts, color=colors, edgecolor="white", linewidth=0.5, alpha=0.7)
        ax.set_xticks(range(5))
        ax.set_xticklabels([f"k={k}" for k in freqs], fontsize=10)
        ax.set_ylabel("# Resid Dims (purity > 50%)" if i == 0 else "", fontsize=10)
        ax.set_xlabel("Dominant Frequency", fontsize=10)

        for bar, count in zip(bars, counts):
            if count > 20:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                        str(count), ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.text(0.98, 0.95,
                f"High purity (>80%): {data['resid_high_purity']}\n"
                f"Moderate: {data['resid_moderate']}\n"
                f"of {data['resid_d']} total dims",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    axes[0, 0].set_ylabel("# MLP Neurons\n(purity > 50%)", fontsize=11)
    axes[1, 0].set_ylabel("# Residual Dims\n(purity > 50%)", fontsize=11)

    fig.suptitle("Per-Neuron & Per-Dimension Fourier Frequency Tuning at Computation Layers",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = PLOT_DIR / "neuron_frequency_tuning.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ============================================================
# PLOT 3: Ablation Curves
# ============================================================
def plot_ablation_curves():
    """Multi-layer frequency ablation results for Gemma 2B (two layer ranges)."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- Panel A: Per-frequency ablation comparison ---
    ax = axes[0]
    freqs = [1, 2, 3, 4, 5]

    # L19-L25 data (from multilayer_freq_ablation_gemma2b.log)
    l19_25_acc = {1: 59.6, 2: 59.0, 3: 91.7, 4: 77.3, 5: 99.0}
    # L13-L19 data (from multilayer_freq_ablation_gemma2b_L13-L19.log)
    l13_19_acc = {1: 98.7, 2: 97.4, 3: 96.3, 4: 95.9, 5: 99.6}

    x = np.arange(5)
    width = 0.35

    bars1 = ax.bar(x - width / 2, [l13_19_acc[k] for k in freqs], width,
                   color=[FREQ_COLORS[k] for k in freqs], alpha=0.5,
                   edgecolor="black", linewidth=0.8, label="L13→L19 (early)")
    bars2 = ax.bar(x + width / 2, [l19_25_acc[k] for k in freqs], width,
                   color=[FREQ_COLORS[k] for k in freqs], alpha=0.9,
                   edgecolor="black", linewidth=0.8, label="L19→L25 (late)")

    ax.axhline(100, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.axhline(10, color="red", linestyle=":", alpha=0.5, linewidth=0.8, label="Chance (10%)")
    ax.set_xticks(x)
    ax.set_xticklabels([FREQ_LABELS[k] for k in freqs], fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("(a) Per-Frequency Ablation", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)

    # Annotate key finding
    ax.annotate("k=5: non-causal\nat BOTH ranges",
                xy=(4 + width / 2, 99.0), fontsize=8, color=FREQ_COLORS[5],
                xytext=(2.5, 50), fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=FREQ_COLORS[5], lw=1.5))

    # --- Panel B: Cumulative ablation (L19-L25 only, since L13-L19 uses different σ² order) ---
    ax = axes[1]
    # Order by σ² at L19: k=5 > k=3 > k=2 > k=4 > k=1
    cum_labels = ["k=5\n(1D)", "+k=3\n(3D)", "+k=2\n(5D)", "+k=4\n(7D)", "+k=1\n(9D)"]
    cum_acc_late = [99.0, 83.1, 30.8, 27.3, 14.0]

    x2 = np.arange(5)
    ax.plot(x2, cum_acc_late, "o-", color="#F44336", linewidth=2.5, markersize=8,
            label="L19→L25")

    ax.fill_between(x2, cum_acc_late, 10, alpha=0.1, color="#F44336")
    ax.axhline(100, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.axhline(10, color="red", linestyle=":", alpha=0.5, linewidth=0.8, label="Chance")

    ax.set_xticks(x2)
    ax.set_xticklabels(cum_labels, fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("(b) Cumulative Ablation (Gemma, L19→L25)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 110)
    ax.grid(alpha=0.3)

    # Annotate cliff
    ax.annotate("5D cliff:\n30.8%", xy=(2, 30.8), fontsize=9, fontweight="bold",
                color="#F44336", xytext=(3.2, 55),
                arrowprops=dict(arrowstyle="->", color="#F44336", lw=1.5))

    # --- Panel C: Cross-model knockout summary ---
    ax = axes[2]
    knockout_data = {
        "Gemma 2B": {
            "single_fourier": 88.7, "single_random": 100.0,
            "multi_fourier": 12.8, "multi_random": 100.0,
        },
        "Phi-3 Mini": {
            "single_fourier": 54.0, "single_random": 100.0,
            "multi_fourier": 12.1, "multi_random": 100.0,
        },
        "LLaMA 3.2-3B": {
            "single_fourier": 22.7, "single_random": 100.0,
            "multi_fourier": 18.8, "multi_random": 99.8,
        },
    }

    x3 = np.arange(3)
    width3 = 0.2
    model_names = list(knockout_data.keys())
    model_cols = [MODEL_COLORS[m] for m in model_names]

    # Four conditions
    conditions = [
        ("Single Fourier 9D", "single_fourier", -1.5),
        ("Single Random 9D", "single_random", -0.5),
        ("Multi Fourier 9D", "multi_fourier", 0.5),
        ("Multi Random 9D", "multi_random", 1.5),
    ]

    hatches = ["", "///", "", "///"]
    alphas = [0.9, 0.4, 0.9, 0.4]

    for ci, (cond_label, cond_key, offset) in enumerate(conditions):
        vals = [knockout_data[m][cond_key] for m in model_names]
        bars = ax.bar(x3 + offset * width3, vals, width3,
                      color=model_cols, alpha=alphas[ci],
                      edgecolor="black", linewidth=0.6,
                      hatch=hatches[ci] if hatches[ci] else None,
                      label=cond_label if ci == 0 or ci == 2 else None)

    ax.axhline(10, color="red", linestyle=":", alpha=0.5, linewidth=0.8)
    ax.set_xticks(x3)
    ax.set_xticklabels(model_names, fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("(c) Causal Knockout: Fourier vs Random", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="gray", alpha=0.9, edgecolor="black", label="Fourier 9D"),
        Patch(facecolor="gray", alpha=0.4, edgecolor="black", hatch="///", label="Random 9D (control)"),
    ]
    leg1 = ax.legend(handles=legend_elements, fontsize=8, loc="upper right", title="Subspace Type")

    # Second legend for single vs multi
    from matplotlib.lines import Line2D
    leg2_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markersize=8, label="Single-layer"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="dimgray", markersize=8, label="Multi-layer"),
    ]

    fig.suptitle("Causal Ablation: Fourier Subspace Is Necessary for Arithmetic",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = PLOT_DIR / "ablation_curves.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  GENERATING MISSING PAPER PLOTS")
    print("=" * 60)

    print("\n[1/3] Layer scan curves (3 models)...")
    plot_layer_scan_curves()

    print("\n[2/3] Neuron-level Fourier analysis...")
    plot_neuron_level_analysis()

    print("\n[3/3] Ablation curves...")
    plot_ablation_curves()

    print(f"\nAll plots saved to {PLOT_DIR}/")
