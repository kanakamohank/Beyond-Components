#!/usr/bin/env python3
"""
Generate paper-quality plots from Fourier decomposition and layer scan results.

Plots:
  1. Fourier spectrum evolution heatmap (Gemma + Phi-3 side by side)
  2. Fourier frequency line plots across layers
  3. Layer scan full-patch transfer curves
  4. Fisher patching dimension sweeps
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("mathematical_toolkit_results")
PLOT_DIR = RESULTS_DIR / "paper_plots"
PLOT_DIR.mkdir(exist_ok=True)

FREQ_LABELS = {1: "k=1\n(ordinal)", 2: "k=2\n(mod-5)", 3: "k=3", 4: "k=4", 5: "k=5\n(parity)"}
FREQ_COLORS = {1: "#2196F3", 2: "#FF9800", 3: "#9C27B0", 4: "#4CAF50", 5: "#F44336"}


def load_fourier_sweep(path):
    """Load Fourier sweep JSON and extract layer→freq_fractions."""
    with open(path) as f:
        data = json.load(f)
    layers = []
    fractions = {k: [] for k in range(1, 6)}
    for layer_key in sorted(data["results"].keys(), key=int):
        result = data["results"][layer_key]
        layers.append(result["layer"])
        for k in range(1, 6):
            fractions[k].append(result["freq_fractions"][str(k)])
    return layers, fractions, data.get("model", "unknown")


def load_fourier_sweep_merged(*paths):
    """Load and merge multiple Fourier sweep JSONs, deduplicating by layer."""
    all_results = {}
    model = "unknown"
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        model = data.get("model", model)
        for layer_key, result in data["results"].items():
            layer = result["layer"]
            if layer not in all_results:
                all_results[layer] = result
    layers = []
    fractions = {k: [] for k in range(1, 6)}
    energies = []
    high_purity = []
    mean_purity = []
    for layer in sorted(all_results.keys()):
        result = all_results[layer]
        layers.append(layer)
        energies.append(result["total_energy"])
        high_purity.append(result["per_neuron"]["high_purity_count"])
        mean_purity.append(result["per_neuron"]["mean_purity"])
        for k in range(1, 6):
            fractions[k].append(result["freq_fractions"][str(k)])
    return layers, fractions, energies, high_purity, mean_purity, model


def plot_fourier_heatmap_sidebyside():
    """Plot 1: Side-by-side Fourier spectrum heatmaps for Gemma and Phi-3."""
    gemma_paths = [
        RESULTS_DIR / "fourier_decomposition_gemma_2b_L5-L14.json",
        RESULTS_DIR / "fourier_decomposition_gemma_2b_L15-L25.json",
    ]
    phi3_paths = [
        RESULTS_DIR / "fourier_decomposition_phi_3_L5-L18.json",
        RESULTS_DIR / "fourier_decomposition_phi_3_L19-L31.json",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    for ax, paths, title in [
        (axes[0], gemma_paths, "Gemma 2B (26 layers)"),
        (axes[1], phi3_paths, "Phi-3 Mini (32 layers)"),
    ]:
        layers, fracs, _, _, _, model = load_fourier_sweep_merged(*paths)
        # Build matrix: rows = frequencies (k=1..5), cols = layers
        matrix = np.array([fracs[k] for k in range(1, 6)])  # (5, n_layers)

        im = ax.imshow(
            matrix * 100,
            aspect="auto",
            cmap="YlOrRd",
            vmin=0,
            vmax=75,
            interpolation="nearest",
        )
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([f"L{l}" for l in layers], fontsize=9)
        ax.set_yticks(range(5))
        ax.set_yticklabels(["k=1\n(ordinal)", "k=2\n(mod-5)", "k=3", "k=4", "k=5\n(parity)"], fontsize=9)
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")

        # Annotate cells with percentage
        for i in range(5):
            for j in range(len(layers)):
                val = matrix[i, j] * 100
                color = "white" if val > 40 else "black"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

    axes[0].set_ylabel("Fourier Frequency", fontsize=11)
    fig.suptitle("Fourier Energy Spectrum Across Layers", fontsize=14, fontweight="bold", y=1.02)
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, label="% of total energy")

    plt.tight_layout()
    out = PLOT_DIR / "fourier_heatmap_cross_model.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_fourier_lines():
    """Plot 2: Line plots of frequency fractions across layers (both models)."""
    gemma_paths = [
        RESULTS_DIR / "fourier_decomposition_gemma_2b_L5-L14.json",
        RESULTS_DIR / "fourier_decomposition_gemma_2b_L15-L25.json",
    ]
    phi3_paths = [
        RESULTS_DIR / "fourier_decomposition_phi_3_L5-L18.json",
        RESULTS_DIR / "fourier_decomposition_phi_3_L19-L31.json",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for ax, paths, title in [
        (axes[0], gemma_paths, "Gemma 2B"),
        (axes[1], phi3_paths, "Phi-3 Mini"),
    ]:
        layers, fracs, _, _, _, _ = load_fourier_sweep_merged(*paths)

        for k in range(1, 6):
            vals = [v * 100 for v in fracs[k]]
            label = FREQ_LABELS[k].replace("\n", " ")
            ax.plot(layers, vals, "o-", color=FREQ_COLORS[k], label=label,
                    linewidth=2, markersize=6)

        # Uniform baseline
        ax.axhline(20, color="gray", linestyle="--", alpha=0.5, label="Uniform (20%)")

        ax.set_xlabel("Layer", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim(0, 80)
        ax.grid(alpha=0.3)
        ax.set_xticks(layers)
        ax.set_xticklabels([f"L{l}" for l in layers], fontsize=9)

    axes[0].set_ylabel("Fourier Energy Fraction (%)", fontsize=11)
    fig.suptitle("Fourier Frequency Evolution Across Layers", fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    out = PLOT_DIR / "fourier_lines_cross_model.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_fourier_stacked_area():
    """Plot 3: Stacked area charts showing frequency composition across layers."""
    gemma_paths = [
        RESULTS_DIR / "fourier_decomposition_gemma_2b_L5-L14.json",
        RESULTS_DIR / "fourier_decomposition_gemma_2b_L15-L25.json",
    ]
    phi3_paths = [
        RESULTS_DIR / "fourier_decomposition_phi_3_L5-L18.json",
        RESULTS_DIR / "fourier_decomposition_phi_3_L19-L31.json",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, paths, title in [
        (axes[0], gemma_paths, "Gemma 2B"),
        (axes[1], phi3_paths, "Phi-3 Mini"),
    ]:
        layers, fracs, _, _, _, _ = load_fourier_sweep_merged(*paths)
        x = range(len(layers))

        # Stack order: k=5 (bottom), k=4, k=3, k=2, k=1 (top)
        stack_order = [5, 4, 3, 2, 1]
        stack_data = np.array([[fracs[k][i] * 100 for i in range(len(layers))] for k in stack_order])

        colors = [FREQ_COLORS[k] for k in stack_order]
        labels = [FREQ_LABELS[k].replace("\n", " ") for k in stack_order]

        ax.stackplot(x, stack_data, colors=colors, labels=labels, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{l}" for l in layers], fontsize=9)
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.2, axis="y")

    axes[0].set_ylabel("Fourier Energy (%)", fontsize=11)
    fig.suptitle("Fourier Decomposition — Stacked Frequency Composition",
                 fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    out = PLOT_DIR / "fourier_stacked_cross_model.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_layer_scan_curves():
    """Plot 4: Full-patch transfer across layers for all models."""
    # Load layer scan data from JSON files
    scan_files = {
        "LLaMA 3.2-3B": RESULTS_DIR / "arithmetic_scan_llama_3b.json",
        "Phi-3 Mini": RESULTS_DIR / "phi3_layer_scan_unembed_v2.json",
        "Gemma 2B": RESULTS_DIR / "arithmetic_scan_gemma_2b.json",
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    model_colors = {"LLaMA 3.2-3B": "#2196F3", "Phi-3 Mini": "#FF9800", "Gemma 2B": "#F44336"}

    for model_name, path in scan_files.items():
        if not path.exists():
            print(f"  Skipping {model_name}: {path} not found")
            continue

        with open(path) as f:
            data = json.load(f)

        # Extract layer scan results
        layers = []
        transfers = []
        for key in sorted(data.keys()):
            if key.startswith("layer_scan_L"):
                layer = int(key.split("_L")[1])
                layers.append(layer)
                transfers.append(data[key].get("full_patch_transfer", 0) * 100)

        if layers:
            ax.plot(layers, transfers, "o-", color=model_colors[model_name],
                    label=model_name, linewidth=2, markersize=4)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Full-Patch Transfer (%)", fontsize=12)
    ax.set_title("Arithmetic Information Across Layers", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(-5, 110)

    plt.tight_layout()
    out = PLOT_DIR / "layer_scan_curves.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_fisher_dimension_sweep():
    """Plot 5: Fisher patching transfer vs subspace dimension."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Standard Fisher
    ax = axes[0]
    std_data = {
        "LLaMA L27": {"dims": [2, 5, 10, 20, 50], "vals": [3.2, 35.8, 100, 100, 100]},
        "Phi-3 L28": {"dims": [2, 5, 10, 20, 50], "vals": [2.2, 17.8, 85.2, 99.6, 100]},
        "Gemma L21": {"dims": [2, 5, 10, 20, 50], "vals": [4.1, 20.7, 85.2, 96.3, 99.3]},
    }
    colors = {"LLaMA L27": "#2196F3", "Phi-3 L28": "#FF9800", "Gemma L21": "#F44336"}

    for label, d in std_data.items():
        ax.plot(d["dims"], d["vals"], "o-", color=colors[label], label=label,
                linewidth=2, markersize=6)

    ax.set_xlabel("Subspace Dimension", fontsize=11)
    ax.set_ylabel("Transfer (% of full-patch)", fontsize=11)
    ax.set_title("Standard Fisher Patching", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(-5, 110)

    # Contrastive Fisher
    ax = axes[1]
    con_data = {
        "LLaMA L27": {"dims": [2, 5, 9], "vals": [8.7, 56.4, 100]},
        "Phi-3 L28": {"dims": [2, 5, 9], "vals": [5.9, 34.1, 93.7]},
        "Gemma L21": {"dims": [2, 5, 9], "vals": [5.9, 29.3, 82.6]},
    }

    for label, d in con_data.items():
        ax.plot(d["dims"], d["vals"], "s--", color=colors[label], label=label,
                linewidth=2, markersize=6)

    ax.set_xlabel("Subspace Dimension", fontsize=11)
    ax.set_title("Contrastive Fisher Patching", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(-5, 110)

    fig.suptitle("Subspace Dimension vs Transfer Rate", fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    out = PLOT_DIR / "fisher_dimension_sweep.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_crt_score_comparison():
    """Plot 6: CRT score (k=2 + k=5 energy) across layers for both models."""
    gemma_paths = [
        RESULTS_DIR / "fourier_decomposition_gemma_2b_L5-L14.json",
        RESULTS_DIR / "fourier_decomposition_gemma_2b_L15-L25.json",
    ]
    phi3_paths = [
        RESULTS_DIR / "fourier_decomposition_phi_3_L5-L18.json",
        RESULTS_DIR / "fourier_decomposition_phi_3_L19-L31.json",
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    for paths, label, color, marker in [
        (gemma_paths, "Gemma 2B", "#F44336", "o"),
        (phi3_paths, "Phi-3 Mini", "#FF9800", "s"),
    ]:
        layers, fracs, _, _, _, _ = load_fourier_sweep_merged(*paths)
        crt = [(fracs[2][i] + fracs[5][i]) * 100 for i in range(len(layers))]
        k2 = [fracs[2][i] * 100 for i in range(len(layers))]
        k5 = [fracs[5][i] * 100 for i in range(len(layers))]

        ax.plot(layers, crt, f"{marker}-", color=color, label=f"{label} CRT (k=2+k=5)",
                linewidth=2.5, markersize=7)
        ax.plot(layers, k5, f"{marker}--", color=color, label=f"{label} k=5 (parity)",
                linewidth=1, markersize=4, alpha=0.6)
        ax.plot(layers, k2, f"{marker}:", color=color, label=f"{label} k=2 (mod-5)",
                linewidth=1, markersize=4, alpha=0.6)

    ax.axhline(20 + 20, color="gray", linestyle="--", alpha=0.4, label="Uniform CRT (40%)")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Energy Fraction (%)", fontsize=12)
    ax.set_title("CRT-Relevant Frequencies Across Layers", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 85)

    plt.tight_layout()
    out = PLOT_DIR / "crt_score_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_neuron_purity_across_layers():
    """Plot 7: Per-neuron Fourier purity statistics across layers."""
    gemma_paths = [
        RESULTS_DIR / "fourier_decomposition_gemma_2b_L5-L14.json",
        RESULTS_DIR / "fourier_decomposition_gemma_2b_L15-L25.json",
    ]
    phi3_paths = [
        RESULTS_DIR / "fourier_decomposition_phi_3_L5-L18.json",
        RESULTS_DIR / "fourier_decomposition_phi_3_L19-L31.json",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, paths, title in [
        (axes[0], gemma_paths, "Gemma 2B"),
        (axes[1], phi3_paths, "Phi-3 Mini"),
    ]:
        layers, _, _, hp, mp, _ = load_fourier_sweep_merged(*paths)

        mean_purity = mp
        high_frac = [h / (2304 if "gemma" in str(paths[0]) else 3072) * 100 for h in hp]

        ax2 = ax.twinx()
        ax.bar(range(len(layers)), mean_purity, color="#2196F3", alpha=0.6, label="Mean purity")
        ax2.plot(range(len(layers)), high_frac, "ro-", label="High purity (>0.8) %", linewidth=2)

        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([f"L{l}" for l in layers], fontsize=9)
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel("Mean Purity", fontsize=11, color="#2196F3")
        ax2.set_ylabel("% Neurons with Purity > 0.8", fontsize=10, color="red")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axhline(0.2, color="gray", linestyle="--", alpha=0.4)
        ax.set_ylim(0, 1.0)
        ax2.set_ylim(0, max(high_frac) * 1.3 if high_frac else 10)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    fig.suptitle("Per-Neuron Fourier Purity Across Layers", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = PLOT_DIR / "neuron_purity_across_layers.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_energy_explosion():
    """Plot 8: Total Fourier energy across layers showing the computation explosion."""
    gemma_paths = [
        RESULTS_DIR / "fourier_decomposition_gemma_2b_L5-L14.json",
        RESULTS_DIR / "fourier_decomposition_gemma_2b_L15-L25.json",
    ]
    phi3_paths = [
        RESULTS_DIR / "fourier_decomposition_phi_3_L5-L18.json",
        RESULTS_DIR / "fourier_decomposition_phi_3_L19-L31.json",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, paths, title, color in [
        (axes[0], gemma_paths, "Gemma 2B", "#F44336"),
        (axes[1], phi3_paths, "Phi-3 Mini", "#FF9800"),
    ]:
        layers, _, energies, hp, _, _ = load_fourier_sweep_merged(*paths)

        ax.semilogy(layers, energies, "o-", color=color, linewidth=2.5, markersize=7,
                    label="Total energy")

        ax2 = ax.twinx()
        ax2.bar(layers, hp, width=0.8, color="#2196F3", alpha=0.4, label="High-purity neurons")
        ax2.set_ylabel("High-Purity Neuron Count", fontsize=10, color="#2196F3")

        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel("Total Fourier Energy (log scale)", fontsize=11, color=color)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(layers)
        ax.set_xticklabels([f"L{l}" for l in layers], fontsize=8, rotation=45)
        ax.grid(alpha=0.3)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    fig.suptitle("Energy Explosion: Active Computation Amplifies Arithmetic Representation",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = PLOT_DIR / "energy_explosion.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("=" * 60)
    print("  GENERATING PAPER PLOTS")
    print("=" * 60)

    print("\n[1/7] Fourier heatmap (cross-model)...")
    plot_fourier_heatmap_sidebyside()

    print("[2/7] Fourier line plots...")
    plot_fourier_lines()

    print("[3/7] Fourier stacked area...")
    plot_fourier_stacked_area()

    print("[4/7] Layer scan curves...")
    plot_layer_scan_curves()

    print("[5/7] Fisher dimension sweep...")
    plot_fisher_dimension_sweep()

    print("[6/7] CRT score comparison...")
    plot_crt_score_comparison()

    print("[7/7] Neuron purity across layers...")
    plot_neuron_purity_across_layers()

    print("\n[8/8] Energy explosion...")
    plot_energy_explosion()

    print(f"\nAll plots saved to {PLOT_DIR}/")
