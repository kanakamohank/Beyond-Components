#!/usr/bin/env python3
"""
Plot Experiment 6: DFT of eigenvector digit scores.

Generates 4 plots:
  1. Digit score waveforms with best-fit Fourier mode overlays (computation layer)
  2. DFT power heatmap per direction (computation vs readout)
  3. Singular value bar chart colored by dominant frequency
  4. Cross-model comparison panel

Usage:
    python plot_eigenvector_dft.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

RESULTS_DIR = Path("mathematical_toolkit_results")
PLOT_DIR = RESULTS_DIR / "paper_plots"
PLOT_DIR.mkdir(exist_ok=True, parents=True)

# Color palette for frequencies
FREQ_COLORS = {
    1: "#2196F3",  # blue  — ordinal
    2: "#FF9800",  # orange — mod-5
    3: "#4CAF50",  # green
    4: "#9C27B0",  # purple
    5: "#F44336",  # red   — parity
}
FREQ_LABELS = {
    1: "k=1 (ordinal)",
    2: "k=2 (mod-5)",
    3: "k=3",
    4: "k=4",
    5: "k=5 (parity)",
}
FREQ_SHORT = {1: "k=1", 2: "k=2", 3: "k=3", 4: "k=4", 5: "k=5"}


def load_data():
    """Load both model results."""
    models = {}
    for name in ["gemma-2b", "phi-3"]:
        path = RESULTS_DIR / f"eigenvector_dft_{name}.json"
        if path.exists():
            with open(path) as f:
                models[name] = json.load(f)
    return models


def fourier_mode(k, N=10):
    """Return the cos and sin basis functions for frequency k."""
    d = np.arange(N)
    cos_k = np.cos(2 * np.pi * k * d / N)
    sin_k = np.sin(2 * np.pi * k * d / N)
    return cos_k, sin_k


def best_fit_fourier(scores, k, N=10):
    """Project digit scores onto cos(k) and sin(k), return the best-fit waveform."""
    scores = np.array(scores)
    scores = scores - scores.mean()  # center
    cos_k, sin_k = fourier_mode(k, N)
    a = np.dot(scores, cos_k) / np.dot(cos_k, cos_k)
    b = np.dot(scores, sin_k) / np.dot(sin_k, sin_k) if k != 5 else 0.0
    # High-res for smooth curve
    d_fine = np.linspace(0, N - 1, 200)
    fit_fine = a * np.cos(2 * np.pi * k * d_fine / N)
    if k != 5:
        fit_fine += b * np.sin(2 * np.pi * k * d_fine / N)
    fit_pts = a * cos_k + b * sin_k
    return fit_fine, d_fine, fit_pts


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1: Waveform panel — digit scores with Fourier overlay
# ═══════════════════════════════════════════════════════════════════════════════
def plot_waveforms(data, model_label, layer_key, layer_label, filename):
    """3×3 grid of digit score waveforms with best-fit Fourier mode overlay."""
    directions = data[layer_key]["directions"]
    n_dirs = min(9, len(directions))

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), dpi=150)
    fig.suptitle(
        f"{model_label} — Eigenvector Digit Scores vs Fourier Modes\n{layer_label}",
        fontsize=15, fontweight="bold", y=0.98,
    )

    digits = np.arange(10)

    for i, ax in enumerate(axes.flat):
        if i >= n_dirs:
            ax.set_visible(False)
            continue

        d = directions[i]
        scores = np.array(d["digit_scores"])
        scores_c = scores - scores.mean()
        dom_k = d["dominant_freq"]
        dom_pct = d["dominant_pct"]
        sv = d["singular_value"]
        color = FREQ_COLORS[dom_k]

        # Plot actual digit scores
        ax.bar(digits, scores_c, color=color, alpha=0.35, width=0.7, zorder=2)
        ax.scatter(digits, scores_c, color=color, s=40, zorder=4, edgecolors="white", linewidths=0.5)

        # Overlay best-fit Fourier mode
        fit_fine, d_fine, fit_pts = best_fit_fourier(scores, dom_k)
        ax.plot(d_fine, fit_fine, color=color, linewidth=2, alpha=0.8, zorder=3)

        # Labels
        ax.set_title(
            f"Dir {i+1}  (σ={sv:.1f})\n{FREQ_SHORT[dom_k]} = {dom_pct:.0f}%",
            fontsize=10, color=color, fontweight="bold",
        )
        ax.set_xticks(digits)
        ax.set_xticklabels([str(d) for d in digits], fontsize=8)
        ax.axhline(0, color="gray", linewidth=0.5, zorder=1)
        ax.set_xlim(-0.5, 9.5)

        if i >= 6:
            ax.set_xlabel("Digit", fontsize=9)
        if i % 3 == 0:
            ax.set_ylabel("Score (centered)", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = PLOT_DIR / filename
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2: DFT power heatmap — computation vs readout
# ═══════════════════════════════════════════════════════════════════════════════
def plot_dft_heatmap(data, model_label, filename):
    """Side-by-side heatmaps: computation layer vs readout layer DFT power."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    fig.suptitle(
        f"{model_label} — DFT Power Spectrum of SVD Directions",
        fontsize=14, fontweight="bold",
    )

    sections = [
        ("computation", f"Computation L{data['comp_layer']} (activations)"),
        ("readout_activations", f"Readout L{data['readout_layer']} (activations)"),
        ("readout", "Readout (W_U)"),
    ]

    for ax_idx, (key, label) in enumerate(sections):
        ax = axes[ax_idx]
        directions = data[key]["directions"]
        n = min(9, len(directions))

        # Build matrix (n_dirs × 5 frequencies)
        mat = np.zeros((n, 5))
        for i in range(n):
            fracs = directions[i]["freq_fractions"]
            for k in range(1, 6):
                mat[i, k - 1] = fracs[str(k)]

        im = ax.imshow(mat.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1,
                       interpolation="nearest")
        ax.set_yticks(range(5))
        ax.set_yticklabels([FREQ_SHORT[k] for k in range(1, 6)], fontsize=10)
        ax.set_xticks(range(n))
        ax.set_xticklabels([f"D{i+1}" for i in range(n)], fontsize=9)
        ax.set_xlabel("SVD Direction", fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")

        # Annotate cells with percentage
        for i in range(n):
            for k in range(5):
                val = mat[i, k]
                color = "white" if val > 0.5 else "black"
                ax.text(i, k, f"{val*100:.0f}", ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")

        # Mark dominant frequency with star
        for i in range(n):
            dom_k = directions[i]["dominant_freq"]
            ax.text(i, dom_k - 1, "★", ha="center", va="center",
                    fontsize=12, color="yellow" if mat[i, dom_k-1] > 0.5 else "gold")

    fig.colorbar(im, ax=axes, shrink=0.8, label="Fraction of DFT power")
    plt.tight_layout(rect=[0, 0, 0.92, 0.93])
    out = PLOT_DIR / filename
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3: Singular value bars colored by dominant frequency
# ═══════════════════════════════════════════════════════════════════════════════
def plot_sv_bars(data, model_label, filename):
    """Bar chart of singular values, colored by dominant Fourier frequency."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)
    fig.suptitle(
        f"{model_label} — Singular Values Colored by Dominant Fourier Frequency",
        fontsize=14, fontweight="bold",
    )

    sections = [
        ("computation", f"Computation L{data['comp_layer']}"),
        ("readout_activations", f"Readout L{data['readout_layer']} (act)"),
        ("readout", "Readout (W_U)"),
    ]

    for ax_idx, (key, label) in enumerate(sections):
        ax = axes[ax_idx]
        directions = data[key]["directions"]
        n = min(9, len(directions))

        svs = [directions[i]["singular_value"] for i in range(n)]
        dom_ks = [directions[i]["dominant_freq"] for i in range(n)]
        colors = [FREQ_COLORS[k] for k in dom_ks]
        purities = [directions[i]["dominant_pct"] for i in range(n)]

        bars = ax.bar(range(n), svs, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5)

        # Annotate with purity
        for i, (bar, pct) in enumerate(zip(bars, purities)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(svs) * 0.02,
                    f"{pct:.0f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")

        ax.set_xticks(range(n))
        ax.set_xticklabels([f"D{i+1}" for i in range(n)], fontsize=9)
        ax.set_ylabel("Singular Value (σ)", fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=FREQ_COLORS[k], label=FREQ_LABELS[k]) for k in range(1, 6)]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=9,
              bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    out = PLOT_DIR / filename
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 4: Cross-model comparison — computation layer waveforms side by side
# ═══════════════════════════════════════════════════════════════════════════════
def plot_cross_model_waveforms(models):
    """Side-by-side comparison of computation layer eigenvectors for both models."""
    fig = plt.figure(figsize=(18, 12), dpi=150)
    fig.suptitle(
        "The Fourier Basis of Digit Arithmetic\n"
        "SVD directions of per-digit mean activations at computation layers",
        fontsize=16, fontweight="bold", y=0.99,
    )

    outer = gridspec.GridSpec(1, 2, wspace=0.15, hspace=0.1,
                              left=0.04, right=0.96, top=0.92, bottom=0.06)

    model_configs = [
        ("gemma-2b", "Gemma 2B — L19", "computation"),
        ("phi-3", "Phi-3 Mini — L26", "computation"),
    ]

    digits = np.arange(10)

    for col, (model_key, title, layer_key) in enumerate(model_configs):
        if model_key not in models:
            continue
        data = models[model_key]
        directions = data[layer_key]["directions"]
        n_dirs = min(9, len(directions))

        inner = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=outer[col],
                                                 wspace=0.3, hspace=0.5)

        # Sort directions by frequency for cleaner presentation
        sorted_dirs = sorted(directions[:n_dirs], key=lambda d: (d["dominant_freq"], -d["dominant_pct"]))

        for i, d in enumerate(sorted_dirs):
            ax = fig.add_subplot(inner[i // 3, i % 3])

            scores = np.array(d["digit_scores"])
            scores_c = scores - scores.mean()
            dom_k = d["dominant_freq"]
            dom_pct = d["dominant_pct"]
            sv = d["singular_value"]
            color = FREQ_COLORS[dom_k]

            # Bars + scatter
            ax.bar(digits, scores_c, color=color, alpha=0.3, width=0.7, zorder=2)
            ax.scatter(digits, scores_c, color=color, s=25, zorder=4,
                      edgecolors="white", linewidths=0.3)

            # Fourier fit overlay
            fit_fine, d_fine, _ = best_fit_fourier(scores, dom_k)
            ax.plot(d_fine, fit_fine, color=color, linewidth=1.5, alpha=0.8, zorder=3)

            ax.set_title(f"{FREQ_SHORT[dom_k]} ({dom_pct:.0f}%)  σ={sv:.0f}",
                        fontsize=8, color=color, fontweight="bold")
            ax.set_xticks(digits)
            ax.set_xticklabels([str(d) for d in digits], fontsize=6)
            ax.axhline(0, color="gray", linewidth=0.4, zorder=1)
            ax.set_xlim(-0.5, 9.5)
            ax.tick_params(axis="y", labelsize=6)

            if i >= 6:
                ax.set_xlabel("Digit", fontsize=7)

        # Model title
        fig.text(0.04 + col * 0.5 + 0.22, 0.935, title,
                fontsize=13, fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # Legend at bottom
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=FREQ_COLORS[k], label=FREQ_LABELS[k]) for k in range(1, 6)]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=10,
              bbox_to_anchor=(0.5, 0.0))

    out = PLOT_DIR / "eigenvector_fourier_cross_model.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 5: Pure Fourier modes reference + actual directions comparison
# ═══════════════════════════════════════════════════════════════════════════════
def plot_fourier_reference(models):
    """Show the theoretical Fourier basis alongside the actual learned directions."""
    fig = plt.figure(figsize=(18, 14), dpi=150)
    fig.suptitle(
        "Theoretical Fourier Basis of ℤ/10ℤ  vs  Learned SVD Directions\n"
        "Models learn near-exact trigonometric modes for digit arithmetic",
        fontsize=15, fontweight="bold", y=0.99,
    )

    outer = gridspec.GridSpec(2, 1, hspace=0.25, top=0.93, bottom=0.05,
                              left=0.04, right=0.96)

    digits = np.arange(10)
    d_fine = np.linspace(0, 9, 200)

    # ── Top row: Theoretical Fourier basis ──
    inner_theory = gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec=outer[0],
                                                     wspace=0.3, hspace=0.4)

    fig.text(0.5, 0.935, "Theoretical Fourier Basis (cos & sin at each frequency)",
            fontsize=12, ha="center", fontstyle="italic")

    for k in range(1, 6):
        color = FREQ_COLORS[k]
        cos_k, sin_k = fourier_mode(k)
        cos_fine = np.cos(2 * np.pi * k * d_fine / 10)
        sin_fine = np.sin(2 * np.pi * k * d_fine / 10)

        # Cos
        ax_cos = fig.add_subplot(inner_theory[0, k - 1])
        ax_cos.bar(digits, cos_k, color=color, alpha=0.3, width=0.7)
        ax_cos.scatter(digits, cos_k, color=color, s=25, zorder=3, edgecolors="white", linewidths=0.3)
        ax_cos.plot(d_fine, cos_fine, color=color, linewidth=1.5, alpha=0.8)
        ax_cos.set_title(f"cos(2π·{k}·d/10)", fontsize=8, color=color, fontweight="bold")
        ax_cos.axhline(0, color="gray", linewidth=0.4)
        ax_cos.set_xlim(-0.5, 9.5)
        ax_cos.set_xticks(digits)
        ax_cos.set_xticklabels([str(d) for d in digits], fontsize=5)
        ax_cos.tick_params(axis="y", labelsize=5)

        # Sin (skip k=5 Nyquist — sin is identically 0)
        ax_sin = fig.add_subplot(inner_theory[1, k - 1])
        if k < 5:
            ax_sin.bar(digits, sin_k, color=color, alpha=0.3, width=0.7)
            ax_sin.scatter(digits, sin_k, color=color, s=25, zorder=3, edgecolors="white", linewidths=0.3)
            ax_sin.plot(d_fine, sin_fine, color=color, linewidth=1.5, alpha=0.8)
            ax_sin.set_title(f"sin(2π·{k}·d/10)", fontsize=8, color=color, fontweight="bold")
        else:
            ax_sin.text(0.5, 0.5, "k=5: only 1 DOF\n(Nyquist = (-1)ᵈ)",
                       ha="center", va="center", fontsize=8, color=color,
                       transform=ax_sin.transAxes, fontstyle="italic")
            ax_sin.set_title("(no sin component)", fontsize=8, color="gray")
        ax_sin.axhline(0, color="gray", linewidth=0.4)
        ax_sin.set_xlim(-0.5, 9.5)
        ax_sin.set_xticks(digits)
        ax_sin.set_xticklabels([str(d) for d in digits], fontsize=5)
        ax_sin.tick_params(axis="y", labelsize=5)

    # ── Bottom row: Actual learned directions from Gemma L19 ──
    model_key = "gemma-2b"
    if model_key in models:
        data = models[model_key]
        directions = data["computation"]["directions"]

        inner_actual = gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec=outer[1],
                                                        wspace=0.3, hspace=0.4)

        fig.text(0.5, 0.47, "Gemma 2B L19 — Actual Learned SVD Directions (sorted by frequency)",
                fontsize=12, ha="center", fontstyle="italic")

        # Sort by frequency then by purity descending
        sorted_dirs = sorted(directions[:9],
                            key=lambda d: (d["dominant_freq"], -d["dominant_pct"]))

        # Group by frequency: cos-like first, sin-like second
        by_freq = {}
        for d in sorted_dirs:
            k = d["dominant_freq"]
            by_freq.setdefault(k, []).append(d)

        for k in range(1, 6):
            color = FREQ_COLORS[k]
            dirs_k = by_freq.get(k, [])

            for row_idx, d in enumerate(dirs_k[:2]):
                ax = fig.add_subplot(inner_actual[row_idx, k - 1])
                scores = np.array(d["digit_scores"])
                scores_c = scores - scores.mean()
                fit_fine, df, _ = best_fit_fourier(scores, k)

                ax.bar(digits, scores_c, color=color, alpha=0.3, width=0.7)
                ax.scatter(digits, scores_c, color=color, s=25, zorder=3,
                          edgecolors="white", linewidths=0.3)
                ax.plot(df, fit_fine, color=color, linewidth=1.5, alpha=0.8)
                pct = d["dominant_pct"]
                sv = d["singular_value"]
                role = "cos-like" if row_idx == 0 else "sin-like"
                ax.set_title(f"Dir (σ={sv:.0f}) {pct:.0f}%\n{role}",
                            fontsize=7, color=color, fontweight="bold")
                ax.axhline(0, color="gray", linewidth=0.4)
                ax.set_xlim(-0.5, 9.5)
                ax.set_xticks(digits)
                ax.set_xticklabels([str(d) for d in digits], fontsize=5)
                ax.tick_params(axis="y", labelsize=5)

            # If k=5 has only 1 direction, blank the second
            if k == 5 and len(dirs_k) < 2:
                ax_blank = fig.add_subplot(inner_actual[1, k - 1])
                ax_blank.text(0.5, 0.5, "k=5: 1 DOF only",
                             ha="center", va="center", fontsize=8,
                             color=color, transform=ax_blank.transAxes, fontstyle="italic")
                ax_blank.axhline(0, color="gray", linewidth=0.4)
                ax_blank.set_xlim(-0.5, 9.5)
                ax_blank.set_xticks(digits)
                ax_blank.set_xticklabels([str(d) for d in digits], fontsize=5)

    out = PLOT_DIR / "eigenvector_fourier_theory_vs_learned.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 6: Purity evolution — computation → readout
# ═══════════════════════════════════════════════════════════════════════════════
def plot_purity_evolution(models):
    """Show how direction purity changes from computation to readout for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    fig.suptitle(
        "Fourier Purity of SVD Directions: Computation → Readout",
        fontsize=14, fontweight="bold",
    )

    for ax_idx, (model_key, model_label) in enumerate([
        ("gemma-2b", "Gemma 2B"),
        ("phi-3", "Phi-3 Mini"),
    ]):
        ax = axes[ax_idx]
        if model_key not in models:
            continue
        data = models[model_key]

        for section_key, section_label, marker, ls in [
            ("computation", f"Computation L{data['comp_layer']}", "o", "-"),
            ("readout_activations", f"Readout L{data['readout_layer']} (act)", "s", "--"),
            ("readout", "Readout (W_U)", "^", ":"),
        ]:
            directions = data[section_key]["directions"]
            purities = [d["dominant_pct"] for d in directions[:9]]
            dom_freqs = [d["dominant_freq"] for d in directions[:9]]
            colors = [FREQ_COLORS[k] for k in dom_freqs]

            ax.scatter(range(1, len(purities) + 1), purities, c=colors,
                      s=60, marker=marker, zorder=3, edgecolors="white", linewidths=0.5)
            ax.plot(range(1, len(purities) + 1), purities, ls=ls, color="gray",
                   alpha=0.4, label=section_label)

        ax.set_xlabel("SVD Direction", fontsize=10)
        ax.set_ylabel("Dominant Freq Purity (%)", fontsize=10)
        ax.set_title(model_label, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.axhline(22.2, color="gray", linewidth=0.8, linestyle=":", alpha=0.5, label="Null (2-DOF)")
        ax.axhline(11.1, color="gray", linewidth=0.8, linestyle="-.", alpha=0.5, label="Null (1-DOF)")
        ax.legend(fontsize=7, loc="lower left")
        ax.set_xticks(range(1, 10))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = PLOT_DIR / "eigenvector_purity_evolution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Loading data...")
    models = load_data()
    print(f"  Loaded: {list(models.keys())}")

    for model_key, model_label in [("gemma-2b", "Gemma 2B"), ("phi-3", "Phi-3 Mini")]:
        if model_key not in models:
            continue
        data = models[model_key]

        print(f"\nGenerating plots for {model_label}...")

        # Plot 1: Waveforms at computation layer
        plot_waveforms(
            data, model_label,
            "computation",
            f"Computation Layer L{data['comp_layer']} — PERFECT FOURIER BASIS"
            if data["computation"]["perfect_fourier"]
            else f"Computation Layer L{data['comp_layer']}",
            f"eigvec_waveforms_{model_key}_comp.png",
        )

        # Plot 1b: Waveforms at W_U
        plot_waveforms(
            data, model_label,
            "readout",
            f"Readout (Unembed W_U SVD)"
            + (" — PERFECT FOURIER BASIS" if data["readout"]["perfect_fourier"] else ""),
            f"eigvec_waveforms_{model_key}_wu.png",
        )

        # Plot 2: DFT heatmap
        plot_dft_heatmap(data, model_label, f"eigvec_dft_heatmap_{model_key}.png")

        # Plot 3: SV bars
        plot_sv_bars(data, model_label, f"eigvec_sv_bars_{model_key}.png")

    # Cross-model plots
    if len(models) >= 2:
        print("\nGenerating cross-model plots...")
        plot_cross_model_waveforms(models)
        plot_fourier_reference(models)
        plot_purity_evolution(models)

    print("\n✓ All plots generated!")
