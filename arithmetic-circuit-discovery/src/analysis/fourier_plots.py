"""
Visualization utilities for Fourier discovery results.

Provides publication-quality plots for:
    1. Power spectrum per layer (bar chart)
    2. Layer-wise power ratio heatmap
    3. Per-head power ratio grid (when head scan is available)
    4. Dominant frequency across layers (line plot)

All functions accept the result objects from ``FourierDiscovery`` and
return ``matplotlib.figure.Figure`` instances for flexible saving/display.

Usage::

    from src.analysis.fourier_plots import plot_all
    figs = plot_all(layer_results, head_results, output_dir="fourier_results")
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CI
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from src.analysis.fourier_discovery import FourierResult, LayerFourierResult

logger = logging.getLogger(__name__)

# Consistent style
COLORS = {
    "primary": "#2196F3",
    "accent": "#FF5722",
    "grid": "#E0E0E0",
    "bg": "#FAFAFA",
}


def plot_power_spectrum_by_layer(
    layer_results: List[LayerFourierResult],
    title: str = "Fourier Power Spectrum by Layer",
    top_n: int = 6,
) -> plt.Figure:
    """Bar chart of the power spectrum for the top-N layers by power ratio.

    Args:
        layer_results: Output from ``FourierDiscovery.run_all_layers``.
        title: Plot title.
        top_n: Number of top layers to plot.

    Returns:
        matplotlib Figure.
    """
    sorted_results = sorted(
        layer_results,
        key=lambda r: r.dominant_frequency_power_ratio,
        reverse=True,
    )[:top_n]

    n_plots = len(sorted_results)
    fig, axes = plt.subplots(
        n_plots, 1, figsize=(10, 2.5 * n_plots), sharex=False
    )
    if n_plots == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    for ax, lr in zip(axes, sorted_results):
        power = lr.resid_pre.power_spectrum
        n_freqs = len(power)
        freqs = np.arange(n_freqs)

        colors = [COLORS["primary"]] * n_freqs
        dom = lr.dominant_frequency
        if 0 <= dom < n_freqs:
            colors[dom] = COLORS["accent"]

        ax.bar(freqs, power, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_ylabel("Power", fontsize=9)
        ax.set_title(
            f"Layer {lr.layer}  (dom freq={dom}, "
            f"ratio={lr.dominant_frequency_power_ratio:.1f}x)",
            fontsize=10,
        )
        ax.set_xlabel("Frequency index k", fontsize=9)
        ax.grid(axis="y", alpha=0.3, color=COLORS["grid"])
        ax.set_facecolor(COLORS["bg"])

    fig.tight_layout()
    return fig


def plot_power_ratio_across_layers(
    layer_results: List[LayerFourierResult],
    title: str = "Dominant-Frequency Power Ratio by Layer",
) -> plt.Figure:
    """Line plot showing power ratio across all layers.

    Helps identify which layers have the strongest periodic structure.

    Args:
        layer_results: Output from ``FourierDiscovery.run_all_layers``.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    sorted_by_layer = sorted(layer_results, key=lambda r: r.layer)
    layers = [r.layer for r in sorted_by_layer]
    ratios = [r.dominant_frequency_power_ratio for r in sorted_by_layer]
    freqs = [r.dominant_frequency for r in sorted_by_layer]

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Power ratio (left axis)
    color1 = COLORS["primary"]
    ax1.plot(layers, ratios, "o-", color=color1, linewidth=2, markersize=6, label="Power ratio")
    ax1.set_xlabel("Layer", fontsize=11)
    ax1.set_ylabel("Power Ratio (signal / noise)", fontsize=11, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.grid(axis="both", alpha=0.3, color=COLORS["grid"])
    ax1.set_facecolor(COLORS["bg"])

    # Dominant frequency (right axis)
    ax2 = ax1.twinx()
    color2 = COLORS["accent"]
    ax2.plot(layers, freqs, "s--", color=color2, linewidth=1.5, markersize=5, alpha=0.7, label="Dom. frequency")
    ax2.set_ylabel("Dominant Frequency k", fontsize=11, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    fig.tight_layout()
    return fig


def plot_head_power_grid(
    head_results: Dict[Tuple[int, int], FourierResult],
    n_layers: int,
    n_heads: int,
    title: str = "Attention Head Fourier Power Ratios",
) -> plt.Figure:
    """Heatmap grid of power ratios for all (layer, head) pairs.

    Args:
        head_results: Output from ``FourierDiscovery.analyze_attention_heads``.
        n_layers: Total number of layers.
        n_heads: Total number of heads per layer.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    grid = np.zeros((n_layers, n_heads))
    for (layer, head), result in head_results.items():
        if layer < n_layers and head < n_heads:
            ratio = result.dominant_frequency_power_ratio
            grid[layer, head] = min(ratio, 50.0)  # Cap for color scale

    fig, ax = plt.subplots(figsize=(max(8, n_heads * 0.6), max(6, n_layers * 0.4)))

    cmap = plt.cm.YlOrRd
    im = ax.imshow(grid, aspect="auto", cmap=cmap, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, label="Power Ratio", shrink=0.8)

    ax.set_xlabel("Head", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Tick labels
    if n_heads <= 20:
        ax.set_xticks(range(n_heads))
    if n_layers <= 30:
        ax.set_yticks(range(n_layers))

    fig.tight_layout()
    return fig


def plot_dominant_frequency_histogram(
    layer_results: List[LayerFourierResult],
    title: str = "Distribution of Dominant Frequencies",
) -> plt.Figure:
    """Histogram of dominant frequencies across all layers.

    Args:
        layer_results: Output from ``FourierDiscovery.run_all_layers``.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    freqs = [r.dominant_frequency for r in layer_results]
    max_freq = max(freqs) if freqs else 1

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(-0.5, max_freq + 1.5, 1)
    ax.hist(freqs, bins=bins, color=COLORS["primary"], edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Dominant Frequency k", fontsize=11)
    ax.set_ylabel("Number of Layers", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(range(max_freq + 1))
    ax.grid(axis="y", alpha=0.3, color=COLORS["grid"])
    ax.set_facecolor(COLORS["bg"])

    fig.tight_layout()
    return fig


def plot_all(
    layer_results: List[LayerFourierResult],
    head_results: Optional[Dict[Tuple[int, int], FourierResult]] = None,
    n_layers: Optional[int] = None,
    n_heads: Optional[int] = None,
    output_dir: str = "fourier_results",
    model_key: str = "model",
    save: bool = True,
    show: bool = False,
) -> Dict[str, plt.Figure]:
    """Generate and optionally save all Fourier discovery plots.

    Args:
        layer_results: Output from ``FourierDiscovery.run_all_layers``.
        head_results: Output from ``FourierDiscovery.analyze_attention_heads``.
        n_layers: Total model layers (needed for head grid).
        n_heads: Total model heads (needed for head grid).
        output_dir: Directory for saving PNG files.
        model_key: Model name for file naming.
        save: Whether to save plots to disk.
        show: Whether to call plt.show() (for interactive use).

    Returns:
        Dict mapping plot names to Figure objects.
    """
    figs: Dict[str, plt.Figure] = {}

    if not layer_results:
        logger.warning("No layer results to plot.")
        return figs

    # 1. Power spectrum bar charts (top layers)
    figs["power_spectra"] = plot_power_spectrum_by_layer(
        layer_results, title=f"Fourier Power Spectra — {model_key}"
    )

    # 2. Power ratio across layers
    figs["power_ratio_line"] = plot_power_ratio_across_layers(
        layer_results, title=f"Power Ratio by Layer — {model_key}"
    )

    # 3. Dominant frequency histogram
    figs["freq_histogram"] = plot_dominant_frequency_histogram(
        layer_results, title=f"Dominant Frequency Distribution — {model_key}"
    )

    # 4. Head power grid (if head results available)
    if head_results and n_layers and n_heads:
        figs["head_grid"] = plot_head_power_grid(
            head_results, n_layers, n_heads,
            title=f"Head Power Ratios — {model_key}",
        )

    # Save
    if save:
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        for name, fig in figs.items():
            path = os.path.join(plots_dir, f"{model_key}_{name}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved plot: {path}")

    if show:
        plt.show()

    return figs
