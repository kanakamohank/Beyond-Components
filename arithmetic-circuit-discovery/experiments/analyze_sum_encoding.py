#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 4c: SVD Direction Sum-Encoding Analysis.

Tests whether SVD direction projections encode cos/sin(f·(a+b)) — the key
prediction from Nanda et al.'s grokking theory.

Phase 4a found directions encode periodic functions of SINGLE operands.
Phase 4b found MLP L23 NEURONS do NOT individually implement sin(f·(a+b)).

This script tests whether the SVD DIRECTIONS (linear combinations of neurons)
encode cos/sin(f·(a+b)) as a 2-D function of (a, b).  If true, the activations
along a direction should show diagonal stripes in the (a, b) heatmap, because
a+b = const defines diagonal lines.

For cos/sin pairs at the same frequency, we also test whether operand values
trace circles: (proj_cos(n), proj_sin(n)) → circle for varying n.

Usage:
    python experiments/analyze_sum_encoding.py \\
        --config configs/arithmetic_pythia_config.yaml \\
        --checkpoint svd_logs/.../model_final.pt \\
        --phase3_results phase3_results/svd_direction_fourier_....json \\
        [--operand_range_end 30] \\
        [--output_dir phase4c_results]
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.arithmetic_dataset import ArithmeticPromptGenerator
from src.models.masked_transformer_circuit import MaskedTransformerCircuit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DirectionSumResult:
    """Result of 2-D cos/sin(a+b) fit for a single SVD direction.

    Attributes:
        layer, head, component, direction_idx: Direction identity.
        effective_strength: σ_k × mask.
        dominant_frequency: From Phase 3 (1-D DFT).
        sum_r2_best: Best R² for cos/sin(f·(a+b)/N) fit across frequencies.
        sum_freq_best: Frequency f with best sum R².
        sum_r2_multi: R² fitting multiple sum frequencies simultaneously.
        sum_freqs_multi: Frequencies used in multi-sum fit.
        sum_r2_at_dominant: R² for cos/sin at the Phase 3 dominant frequency.
        separability_score: How well proj(a,b) ≈ f(a)·g(b). 1.0 = rank-1.
        diag_power_ratio: 2-D DFT power along diagonal (f,f) vs off-diagonal.
    """
    layer: int
    head: Optional[int]
    component: str
    direction_idx: int
    effective_strength: float
    dominant_frequency: int
    sum_r2_best: float
    sum_freq_best: int
    sum_r2_multi: float
    sum_freqs_multi: List[int]
    sum_r2_at_dominant: float
    separability_score: float
    diag_power_ratio: float


@dataclass
class CirclePairResult:
    """Result of circle test for a cos/sin direction pair.

    If two directions at the same frequency form a cos/sin pair,
    projecting operand values onto both should trace a circle.

    Attributes:
        frequency: Shared frequency.
        dir1_label, dir2_label: Direction labels.
        phase_diff_deg: Phase difference between the two directions.
        circle_r2: R² of fitting points to a circle.
        mean_radius: Mean radius of the (proj1, proj2) points.
        radius_std: Std of radii (low → good circle).
        circularity: 1 - (radius_std / mean_radius). 1.0 = perfect circle.
    """
    frequency: int
    dir1_label: str
    dir2_label: str
    phase_diff_deg: float
    circle_r2: float
    mean_radius: float
    radius_std: float
    circularity: float


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def fit_sum_cosine_sine(
    projections_2d: np.ndarray,
    frequency: int,
) -> float:
    """Fit proj(a,b) = A·cos(2πf·(a+b)/N) + B·sin(2πf·(a+b)/N) + C.

    Args:
        projections_2d: Shape (N_a, N_b) — scalar projections over (a, b) grid.
        frequency: The frequency f to test.

    Returns:
        R² of the fit.
    """
    N_a, N_b = projections_2d.shape
    N = max(N_a, N_b)
    a_grid = np.arange(N_a, dtype=np.float64)
    b_grid = np.arange(N_b, dtype=np.float64)
    A_mesh, B_mesh = np.meshgrid(a_grid, b_grid, indexing="ij")
    ab_sum = A_mesh + B_mesh

    theta = 2.0 * np.pi * frequency * ab_sum / N
    flat = projections_2d.flatten()

    X = np.column_stack([
        np.cos(theta).flatten(),
        np.sin(theta).flatten(),
        np.ones(N_a * N_b),
    ])
    coeffs, _, _, _ = np.linalg.lstsq(X, flat, rcond=None)
    fitted = X @ coeffs

    ss_res = np.sum((flat - fitted) ** 2)
    ss_tot = np.sum((flat - np.mean(flat)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def fit_sum_multi_frequency(
    projections_2d: np.ndarray,
    frequencies: List[int],
) -> float:
    """Fit proj(a,b) to multiple cos/sin(f·(a+b)/N) simultaneously.

    Args:
        projections_2d: Shape (N_a, N_b).
        frequencies: List of frequency indices.

    Returns:
        R² of the multi-frequency fit.
    """
    if not frequencies:
        return 0.0

    N_a, N_b = projections_2d.shape
    N = max(N_a, N_b)
    a_grid = np.arange(N_a, dtype=np.float64)
    b_grid = np.arange(N_b, dtype=np.float64)
    A_mesh, B_mesh = np.meshgrid(a_grid, b_grid, indexing="ij")
    ab_sum = A_mesh + B_mesh

    cols = []
    for f in frequencies:
        theta = 2.0 * np.pi * f * ab_sum / N
        cols.append(np.cos(theta).flatten())
        cols.append(np.sin(theta).flatten())
    cols.append(np.ones(N_a * N_b))

    X = np.column_stack(cols)
    flat = projections_2d.flatten()
    coeffs, _, _, _ = np.linalg.lstsq(X, flat, rcond=None)
    fitted = X @ coeffs

    ss_res = np.sum((flat - fitted) ** 2)
    ss_tot = np.sum((flat - np.mean(flat)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def compute_diagonal_power(projections_2d: np.ndarray) -> float:
    """Compute ratio of 2-D DFT power on diagonal (f,f) vs off-diagonal.

    If proj(a,b) depends on a+b, the 2-D DFT concentrates on the
    diagonal where f_a = f_b (because e^{i·f·(a+b)} = e^{i·f·a} · e^{i·f·b}).

    Returns:
        Power ratio: sum of diagonal power / mean of off-diagonal power.
    """
    dft_2d = np.fft.fft2(projections_2d)
    power_2d = np.abs(dft_2d) ** 2
    power_2d[0, 0] = 0.0  # remove DC

    N = min(projections_2d.shape)
    n_half = N // 2 + 1

    # Diagonal power: (f, f) for f = 1..n_half-1
    diag_power = sum(power_2d[f, f] for f in range(1, n_half))

    # Off-diagonal power
    off_diag = []
    for fa in range(n_half):
        for fb in range(n_half):
            if fa == fb or (fa == 0 and fb == 0):
                continue
            off_diag.append(power_2d[fa, fb])

    mean_off = float(np.mean(off_diag)) if off_diag else 1e-12
    return float(diag_power) / mean_off if mean_off > 1e-12 else 0.0


def compute_separability(projections_2d: np.ndarray) -> float:
    """How well proj(a,b) ≈ f(a)·g(b) (rank-1 approximation).

    Returns:
        Fraction of variance explained by rank-1 SVD.
    """
    centered = projections_2d - np.mean(projections_2d)
    _, S, _ = np.linalg.svd(centered)
    total_var = np.sum(S ** 2)
    return float(S[0] ** 2 / total_var) if total_var > 1e-12 else 0.0


def analyze_direction_sum(
    projections_2d: np.ndarray,
    layer: int,
    head: Optional[int],
    component: str,
    direction_idx: int,
    effective_strength: float,
    dominant_frequency: int,
    max_freq: int = 15,
) -> DirectionSumResult:
    """Full 2-D sum-encoding analysis for one SVD direction.

    Args:
        projections_2d: Shape (N_a, N_b) — projections onto Vh[k,:].
        layer, head, component, direction_idx: Direction identity.
        effective_strength: σ_k × mask.
        dominant_frequency: From Phase 3.
        max_freq: Maximum frequency to search.

    Returns:
        DirectionSumResult.
    """
    N = max(projections_2d.shape)

    # Sweep frequencies for best single-freq sum R²
    best_r2 = -1.0
    best_f = 0
    freq_r2s = {}
    for f in range(1, min(max_freq + 1, N // 2)):
        r2 = fit_sum_cosine_sine(projections_2d, f)
        freq_r2s[f] = r2
        if r2 > best_r2:
            best_r2 = r2
            best_f = f

    # R² at dominant frequency
    r2_at_dom = freq_r2s.get(dominant_frequency, 0.0)

    # Find top secondary sum frequencies for multi-freq fit
    sorted_freqs = sorted(freq_r2s.items(), key=lambda x: -x[1])
    top_freqs = [f for f, _ in sorted_freqs[:5] if freq_r2s[f] > 0.01]
    r2_multi = fit_sum_multi_frequency(projections_2d, top_freqs) if top_freqs else 0.0

    # Diagonal power ratio
    diag_ratio = compute_diagonal_power(projections_2d)

    # Separability
    sep_score = compute_separability(projections_2d)

    return DirectionSumResult(
        layer=layer,
        head=head,
        component=component,
        direction_idx=direction_idx,
        effective_strength=effective_strength,
        dominant_frequency=dominant_frequency,
        sum_r2_best=float(best_r2),
        sum_freq_best=best_f,
        sum_r2_multi=float(r2_multi),
        sum_freqs_multi=top_freqs,
        sum_r2_at_dominant=float(r2_at_dom),
        separability_score=float(sep_score),
        diag_power_ratio=float(diag_ratio),
    )


# ---------------------------------------------------------------------------
# Circle test for cos/sin pairs
# ---------------------------------------------------------------------------

def test_circle(
    proj1: np.ndarray,
    proj2: np.ndarray,
    frequency: int,
    label1: str,
    label2: str,
    phase_diff_deg: float,
) -> CirclePairResult:
    """Test whether (proj1[n], proj2[n]) traces a circle for n = 0..N-1.

    Args:
        proj1, proj2: 1-D arrays of length N (projections for operand values).
        frequency: Shared frequency.
        label1, label2: Direction labels.
        phase_diff_deg: Phase difference from Phase 4a.

    Returns:
        CirclePairResult.
    """
    # Center the projections
    c1 = proj1 - np.mean(proj1)
    c2 = proj2 - np.mean(proj2)

    # Compute radii from center
    radii = np.sqrt(c1 ** 2 + c2 ** 2)
    mean_r = float(np.mean(radii))
    std_r = float(np.std(radii))

    # Circularity: perfect circle has std_r / mean_r → 0
    circularity = 1.0 - std_r / mean_r if mean_r > 1e-12 else 0.0

    # R² of fitting to circle: fraction of variance explained by constant radius
    ss_res = np.sum((radii - mean_r) ** 2)
    ss_tot = np.sum((radii - np.mean(radii)) ** 2)
    circle_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0  # constant radius = perfect

    return CirclePairResult(
        frequency=frequency,
        dir1_label=label1,
        dir2_label=label2,
        phase_diff_deg=phase_diff_deg,
        circle_r2=float(circle_r2),
        mean_radius=mean_r,
        radius_std=std_r,
        circularity=float(circularity),
    )


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------

def collect_residual_activations_2d(
    model,
    prompt_gen: ArithmeticPromptGenerator,
    operand_range: range,
    layers: List[int],
    device: torch.device,
    batch_size: int = 16,
) -> Dict[str, np.ndarray]:
    """Collect residual stream activations for all (a, b) pairs.

    Returns:
        Dict mapping hook_name → array of shape (N_a, N_b, d_model).
    """
    op_list = list(operand_range)
    N = len(op_list)

    # Build flat list of prompts in (a, b) order
    flat_prompts = []
    for a in op_list:
        for b in op_list:
            sample = prompt_gen.get_by_operands(a, b)
            if sample is None:
                raise ValueError(f"Missing prompt for ({a}, {b})")
            flat_prompts.append(sample.prompt)

    total = len(flat_prompts)
    hooks = [f"blocks.{l}.hook_resid_pre" for l in layers]

    all_activations: Dict[str, List[np.ndarray]] = {h: [] for h in hooks}

    for start in tqdm(range(0, total, batch_size), desc="Collecting 2-D activations"):
        batch = flat_prompts[start : start + batch_size]
        tokens = model.to_tokens(batch, prepend_bos=True).to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hooks)

        for hook in hooks:
            act = cache[hook][:, -1, :].cpu().numpy()  # (batch, d_model)
            all_activations[hook].append(act)

    result = {}
    for hook in hooks:
        stacked = np.concatenate(all_activations[hook], axis=0)  # (N*N, d_model)
        result[hook] = stacked.reshape(N, N, -1)

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_heatmap(
    projections_2d: np.ndarray,
    label: str,
    sum_r2: float,
    sum_freq: int,
    output_path: str,
):
    """Plot 2-D heatmap of direction projection(a, b).

    Diagonal stripes indicate sum-encoding: proj depends on a+b.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Raw heatmap
    im = axes[0].imshow(projections_2d, origin="lower", aspect="equal", cmap="RdBu_r")
    axes[0].set_xlabel("operand b")
    axes[0].set_ylabel("operand a")
    axes[0].set_title(f"{label}\nProjection(a, b)")
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    # Fitted cos/sin(f*(a+b))
    N_a, N_b = projections_2d.shape
    N = max(N_a, N_b)
    a_grid = np.arange(N_a, dtype=np.float64)
    b_grid = np.arange(N_b, dtype=np.float64)
    A_mesh, B_mesh = np.meshgrid(a_grid, b_grid, indexing="ij")
    ab_sum = A_mesh + B_mesh
    theta = 2.0 * np.pi * sum_freq * ab_sum / N

    flat = projections_2d.flatten()
    X = np.column_stack([
        np.cos(theta).flatten(),
        np.sin(theta).flatten(),
        np.ones(N_a * N_b),
    ])
    coeffs, _, _, _ = np.linalg.lstsq(X, flat, rcond=None)
    fitted = (X @ coeffs).reshape(N_a, N_b)

    im2 = axes[1].imshow(fitted, origin="lower", aspect="equal", cmap="RdBu_r")
    axes[1].set_xlabel("operand b")
    axes[1].set_ylabel("operand a")
    axes[1].set_title(f"Fitted cos/sin(f={sum_freq}·(a+b)/N)\nR²={sum_r2:.4f}")
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_wave_1d(
    projections_2d: np.ndarray,
    label: str,
    dominant_frequency: int,
    effective_strength: float,
    output_path: str,
):
    """Plot 1-D projection vs operand value showing periodic encoding.

    3-panel plot:
      Left:  projection(a, b=0) — how direction responds to operand a
      Middle: projection(a=0, b) — how direction responds to operand b
      Right:  DFT power spectrum of the a-slice

    If a direction encodes cos(2πf·a/N), the left panel shows a clean
    sinusoidal wave and the right panel shows a spike at frequency f.
    """
    N_a, N_b = projections_2d.shape
    N = max(N_a, N_b)

    # Slices: fix one operand, vary the other
    slice_a = projections_2d[:, 0]   # fix b=0, vary a
    slice_b = projections_2d[0, :]   # fix a=0, vary b

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, signal, xlabel, title_suffix in [
        (axes[0], slice_a, "operand a (b=0)", "Response to a"),
        (axes[1], slice_b, "operand b (a=0)", "Response to b"),
    ]:
        n_vals = np.arange(len(signal))

        # Fit cos/sin at dominant frequency
        f = dominant_frequency
        theta = 2.0 * np.pi * f * n_vals / N
        X = np.column_stack([np.cos(theta), np.sin(theta), np.ones(len(signal))])
        coeffs, _, _, _ = np.linalg.lstsq(X, signal, rcond=None)
        fitted = X @ coeffs
        ss_res = np.sum((signal - fitted) ** 2)
        ss_tot = np.sum((signal - np.mean(signal)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        # Plot data + fit
        ax.scatter(n_vals, signal, c=n_vals, cmap="viridis", s=30,
                   edgecolors="k", linewidth=0.3, zorder=3, label="Actual")
        # Smooth fitted curve
        n_smooth = np.linspace(0, len(signal) - 1, 300)
        theta_smooth = 2.0 * np.pi * f * n_smooth / N
        X_smooth = np.column_stack([np.cos(theta_smooth), np.sin(theta_smooth),
                                     np.ones(len(n_smooth))])
        fitted_smooth = X_smooth @ coeffs
        ax.plot(n_smooth, fitted_smooth, "r-", linewidth=2, alpha=0.7,
                label=f"cos/sin fit (f={f}, R²={r2:.3f})")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Projection onto SVD direction")
        ax.set_title(f"{label}\n{title_suffix} — R²={r2:.3f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    # Right panel: DFT power spectrum of a-slice
    dft = np.fft.fft(slice_a)
    n_freqs = len(slice_a) // 2 + 1
    power = np.abs(dft[:n_freqs]) ** 2
    power[0] = 0  # remove DC

    freqs = np.arange(n_freqs)
    axes[2].bar(freqs, power, color="steelblue", alpha=0.7, width=0.8)
    if dominant_frequency < n_freqs:
        axes[2].bar(dominant_frequency, power[dominant_frequency],
                    color="red", alpha=0.9, width=0.8,
                    label=f"Dominant freq={dominant_frequency}")
    axes[2].set_xlabel("Frequency")
    axes[2].set_ylabel("DFT Power")
    axes[2].set_title(f"Fourier Spectrum (a-slice)\nstr={effective_strength:.2f}")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.2)
    axes[2].set_xlim(-0.5, min(n_freqs, 20))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_circle(
    proj1: np.ndarray,
    proj2: np.ndarray,
    label1: str,
    label2: str,
    frequency: int,
    circularity: float,
    output_path: str,
):
    """Plot 2-panel comparison: ideal circle vs actual data.

    Left panel: What a PERFECT cos/sin pair would look like — numbers arranged
    evenly around a circle, colored by value (rainbow = smooth progression).
    Right panel: The actual (proj1, proj2) scatter — if it matches the left,
    the directions form a cos/sin pair and numbers live on a circle.

    How to read:
      - Perfect circle (circularity ≈ 1.0): points sit on a ring, colors
        progress smoothly around it (like a clock face).
      - Not a circle (circularity < 0.5): points are scattered/clustered,
        colors are jumbled — no rotational structure.
    """
    N = len(proj1)
    c1 = proj1 - np.mean(proj1)
    c2 = proj2 - np.mean(proj2)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # --- Left panel: IDEAL circle for this frequency ---
    ideal_theta = 2.0 * np.pi * frequency * np.arange(N) / N
    ideal_x = np.cos(ideal_theta)
    ideal_y = np.sin(ideal_theta)

    sc0 = axes[0].scatter(ideal_x, ideal_y, c=np.arange(N), cmap="hsv",
                          s=60, edgecolors="k", linewidth=0.4, zorder=3)
    # Connect consecutive points
    axes[0].plot(ideal_x, ideal_y, "k-", alpha=0.2, linewidth=0.5)
    # Reference circle
    ref_theta = np.linspace(0, 2 * np.pi, 200)
    axes[0].plot(np.cos(ref_theta), np.sin(ref_theta), "k--", alpha=0.3, linewidth=1)
    # Label a few points
    for i in range(0, N, max(1, N // 8)):
        axes[0].annotate(str(i), (ideal_x[i], ideal_y[i]),
                         fontsize=7, ha="center", va="bottom",
                         xytext=(0, 5), textcoords="offset points")
    axes[0].set_title(f"IDEAL: cos/sin pair at freq={frequency}\n"
                      f"(what a perfect circle looks like)",
                      fontsize=11, fontweight="bold")
    axes[0].set_xlabel("cos(2πf·n/N)")
    axes[0].set_ylabel("sin(2πf·n/N)")
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.2)
    plt.colorbar(sc0, ax=axes[0], label="operand value n", shrink=0.8)

    # --- Right panel: ACTUAL data ---
    sc1 = axes[1].scatter(c1, c2, c=np.arange(N), cmap="hsv",
                          s=60, edgecolors="k", linewidth=0.4, zorder=3)
    # Connect consecutive points
    axes[1].plot(c1, c2, "k-", alpha=0.2, linewidth=0.5)
    # Reference circle at mean radius
    mean_r = np.mean(np.sqrt(c1 ** 2 + c2 ** 2))
    axes[1].plot(mean_r * np.cos(ref_theta), mean_r * np.sin(ref_theta),
                 "k--", alpha=0.3, linewidth=1)
    # Label a few points
    for i in range(0, N, max(1, N // 8)):
        axes[1].annotate(str(i), (c1[i], c2[i]),
                         fontsize=7, ha="center", va="bottom",
                         xytext=(0, 5), textcoords="offset points")

    # Verdict in title
    if circularity >= 0.8:
        verdict = "✓ CIRCLE"
        color = "green"
    elif circularity >= 0.5:
        verdict = "~ WEAK CIRCLE"
        color = "orange"
    else:
        verdict = "✗ NOT A CIRCLE"
        color = "red"

    axes[1].set_title(f"ACTUAL: {verdict} (circularity={circularity:.3f})\n"
                      f"{label1} vs {label2}",
                      fontsize=11, fontweight="bold", color=color)
    axes[1].set_xlabel(f"{label1} (centered)")
    axes[1].set_ylabel(f"{label2} (centered)")
    axes[1].set_aspect("equal")
    axes[1].grid(True, alpha=0.2)
    plt.colorbar(sc1, ax=axes[1], label="operand value n", shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def load_phase3_results(path: str) -> List[dict]:
    """Load Phase 3 survivor directions from JSON (same as causal_validation.py)."""
    with open(path) as f:
        data = json.load(f)

    survivors = []
    for comp in data.get("results", []):
        layer = comp["layer"]
        head = comp.get("head")
        component = comp["component"]
        for d in comp.get("directions", []):
            if d.get("mask_value", 0) > 0.3:
                survivors.append({
                    "layer": layer,
                    "head": head,
                    "component": component,
                    **d,
                })

    return survivors


def direction_label(s: dict) -> str:
    if s.get("head") is not None:
        return f"L{s['layer']}H{s['head']}_{s['component']}_dir{s['direction_idx']}"
    return f"MLP_L{s['layer']}_{s['component']}_dir{s['direction_idx']}"


def main():
    parser = argparse.ArgumentParser(description="Phase 4c: SVD Direction Sum-Encoding Analysis")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--phase3_results", required=True, help="Path to Phase 3 JSON")
    parser.add_argument("--operand_range_end", type=int, default=None,
                        help="Override operand range end (default: from config or 30)")
    parser.add_argument("--output_dir", default="phase4c_results", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_freq", type=int, default=15, help="Max frequency to search")
    parser.add_argument("--device", default=None, help="Device (auto-detect if omitted)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load Phase 3 results
    survivors = load_phase3_results(args.phase3_results)
    logger.info(f"Loaded {len(survivors)} surviving directions from Phase 3")

    if not survivors:
        logger.error("No surviving directions found!")
        return

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Load model
    from transformer_lens import HookedTransformer
    model_name = config["model"]["name"]
    model = HookedTransformer.from_pretrained(model_name, cache_dir="cache_dir")
    model.to(device)
    model.eval()

    # Build circuit for SVD extraction
    circuit = MaskedTransformerCircuit(
        model=model,
        mask_mlp=True,
    )

    # Load masks
    checkpoint_path = args.checkpoint
    torch.serialization.add_safe_globals([torch.nn.modules.container.ParameterDict])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    for key in checkpoint.get("ov_masks", {}):
        if key in circuit.ov_masks:
            circuit.ov_masks[key].data = checkpoint["ov_masks"][key].to(device)
    if circuit.mask_mlp:
        for key in checkpoint.get("mlp_in_masks", {}):
            if key in circuit.mlp_in_masks:
                circuit.mlp_in_masks[key].data = checkpoint["mlp_in_masks"][key].to(device)
        for key in checkpoint.get("mlp_out_masks", {}):
            if key in circuit.mlp_out_masks:
                circuit.mlp_out_masks[key].data = checkpoint["mlp_out_masks"][key].to(device)
    logger.info("Masks loaded successfully")

    # Set up operand range
    arith_cfg = config["arithmetic"]
    range_end = args.operand_range_end or min(arith_cfg.get("operand_range_end", 10), 30)
    op_range = range(arith_cfg.get("operand_range_start", 0), range_end)
    N = len(op_range)
    logger.info(f"Operand range: 0-{range_end - 1} ({N}×{N} = {N*N} prompts)")

    prompt_gen = ArithmeticPromptGenerator(
        operand_range=op_range,
        operation=arith_cfg.get("operation", "add"),
        prompt_template=arith_cfg.get("prompt_template", "{a} + {b} ="),
        shuffle=False,
    )

    # Collect 2-D residual stream activations
    active_layers = sorted(set(s["layer"] for s in survivors))
    logger.info(f"Collecting activations for layers: {active_layers}")

    activations_2d = collect_residual_activations_2d(
        model, prompt_gen, op_range, active_layers, device, args.batch_size
    )
    logger.info("Activation collection complete")

    # Extract projection vectors for each direction
    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ---------------------------------------------------------------
    # Analyze each direction for sum-encoding
    # ---------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("PHASE 4c: SVD DIRECTION SUM-ENCODING ANALYSIS")
    logger.info("=" * 70)

    sum_results: List[DirectionSumResult] = []
    direction_projections_1d: Dict[str, np.ndarray] = {}  # for circle test

    for surv in tqdm(survivors, desc="Analyzing sum-encoding"):
        layer = surv["layer"]
        head = surv.get("head")
        component = surv["component"]
        k = surv["direction_idx"]
        label = direction_label(surv)

        hook = f"blocks.{layer}.hook_resid_pre"
        if hook not in activations_2d:
            logger.warning(f"No activations for {hook}, skipping {label}")
            continue

        act_2d = activations_2d[hook]  # (N_a, N_b, d_model)

        # Get projection vector from SVD cache
        if component == "OV":
            cache_key = f"differential_head_{layer}_{head}_ov"
        elif component in ("MLP_in", "MLP_out"):
            suffix = "in" if component == "MLP_in" else "out"
            cache_key = f"mlp_{layer}_{suffix}"
        else:
            continue

        if cache_key not in circuit.svd_cache:
            logger.warning(f"SVD cache missing for {cache_key}")
            continue

        U, S, Vh, _ = circuit.svd_cache[cache_key]

        if component in ("OV", "MLP_out"):
            vec = Vh[k, :].cpu().numpy()
        elif component == "MLP_in":
            vec = U[:, k].cpu().numpy()
        else:
            continue

        # Handle augmented dimensions
        d_model = act_2d.shape[2]
        if vec.shape[0] == d_model + 1:
            vec = vec[1:]
        elif vec.shape[0] != d_model:
            logger.warning(f"Dim mismatch: vec={vec.shape[0]} act={d_model}")
            continue

        # Project: (N_a, N_b, d_model) @ (d_model,) → (N_a, N_b)
        proj_2d = act_2d @ vec

        # Run sum-encoding analysis
        mask_val = surv.get("mask_value", 0)
        sv = surv.get("singular_value", 0)
        eff_str = sv * mask_val
        dom_freq = surv.get("dominant_frequency", 1)

        result = analyze_direction_sum(
            proj_2d, layer, head, component, k,
            eff_str, dom_freq, args.max_freq,
        )
        sum_results.append(result)

        # Store 1-D projections (a+0 prompts) for circle test
        direction_projections_1d[label] = proj_2d[:, 0]  # fix b=0, vary a

        # Plot 1-D wave for ALL directions (shows periodic single-operand encoding)
        wave_path = os.path.join(plots_dir, f"wave_{label}.png")
        plot_wave_1d(proj_2d, label, dom_freq, eff_str, wave_path)

        # Plot heatmap for top directions
        if result.sum_r2_best >= 0.05 or eff_str >= 1.0:
            plot_path = os.path.join(plots_dir, f"heatmap_{label}.png")
            plot_heatmap(proj_2d, label, result.sum_r2_best, result.sum_freq_best, plot_path)

    # ---------------------------------------------------------------
    # Circle tests for cos/sin pairs
    # ---------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("CIRCLE TESTS FOR COS/SIN PAIRS")
    logger.info("=" * 70)

    # Load Phase 4a results if available (for phase information)
    # Fall back to computing phases from 1-D projections
    circle_results: List[CirclePairResult] = []

    # Group directions by frequency
    freq_groups: Dict[int, List[dict]] = {}
    for surv in survivors:
        f = surv.get("dominant_frequency", 0)
        if f > 0:
            freq_groups.setdefault(f, []).append(surv)

    for freq, group in sorted(freq_groups.items()):
        if len(group) < 2:
            continue

        # Compute phases from 1-D projections (b=0)
        dir_phases = []
        for s in group:
            label = direction_label(s)
            if label not in direction_projections_1d:
                continue
            signal = direction_projections_1d[label]
            dft = np.fft.fft(signal)
            if freq < len(dft):
                phase = np.angle(dft[freq])
            else:
                phase = 0.0
            dir_phases.append((label, phase, signal))

        # Test all pairs for ~90° phase difference
        for i in range(len(dir_phases)):
            for j in range(i + 1, len(dir_phases)):
                l1, p1, s1 = dir_phases[i]
                l2, p2, s2 = dir_phases[j]
                diff = abs(p1 - p2)
                # Normalize to [0, π]
                diff = diff % (2 * np.pi)
                if diff > np.pi:
                    diff = 2 * np.pi - diff
                diff_deg = math.degrees(diff)

                # Only test pairs within 60-120° (cos/sin pair candidates)
                if 60 <= diff_deg <= 120:
                    cr = test_circle(s1, s2, freq, l1, l2, diff_deg)
                    circle_results.append(cr)

                    # Plot circle
                    plot_path = os.path.join(
                        plots_dir,
                        f"circle_f{freq}_{l1}_vs_{l2}.png",
                    )
                    plot_circle(s1, s2, l1, l2, freq, cr.circularity, plot_path)

    # ---------------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)

    # Sort by sum R²
    sum_results.sort(key=lambda r: -r.sum_r2_best)

    lines = []
    lines.append("=" * 90)
    lines.append("PHASE 4c: SVD DIRECTION SUM-ENCODING ANALYSIS")
    lines.append("=" * 90)
    lines.append("")
    lines.append("  Per-Direction cos/sin(f·(a+b)/N) Analysis:")
    lines.append("  {:>30s}  {:>7s}  {:>5s}  {:>8s}  {:>5s}  {:>8s}  {:>8s}  {:>8s}".format(
        "Direction", "Str", "f_dom", "R²_best", "f_sum", "R²_multi", "DiagPow", "Sep",
    ))
    lines.append("  " + "-" * 95)

    n_sum_encoding = 0
    for r in sum_results:
        label = direction_label({
            "layer": r.layer, "head": r.head,
            "component": r.component, "direction_idx": r.direction_idx,
        })
        marker = " ★" if r.sum_r2_best >= 0.3 else ""
        if r.sum_r2_best >= 0.3:
            n_sum_encoding += 1
        lines.append(
            "  {:>30s}  {:>7.2f}  {:>5d}  {:>8.4f}  {:>5d}  {:>8.4f}  {:>8.1f}  {:>8.3f}{}".format(
                label, r.effective_strength, r.dominant_frequency,
                r.sum_r2_best, r.sum_freq_best,
                r.sum_r2_multi, r.diag_power_ratio,
                r.separability_score, marker,
            )
        )

    lines.append("")
    lines.append(f"  Directions with sum R² ≥ 0.3: {n_sum_encoding}/{len(sum_results)}")
    lines.append(f"  Directions with sum R² ≥ 0.1: {sum(1 for r in sum_results if r.sum_r2_best >= 0.1)}/{len(sum_results)}")

    # Circle results
    if circle_results:
        lines.append("")
        lines.append("  Circle Tests (cos/sin pairs with phase diff 60-120°):")
        lines.append("  {:>5s}  {:>25s}  {:>25s}  {:>8s}  {:>10s}  {:>8s}".format(
            "Freq", "Direction 1", "Direction 2", "PhaseDΔ", "Circularity", "Radius",
        ))
        lines.append("  " + "-" * 90)
        for cr in sorted(circle_results, key=lambda x: -x.circularity):
            lines.append(
                "  {:>5d}  {:>25s}  {:>25s}  {:>7.1f}°  {:>10.3f}  {:>8.4f}".format(
                    cr.frequency, cr.dir1_label, cr.dir2_label,
                    cr.phase_diff_deg, cr.circularity, cr.mean_radius,
                )
            )

    summary_text = "\n".join(lines)
    logger.info("\n" + summary_text)

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_data = {
        "timestamp": timestamp,
        "operand_range_end": range_end,
        "n_directions": len(sum_results),
        "n_sum_encoding_r2_03": n_sum_encoding,
        "sum_results": [asdict(r) for r in sum_results],
        "circle_results": [asdict(r) for r in circle_results],
    }

    json_path = os.path.join(args.output_dir, f"sum_encoding_analysis_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(result_data, f, indent=2)
    logger.info(f"Results saved to {json_path}")

    summary_path = json_path.replace(".json", "_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    logger.info(f"Summary saved to {summary_path}")

    # Plots directory
    n_plots = len([f for f in os.listdir(plots_dir) if f.endswith(".png")])
    logger.info(f"Generated {n_plots} plots in {plots_dir}")

    # Final verdict
    logger.info("=" * 70)
    if n_sum_encoding > 0:
        top = sum_results[0]
        top_label = direction_label({
            "layer": top.layer, "head": top.head,
            "component": top.component, "direction_idx": top.direction_idx,
        })
        logger.info(
            f"PHASE 4c: {n_sum_encoding}/{len(sum_results)} directions encode cos/sin(a+b) (R²≥0.3). "
            f"Best: {top_label} R²={top.sum_r2_best:.4f} at f={top.sum_freq_best}"
        )
    else:
        logger.info(
            f"PHASE 4c: No directions with sum R² ≥ 0.3. "
            f"Best R²={sum_results[0].sum_r2_best:.4f} if results exist. "
            f"Sum-encoding may be distributed across directions, not concentrated."
        )
    if circle_results:
        best_circle = max(circle_results, key=lambda x: x.circularity)
        logger.info(
            f"Best circle: freq={best_circle.frequency} "
            f"({best_circle.dir1_label} vs {best_circle.dir2_label}) "
            f"circularity={best_circle.circularity:.3f}"
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
