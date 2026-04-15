#!/usr/bin/env python3
"""
Experiment 7: UMAP Visualization of 9D Fourier Projections

Projects all residual-stream activations into the 9D Fourier subspace
(derived from SVD of per-digit means at the computation layer) and
generates UMAP embeddings to visualize digit clustering.

Generates:
  1. UMAP colored by digit (0-9)  — shows clean cluster separation
  2. UMAP colored by carry status — shows within-cluster carry structure
  3. UMAP colored by dominant frequency — shows Fourier organization
  4. 3D PCA of the 9D projections (no UMAP, for sanity)

Usage:
    python fourier_umap.py --model gemma-2b --device mps
    python fourier_umap.py --model phi-3 --device mps
"""

import argparse
import json
import logging
import sys
import time
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from arithmetic_circuit_scan_updated import (
    generate_teacher_forced_problems,
    filter_correct_teacher_forced,
    generate_direct_answer_problems,
    filter_correct_direct_answer,
    MODEL_MAP,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("mathematical_toolkit_results")
PLOT_DIR = RESULTS_DIR / "paper_plots"
PLOT_DIR.mkdir(exist_ok=True, parents=True)

# ── Color palettes ─────────────────────────────────────────────────────────

DIGIT_COLORS = [
    "#E41A1C",  # 0 - red
    "#377EB8",  # 1 - blue
    "#4DAF4A",  # 2 - green
    "#984EA3",  # 3 - purple
    "#FF7F00",  # 4 - orange
    "#A65628",  # 5 - brown
    "#F781BF",  # 6 - pink
    "#999999",  # 7 - grey
    "#66C2A5",  # 8 - teal
    "#E7298A",  # 9 - magenta
]

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

CARRY_COLORS = {0: "#2196F3", 1: "#F44336"}  # blue=no carry, red=carry


# ─────────────────────────────────────────────────────────────────────────────
# ACTIVATION COLLECTION
# ─────────────────────────────────────────────────────────────────────────────

def collect_all_activations(model, problems, layer, device) -> Tuple[np.ndarray, List[dict]]:
    """
    Collect last-token activations at a given layer for ALL problems.

    Returns:
        activations: (N, d_model) array
        metadata: list of dicts with digit, carry, a, b info for each problem
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    d_model = model.cfg.d_model

    all_acts = []
    metadata = []

    for prob in problems:
        digit = prob["ones_digit"]
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True).to(device)

        holder = {}
        def capture(act, hook, h=holder):
            h["act"] = act.detach()
            return act

        with torch.no_grad():
            with model.hooks(fwd_hooks=[(hook_name, capture)]):
                model(tokens)

        if "act" in holder:
            act_vec = holder["act"][0, -1].cpu().float().numpy()
            all_acts.append(act_vec)
            metadata.append({
                "digit": digit,
                "carry": prob.get("has_carry", 0),
                "a": prob.get("a", 0),
                "b": prob.get("b", 0),
                "ones_a": prob.get("a", 0) % 10,
                "ones_b": prob.get("b", 0) % 10,
            })

    activations = np.stack(all_acts, axis=0)  # (N, d_model)
    logger.info(f"  Collected {activations.shape[0]} activations at L{layer}, shape={activations.shape}")
    return activations, metadata


def compute_fourier_basis(model, train_problems, layer, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the 9D Fourier basis from per-digit mean activations.

    Returns:
        V: (d_model, 9) orthonormal columns
        S: (9,) singular values
        U: (10, 9) left singular vectors (digit patterns)
        centroid: (d_model,) global mean to subtract before projection
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    d_model = model.cfg.d_model

    digit_acts = defaultdict(list)
    for prob in train_problems:
        digit = prob["ones_digit"]
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True).to(device)

        holder = {}
        def capture(act, hook, h=holder):
            h["act"] = act.detach()
            return act

        with torch.no_grad():
            with model.hooks(fwd_hooks=[(hook_name, capture)]):
                model(tokens)

        if "act" in holder:
            act_vec = holder["act"][0, -1].cpu().float().numpy()
            digit_acts[digit].append(act_vec)

    # Per-digit means → (10, d_model)
    means = np.zeros((10, d_model))
    for d in range(10):
        if digit_acts[d]:
            means[d] = np.mean(digit_acts[d], axis=0)
            logger.info(f"    Digit {d}: {len(digit_acts[d])} samples")
        else:
            logger.warning(f"    Digit {d}: NO samples!")

    # Center and SVD
    centroid = means.mean(axis=0, keepdims=True)
    M = means - centroid  # (10, d_model)
    U_full, S_full, Vt_full = np.linalg.svd(M, full_matrices=False)

    V = Vt_full[:9].T   # (d_model, 9) — orthonormal columns
    S = S_full[:9]
    U = U_full[:, :9]   # (10, 9) — digit score patterns

    # Verify orthonormality
    gram = V.T @ V
    err = np.abs(gram - np.eye(9)).max()
    assert err < 1e-5, f"V not orthonormal, max err = {err:.2e}"
    logger.info(f"  9D basis orthonormality check: max err = {err:.2e} ✓")

    return V, S, U, centroid.squeeze()


def assign_frequencies(U_cols: np.ndarray) -> List[int]:
    """Assign each SVD direction to its dominant Fourier frequency."""
    n_dirs = U_cols.shape[0]
    assignments = []
    for i in range(n_dirs):
        scores = U_cols[i]
        scores_centered = scores - scores.mean()
        dft = np.fft.fft(scores_centered)
        power = np.abs(dft) ** 2
        freq_power = np.zeros(6)
        freq_power[0] = power[0]
        for k in range(1, 5):
            freq_power[k] = power[k] + power[10 - k]
        freq_power[5] = power[5]
        dom_k = int(np.argmax(freq_power[1:])) + 1
        assignments.append(dom_k)
    return assignments


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_umap_by_digit(embedding, metadata, model_name, layer, save_path, centroids=None):
    """UMAP colored by digit (0-9), with optional centroid markers."""
    fig, ax = plt.subplots(figsize=(10, 8))
    digits = np.array([m["digit"] for m in metadata])

    for d in range(10):
        mask = digits == d
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=DIGIT_COLORS[d], label=str(d),
            s=12, alpha=0.6, edgecolors="none",
        )

    # Add per-digit mean centroids as large star markers with labels
    if centroids is not None:
        for d in range(10):
            ax.scatter(
                centroids[d, 0], centroids[d, 1],
                c=DIGIT_COLORS[d], s=250, marker="*",
                edgecolors="black", linewidths=0.8, zorder=10,
            )
            ax.text(
                centroids[d, 0], centroids[d, 1] + 0.3, str(d),
                fontsize=11, fontweight="bold", ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          alpha=0.85, edgecolor="gray", linewidth=0.5),
                zorder=11,
            )

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(f"{model_name} L{layer}: 9D Fourier Projection → UMAP (σ-normalized)\nColored by Digit", fontsize=14)
    ax.legend(title="Digit", fontsize=9, title_fontsize=10, ncol=2,
              loc="upper right", markerscale=2.5, framealpha=0.9)
    ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {save_path}")


def plot_umap_by_carry(embedding, metadata, model_name, layer, save_path):
    """UMAP colored by carry status, with digit labels."""
    fig, ax = plt.subplots(figsize=(10, 8))
    digits = np.array([m["digit"] for m in metadata])
    carries = np.array([m["carry"] for m in metadata])

    for carry_val, label, color in [(0, "No Carry", CARRY_COLORS[0]),
                                     (1, "Carry", CARRY_COLORS[1])]:
        mask = carries == carry_val
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=color, label=label,
            s=12, alpha=0.6, edgecolors="none",
        )

    # Add digit centroids as text labels
    for d in range(10):
        mask = digits == d
        if mask.sum() > 0:
            cx = embedding[mask, 0].mean()
            cy = embedding[mask, 1].mean()
            ax.text(cx, cy, str(d), fontsize=14, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="gray"))

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(f"{model_name} L{layer}: 9D Fourier Projection → UMAP\nColored by Carry Status", fontsize=14)
    ax.legend(fontsize=11, loc="upper right", markerscale=2.5, framealpha=0.9)
    ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {save_path}")


def plot_umap_carry_per_digit(embedding, metadata, model_name, layer, save_path):
    """Grid of per-digit UMAP panels showing carry separation within each cluster."""
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    digits = np.array([m["digit"] for m in metadata])
    carries = np.array([m["carry"] for m in metadata])

    for d in range(10):
        ax = axes[d // 5, d % 5]
        mask_d = digits == d

        # Plot all other digits in light gray background
        ax.scatter(
            embedding[~mask_d, 0], embedding[~mask_d, 1],
            c="#E0E0E0", s=4, alpha=0.3, edgecolors="none",
        )

        # This digit: carry vs no-carry
        mask_no_carry = mask_d & (carries == 0)
        mask_carry = mask_d & (carries == 1)

        if mask_no_carry.sum() > 0:
            ax.scatter(
                embedding[mask_no_carry, 0], embedding[mask_no_carry, 1],
                c=CARRY_COLORS[0], s=18, alpha=0.8, edgecolors="none",
                label=f"No carry ({mask_no_carry.sum()})",
            )
        if mask_carry.sum() > 0:
            ax.scatter(
                embedding[mask_carry, 0], embedding[mask_carry, 1],
                c=CARRY_COLORS[1], s=18, alpha=0.8, edgecolors="none",
                label=f"Carry ({mask_carry.sum()})",
            )

        ax.set_title(f"Digit {d}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right", markerscale=1.5)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{model_name} L{layer}: Carry vs No-Carry Within Each Digit Cluster",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {save_path}")


def plot_pca_3d(projections_9d, metadata, model_name, layer, save_path):
    """3D PCA of the 9D projections (sanity check without UMAP)."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=3)
    coords = pca.fit_transform(projections_9d)
    explained = pca.explained_variance_ratio_

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    digits = np.array([m["digit"] for m in metadata])

    for d in range(10):
        mask = digits == d
        ax.scatter(
            coords[mask, 0], coords[mask, 1], coords[mask, 2],
            c=DIGIT_COLORS[d], label=str(d),
            s=10, alpha=0.6,
        )

    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)", fontsize=10)
    ax.set_zlabel(f"PC3 ({explained[2]*100:.1f}%)", fontsize=10)
    ax.set_title(f"{model_name} L{layer}: PCA of 9D Fourier Projections", fontsize=13)
    ax.legend(title="Digit", fontsize=7, title_fontsize=8, ncol=2, loc="upper left")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {save_path}")


def plot_frequency_decomposition(projections_9d, metadata, freq_assignments, singular_values,
                                  model_name, layer, save_path):
    """Show per-frequency 2D sub-projections with per-digit means forming circular patterns."""
    freq_to_dirs = defaultdict(list)
    for i, k in enumerate(freq_assignments):
        freq_to_dirs[k].append(i)

    digits = np.array([m["digit"] for m in metadata])

    # Compute per-digit means in 9D
    digit_means = np.zeros((10, projections_9d.shape[1]))
    for d in range(10):
        mask = digits == d
        if mask.sum() > 0:
            digit_means[d] = projections_9d[mask].mean(axis=0)

    # Separate frequencies into 2-DOF (cos+sin pair) and 1-DOF (Nyquist k=5)
    freqs_2d = [k for k in sorted(freq_to_dirs.keys()) if len(freq_to_dirs[k]) >= 2]
    freqs_1d = [k for k in sorted(freq_to_dirs.keys()) if len(freq_to_dirs[k]) == 1]
    n_panels = len(freqs_2d) + len(freqs_1d)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5.5))
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0

    # ── 2-DOF frequency panels (cos/sin pairs → 2D scatter) ──
    for k in freqs_2d:
        ax = axes[panel_idx]
        panel_idx += 1
        dirs = freq_to_dirs[k][:2]

        # Individual samples as faint background
        for d in range(10):
            mask = digits == d
            ax.scatter(
                projections_9d[mask, dirs[0]], projections_9d[mask, dirs[1]],
                c=DIGIT_COLORS[d], s=6, alpha=0.15, edgecolors="none",
            )

        # Per-digit means as large labeled markers
        mean_x = digit_means[:, dirs[0]]
        mean_y = digit_means[:, dirs[1]]

        # Fit circle to the 10 mean positions
        cx, cy = mean_x.mean(), mean_y.mean()
        radii = np.sqrt((mean_x - cx)**2 + (mean_y - cy)**2)
        r_fit = radii.mean()

        # Draw fitted circle
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(cx + r_fit * np.cos(theta), cy + r_fit * np.sin(theta),
                color="gray", linewidth=1.0, linestyle="--", alpha=0.6, zorder=1)

        # Connect consecutive digits with lines (0→1→2→...→9→0)
        for d in range(10):
            d_next = (d + 1) % 10
            ax.plot(
                [mean_x[d], mean_x[d_next]], [mean_y[d], mean_y[d_next]],
                color="gray", linewidth=0.8, alpha=0.4, zorder=2,
            )

        # Plot mean markers
        for d in range(10):
            ax.scatter(
                mean_x[d], mean_y[d],
                c=DIGIT_COLORS[d], s=180, marker="o",
                edgecolors="black", linewidths=1.0, zorder=5,
            )
            ax.text(
                mean_x[d], mean_y[d], str(d),
                fontsize=9, fontweight="bold", ha="center", va="center",
                color="white" if d not in [7, 8] else "black",
                zorder=6,
            )

        ax.set_xlabel(f"Dir {dirs[0]+1} (σ={singular_values[dirs[0]]:.1f})", fontsize=10)
        ax.set_ylabel(f"Dir {dirs[1]+1} (σ={singular_values[dirs[1]]:.1f})", fontsize=10)
        ax.set_title(f"{FREQ_LABELS.get(k, f'k={k}')}", fontsize=13,
                     color=FREQ_COLORS.get(k, "black"), fontweight="bold")
        ax.set_aspect("equal")
        ax.axhline(cy, color="gray", linewidth=0.3, alpha=0.3)
        ax.axvline(cx, color="gray", linewidth=0.3, alpha=0.3)

        # Annotate circle quality
        r_std = radii.std()
        ax.text(0.02, 0.98, f"r={r_fit:.1f}±{r_std:.1f}",
                transform=ax.transAxes, fontsize=8, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    # ── 1-DOF frequency panels (e.g. k=5 Nyquist → 1D strip) ──
    for k in freqs_1d:
        ax = axes[panel_idx]
        panel_idx += 1
        dir_idx = freq_to_dirs[k][0]

        # Individual samples as strip plot (jitter on y for visibility)
        rng = np.random.RandomState(42)
        for d in range(10):
            mask = digits == d
            n_pts = mask.sum()
            jitter = rng.uniform(-0.3, 0.3, size=n_pts)
            ax.scatter(
                projections_9d[mask, dir_idx], jitter,
                c=DIGIT_COLORS[d], s=8, alpha=0.2, edgecolors="none",
            )

        # Per-digit means as large markers along the axis
        mean_vals = digit_means[:, dir_idx]
        for d in range(10):
            ax.scatter(
                mean_vals[d], 0,
                c=DIGIT_COLORS[d], s=200, marker="o",
                edgecolors="black", linewidths=1.0, zorder=5,
            )
            ax.text(
                mean_vals[d], 0.5, str(d),
                fontsize=9, fontweight="bold", ha="center", va="bottom",
                color=DIGIT_COLORS[d], zorder=6,
            )

        ax.set_xlabel(f"Dir {dir_idx+1} (σ={singular_values[dir_idx]:.1f})", fontsize=10)
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.set_title(f"{FREQ_LABELS.get(k, f'k={k}')} (1 DOF)", fontsize=13,
                     color=FREQ_COLORS.get(k, "black"), fontweight="bold")
        ax.axhline(0, color="gray", linewidth=0.3, alpha=0.3)

        # Annotate: even/odd or relevant pattern
        if k == 5:
            ax.text(0.02, 0.98, "(-1)\u1d48 parity",
                    transform=ax.transAxes, fontsize=8, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    fig.suptitle(f"{model_name} L{layer}: Fourier Frequency Pairs — Per-Digit Mean Positions",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Experiment 7: UMAP of 9D Fourier Projections")
    parser.add_argument("--model", default="gemma-2b", choices=list(MODEL_MAP.keys()))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--comp-layer", type=int, default=None)
    parser.add_argument("--n-per-digit", type=int, default=150,
                        help="Problems per digit for activation collection")
    parser.add_argument("--umap-neighbors", type=int, default=15,
                        help="UMAP n_neighbors parameter")
    parser.add_argument("--umap-min-dist", type=float, default=0.1,
                        help="UMAP min_dist parameter")
    parser.add_argument("--direct-answer", action="store_true",
                        help="Use direct-answer mode (for LLaMA 3B: full answer as single token)")
    args = parser.parse_args()

    model_name_full = MODEL_MAP[args.model]
    device = args.device
    comp_defaults = {"gemma-2b": 19, "phi-3": 26, "llama-3b": 20}
    comp_layer = args.comp_layer or comp_defaults.get(args.model, 20)

    logger.info(f"Model: {args.model} ({model_name_full})")
    logger.info(f"Computation layer: L{comp_layer}")
    logger.info(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(
        model_name_full,
        device=device,
        dtype=torch.float32 if device == "cpu" else torch.float16,
    )
    d_model = model.cfg.d_model
    logger.info(f"d_model = {d_model}")

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 1: Generate problems & filter correct
    # ══════════════════════════════════════════════════════════════════════
    logger.info(f"\n{'═'*60}")
    logger.info(f"  STEP 1: Generate problems")
    logger.info(f"{'═'*60}")

    if args.direct_answer:
        all_problems, _ = generate_direct_answer_problems(n_per_digit=args.n_per_digit)
        logger.info(f"  Generated {len(all_problems)} direct-answer problems")
        correct_problems = filter_correct_direct_answer(model, all_problems, max_n=len(all_problems))
    else:
        all_problems, _ = generate_teacher_forced_problems(n_per_digit=args.n_per_digit)
        logger.info(f"  Generated {len(all_problems)} teacher-forced problems")
        correct_problems = filter_correct_teacher_forced(model, all_problems, max_n=len(all_problems))
    logger.info(f"  Correct: {len(correct_problems)}")

    # Split into train (for computing basis) and test (for visualization)
    min_per_digit = min(
        sum(1 for p in correct_problems if p["ones_digit"] == d)
        for d in range(10)
    )
    train_per_digit = min(80, min_per_digit // 2)
    test_per_digit = min_per_digit - train_per_digit

    train_problems = []
    test_problems = []
    digit_counts = defaultdict(int)
    digit_test_counts = defaultdict(int)

    for prob in correct_problems:
        d = prob["ones_digit"]
        if digit_counts[d] < train_per_digit:
            train_problems.append(prob)
            digit_counts[d] += 1
        elif digit_test_counts[d] < test_per_digit:
            test_problems.append(prob)
            digit_test_counts[d] += 1

    logger.info(f"  Train: {len(train_problems)} ({train_per_digit}/digit)")
    logger.info(f"  Test:  {len(test_problems)} ({test_per_digit}/digit)")

    # Verify disjoint
    train_prompts = set(p["prompt"] for p in train_problems)
    test_prompts = set(p["prompt"] for p in test_problems)
    overlap = train_prompts & test_prompts
    assert len(overlap) == 0, f"Train/test overlap: {len(overlap)} prompts!"
    logger.info(f"  [SANITY] Train/test disjoint ✓ (0 overlap)")

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 2: Compute 9D Fourier basis (from training data only)
    # ══════════════════════════════════════════════════════════════════════
    logger.info(f"\n{'═'*60}")
    logger.info(f"  STEP 2: Compute 9D Fourier subspace (training data only)")
    logger.info(f"{'═'*60}")

    t0 = time.time()
    V, S, U, centroid = compute_fourier_basis(model, train_problems, comp_layer, device)
    elapsed = time.time() - t0
    logger.info(f"  Computed in {elapsed:.1f}s")
    logger.info(f"  Singular values: {np.array2string(S, precision=1, separator=', ')}")

    # Frequency assignments
    freq_assignments = assign_frequencies(U.T)  # U.T is (9, 10)
    logger.info(f"  Frequency assignments: {freq_assignments}")

    freq_to_dirs = defaultdict(list)
    for i, k in enumerate(freq_assignments):
        freq_to_dirs[k].append(i)
    logger.info(f"  Frequency → directions: {dict(freq_to_dirs)}")

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 3: Collect activations and project into 9D
    # ══════════════════════════════════════════════════════════════════════
    logger.info(f"\n{'═'*60}")
    logger.info(f"  STEP 3: Collect test activations & project into 9D")
    logger.info(f"{'═'*60}")

    t0 = time.time()
    activations, metadata = collect_all_activations(model, test_problems, comp_layer, device)
    elapsed = time.time() - t0
    logger.info(f"  Collected in {elapsed:.1f}s")

    # Center activations (critical: SVD was computed on centered means)
    activations_centered = activations - centroid[np.newaxis, :]  # (N, d_model)

    # Project into 9D Fourier subspace
    projections_9d = activations_centered @ V  # (N, 9)
    logger.info(f"  9D projections shape: {projections_9d.shape}")

    # Normalize by singular values so each frequency contributes equally
    projections_normalized = projections_9d / S[np.newaxis, :]  # (N, 9)
    logger.info(f"  Normalized projections (divided by σ): each dim has unit-scale")

    # Compute per-digit mean projections (for centroid markers)
    digits_arr = np.array([m["digit"] for m in metadata])
    digit_mean_proj = np.zeros((10, 9))
    digit_mean_norm = np.zeros((10, 9))
    for d in range(10):
        mask = digits_arr == d
        if mask.sum() > 0:
            digit_mean_proj[d] = projections_9d[mask].mean(axis=0)
            digit_mean_norm[d] = projections_normalized[mask].mean(axis=0)

    # Quick cluster quality check: silhouette score
    try:
        from sklearn.metrics import silhouette_score
        digits = digits_arr
        sil_raw = silhouette_score(projections_9d, digits, sample_size=min(2000, len(digits)))
        sil_norm = silhouette_score(projections_normalized, digits, sample_size=min(2000, len(digits)))
        logger.info(f"  [QUALITY] Silhouette (raw 9D):  {sil_raw:.3f}")
        logger.info(f"  [QUALITY] Silhouette (norm 9D): {sil_norm:.3f}")
    except ImportError:
        logger.info("  sklearn not available for silhouette score")

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 4: Run UMAP
    # ══════════════════════════════════════════════════════════════════════
    logger.info(f"\n{'═'*60}")
    logger.info(f"  STEP 4: Run UMAP on 9D projections")
    logger.info(f"{'═'*60}")

    try:
        import umap
    except ImportError:
        logger.error("umap-learn not installed! Install with: pip install umap-learn")
        logger.info("Falling back to PCA-only visualization...")
        umap = None

    if umap is not None:
        # Run UMAP on σ-normalized projections (better frequency balance)
        t0 = time.time()
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            metric="euclidean",
            random_state=42,
        )
        embedding = reducer.fit_transform(projections_normalized)
        elapsed = time.time() - t0
        logger.info(f"  UMAP (normalized) completed in {elapsed:.1f}s, shape={embedding.shape}")

        # Transform per-digit means through the same UMAP for centroid markers
        embedding_means = reducer.transform(digit_mean_norm)

        # ── Generate UMAP plots ──────────────────────────────────────────
        logger.info(f"\n{'═'*60}")
        logger.info(f"  STEP 5: Generate plots")
        logger.info(f"{'═'*60}")

        prefix = f"exp7_umap_{args.model}"

        # Plot 1: UMAP by digit
        plot_umap_by_digit(
            embedding, metadata, args.model, comp_layer,
            PLOT_DIR / f"{prefix}_by_digit.png",
            centroids=embedding_means,
        )

        # Plot 2: UMAP by carry
        plot_umap_by_carry(
            embedding, metadata, args.model, comp_layer,
            PLOT_DIR / f"{prefix}_by_carry.png"
        )

        # Plot 3: Per-digit carry panels
        plot_umap_carry_per_digit(
            embedding, metadata, args.model, comp_layer,
            PLOT_DIR / f"{prefix}_carry_per_digit.png"
        )

    # Plot 4: 3D PCA (always, even without UMAP)
    plot_pca_3d(
        projections_9d, metadata, args.model, comp_layer,
        PLOT_DIR / f"exp7_pca3d_{args.model}.png"
    )

    # Plot 5: Per-frequency 2D projections
    plot_frequency_decomposition(
        projections_9d, metadata, freq_assignments, S,
        args.model, comp_layer,
        PLOT_DIR / f"exp7_freq_pairs_{args.model}.png"
    )

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 6: Save metadata as JSON
    # ══════════════════════════════════════════════════════════════════════
    results = {
        "model": args.model,
        "comp_layer": comp_layer,
        "n_problems": len(test_problems),
        "singular_values": S.tolist(),
        "freq_assignments": freq_assignments,
        "d_model": d_model,
    }

    if umap is not None:
        # Compute per-digit cluster stats
        digits = np.array([m["digit"] for m in metadata])
        carries = np.array([m["carry"] for m in metadata])
        cluster_stats = {}
        for d in range(10):
            mask = digits == d
            if mask.sum() > 0:
                cluster_center = embedding[mask].mean(axis=0)
                spread = np.sqrt(np.mean(np.sum((embedding[mask] - cluster_center) ** 2, axis=1)))
                n_carry = int((carries[mask] == 1).sum())
                n_no_carry = int((carries[mask] == 0).sum())
                cluster_stats[str(d)] = {
                    "centroid": cluster_center.tolist(),
                    "spread": float(spread),
                    "n_carry": n_carry,
                    "n_no_carry": n_no_carry,
                }
        results["cluster_stats"] = cluster_stats

        try:
            from sklearn.metrics import silhouette_score
            results["silhouette_9d"] = float(silhouette_score(
                projections_9d, digits, sample_size=min(2000, len(digits))))
            results["silhouette_umap"] = float(silhouette_score(
                embedding, digits))
        except ImportError:
            pass

    suffix = "_direct" if args.direct_answer else ""
    out_path = RESULTS_DIR / f"fourier_umap_{args.model}{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n  Results saved to {out_path}")

    logger.info(f"\n{'═'*60}")
    logger.info(f"  DONE — Experiment 7 complete for {args.model}")
    logger.info(f"{'═'*60}")


if __name__ == "__main__":
    main()
