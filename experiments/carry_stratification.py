#!/usr/bin/env python3
"""
Experiment 1: Carry Stratification of 9D Fourier Subspace

Tests whether the Fourier structure differs between carry (a%10+b%10 >= 10)
and no-carry (a%10+b%10 < 10) arithmetic problems.

Key questions:
  1. Do carry and no-carry problems produce the same 9D Fourier basis?
  2. Are frequency purities and singular value profiles preserved?
  3. Does the carry direction live inside or outside the Fourier subspace?

Design note:
  - Digit 9 NEVER has carry (max ones sum = 9+9 = 18 < 19)
  - Digit 0 has very few no-carry samples (needs a%10 + b%10 = 0)
  - We handle this by tracking sample counts and skipping under-sampled digits

Usage:
    python carry_stratification.py --model gemma-2b --device mps
    python carry_stratification.py --model phi-3 --device mps
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# Minimum samples per digit to include in SVD analysis
MIN_SAMPLES_PER_DIGIT = 5

FREQ_LABELS = {1: "k=1 (ordinal)", 2: "k=2 (mod-5)", 3: "k=3", 4: "k=4", 5: "k=5 (parity)"}
FREQ_COLORS = {1: "#2196F3", 2: "#FF9800", 3: "#4CAF50", 4: "#9C27B0", 5: "#F44336"}


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_basis_from_means(means, counts, label="", min_samples=MIN_SAMPLES_PER_DIGIT):
    """
    Compute 9D (or fewer) Fourier basis from per-digit means.

    Only includes digits with >= min_samples. Returns:
        V: (d_model, n_dirs) orthonormal columns
        S: (n_dirs,) singular values
        U: (n_used_digits, n_dirs) left singular vectors
        centroid: (d_model,) global mean of used digits
        used_digits: list of digit indices included
    """
    used_digits = [d for d in range(10) if counts[d] >= min_samples]
    n_used = len(used_digits)

    if n_used < 3:
        logger.warning(f"  [{label}] Only {n_used} digits with enough samples — cannot compute basis")
        return None, None, None, None, used_digits

    # Select used digit means
    used_means = means[used_digits]  # (n_used, d_model)

    # Center
    centroid = used_means.mean(axis=0, keepdims=True)  # (1, d_model)
    M = used_means - centroid  # (n_used, d_model)

    # SVD
    U_full, S_full, Vt_full = np.linalg.svd(M, full_matrices=False)
    n_dirs = min(9, n_used - 1)  # at most n_used-1 non-trivial directions

    V = Vt_full[:n_dirs].T   # (d_model, n_dirs)
    S = S_full[:n_dirs]
    U = U_full[:, :n_dirs]   # (n_used, n_dirs)

    # Verify orthonormality
    gram = V.T @ V
    err = np.abs(gram - np.eye(n_dirs)).max()
    assert err < 1e-5, f"[{label}] V not orthonormal, max err = {err:.2e}"

    logger.info(f"  [{label}] Basis: {n_dirs}D from {n_used} digits, "
                f"orthonormality err = {err:.2e} ✓")

    return V, S, U, centroid.squeeze(), used_digits


def analyze_group_dft(label, U, S, used_digits):
    """
    Perform DFT analysis on SVD directions.
    U rows correspond to used_digits, not necessarily 0-9.

    Reconstructs full 10-digit score patterns (inserting zeros for missing
    digits) so DFT frequencies correspond to the correct ℤ/10ℤ frequencies.

    NOTE: Zero-filling missing digits introduces mild spectral distortion
    (~10% per missing digit). This is acceptable for comparative analysis.
    """
    n_used = len(used_digits)
    n_dirs = U.shape[1]
    missing_digits = [d for d in range(10) if d not in used_digits]

    if missing_digits:
        logger.warning(f"  [{label}] Missing digits {missing_digits} — zero-filled in DFT")

    # U is (n_used, n_dirs). U[i, j] = score of used_digits[i] on direction j.
    # Reconstruct full (n_dirs, 10) pattern.
    full_scores = np.zeros((n_dirs, 10))
    for j in range(n_dirs):
        for i, d in enumerate(used_digits):
            full_scores[j, d] = U[i, j]

    # Center each row
    scores_centered = full_scores - full_scores.mean(axis=1, keepdims=True)

    # DFT on 10-point patterns
    dft = np.fft.fft(scores_centered, axis=1)  # (n_dirs, 10) complex
    power = np.abs(dft) ** 2

    # Group frequencies
    freq_power = np.zeros((n_dirs, 6))
    freq_power[:, 0] = power[:, 0]
    for k in range(1, 5):
        freq_power[:, k] = power[:, k] + power[:, 10 - k]
    freq_power[:, 5] = power[:, 5]

    # Normalize to fractions (exclude DC)
    total_power = freq_power[:, 1:].sum(axis=1, keepdims=True)
    total_power = np.maximum(total_power, 1e-10)
    freq_frac = freq_power[:, 1:] / total_power  # (n_dirs, 5) for k=1..5

    logger.info(f"\n{'='*70}")
    logger.info(f"  {label}")
    logger.info(f"{'='*70}")
    logger.info(f"  {'Dir':>4}  {'σ':>8}  {'k=1':>7}  {'k=2':>7}  {'k=3':>7}  "
                f"{'k=4':>7}  {'k=5':>7}  {'Dominant':>12}")
    logger.info(f"  {'─'*65}")

    results = []
    for i in range(n_dirs):
        dominant_k = np.argmax(freq_frac[i]) + 1
        dominant_pct = freq_frac[i, dominant_k - 1] * 100

        logger.info(
            f"  {i+1:>4}  {S[i]:>8.2f}  "
            f"{freq_frac[i,0]*100:>6.1f}%  {freq_frac[i,1]*100:>6.1f}%  "
            f"{freq_frac[i,2]*100:>6.1f}%  {freq_frac[i,3]*100:>6.1f}%  "
            f"{freq_frac[i,4]*100:>6.1f}%  "
            f"k={dominant_k} ({dominant_pct:.0f}%)"
        )

        results.append({
            "direction": i + 1,
            "singular_value": float(S[i]),
            "freq_fractions": {str(k): float(freq_frac[i, k-1]) for k in range(1, 6)},
            "dominant_freq": int(dominant_k),
            "dominant_pct": float(dominant_pct),
        })

    # Check Fourier basis quality
    dom_counts = defaultdict(int)
    for r in results:
        dom_counts[r["dominant_freq"]] += 1

    logger.info(f"\n  Frequency assignment summary (n_dirs={n_dirs}):")
    for k in range(1, 6):
        dof = 1 if k == 5 else 2
        actual = dom_counts.get(k, 0)
        match = "✓" if actual == dof else f"(expected {dof})"
        logger.info(f"    k={k}: {actual} directions {match}")

    # Perfect check only meaningful with 9 directions (all 10 digits)
    if n_dirs == 9:
        perfect = all(dom_counts.get(k, 0) == (1 if k == 5 else 2) for k in range(1, 6))
        if perfect:
            logger.info(f"\n  ★ PERFECT FOURIER BASIS ★")
        else:
            logger.info(f"\n  Not a perfect Fourier basis")
    else:
        perfect = False
        logger.info(f"\n  [NOTE] Only {n_dirs} directions (need 9 for perfect Fourier test)")

    mean_purity = np.mean([r["dominant_pct"] for r in results])
    logger.info(f"  Mean purity: {mean_purity:.1f}%")

    return results, perfect, mean_purity


def principal_angles(V1, V2):
    """
    Compute principal angles between two subspaces.

    Args:
        V1: (d, k1) orthonormal columns
        V2: (d, k2) orthonormal columns

    Returns:
        angles: min(k1, k2) principal angles in radians (sorted ascending)
        cos_angles: cosines of principal angles (sorted descending)
    """
    # SVD of V1.T @ V2
    M = V1.T @ V2  # (k1, k2)
    _, sigmas, _ = np.linalg.svd(M)

    # Clamp to [0, 1] for numerical safety
    cos_angles = np.clip(sigmas, 0.0, 1.0)
    angles = np.arccos(cos_angles)

    return angles, cos_angles


def analyze_carry_directions(means_all, means_carry, means_nocarry,
                              counts_carry, counts_nocarry,
                              V_all, centroid_all):
    """
    Analyze where the carry direction lives relative to the Fourier subspace.

    For each digit with enough carry and no-carry samples:
      δ_d = mean_carry[d] - mean_nocarry[d]

    Then decompose δ into Fourier component and orthogonal component.
    """
    results = {}
    valid_digits = []
    deltas = []

    for d in range(10):
        if counts_carry[d] >= MIN_SAMPLES_PER_DIGIT and counts_nocarry[d] >= MIN_SAMPLES_PER_DIGIT:
            delta = means_carry[d] - means_nocarry[d]
            valid_digits.append(d)
            deltas.append(delta)

            # Project delta onto Fourier subspace
            delta_centered = delta  # difference already removes shared mean
            proj_fourier = V_all @ (V_all.T @ delta_centered)  # component in subspace
            proj_ortho = delta_centered - proj_fourier           # component outside

            norm_total = np.linalg.norm(delta_centered)
            norm_fourier = np.linalg.norm(proj_fourier)
            norm_ortho = np.linalg.norm(proj_ortho)

            # Use variance fractions (squared norms) so they sum to 100%
            var_total = norm_total ** 2
            frac_fourier = (norm_fourier ** 2 / var_total * 100) if var_total > 1e-20 else 0
            frac_ortho = (norm_ortho ** 2 / var_total * 100) if var_total > 1e-20 else 0

            results[d] = {
                "norm_total": float(norm_total),
                "norm_fourier": float(norm_fourier),
                "norm_ortho": float(norm_ortho),
                "var_frac_fourier_pct": float(frac_fourier),
                "var_frac_ortho_pct": float(frac_ortho),
                "n_carry": int(counts_carry[d]),
                "n_nocarry": int(counts_nocarry[d]),
            }

            logger.info(
                f"  Digit {d}: ||δ||={norm_total:.2f}, "
                f"Fourier={frac_fourier:.1f}%, Ortho={frac_ortho:.1f}% (variance) "
                f"(carry={counts_carry[d]}, no-carry={counts_nocarry[d]})"
            )

    # Average carry direction
    if len(deltas) >= 2:
        avg_delta = np.mean(deltas, axis=0)
        proj_f = V_all @ (V_all.T @ avg_delta)
        proj_o = avg_delta - proj_f
        norm_t = np.linalg.norm(avg_delta)
        norm_f = np.linalg.norm(proj_f)
        norm_o = np.linalg.norm(proj_o)
        var_t = norm_t ** 2
        vf = (norm_f ** 2 / var_t * 100) if var_t > 1e-20 else 0
        vo = (norm_o ** 2 / var_t * 100) if var_t > 1e-20 else 0

        logger.info(f"\n  Average carry direction (digits {valid_digits}):")
        logger.info(f"    ||δ_avg|| = {norm_t:.2f}")
        logger.info(f"    Fourier component: {vf:.1f}% of variance")
        logger.info(f"    Orthogonal component: {vo:.1f}% of variance")

        results["average"] = {
            "digits_used": valid_digits,
            "norm_total": float(norm_t),
            "var_frac_fourier_pct": float(vf),
            "var_frac_ortho_pct": float(vo),
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_sv_comparison(S_carry, S_nocarry, S_all, model_name, layer, save_path):
    """Compare singular value profiles across groups."""
    fig, ax = plt.subplots(figsize=(8, 5))

    n_all = len(S_all)
    n_carry = len(S_carry)
    n_nocarry = len(S_nocarry)
    x_all = np.arange(1, n_all + 1)
    x_carry = np.arange(1, n_carry + 1)
    x_nocarry = np.arange(1, n_nocarry + 1)

    ax.plot(x_all, S_all, "k-o", label="All problems", linewidth=2, markersize=7)
    ax.plot(x_carry, S_carry, "r--s", label="Carry only", linewidth=1.5, markersize=6)
    ax.plot(x_nocarry, S_nocarry, "b--^", label="No carry only", linewidth=1.5, markersize=6)

    ax.set_xlabel("SVD Direction", fontsize=12)
    ax.set_ylabel("Singular Value (σ)", fontsize=12)
    ax.set_title(f"{model_name} L{layer}: Singular Value Profile — Carry Stratification",
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, max(n_all, n_carry, n_nocarry) + 1))

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {save_path}")


def plot_purity_comparison(results_carry, results_nocarry, results_all,
                            model_name, layer, save_path):
    """Compare frequency purities side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, results, title in zip(
        axes,
        [results_all, results_carry, results_nocarry],
        ["All Problems", "Carry Only", "No Carry Only"],
    ):
        n_dirs = len(results)
        x = np.arange(n_dirs)
        bottom = np.zeros(n_dirs)

        for k in range(1, 6):
            fracs = [r["freq_fractions"].get(str(k), 0) for r in results]
            ax.bar(x, fracs, bottom=bottom, color=FREQ_COLORS[k],
                   label=FREQ_LABELS[k], edgecolor="white", linewidth=0.5)
            bottom += np.array(fracs)

        ax.set_xlabel("SVD Direction", fontsize=10)
        ax.set_ylabel("DFT Power Fraction", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"D{i+1}" for i in range(n_dirs)], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, loc="upper right", ncol=2)

    fig.suptitle(f"{model_name} L{layer}: Frequency Purity — Carry Stratification",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {save_path}")


def plot_principal_angles(angles_carry_nocarry, angles_carry_all, angles_nocarry_all,
                           model_name, layer, save_path):
    """Plot principal angle spectra."""
    fig, ax = plt.subplots(figsize=(8, 5))

    n1 = len(angles_carry_nocarry)
    n2 = len(angles_carry_all)
    n3 = len(angles_nocarry_all)

    ax.plot(range(1, n1+1), np.degrees(angles_carry_nocarry), "go-",
            label="Carry vs No-Carry", linewidth=2, markersize=8)
    ax.plot(range(1, n2+1), np.degrees(angles_carry_all), "r--s",
            label="Carry vs All", linewidth=1.5, markersize=6)
    ax.plot(range(1, n3+1), np.degrees(angles_nocarry_all), "b--^",
            label="No-Carry vs All", linewidth=1.5, markersize=6)

    ax.axhline(90, color="gray", linestyle=":", alpha=0.5, label="Orthogonal (90°)")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.3)

    ax.set_xlabel("Principal Angle Index", fontsize=12)
    ax.set_ylabel("Angle (degrees)", fontsize=12)
    ax.set_title(f"{model_name} L{layer}: Principal Angles Between Fourier Subspaces\n"
                 f"(0° = identical, 90° = orthogonal)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(-5, 95)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {save_path}")


def plot_carry_direction(carry_dir_results, model_name, layer, save_path):
    """Plot carry direction decomposition per digit."""
    per_digit = {d: v for d, v in carry_dir_results.items() if isinstance(d, int)}
    if len(per_digit) == 0:
        return

    digits = sorted(per_digit.keys())
    frac_fourier = [per_digit[d]["var_frac_fourier_pct"] for d in digits]
    frac_ortho = [per_digit[d]["var_frac_ortho_pct"] for d in digits]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(digits))
    width = 0.35

    ax.bar(x - width/2, frac_fourier, width, label="Inside Fourier 9D",
           color="#2196F3", edgecolor="white")
    ax.bar(x + width/2, frac_ortho, width, label="Orthogonal to Fourier",
           color="#F44336", edgecolor="white")

    ax.set_xlabel("Digit", fontsize=12)
    ax.set_ylabel("Fraction of variance (%)", fontsize=12)
    ax.set_title(f"{model_name} L{layer}: Carry Direction Decomposition\n"
                 f"δ_d = mean(carry) − mean(no-carry) per digit", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in digits])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Add average line if available
    if "average" in carry_dir_results:
        avg = carry_dir_results["average"]
        ax.axhline(avg["var_frac_fourier_pct"], color="#2196F3", linestyle="--",
                    alpha=0.7, linewidth=1.5)
        ax.axhline(avg["var_frac_ortho_pct"], color="#F44336", linestyle="--",
                    alpha=0.7, linewidth=1.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Carry Stratification")
    parser.add_argument("--model", default="gemma-2b", choices=list(MODEL_MAP.keys()))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--comp-layer", type=int, default=None)
    parser.add_argument("--n-per-digit", type=int, default=200,
                        help="Problems per digit (more than usual for carry balance)")
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
    #  STEP 1: Generate problems, filter, stratify
    # ══════════════════════════════════════════════════════════════════════
    logger.info(f"\n{'═'*60}")
    logger.info(f"  STEP 1: Generate & stratify problems")
    logger.info(f"{'═'*60}")

    if args.direct_answer:
        all_problems, _ = generate_direct_answer_problems(n_per_digit=args.n_per_digit)
        logger.info(f"  Generated {len(all_problems)} direct-answer problems")
        correct = filter_correct_direct_answer(model, all_problems, max_n=len(all_problems))
    else:
        all_problems, _ = generate_teacher_forced_problems(n_per_digit=args.n_per_digit)
        logger.info(f"  Generated {len(all_problems)} teacher-forced problems")
        correct = filter_correct_teacher_forced(model, all_problems, max_n=len(all_problems))
    logger.info(f"  Correct: {len(correct)}")

    # Split by carry status
    carry_problems = [p for p in correct if p.get("has_carry", 0) == 1]
    nocarry_problems = [p for p in correct if p.get("has_carry", 0) == 0]

    logger.info(f"  Carry:    {len(carry_problems)}")
    logger.info(f"  No-carry: {len(nocarry_problems)}")

    # Per-digit carry distribution
    logger.info(f"\n  Per-digit carry distribution:")
    for d in range(10):
        n_c = sum(1 for p in carry_problems if p["ones_digit"] == d)
        n_nc = sum(1 for p in nocarry_problems if p["ones_digit"] == d)
        flag = " ← SPARSE" if n_c < MIN_SAMPLES_PER_DIGIT or n_nc < MIN_SAMPLES_PER_DIGIT else ""
        logger.info(f"    Digit {d}: carry={n_c:>4}, no-carry={n_nc:>4}{flag}")

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 2: Collect activations (single pass) & compute per-digit means
    # ══════════════════════════════════════════════════════════════════════
    logger.info(f"\n{'═'*60}")
    logger.info(f"  STEP 2: Collect activations & compute per-digit means")
    logger.info(f"{'═'*60}")

    t0 = time.time()
    hook_name = f"blocks.{comp_layer}.hook_resid_post"

    # Collect individual activations in a SINGLE pass over all correct problems
    digit_acts_carry = defaultdict(list)    # digit → [activation vectors]
    digit_acts_nocarry = defaultdict(list)
    for prob in correct:
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
            if prob.get("has_carry", 0) == 1:
                digit_acts_carry[digit].append(act_vec)
            else:
                digit_acts_nocarry[digit].append(act_vec)

    # Compute per-digit means for each group
    means_carry = np.zeros((10, d_model))
    means_nocarry = np.zeros((10, d_model))
    means_all = np.zeros((10, d_model))
    counts_carry = np.zeros(10, dtype=int)
    counts_nocarry = np.zeros(10, dtype=int)
    counts_all = np.zeros(10, dtype=int)

    for d in range(10):
        nc = len(digit_acts_carry[d])
        nn = len(digit_acts_nocarry[d])
        counts_carry[d] = nc
        counts_nocarry[d] = nn
        counts_all[d] = nc + nn

        if nc > 0:
            means_carry[d] = np.mean(digit_acts_carry[d], axis=0)
        if nn > 0:
            means_nocarry[d] = np.mean(digit_acts_nocarry[d], axis=0)
        if nc + nn > 0:
            all_acts = digit_acts_carry[d] + digit_acts_nocarry[d]
            means_all[d] = np.mean(all_acts, axis=0)

        logger.info(f"    Digit {d}: carry={nc:>4}, no-carry={nn:>4}, total={nc+nn:>4}")

    elapsed = time.time() - t0
    logger.info(f"\n  Activations collected in {elapsed:.1f}s ({len(correct)} forward passes)")

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 3: Compute Fourier basis for each group
    # ══════════════════════════════════════════════════════════════════════
    logger.info(f"\n{'═'*60}")
    logger.info(f"  STEP 3: SVD & DFT for each group")
    logger.info(f"{'═'*60}")

    V_all, S_all, U_all, cent_all, digits_all = compute_basis_from_means(
        means_all, counts_all, "ALL")
    V_carry, S_carry, U_carry, cent_carry, digits_carry = compute_basis_from_means(
        means_carry, counts_carry, "CARRY")
    V_nocarry, S_nocarry, U_nocarry, cent_nocarry, digits_nocarry = compute_basis_from_means(
        means_nocarry, counts_nocarry, "NO-CARRY")

    if V_all is None or V_carry is None or V_nocarry is None:
        logger.error("  Cannot proceed — insufficient data for one or more groups")
        return

    logger.info(f"\n  Digits used — ALL: {digits_all}, CARRY: {digits_carry}, NO-CARRY: {digits_nocarry}")

    # DFT analysis for each group
    dft_all, perfect_all, purity_all = analyze_group_dft(
        f"ALL ({len(correct)} problems)", U_all, S_all, digits_all)
    dft_carry, perfect_carry, purity_carry = analyze_group_dft(
        f"CARRY ({len(carry_problems)} problems)", U_carry, S_carry, digits_carry)
    dft_nocarry, perfect_nocarry, purity_nocarry = analyze_group_dft(
        f"NO-CARRY ({len(nocarry_problems)} problems)", U_nocarry, S_nocarry, digits_nocarry)

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 4: Subspace alignment (principal angles)
    # ══════════════════════════════════════════════════════════════════════
    logger.info(f"\n{'═'*60}")
    logger.info(f"  STEP 4: Subspace alignment")
    logger.info(f"{'═'*60}")

    k_min = min(V_carry.shape[1], V_nocarry.shape[1], V_all.shape[1])

    angles_cn, cos_cn = principal_angles(V_carry[:, :k_min], V_nocarry[:, :k_min])
    angles_ca, cos_ca = principal_angles(V_carry[:, :k_min], V_all[:, :k_min])
    angles_na, cos_na = principal_angles(V_nocarry[:, :k_min], V_all[:, :k_min])

    logger.info(f"\n  Principal angles (Carry vs No-Carry):")
    for i, (a, c) in enumerate(zip(angles_cn, cos_cn)):
        logger.info(f"    Angle {i+1}: {np.degrees(a):.2f}° (cos={c:.4f})")

    logger.info(f"\n  Principal angles (Carry vs All):")
    for i, (a, c) in enumerate(zip(angles_ca, cos_ca)):
        logger.info(f"    Angle {i+1}: {np.degrees(a):.2f}° (cos={c:.4f})")

    logger.info(f"\n  Principal angles (No-Carry vs All):")
    for i, (a, c) in enumerate(zip(angles_na, cos_na)):
        logger.info(f"    Angle {i+1}: {np.degrees(a):.2f}° (cos={c:.4f})")

    # Summary: mean angle
    mean_angle_cn = np.degrees(angles_cn.mean())
    mean_angle_ca = np.degrees(angles_ca.mean())
    mean_angle_na = np.degrees(angles_na.mean())
    logger.info(f"\n  Mean principal angles:")
    logger.info(f"    Carry vs No-Carry: {mean_angle_cn:.2f}°")
    logger.info(f"    Carry vs All:      {mean_angle_ca:.2f}°")
    logger.info(f"    No-Carry vs All:   {mean_angle_na:.2f}°")

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 5: Carry direction analysis
    # ══════════════════════════════════════════════════════════════════════
    logger.info(f"\n{'═'*60}")
    logger.info(f"  STEP 5: Carry direction geometry")
    logger.info(f"{'═'*60}")

    carry_dir_results = analyze_carry_directions(
        means_all, means_carry, means_nocarry,
        counts_carry, counts_nocarry,
        V_all, cent_all,
    )

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 6: Generate plots
    # ══════════════════════════════════════════════════════════════════════
    logger.info(f"\n{'═'*60}")
    logger.info(f"  STEP 6: Generate plots")
    logger.info(f"{'═'*60}")

    prefix = f"exp1_carry_{args.model}"

    plot_sv_comparison(
        S_carry, S_nocarry, S_all, args.model, comp_layer,
        PLOT_DIR / f"{prefix}_sv_profile.png"
    )

    plot_purity_comparison(
        dft_carry, dft_nocarry, dft_all, args.model, comp_layer,
        PLOT_DIR / f"{prefix}_purity.png"
    )

    plot_principal_angles(
        angles_cn, angles_ca, angles_na, args.model, comp_layer,
        PLOT_DIR / f"{prefix}_angles.png"
    )

    plot_carry_direction(
        carry_dir_results, args.model, comp_layer,
        PLOT_DIR / f"{prefix}_direction.png"
    )

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 7: Save results JSON
    # ══════════════════════════════════════════════════════════════════════
    results = {
        "model": args.model,
        "comp_layer": comp_layer,
        "d_model": d_model,
        "n_total": len(correct),
        "n_carry": len(carry_problems),
        "n_nocarry": len(nocarry_problems),
        "counts_carry": counts_carry.tolist(),
        "counts_nocarry": counts_nocarry.tolist(),
        "all": {
            "singular_values": S_all.tolist(),
            "perfect_fourier": perfect_all,
            "mean_purity": purity_all,
            "dft": dft_all,
            "digits_used": digits_all,
        },
        "carry": {
            "singular_values": S_carry.tolist(),
            "perfect_fourier": perfect_carry,
            "mean_purity": purity_carry,
            "dft": dft_carry,
            "digits_used": digits_carry,
        },
        "nocarry": {
            "singular_values": S_nocarry.tolist(),
            "perfect_fourier": perfect_nocarry,
            "mean_purity": purity_nocarry,
            "dft": dft_nocarry,
            "digits_used": digits_nocarry,
        },
        "subspace_alignment": {
            "carry_vs_nocarry": {
                "angles_deg": np.degrees(angles_cn).tolist(),
                "mean_angle_deg": float(mean_angle_cn),
            },
            "carry_vs_all": {
                "angles_deg": np.degrees(angles_ca).tolist(),
                "mean_angle_deg": float(mean_angle_ca),
            },
            "nocarry_vs_all": {
                "angles_deg": np.degrees(angles_na).tolist(),
                "mean_angle_deg": float(mean_angle_na),
            },
        },
        "carry_direction": {
            str(k): v for k, v in carry_dir_results.items()
        },
    }

    suffix = "_direct" if args.direct_answer else ""
    out_path = RESULTS_DIR / f"carry_stratification_{args.model}{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n  Results saved to {out_path}")

    # ══════════════════════════════════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════════════════════════════════
    logger.info(f"\n{'═'*60}")
    logger.info(f"  SUMMARY — Experiment 1: Carry Stratification ({args.model})")
    logger.info(f"{'═'*60}")
    logger.info(f"  Perfect Fourier — ALL: {perfect_all}, CARRY: {perfect_carry}, NO-CARRY: {perfect_nocarry}")
    logger.info(f"  Mean purity    — ALL: {purity_all:.1f}%, CARRY: {purity_carry:.1f}%, NO-CARRY: {purity_nocarry:.1f}%")
    logger.info(f"  Subspace alignment (carry vs no-carry): mean angle = {mean_angle_cn:.2f}°")
    if "average" in carry_dir_results:
        avg = carry_dir_results["average"]
        logger.info(f"  Carry direction: {avg['var_frac_fourier_pct']:.1f}% variance inside Fourier, "
                    f"{avg['var_frac_ortho_pct']:.1f}% orthogonal")
    logger.info(f"\n  DONE ✓")


if __name__ == "__main__":
    main()
