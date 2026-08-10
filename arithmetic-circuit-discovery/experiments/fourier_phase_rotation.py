#!/usr/bin/env python3
"""
Frequency-Specific Fourier Phase Rotation
==========================================

FIXES the critical bug in fisher_phase_shift.py:
  OLD: rotated in top-2 PCA plane (k=5 parity + k=3 for Gemma L19 — WRONG)
  NEW: rotates in frequency-SPECIFIC 2D Fourier planes from digit-mean SVD

For frequency k, the encoding is (cos(2pi*k*d/10), sin(2pi*k*d/10)).
To shift digit d -> d+j, rotate each freq-k plane by angle 2*pi*k*j/10.

Tests:
  A. COHERENT: rotate ALL freq planes simultaneously (correct multi-freq shift)
  B. K=1 ONLY: rotate only k=1 ordinal plane (diagnostic)
  C. TOP-2 PCA: original (buggy) approach as control

Usage:
    python fourier_phase_rotation.py --model gemma-2b --layers 19,25
    python fourier_phase_rotation.py --model phi-3-mini --layers 26,28
    python fourier_phase_rotation.py --model llama-3b --layers 20,27 --direct-answer
    python fourier_phase_rotation.py --model gemma-2b --layers 19 --dry-run
"""

import os
import sys
import torch
import numpy as np
import json
import gc
import argparse
import logging
import warnings
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from arithmetic_circuit_scan_updated import (
    generate_direct_answer_problems,
    filter_correct_direct_answer,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("mathematical_toolkit_results")
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_MAP = {
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "gemma-2b": "google/gemma-2-2b",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "llama-3b": "meta-llama/Llama-3.2-3B",
}

# Default computation and readout layers per model
COMP_LAYERS = {"gemma-2b": 19, "phi-3-mini": 26, "llama-3b": 20}
READOUT_LAYERS = {"gemma-2b": 25, "phi-3-mini": 28, "llama-3b": 27}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ═══════════════════════════════════════════════════════════════
# STEP 1: PROBLEM GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_test_problems(max_operand=30, few_shot=True):
    """Generate arithmetic problems."""
    few_shot_prefix = ""
    if few_shot:
        few_shot_prefix = "Calculate:\n12 + 7 = 19\n34 + 15 = 49\n"
    problems = []
    for a in range(max_operand):
        for b in range(max_operand):
            answer = a + b
            prompt = f"{few_shot_prefix}{a} + {b} = "
            first_digit = str(answer)[0]
            problems.append({
                "a": a, "b": b, "answer": answer,
                "prompt": prompt,
                "ones_digit": answer % 10,
                "tens_digit": (answer // 10) % 10,
                "first_digit": int(first_digit),
                "n_digits": len(str(answer)),
                "carry": 1 if (a % 10 + b % 10) >= 10 else 0,
            })
    return problems


def filter_correct_problems(model, problems, n_test=200, seed=42):
    """Keep problems where model predicts correct first digit."""
    import random
    rng = random.Random(seed)
    subset = rng.sample(problems, min(n_test, len(problems)))
    correct = []
    for prob in subset:
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
        with torch.no_grad():
            logits = model(tokens)
        pred_tok = logits[0, -1].argmax(dim=-1).item()
        pred_str = model.tokenizer.decode([pred_tok]).strip()
        try:
            if pred_str == str(prob["answer"])[0]:
                correct.append(prob)
        except ValueError:
            continue
    logger.info(f"  Model correct on {len(correct)}/{len(subset)} = {len(correct)/len(subset)*100:.1f}%")
    return correct


def generate_single_digit_problems(few_shot=True):
    """Generate only single-digit answer problems (a+b < 10).
    
    This ensures first_digit == ones_digit == answer, so the exact_mod10
    metric directly measures whether the Fourier rotation shifted the digit.
    Yields 55 problems (all a,b >= 0 with a+b <= 9).
    """
    few_shot_prefix = ""
    if few_shot:
        few_shot_prefix = "Calculate:\n12 + 7 = 19\n34 + 15 = 49\n"
    problems = []
    for a in range(10):
        for b in range(10 - a):  # a + b <= 9
            answer = a + b
            prompt = f"{few_shot_prefix}{a} + {b} = "
            problems.append({
                "a": a, "b": b, "answer": answer,
                "prompt": prompt,
                "ones_digit": answer,
                "tens_digit": 0,
                "first_digit": answer,
                "n_digits": 1,
                "carry": 0,
            })
    logger.info(f"  Generated {len(problems)} single-digit problems (a+b < 10)")
    return problems


def filter_correct_single_digit(model, problems):
    """Filter single-digit problems where model predicts correct answer."""
    correct = []
    for prob in problems:
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
        with torch.no_grad():
            logits = model(tokens)
        pred_tok = logits[0, -1].argmax(dim=-1).item()
        pred_str = model.tokenizer.decode([pred_tok]).strip()
        try:
            if int(pred_str) == prob["answer"]:
                correct.append(prob)
        except ValueError:
            continue
    logger.info(f"  Model correct on {len(correct)}/{len(problems)} single-digit = {len(correct)/len(problems)*100:.1f}%")
    return correct


def generate_single_digit_direct_answer_problems():
    """Generate all 55 single-digit problems in direct-answer format (no few-shot prefix).
    
    Same as generate_single_digit_problems() but uses the direct-answer prompt
    format 'a + b = ' instead of few-shot 'Calculate:\n12 + 7 = 19\n...'.
    This gives LLaMA and other direct-answer models the full 55-problem test set.
    """
    problems = []
    for a in range(10):
        for b in range(10 - a):  # a + b <= 9
            answer = a + b
            prompt = f"{a} + {b} = "
            problems.append({
                "a": a, "b": b, "answer": answer,
                "prompt": prompt,
                "ones_digit": answer,
                "tens_digit": 0,
                "first_digit": answer,
                "n_digits": 1,
                "carry": 0,
                "target_str": str(answer),
            })
    logger.info(f"  Generated {len(problems)} single-digit direct-answer problems (a+b < 10)")
    return problems


# ═══════════════════════════════════════════════════════════════
# STEP 2: COMPUTE FOURIER BASIS VIA HYBRID SVD+DFT
# ═══════════════════════════════════════════════════════════════

def compute_digit_fourier_basis(model, problems, layer, n_problems=500):
    """
    Compute the 9D Fourier basis using HYBRID SVD+DFT method.

    Pure SVD has sign/axis ambiguity (v or -v, axes may mix cos+sin).
    Pure DFT projection gives correct axes but non-orthogonal d_model vectors.

    Hybrid SVD+DFT method:
    1. SVD of centered digit means → orthogonal 9D subspace (U9, S9, V9)
    2. Project DFT waves into SVD digit-loading space (U9.T @ wave)
       - These ARE orthogonal because U9@U9.T is the centering projector
         and DFT waves for k≥1 are centered (sum to zero)
    3. Build rotation matrix R from DFT projections in SVD space
    4. Final basis = V9.T @ R → orthonormal by construction

    GUARANTEES:
    - Orthonormality (from SVD subspace + orthogonal R)
    - v_cos is always the Cosine axis (from DFT wave projection)
    - v_sin is always the Sine axis (from DFT wave projection)
    - Rotation direction is always forward (d → d+j for positive j)
    - Same variance capture as SVD (100% of 9D total)

    Returns:
        basis: (d_model, 9) basis vectors (columns), cos/sin ordered
        freq_assignments: list of 9 ints — frequency per direction
        singular_values: (9,) array — signal strength per direction
        digit_scores: (9, 10) array — how each direction scores each digit
        freq_purities: list of 9 floats — DFT purity (verified, not assumed)
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    device = next(model.parameters()).device
    digit_acts = defaultdict(list)

    subset = problems[:n_problems]
    for i, prob in enumerate(subset):
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
            act = cache[hook_name][0, -1].cpu().float().numpy()
            digit_acts[prob["ones_digit"]].append(act)
            del cache
        if (i + 1) % 100 == 0:
            logger.info(f"    Collecting activations: {i+1}/{len(subset)}")

    # Per-digit means
    d_model = model.cfg.d_model
    digit_means = np.zeros((10, d_model))
    counts = []
    for d in range(10):
        n = len(digit_acts[d])
        counts.append(n)
        if n > 0:
            digit_means[d] = np.mean(digit_acts[d], axis=0)
        else:
            logger.warning(f"  No activations for digit {d}!")
    logger.info(f"  Digit counts: {counts} (min={min(counts)})")

    # Center
    M = digit_means - digit_means.mean(axis=0, keepdims=True)  # (10, d_model)

    # ─── Step 1: SVD for orthogonal 9D subspace ───
    U_full, S_full, Vt_full = np.linalg.svd(M, full_matrices=False)
    n_keep = min(9, len(S_full))
    U9 = U_full[:, :n_keep]       # (10, 9) — digit loadings in SVD space
    S9 = S_full[:n_keep]            # (9,) — singular values
    V9 = Vt_full[:n_keep, :]       # (9, d_model) — right singular vectors
    total_var = np.sum(S_full ** 2)

    logger.info(f"\n  SVD singular values: {S9.round(2)}")
    logger.info(f"  SVD total variance: {total_var:.1f}")

    # ─── Step 2: DFT rotation within SVD space ───
    # Project DFT waves into SVD digit-loading space.
    # Since U9@U9.T = centering projector and DFT waves (k≥1) are centered,
    # these projections preserve orthogonality: (U9.T @ cos_k1) · (U9.T @ cos_k2) = 0
    digits = np.arange(10)
    R_cols = []           # columns of rotation matrix in SVD space
    freq_assignments = []
    raw_sigma_cos = []    # DFT projection norms (for elliptical ratio)
    raw_sigma_sin = []

    logger.info(f"\n  Hybrid SVD+DFT (orthogonal subspace + DFT axis labeling):")
    for k in range(1, 6):
        cos_wave = np.cos(2 * np.pi * k * digits / 10)
        sin_wave = np.sin(2 * np.pi * k * digits / 10)

        # Project into SVD digit-loading space
        c_cos = U9.T @ cos_wave   # (9,) — cos wave in SVD coordinates
        c_sin = U9.T @ sin_wave   # (9,) — sin wave in SVD coordinates

        # Signal strength: ||M.T @ wave|| = ||diag(S9) @ (U9.T @ wave)||
        sc_cos = S9 * c_cos       # S-weighted coordinates
        sigma_cos = np.linalg.norm(sc_cos)
        raw_sigma_cos.append(sigma_cos)

        # Normalize in SVD space for rotation matrix R
        c_cos_norm = np.linalg.norm(c_cos)
        if c_cos_norm < 1e-10:
            logger.warning(f"  k={k}: cos direction has near-zero projection!")
            continue
        c_cos_hat = c_cos / c_cos_norm
        R_cols.append(c_cos_hat)
        freq_assignments.append(k)

        if k < 5:  # k=1..4 have sin components; k=5 (Nyquist) sin=0
            sc_sin = S9 * c_sin
            sigma_sin_raw = np.linalg.norm(sc_sin)

            # Gram-Schmidt: orthogonalize sin against cos in SVD space
            # (should be near-zero overlap since DFT waves are orthogonal)
            overlap = np.dot(c_sin, c_cos_hat)
            c_sin_perp = c_sin - overlap * c_cos_hat
            c_sin_norm = np.linalg.norm(c_sin_perp)

            if c_sin_norm < 1e-10:
                logger.warning(f"  k={k}: sin direction collapsed after orthogonalization!")
                continue

            c_sin_hat = c_sin_perp / c_sin_norm
            R_cols.append(c_sin_hat)
            freq_assignments.append(k)

            # σ_sin: project orthogonalized unit direction through S
            sigma_sin = np.linalg.norm(S9 * c_sin_hat)
            raw_sigma_sin.append(sigma_sin)

            logger.info(f"    k={k}: σ_cos={sigma_cos:.2f}, σ_sin={sigma_sin:.2f}, "
                        f"ratio={sigma_cos/sigma_sin:.2f}x, "
                        f"GS overlap={abs(overlap):.2e}")
        else:
            logger.info(f"    k={k}: σ_cos={sigma_cos:.2f} (Nyquist, sin=0)")

    # ─── Step 3: Build orthonormal basis in d_model space ───
    R = np.column_stack(R_cols)    # (9, n_dirs) rotation in SVD space
    basis = V9.T @ R               # (d_model, n_dirs) — orthonormal by construction
    n_dirs = basis.shape[1]

    # ─── Step 4: Compute signal strengths and digit scores ───
    # digit_scores = basis.T @ M.T = R.T @ V9 @ M.T = R.T @ diag(S9) @ U9.T
    digit_scores = R.T @ np.diag(S9) @ U9.T   # (n_dirs, 10)

    # Signal strength per direction: norm of digit scores
    svals = np.linalg.norm(digit_scores, axis=1)   # (n_dirs,)

    # ─── Step 5: Verify DFT purity of each direction ───
    freq_purities = []
    for i in range(n_dirs):
        scores = digit_scores[i]
        scores_c = scores - scores.mean()
        fft_vals = np.fft.fft(scores_c)
        power = np.abs(fft_vals) ** 2
        combined = np.zeros(6)
        combined[0] = power[0]
        for kk in range(1, 5):
            combined[kk] = power[kk] + power[10 - kk]
        combined[5] = power[5]
        total_power = combined[1:].sum()
        if total_power > 1e-10:
            target_k = freq_assignments[i]
            purity = float(combined[target_k] / total_power)
        else:
            purity = 0.0
        freq_purities.append(purity)

    # ─── Logging ───
    logger.info(f"\n  Hybrid SVD+DFT basis: {n_dirs} directions")
    logger.info(f"  Signal strengths (σ): {svals.round(2)}")
    logger.info(f"  Freq assignments: {freq_assignments}")
    logger.info(f"  Freq purities: {[f'{p:.1%}' for p in freq_purities]}")
    captured_var = np.sum(svals ** 2)
    var_frac = captured_var / total_var if total_var > 0 else 0
    logger.info(f"  Variance captured: {var_frac:.1%} of SVD total "
                f"({captured_var:.1f} / {total_var:.1f})")
    logger.info(f"  SVD singular values: {S9.round(2)}")

    # Verify orthonormality
    gram = basis.T @ basis
    off_diag = gram - np.eye(n_dirs)
    max_off_diag = np.max(np.abs(off_diag))
    logger.info(f"  Orthonormality: max off-diag = {max_off_diag:.2e} "
                f"({'PERFECT' if max_off_diag < 1e-10 else 'OK' if max_off_diag < 0.01 else 'WARN'})")

    return basis, freq_assignments, svals, digit_scores, freq_purities


# ═══════════════════════════════════════════════════════════════
# STEP 3: SANITY CHECKS
# ═══════════════════════════════════════════════════════════════

def sanity_check_basis(basis, freq_assignments, svals, digit_scores, freq_purities,
                       model_key=None, layer=None):
    """
    Thorough sanity checks on the computed Fourier basis.
    Returns True if all critical checks pass.
    """
    n_dirs = basis.shape[1]
    passed = True

    logger.info(f"\n  {'='*50}")
    logger.info(f"  SANITY CHECKS (Layer {layer})")
    logger.info(f"  {'='*50}")

    # CHECK 1: Orthonormality (DFT basis: within-plane exact, cross-plane approximate)
    gram = basis.T @ basis  # (n, n) should be near-identity
    off_diag_max = np.max(np.abs(gram - np.eye(n_dirs)))
    diag_err = np.max(np.abs(np.diag(gram) - 1.0))
    if off_diag_max < 0.05 and diag_err < 1e-5:
        logger.info(f"  [CHECK 1] Orthonormality: PASS (off-diag max={off_diag_max:.2e})")
    elif off_diag_max < 0.15:
        logger.warning(f"  [CHECK 1] Orthonormality: ACCEPTABLE (off-diag max={off_diag_max:.2e}, "
                       f"cross-freq overlap expected with DFT basis)")
    else:
        logger.error(f"  [CHECK 1] Orthonormality: FAIL (off-diag max={off_diag_max:.2e})")
        passed = False

    # CHECK 2: Frequency pair counts
    freq_counts = defaultdict(int)
    for f in freq_assignments:
        freq_counts[f] += 1
    expected = {1: 2, 2: 2, 3: 2, 4: 2, 5: 1}
    pair_ok = True
    for k in range(1, 6):
        exp = expected[k]
        got = freq_counts.get(k, 0)
        status = "PASS" if got == exp else f"WARN (expected {exp})"
        if got != exp:
            pair_ok = False
        logger.info(f"  [CHECK 2] Freq k={k}: {got} directions — {status}")
    if pair_ok:
        logger.info(f"  [CHECK 2] Frequency pairing: PERFECT FOURIER BASIS")
    else:
        logger.warning(f"  [CHECK 2] Frequency pairing: NOT perfect — rotation still valid")

    # CHECK 3: Minimum purity threshold
    min_purity = min(freq_purities)
    if min_purity > 0.4:
        logger.info(f"  [CHECK 3] Min purity: PASS ({min_purity:.1%})")
    else:
        logger.warning(f"  [CHECK 3] Min purity: LOW ({min_purity:.1%}) — some directions may be mixed")

    # CHECK 4: k=1 directions exist (critical for the experiment)
    k1_indices = [i for i, f in enumerate(freq_assignments) if f == 1]
    if len(k1_indices) >= 2:
        logger.info(f"  [CHECK 4] k=1 directions: PASS (indices {k1_indices})")
    elif len(k1_indices) == 1:
        logger.warning(f"  [CHECK 4] k=1 directions: only 1 found (index {k1_indices})")
    else:
        logger.error(f"  [CHECK 4] k=1 directions: NONE FOUND — k=1 rotation impossible")
        passed = False

    # CHECK 5: DFT basis axis verification — confirm cos direction has cos pattern
    digits = np.arange(10)
    axis_ok = True
    for i in range(n_dirs):
        k = freq_assignments[i]
        scores = digit_scores[i]  # (10,)
        scores_c = scores - scores.mean()
        # Check if this direction correlates with cos or sin of freq k
        cos_wave = np.cos(2 * np.pi * k * digits / 10)
        sin_wave = np.sin(2 * np.pi * k * digits / 10)
        cos_corr = np.abs(np.corrcoef(scores_c, cos_wave)[0, 1]) if np.std(scores_c) > 1e-10 else 0
        sin_corr = np.abs(np.corrcoef(scores_c, sin_wave)[0, 1]) if np.std(scores_c) > 1e-10 else 0
        # For DFT basis: even-indexed dirs within each freq should be cos,
        # odd-indexed should be sin
        freq_dirs = [j for j, f in enumerate(freq_assignments) if f == k]
        pos_in_freq = freq_dirs.index(i) if i in freq_dirs else -1
        expected_axis = "cos" if pos_in_freq == 0 else "sin"
        actual_axis = "cos" if cos_corr > sin_corr else "sin"
        ok = (expected_axis == actual_axis) or (k == 5)  # k=5 is cos-only
        if not ok:
            axis_ok = False
        logger.info(f"  [CHECK 5] Dir {i} (k={k}): expected={expected_axis}, "
                    f"cos_corr={cos_corr:.3f}, sin_corr={sin_corr:.3f} — "
                    f"{'PASS' if ok else 'WARN'}")
    if axis_ok:
        logger.info(f"  [CHECK 5] Axis alignment: ALL CORRECT (DFT basis verified)")
    else:
        logger.warning(f"  [CHECK 5] Axis alignment: some mismatches (check DFT projection)")

    # CHECK 6: Singular value sanity
    if svals[0] > 0 and svals[-1] / svals[0] > 0.01:
        logger.info(f"  [CHECK 6] SV range: PASS (ratio last/first = {svals[-1]/svals[0]:.3f})")
    else:
        logger.warning(f"  [CHECK 6] SV range: very spread (ratio = {svals[-1]/svals[0]:.3f})")

    logger.info(f"  {'='*50}")
    logger.info(f"  OVERALL: {'ALL CHECKS PASSED' if passed else 'SOME CHECKS FAILED'}")
    logger.info(f"  {'='*50}\n")

    return passed


# ═══════════════════════════════════════════════════════════════
# STEP 4: BUILD FREQUENCY PLANES
# ═══════════════════════════════════════════════════════════════

def build_frequency_planes(basis, freq_assignments, svals):
    """
    Group basis vectors into frequency-specific 2D planes.

    Returns:
        planes: dict mapping freq_k -> dict with:
            'vecs': np.ndarray (d_model, n_dirs)
            'svals': np.ndarray (n_dirs,) — singular values for elliptical rotation
    """
    freq_indices = defaultdict(list)
    for i, k in enumerate(freq_assignments):
        freq_indices[k].append(i)

    planes = {}
    for k, indices in freq_indices.items():
        planes[k] = {
            'vecs': basis[:, indices],   # (d_model, n_dirs)
            'svals': np.array([svals[i] for i in indices]),
        }

    return planes


# ═══════════════════════════════════════════════════════════════
# STEP 5: PHASE ROTATION EXPERIMENT
# ═══════════════════════════════════════════════════════════════

def compute_rotation_delta(act_1d, freq_planes_torch, shift_j, mode="coherent"):
    """
    Compute the activation delta for a frequency-specific rotation.

    With DFT basis: v1 is ALWAYS the cosine axis, v2 is ALWAYS the sine axis.
    This guarantees the elliptical rotation formula is applied to the correct
    axes (fixing the SVD sign/axis ambiguity trap).

    Uses ELLIPTICAL rotation to account for σ_cos ≠ σ_sin:
        new_c1 = c1·cos(θ) − c2·(σ_cos/σ_sin)·sin(θ)
        new_c2 = c1·(σ_sin/σ_cos)·sin(θ) + c2·cos(θ)
    which reduces to circular rotation when σ_cos = σ_sin.

    Args:
        act_1d: (d_model,) torch tensor — clean activation
        freq_planes_torch: dict[int, dict] — freq_k -> {'vecs': tensor, 'svals': tensor}
        shift_j: int — desired digit shift (1..9)
        mode: "coherent" (all freqs) or "k1_only" (only k=1)

    Returns:
        delta: (d_model,) torch tensor
    """
    delta = torch.zeros_like(act_1d)

    for k, plane in freq_planes_torch.items():
        if mode == "k1_only" and k != 1:
            continue

        vecs = plane['vecs']
        sv = plane['svals']

        theta = 2.0 * np.pi * k * shift_j / 10.0
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        if k == 5 and vecs.shape[1] == 1:
            # Nyquist: 1D only. cos(pi*j) = (-1)^j
            v = vecs[:, 0]
            c = torch.dot(act_1d.float(), v)
            new_c = c * cos_t  # cos(pi*j) = (-1)^j
            delta = delta + (new_c - c) * v
        elif vecs.shape[1] >= 2:
            v1 = vecs[:, 0]
            v2 = vecs[:, 1]
            c1 = torch.dot(act_1d.float(), v1)
            c2 = torch.dot(act_1d.float(), v2)
            # Elliptical rotation: account for σ₁ ≠ σ₂
            # Digit means trace ellipse with semi-axes ∝ (σ₁, σ₂)
            # Normalize → rotate on circle → rescale back
            sv1, sv2 = float(sv[0]), float(sv[1])
            ratio = sv1 / sv2 if sv2 > 1e-10 else 1.0
            new_c1 = c1 * cos_t - c2 * ratio * sin_t
            new_c2 = c1 / ratio * sin_t + c2 * cos_t
            delta = delta + (new_c1 - c1) * v1 + (new_c2 - c2) * v2
        elif vecs.shape[1] == 1:
            # Single direction for non-Nyquist (unusual)
            v = vecs[:, 0]
            c = torch.dot(act_1d.float(), v)
            new_c = c * cos_t
            delta = delta + (new_c - c) * v

    return delta


def run_fourier_phase_shift(model, problems, layer, freq_planes, basis, n_shifts=9,
                            digit_scores=None):
    """
    Core experiment: frequency-specific phase rotation.

    Modes:
      A. COHERENT: rotate ALL freq planes by freq-specific angles
      B. K1_ONLY: rotate only k=1 plane
      C. TOP2_PCA: rotate top-2 PCA directions (original buggy approach, as control)
      D. MEAN_SUB: replace 9D Fourier component with target digit's mean (noise-free)

    For each shift j=1..9, check if ones digit shifts by j mod 10.
    """
    device = next(model.parameters()).device
    hook_name = f"blocks.{layer}.hook_resid_post"

    # Convert freq planes to torch (preserving svals for elliptical rotation)
    freq_planes_torch = {}
    for k, plane in freq_planes.items():
        freq_planes_torch[k] = {
            'vecs': torch.tensor(plane['vecs'], dtype=torch.float32, device=device),
            'svals': torch.tensor(plane['svals'], dtype=torch.float32, device=device),
        }

    # Top-2 PCA control: columns 0,1 of the ORIGINAL basis (highest singular values)
    # NOT sorted by frequency — these are the directions the buggy experiment rotated in
    top2_v1 = torch.tensor(basis[:, 0], dtype=torch.float32, device=device)
    top2_v2 = torch.tensor(basis[:, 1], dtype=torch.float32, device=device)

    # Full basis as torch tensor for mean-substitution mode
    basis_torch = torch.tensor(basis, dtype=torch.float32, device=device)  # (d_model, 9)

    # Digit-mean projections in 9D Fourier space (if available)
    digit_mean_F = None
    if digit_scores is not None:
        # digit_scores is (9, 10): digit_scores[:, d] = 9D projection of digit d's mean
        digit_mean_F = torch.tensor(digit_scores, dtype=torch.float32, device=device)  # (9, 10)

    modes = ["coherent", "k1_only", "top2_pca"]
    if digit_mean_F is not None:
        modes.append("mean_sub")
    results = {}

    for mode in modes:
        logger.info(f"\n  --- Mode: {mode.upper()} at Layer {layer} ---")

        per_shift = {}
        total_exact = 0
        total_changed = 0
        total_numeric = 0
        total_tests = 0
        n_single_total = 0
        shift_histogram = defaultdict(int)

        for j in range(1, n_shifts + 1):
            n_exact = 0
            n_changed = 0
            n_numeric = 0
            n_total = 0
            n_single = 0

            for prob in problems:
                tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
                original_digit = prob["first_digit"]
                original_ones = prob["ones_digit"]
                is_single_digit = (prob["n_digits"] == 1)

                # Get clean activation
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens, names_filter=hook_name)
                    clean_act = cache[hook_name][0, -1].clone()
                    del cache

                # Compute delta based on mode
                if mode == "mean_sub":
                    # Mean-substitution: replace 9D Fourier component with target digit's mean
                    target_digit = (original_ones + j) % 10
                    # Current 9D projection: basis.T @ act
                    h_F = basis_torch.T @ clean_act.float()  # (9,)
                    # Target: digit_scores[:, target_digit]
                    target_F = digit_mean_F[:, target_digit]  # (9,)
                    # Delta = basis @ (target - current)
                    delta = basis_torch @ (target_F - h_F)
                elif mode == "top2_pca":
                    # Original (buggy) approach: rotate by j*2pi/10 in top-2 PCA
                    theta = j * 2.0 * np.pi / 10.0
                    c1 = torch.dot(clean_act.float(), top2_v1)
                    c2 = torch.dot(clean_act.float(), top2_v2)
                    cos_t = np.cos(theta)
                    sin_t = np.sin(theta)
                    new_c1 = c1 * cos_t - c2 * sin_t
                    new_c2 = c1 * sin_t + c2 * cos_t
                    delta = (new_c1 - c1) * top2_v1 + (new_c2 - c2) * top2_v2
                else:
                    delta = compute_rotation_delta(
                        clean_act, freq_planes_torch, j, mode=mode
                    )

                # Hook to apply delta
                def rotation_hook(act, hook, d=delta):
                    act[:, -1, :] = act[:, -1, :] + d.to(act.dtype).unsqueeze(0)
                    return act

                with torch.no_grad():
                    with model.hooks(fwd_hooks=[(hook_name, rotation_hook)]):
                        logits = model(tokens)

                pred_tok = logits[0, -1].argmax(dim=-1).item()
                pred_str = model.tokenizer.decode([pred_tok]).strip()

                is_numeric = False
                pred_digit = -1
                try:
                    pred_digit = int(pred_str)
                    is_numeric = True
                except ValueError:
                    pass

                # Non-numeric = definitely changed (matches fisher_phase_shift.py)
                digit_changed = (not is_numeric) or (pred_digit != original_digit)
                if is_single_digit:
                    expected = (original_ones + j) % 10
                    n_single += 1
                else:
                    expected = -99
                is_exact = is_numeric and is_single_digit and (pred_digit == expected)

                n_total += 1
                if is_numeric:
                    n_numeric += 1
                if digit_changed:
                    n_changed += 1
                if is_exact:
                    n_exact += 1

                if is_numeric and pred_digit >= 0:
                    actual_shift = (pred_digit - original_digit) % 10
                    shift_histogram[actual_shift] += 1

            per_shift[j] = {
                "change_rate": n_changed / n_total * 100 if n_total > 0 else 0,
                "exact_rate": n_exact / n_single * 100 if n_single > 0 else 0,
                "n_changed": n_changed,
                "n_exact": n_exact,
                "n_total": n_total,
                "n_single": n_single,
            }
            total_exact += n_exact
            total_changed += n_changed
            total_numeric += n_numeric
            total_tests += n_total
            n_single_total += n_single

            logger.info(f"    Shift j={j}: changed={per_shift[j]['change_rate']:.1f}%, "
                        f"exact_mod10={per_shift[j]['exact_rate']:.1f}% "
                        f"(of {n_single} single-digit)")

        overall_change = total_changed / total_tests * 100 if total_tests > 0 else 0
        overall_exact = total_exact / n_single_total * 100 if n_single_total > 0 else 0

        logger.info(f"\n  {mode.upper()} SUMMARY (Layer {layer}):")
        logger.info(f"    Digit changed: {overall_change:.1f}%")
        logger.info(f"    Exact mod-10 shift: {overall_exact:.1f}%")
        logger.info(f"    Shift histogram: {dict(sorted(shift_histogram.items()))}")

        results[mode] = {
            "digit_change_rate": overall_change,
            "exact_mod10_rate": overall_exact,
            "numeric_rate": total_numeric / total_tests * 100 if total_tests > 0 else 0,
            "per_shift": per_shift,
            "shift_histogram": dict(shift_histogram),
            "n_total_tests": total_tests,
            "n_single_total": n_single_total,
        }

    return results


# ═══════════════════════════════════════════════════════════════
# STEP 6a: FOURIER-UNEMBED ALIGNMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_fourier_unembed_alignment(model, basis, layer):
    """
    Diagnose the gap between encoding space (Fourier basis) and readout space (W_U digit columns).

    Computes:
    1. Unembed digit subspace: PCA of W_U[:, digit_tokens] (centered)
    2. Subspace alignment: principal angles between Fourier and unembed digit subspaces
    3. Per-direction overlap: how much of each Fourier direction is "readable" by W_U
    4. Converse: how much of each unembed digit direction lives in the Fourier subspace
    5. Rotation feasibility: can we rotate in the unembed space instead?

    Returns dict with all alignment metrics.
    """
    device = next(model.parameters()).device

    # Resolve digit token IDs
    digit_token_ids = []
    for d in range(10):
        toks = model.to_tokens(str(d), prepend_bos=False)[0]
        digit_token_ids.append(toks[-1].item())
    logger.info(f"  Digit token IDs: {digit_token_ids}")

    # Extract W_U columns for digit tokens: (d_model, 10)
    W_U = model.W_U  # (d_model, d_vocab)
    D = W_U[:, digit_token_ids].float()  # (d_model, 10)

    # Center the digit columns (remove mean across digits)
    D_mean = D.mean(dim=1, keepdim=True)  # (d_model, 1)
    D_centered = D - D_mean  # (d_model, 10)

    # SVD of centered digit unembed matrix → unembed digit subspace
    U_d, S_d, Vh_d = torch.linalg.svd(D_centered, full_matrices=False)
    # U_d: (d_model, 10), S_d: (10,), Vh_d: (10, 10)
    # Only 9 non-trivial directions (10 digits - 1 mean = 9 DoF)
    n_unembed_dirs = min(9, (S_d > 1e-6).sum().item())
    U_digit = U_d[:, :n_unembed_dirs]  # (d_model, n_unembed_dirs)
    S_digit = S_d[:n_unembed_dirs]

    logger.info(f"\n  === FOURIER-UNEMBED ALIGNMENT ANALYSIS (Layer {layer}) ===")
    logger.info(f"  Unembed digit subspace: {n_unembed_dirs} directions")
    logger.info(f"  Unembed singular values: {[f'{s:.4f}' for s in S_digit.cpu().tolist()]}")

    # Convert Fourier basis to torch
    F = torch.tensor(basis, dtype=torch.float32, device=device)  # (d_model, 9)
    n_fourier = F.shape[1]

    # === METRIC 1: Per-Fourier-direction projection onto unembed subspace ===
    # For each Fourier direction f_j, compute ||P_U f_j||² / ||f_j||²
    # P_U = U_digit @ U_digit^T is the projector onto unembed digit subspace
    F_proj = U_digit.T @ F  # (n_unembed, 9) — projection coefficients
    fourier_in_unembed = (F_proj ** 2).sum(dim=0)  # (9,) — ||P_U f_j||² (f_j is unit)
    logger.info(f"\n  Fourier→Unembed overlap (fraction of each Fourier dir in unembed subspace):")
    for j in range(n_fourier):
        logger.info(f"    Fourier dir {j}: {fourier_in_unembed[j].item():.4f} "
                     f"({fourier_in_unembed[j].item()*100:.1f}%)")
    logger.info(f"    MEAN: {fourier_in_unembed.mean().item():.4f} "
                f"({fourier_in_unembed.mean().item()*100:.1f}%)")

    # === METRIC 2: Per-unembed-direction projection onto Fourier subspace ===
    # For each unembed digit direction u_i, compute ||P_F u_i||² / ||u_i||²
    U_proj = F.T @ U_digit  # (9, n_unembed) — projection coefficients
    unembed_in_fourier = (U_proj ** 2).sum(dim=0)  # (n_unembed,)
    logger.info(f"\n  Unembed→Fourier overlap (fraction of each unembed dir in Fourier subspace):")
    for j in range(n_unembed_dirs):
        logger.info(f"    Unembed dir {j} (σ={S_digit[j].item():.4f}): "
                     f"{unembed_in_fourier[j].item():.4f} ({unembed_in_fourier[j].item()*100:.1f}%)")
    logger.info(f"    MEAN: {unembed_in_fourier.mean().item():.4f} "
                f"({unembed_in_fourier.mean().item()*100:.1f}%)")

    # === METRIC 3: Principal angles between subspaces ===
    # cos(θ_i) = singular values of F^T @ U_digit
    cross = F.T @ U_digit  # (9, n_unembed)
    _, sigmas, _ = torch.linalg.svd(cross)
    # sigmas are cos(principal angles), clamped to [0,1]
    cos_angles = sigmas.clamp(0, 1)
    angles_deg = torch.acos(cos_angles) * 180 / torch.pi
    logger.info(f"\n  Principal angles between Fourier and Unembed subspaces:")
    for j in range(len(cos_angles)):
        logger.info(f"    Angle {j}: {angles_deg[j].item():.1f}° (cos={cos_angles[j].item():.4f})")
    logger.info(f"    Mean angle: {angles_deg.mean().item():.1f}°")

    # === METRIC 4: Total subspace overlap (Grassmann metric) ===
    total_overlap = (cos_angles ** 2).sum().item() / min(n_fourier, n_unembed_dirs)
    logger.info(f"\n  Total subspace overlap: {total_overlap:.4f} ({total_overlap*100:.1f}%)")
    logger.info(f"    1.0 = identical subspaces, 0.0 = orthogonal subspaces")

    # === METRIC 5: How much of the digit-discriminative info in W_U lives in Fourier? ===
    # Weighted by singular values (σ²-weighted overlap)
    weights = S_digit ** 2
    weighted_overlap = (unembed_in_fourier * weights).sum().item() / weights.sum().item()
    logger.info(f"\n  σ²-weighted overlap (unembed→Fourier): {weighted_overlap:.4f} ({weighted_overlap*100:.1f}%)")
    logger.info(f"    This is the fraction of digit-discriminative W_U variance in the Fourier subspace")

    # === METRIC 6: Per-digit-token analysis ===
    # For each digit token, how much of its W_U direction (centered) is in the Fourier subspace?
    logger.info(f"\n  Per-digit W_U overlap with Fourier subspace:")
    per_digit_overlap = []
    for d in range(10):
        w_d = D_centered[:, d]  # (d_model,)
        w_d_proj = F @ (F.T @ w_d)  # projection onto Fourier subspace
        overlap = (w_d_proj ** 2).sum().item() / (w_d ** 2).sum().item()
        per_digit_overlap.append(overlap)
        logger.info(f"    Digit {d}: {overlap:.4f} ({overlap*100:.1f}%)")
    logger.info(f"    MEAN: {np.mean(per_digit_overlap):.4f} ({np.mean(per_digit_overlap)*100:.1f}%)")

    # === METRIC 7: DFT structure in unembed digit vectors ===
    # Do the unembed digit vectors have Fourier structure?
    logger.info(f"\n  DFT analysis of unembed digit vectors (centered):")
    D_centered_np = D_centered.detach().cpu().numpy()  # (d_model, 10)
    # For each unembed SVD direction, check if the digit loadings are Fourier
    Vh_np = Vh_d.detach().cpu().numpy()  # (10, 10) — digit loadings
    for j in range(min(n_unembed_dirs, 9)):
        loadings = Vh_np[j, :]  # (10,) — loading of each digit on this direction
        # DFT of the loadings
        dft = np.fft.fft(loadings)
        dft_power = np.abs(dft) ** 2
        dft_power[0] = 0  # ignore DC
        total_power = dft_power.sum()
        dominant_k = np.argmax(dft_power[1:6]) + 1  # k=1..5
        dominant_frac = dft_power[dominant_k] / total_power if total_power > 0 else 0
        # Also check k and 10-k (conjugate pair)
        conj_k = 10 - dominant_k if dominant_k < 5 else dominant_k
        pair_frac = (dft_power[dominant_k] + dft_power[conj_k]) / total_power if total_power > 0 else 0
        logger.info(f"    Unembed dir {j} (σ={S_digit[j].item():.4f}): "
                     f"dominant k={dominant_k}, purity={pair_frac*100:.1f}%, "
                     f"loadings=[{', '.join(f'{v:.3f}' for v in loadings)}]")

    results = {
        "n_fourier_dirs": n_fourier,
        "n_unembed_dirs": n_unembed_dirs,
        "unembed_singular_values": S_digit.cpu().tolist(),
        "fourier_in_unembed": fourier_in_unembed.cpu().tolist(),
        "unembed_in_fourier": unembed_in_fourier.cpu().tolist(),
        "principal_angles_deg": angles_deg.cpu().tolist(),
        "cos_principal_angles": cos_angles.cpu().tolist(),
        "total_subspace_overlap": total_overlap,
        "sigma2_weighted_overlap": weighted_overlap,
        "per_digit_overlap": per_digit_overlap,
    }
    return results


# ═══════════════════════════════════════════════════════════════
# STEP 6b: LOGIT LENS ANALYSIS
# ═══════════════════════════════════════════════════════════════

def run_logit_lens_analysis(model, problems, layer, freq_planes, basis, n_shifts=9,
                            digit_scores=None):
    """
    Logit lens analysis: project rotated activations through LN_final + W_U
    to check if Fourier rotation shifts digit logits in representation space.

    This bypasses downstream layers entirely — it directly reads what the
    representation at this layer "thinks" the output digit should be.

    Key question: Is the rotation working in representation space even though
    the full model output shows only ~10% exact_mod10?

    Much faster than full forward pass: only ONE forward pass per problem
    (to get clean activation), then all shifts are cheap matrix ops.
    """
    device = next(model.parameters()).device
    hook_name = f"blocks.{layer}.hook_resid_post"

    # Resolve digit token IDs
    digit_token_ids = []
    for d in range(10):
        toks = model.to_tokens(str(d), prepend_bos=False)[0]
        digit_token_ids.append(toks[-1].item())
    digit_token_ids_t = torch.tensor(digit_token_ids, device=device)
    logger.info(f"  Digit token IDs: {digit_token_ids}")
    logger.info(f"  Digit tokens decode: {[model.tokenizer.decode([t]) for t in digit_token_ids]}")

    # Convert freq planes to torch
    freq_planes_torch = {}
    for k, plane in freq_planes.items():
        freq_planes_torch[k] = {
            'vecs': torch.tensor(plane['vecs'], dtype=torch.float32, device=device),
            'svals': torch.tensor(plane['svals'], dtype=torch.float32, device=device),
        }

    basis_torch = torch.tensor(basis, dtype=torch.float32, device=device)
    digit_mean_F = None
    if digit_scores is not None:
        digit_mean_F = torch.tensor(digit_scores, dtype=torch.float32, device=device)

    # Pre-cache all clean activations (ONE forward pass per problem)
    logger.info(f"\n  Caching clean activations for {len([p for p in problems if p['n_digits'] == 1])} single-digit problems...")
    clean_cache = {}  # idx -> (clean_act, original_ones)
    for i, prob in enumerate(problems):
        if prob["n_digits"] != 1:
            continue
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
            clean_cache[i] = {
                'act': cache[hook_name][0, -1].clone(),
                'ones': prob["ones_digit"],
            }
            del cache

    n_problems = len(clean_cache)
    logger.info(f"  Cached {n_problems} activations")

    # Clean logit-lens baseline: what does the logit lens predict without rotation?
    logger.info(f"\n  --- Logit Lens BASELINE at Layer {layer} ---")
    clean_correct = 0
    clean_digit_preds = {}
    with torch.no_grad():
        for i, cd in clean_cache.items():
            h = cd['act'].unsqueeze(0).unsqueeze(0)  # (1, 1, d_model)
            normed = model.ln_final(h)
            logits_all = model.unembed(normed)[0, 0]  # (d_vocab,)
            digit_logits = logits_all[digit_token_ids_t]  # (10,)
            pred = digit_logits.argmax().item()
            clean_digit_preds[i] = pred
            if pred == cd['ones']:
                clean_correct += 1

    clean_acc = clean_correct / n_problems * 100 if n_problems > 0 else 0
    logger.info(f"  Clean logit-lens accuracy: {clean_correct}/{n_problems} = {clean_acc:.1f}%")

    # Run logit lens for each mode
    modes = ["coherent"]
    if digit_mean_F is not None:
        modes.append("mean_sub")

    results = {"clean_logit_lens_accuracy": clean_acc, "n_problems": n_problems}

    for mode in modes:
        logger.info(f"\n  --- Logit Lens: {mode.upper()} at Layer {layer} ---")

        per_shift = {}
        total_exact = 0
        total_changed = 0
        total_rank_sum = 0
        total_prob_sum = 0.0
        total_logit_diff_sum = 0.0
        n_single_total = 0

        for j in range(1, n_shifts + 1):
            n_exact = 0
            n_changed = 0
            n_single = 0
            rank_sum = 0
            prob_sum = 0.0
            logit_diff_sum = 0.0

            with torch.no_grad():
                for i, cd in clean_cache.items():
                    clean_act = cd['act']
                    original_ones = cd['ones']
                    target_digit = (original_ones + j) % 10
                    n_single += 1

                    # Compute rotation delta
                    if mode == "mean_sub":
                        h_F = basis_torch.T @ clean_act.float()
                        target_F = digit_mean_F[:, target_digit]
                        delta = basis_torch @ (target_F - h_F)
                    else:
                        delta = compute_rotation_delta(
                            clean_act, freq_planes_torch, j, mode="coherent"
                        )

                    rotated_act = clean_act.float() + delta

                    # Rotated logit lens
                    rot_h = rotated_act.unsqueeze(0).unsqueeze(0)  # (1, 1, d_model)
                    rot_normed = model.ln_final(rot_h)
                    rot_logits_all = model.unembed(rot_normed)[0, 0]  # (d_vocab,)
                    rot_digit_logits = rot_logits_all[digit_token_ids_t]  # (10,)

                    # Predictions
                    clean_pred = clean_digit_preds[i]
                    rot_pred = rot_digit_logits.argmax().item()

                    # Metrics
                    if rot_pred != clean_pred:
                        n_changed += 1
                    if rot_pred == target_digit:
                        n_exact += 1

                    # Rank of target digit (0=best, 9=worst)
                    sorted_indices = rot_digit_logits.argsort(descending=True)
                    target_rank = (sorted_indices == target_digit).nonzero(as_tuple=True)[0].item()
                    rank_sum += target_rank

                    # Probability of target (softmax over digit tokens)
                    rot_probs = torch.softmax(rot_digit_logits, dim=0)
                    target_prob = rot_probs[target_digit].item()
                    prob_sum += target_prob

                    # Logit diff: target - original (positive = rotation helped)
                    logit_diff = (rot_digit_logits[target_digit] - rot_digit_logits[original_ones]).item()
                    logit_diff_sum += logit_diff

            exact_rate = n_exact / n_single * 100 if n_single > 0 else 0
            change_rate = n_changed / n_single * 100 if n_single > 0 else 0
            mean_rank = rank_sum / n_single if n_single > 0 else -1
            mean_prob = prob_sum / n_single if n_single > 0 else 0
            mean_logit_diff = logit_diff_sum / n_single if n_single > 0 else 0

            per_shift[j] = {
                "n_exact": n_exact,
                "n_changed": n_changed,
                "n_single": n_single,
                "exact_rate": exact_rate,
                "change_rate": change_rate,
                "mean_target_rank": float(mean_rank),
                "mean_target_prob": float(mean_prob),
                "mean_logit_diff": float(mean_logit_diff),
            }
            total_exact += n_exact
            total_changed += n_changed
            total_rank_sum += rank_sum
            total_prob_sum += prob_sum
            total_logit_diff_sum += logit_diff_sum
            n_single_total += n_single

            logger.info(f"    Shift j={j}: exact={exact_rate:.1f}%, "
                        f"changed={change_rate:.1f}%, "
                        f"rank={mean_rank:.2f}, prob={mean_prob:.3f}, "
                        f"Δlogit={mean_logit_diff:+.2f}")

        overall_exact = total_exact / n_single_total * 100 if n_single_total > 0 else 0
        overall_changed = total_changed / n_single_total * 100 if n_single_total > 0 else 0
        overall_rank = total_rank_sum / n_single_total if n_single_total > 0 else -1
        overall_prob = total_prob_sum / n_single_total if n_single_total > 0 else 0
        overall_logit_diff = total_logit_diff_sum / n_single_total if n_single_total > 0 else 0

        logger.info(f"\n  LOGIT LENS {mode.upper()} SUMMARY (Layer {layer}):")
        logger.info(f"    Exact mod-10 shift: {overall_exact:.1f}% (full-model baseline ~10%)")
        logger.info(f"    Digit changed: {overall_changed:.1f}%")
        logger.info(f"    Mean target rank: {overall_rank:.2f} (0=best, 9=worst, chance=4.5)")
        logger.info(f"    Mean target prob: {overall_prob:.3f} (chance=0.10)")
        logger.info(f"    Mean logit diff (target−original): {overall_logit_diff:+.2f}")

        results[f"logit_lens_{mode}"] = {
            "exact_mod10_rate": overall_exact,
            "digit_change_rate": overall_changed,
            "mean_target_rank": float(overall_rank),
            "mean_target_prob": float(overall_prob),
            "mean_logit_diff": float(overall_logit_diff),
            "per_shift": per_shift,
            "n_single_total": n_single_total,
        }

    return results


# ═══════════════════════════════════════════════════════════════
# STEP 6c: ALGORITHM VERIFICATION TESTS
# ═══════════════════════════════════════════════════════════════

def run_algorithm_tests():
    """
    Pure-math verification of the rotation logic (no model needed).
    Tests that the algorithm is correct before running on real data.
    """
    logger.info("\n" + "=" * 60)
    logger.info("ALGORITHM VERIFICATION TESTS")
    logger.info("=" * 60)
    passed = 0
    total = 0

    # TEST 1: Rotation by 0 gives zero delta
    total += 1
    d_model = 100
    basis = np.random.randn(d_model, 2)
    basis, _ = np.linalg.qr(basis)
    planes = {1: {'vecs': torch.tensor(basis, dtype=torch.float32),
                   'svals': torch.tensor([1.0, 1.0])}}
    act = torch.randn(d_model)
    delta = compute_rotation_delta(act, planes, shift_j=0, mode="coherent")
    if delta.abs().max().item() < 1e-5:
        logger.info(f"  TEST 1 [j=0 -> zero delta]: PASS (max={delta.abs().max():.2e})")
        passed += 1
    else:
        logger.error(f"  TEST 1 [j=0 -> zero delta]: FAIL (max={delta.abs().max():.2e})")

    # TEST 2: Rotation by 10 (full cycle) gives zero delta for k=1
    total += 1
    delta_10 = compute_rotation_delta(act, planes, shift_j=10, mode="coherent")
    if delta_10.abs().max().item() < 1e-5:
        logger.info(f"  TEST 2 [j=10 -> full cycle]: PASS (max={delta_10.abs().max():.2e})")
        passed += 1
    else:
        logger.error(f"  TEST 2 [j=10 -> full cycle]: FAIL (max={delta_10.abs().max():.2e})")

    # TEST 3: Rotation preserves norm of projection onto the plane (equal svals = circular)
    total += 1
    v1 = planes[1]['vecs'][:, 0]
    v2 = planes[1]['vecs'][:, 1]
    c1_before = torch.dot(act.float(), v1)
    c2_before = torch.dot(act.float(), v2)
    norm_before = (c1_before**2 + c2_before**2).sqrt().item()

    act_rotated = act + compute_rotation_delta(act, planes, shift_j=3, mode="coherent")
    c1_after = torch.dot(act_rotated.float(), v1)
    c2_after = torch.dot(act_rotated.float(), v2)
    norm_after = (c1_after**2 + c2_after**2).sqrt().item()

    if abs(norm_before - norm_after) / (norm_before + 1e-10) < 1e-5:
        logger.info(f"  TEST 3 [norm preserved (equal sv)]: PASS ({norm_before:.4f} -> {norm_after:.4f})")
        passed += 1
    else:
        logger.error(f"  TEST 3 [norm preserved (equal sv)]: FAIL ({norm_before:.4f} -> {norm_after:.4f})")

    # TEST 4: Rotation doesn't affect orthogonal complement
    total += 1
    ortho = torch.randn(d_model)
    ortho = ortho - torch.dot(ortho, v1) * v1 - torch.dot(ortho, v2) * v2
    proj_before = torch.dot(act.float(), ortho).item()
    proj_after = torch.dot(act_rotated.float(), ortho).item()
    if abs(proj_before - proj_after) < 1e-5:
        logger.info(f"  TEST 4 [ortho unchanged]: PASS (diff={abs(proj_before-proj_after):.2e})")
        passed += 1
    else:
        logger.error(f"  TEST 4 [ortho unchanged]: FAIL (diff={abs(proj_before-proj_after):.2e})")

    # TEST 5: Coherent rotation of synthetic Fourier signal (equal svals)
    total += 1
    d_model_syn = 20
    b1 = np.zeros(d_model_syn); b1[0] = 1.0
    b2 = np.zeros(d_model_syn); b2[1] = 1.0
    syn_planes = {1: {'vecs': torch.tensor(np.stack([b1, b2], axis=1), dtype=torch.float32),
                       'svals': torch.tensor([1.0, 1.0])}}

    d_orig = 3
    act_syn = torch.zeros(d_model_syn)
    act_syn[0] = np.cos(2 * np.pi * 1 * d_orig / 10)
    act_syn[1] = np.sin(2 * np.pi * 1 * d_orig / 10)

    delta_syn = compute_rotation_delta(act_syn, syn_planes, shift_j=2, mode="coherent")
    act_shifted = act_syn + delta_syn
    expected_c1 = np.cos(2 * np.pi * 1 * 5 / 10)
    expected_c2 = np.sin(2 * np.pi * 1 * 5 / 10)
    err = abs(act_shifted[0].item() - expected_c1) + abs(act_shifted[1].item() - expected_c2)
    if err < 1e-5:
        logger.info(f"  TEST 5 [synthetic d=3+2=5]: PASS (err={err:.2e})")
        passed += 1
    else:
        logger.error(f"  TEST 5 [synthetic d=3+2=5]: FAIL (err={err:.2e})")
        logger.error(f"    Got: ({act_shifted[0]:.4f}, {act_shifted[1]:.4f})")
        logger.error(f"    Expected: ({expected_c1:.4f}, {expected_c2:.4f})")

    # TEST 6: k=5 Nyquist parity flip
    total += 1
    b5 = np.zeros(d_model_syn); b5[2] = 1.0
    parity_planes = {5: {'vecs': torch.tensor(b5.reshape(-1, 1), dtype=torch.float32),
                          'svals': torch.tensor([1.0])}}
    act_p = torch.zeros(d_model_syn)
    act_p[2] = 1.0
    delta_p = compute_rotation_delta(act_p, parity_planes, shift_j=1, mode="coherent")
    act_p_shifted = act_p + delta_p
    if abs(act_p_shifted[2].item() - (-1.0)) < 1e-5:
        logger.info(f"  TEST 6 [k=5 parity flip]: PASS ({act_p_shifted[2]:.4f})")
        passed += 1
    else:
        logger.error(f"  TEST 6 [k=5 parity flip]: FAIL ({act_p_shifted[2]:.4f}, expected -1.0)")

    # TEST 7: Multi-frequency coherent rotation (equal svals)
    total += 1
    b3 = np.zeros(d_model_syn); b3[3] = 1.0
    b4 = np.zeros(d_model_syn); b4[4] = 1.0
    multi_planes = {
        1: {'vecs': torch.tensor(np.stack([b1, b2], axis=1), dtype=torch.float32),
            'svals': torch.tensor([1.0, 1.0])},
        2: {'vecs': torch.tensor(np.stack([b3, b4], axis=1), dtype=torch.float32),
            'svals': torch.tensor([1.0, 1.0])},
    }
    d_orig = 3
    act_multi = torch.zeros(d_model_syn)
    act_multi[0] = np.cos(2 * np.pi * 1 * d_orig / 10)  # k=1 cos
    act_multi[1] = np.sin(2 * np.pi * 1 * d_orig / 10)  # k=1 sin
    act_multi[3] = np.cos(2 * np.pi * 2 * d_orig / 10)  # k=2 cos
    act_multi[4] = np.sin(2 * np.pi * 2 * d_orig / 10)  # k=2 sin

    delta_multi = compute_rotation_delta(act_multi, multi_planes, shift_j=4, mode="coherent")
    act_m_shifted = act_multi + delta_multi
    d_target = 7  # 3 + 4
    exp_vals = [
        np.cos(2 * np.pi * 1 * d_target / 10),
        np.sin(2 * np.pi * 1 * d_target / 10),
        np.cos(2 * np.pi * 2 * d_target / 10),
        np.sin(2 * np.pi * 2 * d_target / 10),
    ]
    err_multi = sum(abs(act_m_shifted[i].item() - exp_vals[idx])
                    for idx, i in enumerate([0, 1, 3, 4]))
    if err_multi < 1e-5:
        logger.info(f"  TEST 7 [multi-freq d=3+4=7]: PASS (err={err_multi:.2e})")
        passed += 1
    else:
        logger.error(f"  TEST 7 [multi-freq d=3+4=7]: FAIL (err={err_multi:.2e})")

    # TEST 8: ELLIPTICAL rotation with unequal singular values
    # This is the critical test for the bug fix. With σ₁=2, σ₂=1 (ratio=2),
    # the digit encoding is an ellipse: c1 = 2·cos(α), c2 = 1·sin(α).
    # A circular rotation would NOT map d→d+j correctly on this ellipse.
    total += 1
    sv1_test, sv2_test = 2.0, 1.0  # 2:1 ratio — exaggerated for clear test
    ellip_planes = {1: {'vecs': torch.tensor(np.stack([b1, b2], axis=1), dtype=torch.float32),
                         'svals': torch.tensor([sv1_test, sv2_test])}}
    d_orig = 3
    act_ellip = torch.zeros(d_model_syn)
    # Encode as elliptical: c1 = σ₁·cos(2πkd/10), c2 = σ₂·sin(2πkd/10)
    alpha = 2 * np.pi * 1 * d_orig / 10
    act_ellip[0] = sv1_test * np.cos(alpha)   # c1 = σ₁·cos(α)
    act_ellip[1] = sv2_test * np.sin(alpha)   # c2 = σ₂·sin(α)

    # Shift by j=4 → target digit 7
    d_target_e = 7
    alpha_target = 2 * np.pi * 1 * d_target_e / 10
    expected_e_c1 = sv1_test * np.cos(alpha_target)
    expected_e_c2 = sv2_test * np.sin(alpha_target)

    delta_ellip = compute_rotation_delta(act_ellip, ellip_planes, shift_j=4, mode="coherent")
    act_e_shifted = act_ellip + delta_ellip
    err_ellip = abs(act_e_shifted[0].item() - expected_e_c1) + abs(act_e_shifted[1].item() - expected_e_c2)
    if err_ellip < 1e-5:
        logger.info(f"  TEST 8 [elliptical σ₁/σ₂=2, d=3+4=7]: PASS (err={err_ellip:.2e})")
        passed += 1
    else:
        logger.error(f"  TEST 8 [elliptical σ₁/σ₂=2, d=3+4=7]: FAIL (err={err_ellip:.2e})")
        logger.error(f"    Got: ({act_e_shifted[0]:.4f}, {act_e_shifted[1]:.4f})")
        logger.error(f"    Expected: ({expected_e_c1:.4f}, {expected_e_c2:.4f})")

    logger.info(f"\n  ALGORITHM TESTS: {passed}/{total} passed")
    logger.info("=" * 60 + "\n")

    return passed == total


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Frequency-Specific Fourier Phase Rotation")
    parser.add_argument("--model", type=str, default="gemma-2b",
                        help="Model key (gemma-2b, phi-3-mini, llama-3b)")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layers (default: comp + readout for model)")
    parser.add_argument("--operand-range", type=int, default=30)
    parser.add_argument("--n-basis-problems", type=int, default=500,
                        help="Problems for digit-mean SVD computation")
    parser.add_argument("--n-test-problems", type=int, default=100,
                        help="Correct problems for phase shift testing")
    parser.add_argument("--direct-answer", action="store_true",
                        help="Use direct-answer mode (for LLaMA)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Only compute basis + sanity checks, skip phase shift")
    parser.add_argument("--logit-lens", action="store_true",
                        help="Run logit lens analysis (project rotated activations through LN_final + W_U)")
    args = parser.parse_args()

    # Run algorithm tests first (no model needed)
    algo_ok = run_algorithm_tests()
    if not algo_ok:
        logger.error("ALGORITHM TESTS FAILED — aborting.")
        return

    model_name = MODEL_MAP.get(args.model, args.model)
    if args.layers:
        layers = [int(x.strip()) for x in args.layers.split(",")]
    else:
        layers = [COMP_LAYERS.get(args.model, 20), READOUT_LAYERS.get(args.model, 27)]

    logger.info(f"\n{'='*60}")
    logger.info(f"FOURIER PHASE ROTATION EXPERIMENT")
    logger.info(f"{'='*60}")
    logger.info(f"Model: {args.model} ({model_name})")
    logger.info(f"Layers: {layers}")
    logger.info(f"Operand range: {args.operand_range}")
    logger.info(f"Basis problems: {args.n_basis_problems}")
    logger.info(f"Test problems: single-digit only (a+b < 10, ~55 problems)")
    logger.info(f"Direct answer: {args.direct_answer}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Logit lens: {args.logit_lens}")

    # Estimate runtime (55 single-digit problems typical)
    n_forward_basis = args.operand_range ** 2 * len(layers)  # all problems for basis
    n_est_test = 55  # approximate single-digit count
    n_forward_test = n_est_test * 9 * 3 * len(layers)  # 9 shifts, 3 modes, ~55 problems
    logger.info(f"\n  Estimated forward passes: {n_forward_basis} (basis) + ~{n_forward_test} (test)")
    logger.info(f"  Estimated runtime: ~{(n_forward_basis + n_forward_test) * 0.1 / 60:.0f} min on MPS")

    # Load model
    from transformer_lens import HookedTransformer
    device = torch.device(args.device) if args.device else get_device()
    logger.info(f"Loading {model_name} on {device}...")
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=torch.float32)
    model.eval()

    # Generate problems: use ALL problems for basis, SINGLE-DIGIT for testing
    # Basis computation needs diverse problems for robust digit-mean estimates
    # Testing needs single-digit answers so first_digit == ones_digit == answer
    if args.direct_answer:
        da_problems, _ = generate_direct_answer_problems(n_per_digit=100, operand_max=args.operand_range)
        da_correct = filter_correct_direct_answer(model, da_problems, max_n=len(da_problems))
        basis_problems = []
        for p in da_correct:
            answer = p["answer"]
            basis_problems.append({
                "a": p["a"], "b": p["b"], "answer": answer,
                "prompt": p["prompt"],
                "ones_digit": answer % 10,
                "tens_digit": (answer // 10) % 10,
                "first_digit": int(str(answer)[0]),
                "n_digits": len(str(answer)),
                "carry": 1 if (p["a"] % 10 + p["b"] % 10) >= 10 else 0,
            })
        # For direct-answer, single-digit test: generate ALL 55 explicitly
        sd_da_problems = generate_single_digit_direct_answer_problems()
        test_problems = filter_correct_single_digit(model, sd_da_problems)
    else:
        # Basis: use ALL correct problems for robust digit-mean estimates
        all_problems = generate_test_problems(max_operand=args.operand_range)
        basis_problems = filter_correct_problems(model, all_problems, n_test=len(all_problems))
        # Test: single-digit answers only (a+b < 10) for clean metrics
        sd_problems = generate_single_digit_problems()
        test_problems = filter_correct_single_digit(model, sd_problems)

    if len(basis_problems) < 20:
        logger.error(f"Only {len(basis_problems)} basis problems — aborting.")
        return
    if len(test_problems) < 10:
        logger.error(f"Only {len(test_problems)} single-digit test problems — aborting.")
        return

    logger.info(f"Basis problems: {len(basis_problems)} (all answer sizes)")
    logger.info(f"Test problems: {len(test_problems)} (single-digit only, a+b < 10)")

    all_results = {
        "model": args.model,
        "model_name": model_name,
        "layers": layers,
        "n_test_problems": len(test_problems),
        "operand_range": args.operand_range,
        "direct_answer": args.direct_answer,
        "experiments": {},
    }

    for layer in layers:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# LAYER {layer}")
        logger.info(f"{'#'*60}")

        # Compute Fourier basis (using ALL problems for robust digit-mean estimates)
        basis, freq_assignments, svals, digit_scores, purities = compute_digit_fourier_basis(
            model, basis_problems, layer, n_problems=len(basis_problems)
        )

        # Sanity checks
        checks_ok = sanity_check_basis(
            basis, freq_assignments, svals, digit_scores, purities,
            model_key=args.model, layer=layer
        )

        all_results["experiments"][f"L{layer}"] = {
            "freq_assignments": freq_assignments,
            "singular_values": svals.tolist(),
            "freq_purities": purities,
            "sanity_passed": checks_ok,
        }

        if args.dry_run:
            logger.info(f"  DRY RUN — skipping phase shift at layer {layer}")
            continue

        if not checks_ok:
            logger.warning(f"  Sanity checks failed at layer {layer} — running anyway")

        # Build frequency planes (with svals for elliptical rotation)
        freq_planes = build_frequency_planes(basis, freq_assignments, svals)

        # Log SV ratios for each frequency plane (shows elliptical correction)
        for k in sorted(freq_planes.keys()):
            plane_sv = freq_planes[k]['svals']
            n_dirs = len(plane_sv)
            if n_dirs >= 2:
                ratio = plane_sv[0] / plane_sv[1] if plane_sv[1] > 1e-10 else float('inf')
                logger.info(f"  k={k}: {n_dirs} dirs, σ=[{', '.join(f'{s:.1f}' for s in plane_sv)}], "
                            f"ratio={ratio:.2f}x")
            else:
                logger.info(f"  k={k}: {n_dirs} dir, σ=[{plane_sv[0]:.1f}]")

        # Run phase shift (pass digit_scores for mean-substitution mode)
        if not args.logit_lens:
            layer_results = run_fourier_phase_shift(
                model, test_problems, layer, freq_planes, basis, n_shifts=9,
                digit_scores=digit_scores
            )
            all_results["experiments"][f"L{layer}"]["phase_shift"] = layer_results

        # Logit lens analysis (if requested)
        if args.logit_lens:
            # Fourier-Unembed alignment diagnostic
            logger.info(f"\n{'#'*60}")
            logger.info(f"# FOURIER-UNEMBED ALIGNMENT at Layer {layer}")
            logger.info(f"{'#'*60}")
            align_results = analyze_fourier_unembed_alignment(model, basis, layer)
            all_results["experiments"][f"L{layer}"]["alignment"] = align_results

            logger.info(f"\n{'#'*60}")
            logger.info(f"# LOGIT LENS ANALYSIS at Layer {layer}")
            logger.info(f"{'#'*60}")
            ll_results = run_logit_lens_analysis(
                model, test_problems, layer, freq_planes, basis, n_shifts=9,
                digit_scores=digit_scores
            )
            all_results["experiments"][f"L{layer}"]["logit_lens"] = ll_results

        gc.collect()

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL COMPARISON")
    logger.info(f"{'='*60}")
    for layer_key, layer_data in all_results["experiments"].items():
        if "phase_shift" in layer_data:
            ps = layer_data["phase_shift"]
            logger.info(f"\n  {layer_key}:")
            for mode_name, mode_data in ps.items():
                logger.info(f"    {mode_name:>12}: digit_changed={mode_data['digit_change_rate']:.1f}%, "
                            f"exact_mod10={mode_data['exact_mod10_rate']:.1f}%")
            # Highlight the key comparison
            coherent = ps.get("coherent", {})
            top2 = ps.get("top2_pca", {})
            k1 = ps.get("k1_only", {})
            logger.info(f"    {'─'*50}")
            logger.info(f"    COHERENT vs TOP2_PCA exact_mod10: "
                        f"{coherent.get('exact_mod10_rate', 0):.1f}% vs "
                        f"{top2.get('exact_mod10_rate', 0):.1f}%")

    # Logit lens summary
    for layer_key, layer_data in all_results["experiments"].items():
        if "logit_lens" in layer_data:
            ll = layer_data["logit_lens"]
            logger.info(f"\n  LOGIT LENS {layer_key}:")
            logger.info(f"    Clean accuracy: {ll.get('clean_logit_lens_accuracy', 0):.1f}%")
            for key, val in ll.items():
                if key.startswith("logit_lens_"):
                    mode_name = key.replace("logit_lens_", "")
                    logger.info(f"    {mode_name:>12}: exact={val['exact_mod10_rate']:.1f}%, "
                                f"changed={val['digit_change_rate']:.1f}%, "
                                f"rank={val['mean_target_rank']:.2f}, "
                                f"prob={val['mean_target_prob']:.3f}, "
                                f"Δlogit={val['mean_logit_diff']:+.2f}")

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = "_direct" if args.direct_answer else ""
    output_file = RESULTS_DIR / f"fourier_phase_rotation_{args.model}{suffix}_{timestamp}.json"

    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    with open(output_file, 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2)
    logger.info(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
