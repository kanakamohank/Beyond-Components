#!/usr/bin/env python3
"""
Experiment 9: Causal Knockout of the 9D Fourier Subspace

Tests whether the 9D Fourier subspace is NECESSARY for digit arithmetic
by ablating it at the computation layer and measuring accuracy drop.

Conditions:
  1. Baseline           — no modification, establish raw accuracy
  2. Fourier-9D ablation — zero out projection onto 9D Fourier subspace
  3. Random-9D ablation  — zero out random 9D orthonormal subspace (control)
  4. Per-frequency ablation — ablate individual frequency pairs (k=1..5)
  5. Progressive ablation — ablate top-k directions (k=1..9) by singular value

Ablation method:
  At the computation layer, last-token position:
    h_ablated = h - P @ h
  where P = V @ V.T is the projection matrix onto the target subspace,
  and V has shape (d_model, n_dims) with orthonormal columns.

  This zeroes out the component of the residual stream that lies in
  the target subspace while preserving all other information.

Usage:
    python fourier_knockout.py --model gemma-2b --device mps
    python fourier_knockout.py --model phi-3 --comp-layer 26 --device mps
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
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from arithmetic_circuit_scan_updated import (
    generate_teacher_forced_problems,
    filter_correct_teacher_forced,
    generate_single_digit_problems,
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
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# SANITY CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def check_projection_matrix(P: np.ndarray, rank: int, label: str = ""):
    """Verify a projection matrix is symmetric, idempotent, and correct rank."""
    # Symmetry
    sym_err = np.abs(P - P.T).max()
    assert sym_err < 1e-5, \
        f"[SANITY] {label}: P not symmetric, max err = {sym_err:.2e}"

    # Idempotency: P @ P = P
    P2 = P @ P
    idem_err = np.abs(P2 - P).max()
    assert idem_err < 1e-4, \
        f"[SANITY] {label}: P not idempotent, max err = {idem_err:.2e}"

    # Rank check via trace (trace of projection matrix = rank)
    actual_rank = np.trace(P)
    assert abs(actual_rank - rank) < 0.5, \
        f"[SANITY] {label}: Expected rank {rank}, trace = {actual_rank:.2f}"

    logger.info(f"  [SANITY] {label}: P is symmetric (err={sym_err:.1e}), "
                f"idempotent (err={idem_err:.1e}), rank={actual_rank:.1f} ✓")


def check_orthonormal(V: np.ndarray, label: str = ""):
    """Verify columns of V are orthonormal."""
    gram = V.T @ V  # should be identity
    eye = np.eye(V.shape[1])
    err = np.abs(gram - eye).max()
    assert err < 1e-5, \
        f"[SANITY] {label}: columns not orthonormal, max err = {err:.2e}"


def make_random_orthonormal_basis(d_model: int, n_dims: int, seed: int = 42) -> np.ndarray:
    """Generate a random orthonormal basis of given dimension via QR decomposition."""
    rng = np.random.RandomState(seed)
    # Generate random Gaussian matrix and QR-decompose
    A = rng.randn(d_model, n_dims)
    Q, R = np.linalg.qr(A)
    # Q is (d_model, n_dims) with orthonormal columns
    assert Q.shape == (d_model, n_dims)
    check_orthonormal(Q, label=f"random-{n_dims}D")
    return Q


# ─────────────────────────────────────────────────────────────────────────────
# DFT FREQUENCY ASSIGNMENT (from eigenvector_dft.py)
# ─────────────────────────────────────────────────────────────────────────────

def assign_frequencies(U_cols: np.ndarray) -> List[int]:
    """
    Assign each SVD direction to its dominant Fourier frequency.

    Args:
        U_cols: (9, 10) — each row is a digit score pattern (U.T from SVD)

    Returns:
        List of dominant frequencies [k1, k2, ..., k9] where ki ∈ {1,2,3,4,5}
    """
    n_dirs = U_cols.shape[0]
    assignments = []

    for i in range(n_dirs):
        scores = U_cols[i]
        scores_centered = scores - scores.mean()

        # DFT
        dft = np.fft.fft(scores_centered)
        power = np.abs(dft) ** 2

        # Group into frequencies 1..5
        freq_power = np.zeros(6)  # idx 0=DC, 1..5=frequencies
        freq_power[0] = power[0]
        for k in range(1, 5):
            freq_power[k] = power[k] + power[10 - k]
        freq_power[5] = power[5]

        # Dominant frequency (skip DC)
        dom_k = int(np.argmax(freq_power[1:])) + 1
        assignments.append(dom_k)

    return assignments


# ─────────────────────────────────────────────────────────────────────────────
# ACTIVATION COLLECTION (reused from eigenvector_dft.py)
# ─────────────────────────────────────────────────────────────────────────────

def collect_per_digit_means(model, problems, layer, device) -> np.ndarray:
    """Collect per-digit mean activations at a given layer (last token position)."""
    hook_name = f"blocks.{layer}.hook_resid_post"
    d_model = model.cfg.d_model

    digit_acts = defaultdict(list)
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
            digit_acts[digit].append(act_vec)

    means = np.zeros((10, d_model))
    for d in range(10):
        if digit_acts[d]:
            means[d] = np.mean(digit_acts[d], axis=0)
            logger.info(f"    Digit {d}: {len(digit_acts[d])} samples")
        else:
            logger.warning(f"    Digit {d}: NO samples!")

    return means


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_accuracy(
    model,
    problems: List[dict],
    layer: int,
    projection_matrix: Optional[torch.Tensor],
    device: str,
    mode: str = "ablate",
    direct_answer: bool = False,
) -> Dict:
    """
    Evaluate arithmetic accuracy with optional subspace ablation.

    Args:
        model: HookedTransformer
        problems: list of problem dicts with "prompt", "target_str", "ones_digit"
        layer: which layer to hook
        projection_matrix: (d_model, d_model) tensor, or None for baseline
        device: "cpu" / "mps" / "cuda"
        mode: "ablate" (remove subspace) or "keep" (keep only subspace)

    Returns:
        Dict with overall accuracy, per-digit accuracy, and counts
    """
    hook_name = f"blocks.{layer}.hook_resid_post"

    total = 0
    correct = 0
    per_digit_total = defaultdict(int)
    per_digit_correct = defaultdict(int)
    predictions = []

    for prob in problems:
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True).to(device)
        target_digit = prob["ones_digit"]

        if projection_matrix is not None:
            P = projection_matrix

            def ablation_hook(act, hook, proj=P, m=mode):
                # Only modify last token position
                h = act[:, -1, :].float()  # (batch, d_model)
                projected = h @ proj       # (batch, d_model)
                if m == "ablate":
                    # Remove the subspace component
                    act[:, -1, :] = (h - projected).to(act.dtype)
                elif m == "keep":
                    # Keep only the subspace component
                    act[:, -1, :] = projected.to(act.dtype)
                return act

            with torch.no_grad():
                with model.hooks(fwd_hooks=[(hook_name, ablation_hook)]):
                    logits = model(tokens)
        else:
            with torch.no_grad():
                logits = model(tokens)

        pred_tok = logits[0, -1].argmax().item()
        pred_str = model.tokenizer.decode([pred_tok]).strip()

        total += 1
        per_digit_total[target_digit] += 1

        try:
            pred_val = int(pred_str)
            pred_digit = pred_val % 10 if direct_answer else pred_val
            if pred_digit == target_digit:
                correct += 1
                per_digit_correct[target_digit] += 1
            predictions.append(pred_digit)
        except ValueError:
            predictions.append(-1)  # non-numeric prediction

    accuracy = correct / total if total > 0 else 0.0
    per_digit_acc = {}
    for d in range(10):
        n = per_digit_total[d]
        c = per_digit_correct[d]
        per_digit_acc[d] = c / n if n > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_digit_accuracy": per_digit_acc,
        "per_digit_total": dict(per_digit_total),
        "per_digit_correct": dict(per_digit_correct),
    }


def evaluate_accuracy_multi_layer(
    model,
    problems: List[dict],
    layer_projections: Dict[int, torch.Tensor],
    device: str,
    mode: str = "ablate",
    direct_answer: bool = False,
) -> Dict:
    """
    Evaluate accuracy with ablation at MULTIPLE layers simultaneously.

    Args:
        layer_projections: {layer_idx: projection_matrix} — one per layer
    """
    total = 0
    correct = 0
    per_digit_total = defaultdict(int)
    per_digit_correct = defaultdict(int)

    # Build hook list
    def make_hook(proj, m):
        def hook_fn(act, hook, p=proj, md=m):
            h = act[:, -1, :].float()
            projected = h @ p
            if md == "ablate":
                act[:, -1, :] = (h - projected).to(act.dtype)
            elif md == "keep":
                act[:, -1, :] = projected.to(act.dtype)
            return act
        return hook_fn

    fwd_hooks = []
    for layer_idx, proj in layer_projections.items():
        hook_name = f"blocks.{layer_idx}.hook_resid_post"
        fwd_hooks.append((hook_name, make_hook(proj, mode)))

    for prob in problems:
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True).to(device)
        target_digit = prob["ones_digit"]

        with torch.no_grad():
            with model.hooks(fwd_hooks=fwd_hooks):
                logits = model(tokens)

        pred_tok = logits[0, -1].argmax().item()
        pred_str = model.tokenizer.decode([pred_tok]).strip()

        total += 1
        per_digit_total[target_digit] += 1
        try:
            pred_val = int(pred_str)
            pred_digit = pred_val % 10 if direct_answer else pred_val
            if pred_digit == target_digit:
                correct += 1
                per_digit_correct[target_digit] += 1
        except ValueError:
            pass

    accuracy = correct / total if total > 0 else 0.0
    per_digit_acc = {}
    for d in range(10):
        n = per_digit_total[d]
        c = per_digit_correct[d]
        per_digit_acc[d] = c / n if n > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_digit_accuracy": per_digit_acc,
        "per_digit_total": dict(per_digit_total),
        "per_digit_correct": dict(per_digit_correct),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Experiment 9: Causal Knockout")
    parser.add_argument("--model", default="gemma-2b", choices=list(MODEL_MAP.keys()))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--comp-layer", type=int, default=None)
    parser.add_argument("--n-train-per-digit", type=int, default=100,
                        help="Problems per digit for computing the subspace (training set)")
    parser.add_argument("--n-test-per-digit", type=int, default=50,
                        help="Problems per digit for evaluation (test set)")
    parser.add_argument("--direct-answer", action="store_true",
                        help="Use direct-answer mode (for LLaMA 3B: full answer as single token)")
    args = parser.parse_args()

    model_name = MODEL_MAP[args.model]
    device = args.device

    comp_defaults = {"gemma-2b": 19, "phi-3": 26, "llama-3b": 20}
    comp_layer = args.comp_layer or comp_defaults.get(args.model, 20)

    logger.info(f"Model: {args.model} ({model_name})")
    logger.info(f"Computation layer: L{comp_layer}")
    logger.info(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=torch.float32 if device == "cpu" else torch.float16,
    )
    model.eval()
    d_model = model.cfg.d_model
    logger.info(f"d_model = {d_model}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: Generate SEPARATE train and test problem sets
    # Train set: used to compute the 9D subspace (per-digit means → SVD)
    # Test set: used to evaluate accuracy under ablation
    # This prevents data leakage.
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "═" * 60)
    logger.info("  STEP 1: Generate train/test problem sets")
    logger.info("═" * 60)

    # Generate a large pool and split
    n_train = args.n_train_per_digit
    n_test = args.n_test_per_digit
    n_total_needed = n_train + n_test

    if args.direct_answer:
        all_problems, _ = generate_direct_answer_problems(n_per_digit=n_total_needed)
        logger.info(f"  Generated {len(all_problems)} direct-answer problems")
        all_correct = filter_correct_direct_answer(model, all_problems, max_n=len(all_problems))
        logger.info(f"  Correct: {len(all_correct)}")
    else:
        try:
            all_problems, by_digit_pool = generate_teacher_forced_problems(
                n_per_digit=n_total_needed
            )
            logger.info(f"  Generated {len(all_problems)} total problems")

            # Filter for correctness
            all_correct = filter_correct_teacher_forced(
                model, all_problems, max_n=n_total_needed * 10
            )
            logger.info(f"  Correct: {len(all_correct)}")

        except Exception as e:
            logger.warning(f"  Teacher-forced failed ({e}), trying single-digit fallback...")
            all_problems = generate_single_digit_problems()
            all_correct = filter_correct_teacher_forced(model, all_problems, max_n=500)

    # Balance and split into train/test by digit
    by_digit = defaultdict(list)
    for p in all_correct:
        by_digit[p["ones_digit"]].append(p)

    min_count = min(len(by_digit[d]) for d in range(10))
    logger.info(f"  Min per-digit count: {min_count}")

    # Determine actual train/test split
    actual_train = min(n_train, min_count - 10)  # reserve at least 10 for test
    actual_test = min(n_test, min_count - actual_train)
    assert actual_train >= 10, \
        f"[SANITY] Insufficient training data: {actual_train} per digit (need ≥10)"
    assert actual_test >= 10, \
        f"[SANITY] Insufficient test data: {actual_test} per digit (need ≥10)"

    train_problems = []
    test_problems = []
    for d in range(10):
        digit_probs = by_digit[d][:min_count]
        train_problems.extend(digit_probs[:actual_train])
        test_problems.extend(digit_probs[actual_train:actual_train + actual_test])

    logger.info(f"  Train: {len(train_problems)} ({actual_train}/digit)")
    logger.info(f"  Test:  {len(test_problems)} ({actual_test}/digit)")

    # Sanity: no overlap between train and test
    train_prompts = set(p["prompt"] for p in train_problems)
    test_prompts = set(p["prompt"] for p in test_problems)
    overlap = train_prompts & test_prompts
    assert len(overlap) == 0, \
        f"[SANITY] Train/test overlap: {len(overlap)} shared prompts!"
    logger.info(f"  [SANITY] Train/test disjoint ✓ (0 overlap)")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: Compute the 9D Fourier subspace from TRAINING data
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "═" * 60)
    logger.info("  STEP 2: Compute 9D Fourier subspace (training data only)")
    logger.info("═" * 60)

    digit_means = collect_per_digit_means(model, train_problems, comp_layer, device)

    # Center and SVD
    centroid = digit_means.mean(axis=0, keepdims=True)  # (1, d_model)
    M_centered = digit_means - centroid                   # (10, d_model)

    # Centering sanity
    col_means = np.abs(M_centered.mean(axis=0))
    assert col_means.max() < 1e-5, \
        f"[SANITY] Centering failed: max col mean = {col_means.max():.2e}"

    U, S, Vt = np.linalg.svd(M_centered, full_matrices=False)
    # M_centered = U @ diag(S) @ Vt
    # Vt: (10, d_model) — rows are activation-space directions
    # U:  (10, 10) — columns are digit loading patterns
    # We use the top 9 rows of Vt as the 9D subspace basis

    logger.info(f"  SVD singular values: {S[:9].round(2)}")
    logger.info(f"  σ₁/σ₂ = {S[0]/S[1]:.2f}, σ₁/σ₉ = {S[0]/S[8]:.2f}")

    # The 9D subspace basis: columns of V_9 are the 9 directions
    V_9 = Vt[:9].T  # (d_model, 9) — columns are orthonormal basis vectors
    check_orthonormal(V_9, label="Fourier-9D basis")

    # Projection matrix: P = V_9 @ V_9.T (d_model, d_model)
    P_9 = V_9 @ V_9.T
    check_projection_matrix(P_9, rank=9, label="Fourier-9D")

    # Assign frequencies to each direction
    freq_assignments = assign_frequencies(U[:, :9].T)  # U.T rows = digit patterns
    logger.info(f"  Frequency assignments: {freq_assignments}")

    # Group directions by frequency for per-frequency ablation
    freq_to_dirs = defaultdict(list)
    for dir_idx, freq in enumerate(freq_assignments):
        freq_to_dirs[freq].append(dir_idx)
    logger.info(f"  Frequency → directions: {dict(freq_to_dirs)}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: Build all ablation conditions
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "═" * 60)
    logger.info("  STEP 3: Build ablation projection matrices")
    logger.info("═" * 60)

    # Convert to torch for hook use
    P_fourier_9d = torch.tensor(P_9, dtype=torch.float32, device=device)

    # Random 9D control
    V_rand = make_random_orthonormal_basis(d_model, 9, seed=42)
    P_rand_9d = torch.tensor(V_rand @ V_rand.T, dtype=torch.float32, device=device)
    check_projection_matrix(V_rand @ V_rand.T, rank=9, label="Random-9D")

    # Second random 9D (different seed, to check stability)
    V_rand2 = make_random_orthonormal_basis(d_model, 9, seed=123)
    P_rand_9d_2 = torch.tensor(V_rand2 @ V_rand2.T, dtype=torch.float32, device=device)

    # Sanity: Fourier and random subspaces should have low overlap
    # Principal angle cosines between two 9D subspaces
    cos_angles = np.linalg.svd(V_9.T @ V_rand, compute_uv=False)
    logger.info(f"  [SANITY] Fourier↔Random principal cosines: {cos_angles[:3].round(3)}")
    assert cos_angles[0] < 0.5, \
        f"[SANITY] Random basis too aligned with Fourier (cos₁={cos_angles[0]:.3f})"
    logger.info(f"  [SANITY] Fourier↔Random alignment is low ✓ (max cos = {cos_angles[0]:.3f})")

    # Per-frequency projection matrices
    per_freq_projections = {}
    for k in range(1, 6):
        if k in freq_to_dirs:
            dir_indices = freq_to_dirs[k]
            V_k = Vt[dir_indices].T  # (d_model, n_dirs_for_this_freq)
            P_k = V_k @ V_k.T
            n_dirs = len(dir_indices)
            check_projection_matrix(P_k, rank=n_dirs, label=f"k={k} ({n_dirs}D)")
            per_freq_projections[k] = torch.tensor(P_k, dtype=torch.float32, device=device)

    # Progressive ablation: top-1, top-2, ..., top-9 directions
    progressive_projections = {}
    for n in [1, 2, 3, 5, 7, 9]:
        V_n = Vt[:n].T  # (d_model, n)
        P_n = V_n @ V_n.T
        check_projection_matrix(P_n, rank=n, label=f"Progressive-{n}D")
        progressive_projections[n] = torch.tensor(P_n, dtype=torch.float32, device=device)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4: Run all conditions
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "═" * 60)
    logger.info("  STEP 4: Evaluate all ablation conditions")
    logger.info("═" * 60)

    results = {}

    def run_condition(name, proj_matrix, mode="ablate"):
        logger.info(f"\n  ── {name} ──")
        t0 = time.time()
        res = evaluate_accuracy(model, test_problems, comp_layer, proj_matrix, device, mode,
                               direct_answer=args.direct_answer)
        elapsed = time.time() - t0
        acc = res["accuracy"]
        logger.info(f"  Accuracy: {res['correct']}/{res['total']} = {acc*100:.1f}%  ({elapsed:.1f}s)")
        per_d = res["per_digit_accuracy"]
        logger.info(f"  Per-digit: " +
                    " ".join(f"{d}:{per_d[d]*100:.0f}%" for d in range(10)))
        results[name] = res
        return res

    # 1. Baseline (no hook)
    baseline = run_condition("baseline", None)
    baseline_acc = baseline["accuracy"]
    logger.info(f"\n  *** Baseline accuracy: {baseline_acc*100:.1f}% ***")

    # Sanity: baseline should be high (>70%) — these are pre-filtered correct problems
    assert baseline_acc > 0.70, \
        f"[SANITY] Baseline accuracy too low: {baseline_acc*100:.1f}% " \
        f"(expected >70% on pre-filtered problems)"

    # 2. Fourier-9D ablation (THE KEY TEST)
    run_condition("fourier_9d_ablate", P_fourier_9d, mode="ablate")

    # 3. Random-9D ablation (dimensionality control)
    run_condition("random_9d_ablate", P_rand_9d, mode="ablate")
    run_condition("random_9d_ablate_seed2", P_rand_9d_2, mode="ablate")

    # 4. Per-frequency ablation
    for k in sorted(per_freq_projections.keys()):
        n_dirs = len(freq_to_dirs[k])
        dof_label = f"{n_dirs}D" if n_dirs > 1 else "1D"
        run_condition(f"freq_k{k}_ablate ({dof_label})", per_freq_projections[k], mode="ablate")

    # 5. Progressive ablation (top-k directions by singular value)
    for n in sorted(progressive_projections.keys()):
        run_condition(f"progressive_{n}d_ablate", progressive_projections[n], mode="ablate")

    # 6. MULTI-LAYER ablation — compute 9D subspace at each layer, ablate simultaneously
    logger.info(f"\n  ── MULTI-LAYER ABLATION ──")
    logger.info(f"  Computing per-layer Fourier subspaces...")

    # Layer ranges to test
    readout_defaults = {"gemma-2b": 25, "phi-3": 31, "llama-3b": 27}
    readout_layer = readout_defaults.get(args.model, comp_layer + 6)
    mid_layer = (comp_layer + readout_layer) // 2

    multi_layer_configs = [
        ("comp+mid", [comp_layer, mid_layer]),
        ("comp+readout", [comp_layer, readout_layer]),
        ("comp+mid+readout", [comp_layer, mid_layer, readout_layer]),
        ("all_comp_to_readout", list(range(comp_layer, readout_layer + 1))),
    ]

    for config_name, layers in multi_layer_configs:
        layer_projs = {}
        for l in layers:
            logger.info(f"    Computing 9D subspace at L{l}...")
            means_l = collect_per_digit_means(model, train_problems, l, device)
            cent_l = means_l.mean(axis=0, keepdims=True)
            M_l = means_l - cent_l
            _, _, Vt_l = np.linalg.svd(M_l, full_matrices=False)
            V_l = Vt_l[:9].T  # (d_model, 9)
            P_l = V_l @ V_l.T
            layer_projs[l] = torch.tensor(P_l, dtype=torch.float32, device=device)

        layer_str = "+".join(f"L{l}" for l in layers)
        name = f"multi_{config_name} ({layer_str})"
        logger.info(f"\n  ── {name} ──")
        t0 = time.time()
        res = evaluate_accuracy_multi_layer(
            model, test_problems, layer_projs, device, mode="ablate",
            direct_answer=args.direct_answer
        )
        elapsed = time.time() - t0
        acc = res["accuracy"]
        logger.info(f"  Accuracy: {res['correct']}/{res['total']} = {acc*100:.1f}%  ({elapsed:.1f}s)")
        per_d = res["per_digit_accuracy"]
        logger.info(f"  Per-digit: " +
                    " ".join(f"{d}:{per_d[d]*100:.0f}%" for d in range(10)))
        results[name] = res

    # Multi-layer random control
    logger.info(f"\n  ── Multi-layer RANDOM control ──")
    all_layers = list(range(comp_layer, readout_layer + 1))
    rand_layer_projs = {}
    for l in all_layers:
        V_r = make_random_orthonormal_basis(d_model, 9, seed=42 + l)
        rand_layer_projs[l] = torch.tensor(
            V_r @ V_r.T, dtype=torch.float32, device=device
        )
    name = f"multi_random_all ({'+'.join(f'L{l}' for l in all_layers)})"
    logger.info(f"\n  ── {name} ──")
    t0 = time.time()
    res = evaluate_accuracy_multi_layer(
        model, test_problems, rand_layer_projs, device, mode="ablate",
        direct_answer=args.direct_answer
    )
    elapsed = time.time() - t0
    acc = res["accuracy"]
    logger.info(f"  Accuracy: {res['correct']}/{res['total']} = {acc*100:.1f}%  ({elapsed:.1f}s)")
    per_d = res["per_digit_accuracy"]
    logger.info(f"  Per-digit: " +
                " ".join(f"{d}:{per_d[d]*100:.0f}%" for d in range(10)))
    results[name] = res

    # ══════════════════════════════════════════════════════════════════════
    # STEP 5: Summary
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "═" * 60)
    logger.info("  SUMMARY")
    logger.info("═" * 60)

    logger.info(f"\n  {'Condition':<35} {'Accuracy':>10} {'Δ from baseline':>15}")
    logger.info(f"  {'─'*35} {'─'*10} {'─'*15}")
    for name, res in results.items():
        acc = res["accuracy"] * 100
        delta = (res["accuracy"] - baseline_acc) * 100
        marker = ""
        if name == "baseline":
            marker = " ◀ REFERENCE"
        elif abs(delta) > 30:
            marker = " ★★★" if delta < 0 else " !!!"
        elif abs(delta) > 10:
            marker = " ★★"
        elif abs(delta) > 5:
            marker = " ★"
        logger.info(f"  {name:<35} {acc:>9.1f}% {delta:>+14.1f}%{marker}")

    # Chance level analysis
    chance = 10.0  # 10% for 10 digits
    fourier_acc = results.get("fourier_9d_ablate", {}).get("accuracy", -1) * 100
    random_acc = results.get("random_9d_ablate", {}).get("accuracy", -1) * 100

    logger.info(f"\n  Key comparisons:")
    logger.info(f"    Baseline:        {baseline_acc*100:.1f}%")
    logger.info(f"    Fourier ablated: {fourier_acc:.1f}%")
    logger.info(f"    Random ablated:  {random_acc:.1f}%")
    logger.info(f"    Chance level:    {chance:.1f}%")

    if fourier_acc < 20 and random_acc > 50:
        logger.info(f"\n  ★ STRONG NECESSITY: Fourier ablation → near-chance, "
                    f"random ablation → still functional")
        logger.info(f"    The 9D Fourier subspace is NECESSARY for arithmetic.")
    elif fourier_acc < random_acc - 20:
        logger.info(f"\n  ★ MODERATE NECESSITY: Fourier ablation hurts more than random "
                    f"({fourier_acc:.1f}% vs {random_acc:.1f}%)")
    else:
        logger.info(f"\n  ⚠ WEAK/NO NECESSITY: Fourier ablation similar to random "
                    f"({fourier_acc:.1f}% vs {random_acc:.1f}%)")

    # ── Save results ──────────────────────────────────────────────────────
    output = {
        "model": model_name,
        "model_short": args.model,
        "comp_layer": comp_layer,
        "d_model": d_model,
        "n_train_per_digit": actual_train,
        "n_test_per_digit": actual_test,
        "n_test_total": len(test_problems),
        "svd_singular_values": S[:9].tolist(),
        "freq_assignments": freq_assignments,
        "freq_to_dirs": {str(k): v for k, v in freq_to_dirs.items()},
        "fourier_random_principal_cosines": cos_angles[:3].tolist(),
        "conditions": {},
    }
    for name, res in results.items():
        output["conditions"][name] = {
            "accuracy": res["accuracy"],
            "correct": res["correct"],
            "total": res["total"],
            "per_digit_accuracy": {str(k): v for k, v in res["per_digit_accuracy"].items()},
        }

    suffix = "_direct" if args.direct_answer else ""
    out_path = RESULTS_DIR / f"fourier_knockout_{args.model}{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
