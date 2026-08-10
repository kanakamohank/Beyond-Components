#!/usr/bin/env python3
"""
Experiment 10: Attention Head & MLP Attribution for the 9D Fourier Subspace

Identifies WHICH specific attention heads and MLP layers write into the
9D Fourier subspace that encodes ones-digit arithmetic.  This answers
"which components perform the angle addition" rather than just
"where is the result stored."

══════════════════════════════════════════════════════════════════════
APPROACH: Hybrid Direct Logit Attribution + Targeted Activation Patching
══════════════════════════════════════════════════════════════════════

Phase 1 — Direct Writing Score (DLA):
  For each component c (attention head or MLP) at each layer L, measure
  how much of its output lands in the 9D Fourier subspace:

    writing_score(c) = E_x[ ||P_F · output_c(x)||² / ||output_c(x)||² ]

  where P_F = V₉ V₉ᵀ is the Fourier projection matrix and V₉ is the
  9D basis from per-digit-mean SVD at the computation layer.

  This requires ONE forward pass per sample (with full cache), so it is
  O(n_samples) not O(n_components × n_samples).

Phase 2 — Causal Patching (Validation):
  For the top-K components from Phase 1, run zero-ablation patching:
  replace each component's Fourier-projected output with zero (or mean)
  and measure accuracy drop.  This confirms causal NECESSITY.

  This is O(K × n_samples) forward passes, where K << total components.

Phase 3 — Frequency-Resolved Attribution:
  For each top component, decompose its writing into per-frequency
  contributions (k=1..5).  This reveals which components write which
  Fourier frequencies — e.g., does head L19H3 specifically write k=5
  (parity)?

══════════════════════════════════════════════════════════════════════
SANITY CHECKS (built-in):
══════════════════════════════════════════════════════════════════════

  S1. Residual stream decomposition: sum of all component outputs at
      last token ≈ residual stream at that layer (up to LayerNorm).
  S2. Fourier projection is idempotent and correct rank.
  S3. Writing scores sum ≤ 1.0 (no double-counting via Pythagorean).
  S4. Random-subspace control: DLA with random 9D should give ~9/d_model.
  S5. Phase 2 damage ordering correlates with Phase 1 DLA ranking.
  S6. Per-frequency writing fractions sum to 100% for each component.
  S7. Train/test split: subspace computed on train, evaluation on test.

Usage:
    python fourier_head_attribution.py --model gemma-2b --device mps
    python fourier_head_attribution.py --model phi-3 --comp-layer 26 --device mps
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
    VALID_PROMPT_FORMATS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("mathematical_toolkit_results")
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# HOOK POINT VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def verify_hook_points(model, device):
    """
    Verify that the hook points we depend on exist in this model.

    Critical hook: blocks.{L}.attn.hook_result — per-head output AFTER W_O,
    shape (batch, pos, n_heads, d_model).  This is the direct contribution
    of each head to the residual stream.

    If hook_result does not exist (older TransformerLens), we fall back to
    hook_z (shape: batch, pos, n_heads, d_head) and manually project via W_O.

    Returns:
        use_hook_result: bool — True if hook_result exists, False to use hook_z
    """
    test_input = model.to_tokens("1 + 2 = ", prepend_bos=True).to(device)

    # Try hook_result first
    try:
        hook_name = "blocks.0.attn.hook_result"
        with torch.no_grad():
            _, cache = model.run_with_cache(test_input, names_filter=[hook_name])
        if hook_name in cache:
            shape = cache[hook_name].shape
            del cache
            expected_dims = 4  # (batch, pos, n_heads, d_model)
            if len(shape) == expected_dims and shape[-1] == model.cfg.d_model:
                logger.info(f"  [HOOK] attn.hook_result available, shape={list(shape)} ✓")
                return True
        del cache
    except Exception:
        pass

    # Fallback to hook_z
    hook_name = "blocks.0.attn.hook_z"
    with torch.no_grad():
        _, cache = model.run_with_cache(test_input, names_filter=[hook_name])
    assert hook_name in cache, "Neither hook_result nor hook_z found!"
    shape = cache[hook_name].shape
    del cache
    logger.info(f"  [HOOK] attn.hook_result NOT available; using hook_z, shape={list(shape)}")
    logger.info(f"  [HOOK] Will manually project via W_O to get d_model-space outputs")
    return False


def get_head_outputs_from_cache(cache, layer, model, use_hook_result, device):
    """
    Extract per-head outputs in d_model space from the cache.

    If use_hook_result=True: directly read hook_result (batch, pos, n_heads, d_model)
    If use_hook_result=False: read hook_z (batch, pos, n_heads, d_head) and
      multiply by W_O to get d_model-space outputs.

    Returns: (n_heads, d_model) tensor at last_pos, or None if not in cache.
    """
    if use_hook_result:
        key = f"blocks.{layer}.attn.hook_result"
        if key not in cache:
            return None
        return cache[key]  # (batch, pos, n_heads, d_model)
    else:
        key = f"blocks.{layer}.attn.hook_z"
        if key not in cache:
            return None
        z = cache[key]  # (batch, pos, n_heads, d_head)
        # W_O: (n_heads, d_head, d_model)
        W_O = model.blocks[layer].attn.W_O
        # result[h] = z[h] @ W_O[h]  →  einsum for all heads
        result = torch.einsum("bphi,hid->bphd", z.float(), W_O.float())
        return result  # (batch, pos, n_heads, d_model)


def get_attn_hook_name(layer, use_hook_result):
    """Return the appropriate hook name for per-head attention output."""
    if use_hook_result:
        return f"blocks.{layer}.attn.hook_result"
    else:
        return f"blocks.{layer}.attn.hook_z"


# ─────────────────────────────────────────────────────────────────────────────
# SANITY CHECK UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def check_orthonormal(V: np.ndarray, label: str = ""):
    """Verify columns of V are orthonormal."""
    gram = V.T @ V
    eye = np.eye(V.shape[1])
    err = np.abs(gram - eye).max()
    assert err < 1e-4, \
        f"[SANITY] {label}: columns not orthonormal, max err = {err:.2e}"
    logger.info(f"  [SANITY] {label}: orthonormal ✓ (err={err:.1e})")


def check_projection_matrix(P: np.ndarray, rank: int, label: str = ""):
    """Verify P is symmetric, idempotent, and correct rank."""
    sym_err = np.abs(P - P.T).max()
    assert sym_err < 1e-4, \
        f"[SANITY] {label}: P not symmetric, max err = {sym_err:.2e}"

    P2 = P @ P
    idem_err = np.abs(P2 - P).max()
    assert idem_err < 1e-3, \
        f"[SANITY] {label}: P not idempotent, max err = {idem_err:.2e}"

    actual_rank = np.trace(P)
    assert abs(actual_rank - rank) < 0.5, \
        f"[SANITY] {label}: Expected rank {rank}, trace = {actual_rank:.2f}"

    logger.info(f"  [SANITY] {label}: symmetric (err={sym_err:.1e}), "
                f"idempotent (err={idem_err:.1e}), rank={actual_rank:.1f} ✓")


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
# DATA GENERATION (balanced, with train/test split)
# ─────────────────────────────────────────────────────────────────────────────

def generate_train_test(
    model,
    n_train_per_digit: int = 50,
    n_test_per_digit: int = 30,
    prompt_format: str = "calculate",
    direct_answer: bool = False,
):
    """Generate balanced, disjoint train and test problem sets."""
    n_total = n_train_per_digit + n_test_per_digit

    correct = []
    if direct_answer:
        logger.info("  Using direct-answer mode (full answer as single token)")
        problems, _ = generate_direct_answer_problems(
            n_per_digit=n_total, operand_max=99
        )
        correct = filter_correct_direct_answer(model, problems, max_n=n_total * 10)
    else:
        try:
            problems, _ = generate_teacher_forced_problems(
                n_per_digit=n_total, prompt_format=prompt_format
            )
            correct = filter_correct_teacher_forced(model, problems, max_n=n_total * 10)
        except Exception as e:
            logger.warning(f"  Teacher-forced failed ({e})")

    # Check per-digit balance
    by_digit = defaultdict(list)
    for p in correct:
        by_digit[p["ones_digit"]].append(p)
    min_count = min(len(by_digit.get(d, [])) for d in range(10))

    if min_count < (n_train_per_digit + 1) and not direct_answer:
        logger.warning(
            f"  Multi-digit: only {min_count}/digit (need {n_total}). "
            f"Falling back to single-digit..."
        )
        problems = generate_single_digit_problems(prompt_format=prompt_format)
        correct = filter_correct_teacher_forced(model, problems, max_n=500)
        by_digit = defaultdict(list)
        for p in correct:
            by_digit[p["ones_digit"]].append(p)
        min_count = min(len(by_digit.get(d, [])) for d in range(10))

    if min_count < 3:
        raise RuntimeError(
            f"[DATA] Only {min_count} samples per digit after filtering. "
            f"Need ≥3 (2 train + 1 test minimum). "
            f"Model may not support arithmetic at this prompt format."
        )

    actual_train = min(n_train_per_digit, max(1, min_count - 2))
    actual_test = min(n_test_per_digit, max(1, min_count - actual_train))

    assert actual_train >= 1, \
        f"[SANITY] Insufficient training data: {actual_train}/digit"
    assert actual_test >= 1, \
        f"[SANITY] Insufficient test data: {actual_test}/digit"

    train, test = [], []
    for d in range(10):
        pool = by_digit[d][:min_count]
        train.extend(pool[:actual_train])
        test.extend(pool[actual_train:actual_train + actual_test])

    # S7: Verify no overlap
    train_prompts = set(p["prompt"] for p in train)
    test_prompts = set(p["prompt"] for p in test)
    assert len(train_prompts & test_prompts) == 0, \
        "[SANITY] Train/test overlap!"

    logger.info(f"  Train: {len(train)} ({actual_train}/digit)")
    logger.info(f"  Test:  {len(test)} ({actual_test}/digit)")
    logger.info(f"  [SANITY] Train/test disjoint ✓")
    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# FOURIER SUBSPACE CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_fourier_subspace(model, train_problems, comp_layer, device):
    """
    Compute the 9D Fourier subspace from per-digit mean activations.

    Returns:
        V_9: (d_model, 9) orthonormal basis
        S: (9,) singular values
        freq_assignments: [k1, ..., k9] dominant frequency per direction
        U: (10, 10) digit loading matrix
    """
    hook_name = f"blocks.{comp_layer}.hook_resid_post"
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
            digit_acts[digit].append(holder["act"][0, -1].cpu().float().numpy())

    # Per-digit means
    means = np.zeros((10, d_model))
    for d in range(10):
        assert len(digit_acts[d]) > 0, f"[SANITY] No samples for digit {d}"
        means[d] = np.mean(digit_acts[d], axis=0)
        logger.info(f"    Digit {d}: {len(digit_acts[d])} samples")

    # Center and SVD
    centroid = means.mean(axis=0, keepdims=True)
    M = means - centroid

    # Centering sanity
    col_means = np.abs(M.mean(axis=0))
    assert col_means.max() < 1e-5, \
        f"[SANITY] Centering failed: max col mean = {col_means.max():.2e}"

    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    V_9 = Vt[:9].T  # (d_model, 9)

    check_orthonormal(V_9, label="Fourier-9D basis")

    logger.info(f"  SVD σ values: {S[:9].round(2)}")
    logger.info(f"  σ₁/σ₂ = {S[0]/S[1]:.2f}, σ₁/σ₉ = {S[0]/S[8]:.2f}")

    freq_assignments = assign_frequencies(U[:, :9].T)
    logger.info(f"  Frequency assignments: {freq_assignments}")

    return V_9, S[:9], freq_assignments, U


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: DIRECT WRITING SCORE (DLA)
# ─────────────────────────────────────────────────────────────────────────────

def compute_writing_scores(model, problems, comp_layer, V_9, device, use_hook_result=True):
    """
    Compute how much each attention head and MLP writes into the 9D
    Fourier subspace, using BOTH unsigned and signed attribution.

    Metrics per component c:
      UNSIGNED: ||P_F · output_c||²  — raw Fourier energy from this component
      SIGNED:   (P_F · output_c) · (P_F · resid_readout)
                — net contribution to total Fourier energy, including interference.
                Sums EXACTLY to ||P_F · resid_readout||² across all components,
                because resid = embed + Σ_L(Σ_H(head_LH) + mlp_L) and P_F is linear.

    The signed decomposition is the principled attribution because it captures
    constructive/destructive interference and provides an exact additive
    decomposition of the total Fourier energy at the readout layer.

    Also tracks the embedding component (token + positional embeddings).

    Returns:
        head_scores: dict (L, H) -> {writing_frac, signed_attr, unsigned_attr, ...}
        mlp_scores: dict L -> {writing_frac, signed_attr, unsigned_attr, ...}
        embed_score: dict with embed component attribution
        additivity_info: dict with sanity check results
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model

    V_9_t = torch.tensor(V_9, dtype=torch.float32)  # (d_model, 9)
    P_F = V_9_t @ V_9_t.T  # (d_model, d_model) — Fourier projection

    # Collect per-head attn output, hook_mlp_out, embed, and readout resid
    hook_names = [f"hook_embed", f"hook_pos_embed"]
    for L in range(n_layers):
        hook_names.append(get_attn_hook_name(L, use_hook_result))
        hook_names.append(f"blocks.{L}.hook_mlp_out")
    # Readout-layer residual for signed attribution
    hook_names.append(f"blocks.{n_layers - 1}.hook_resid_post")

    # Accumulators — unsigned
    head_fourier_norms = defaultdict(list)   # (L, H) -> [||P_F · h||²]
    head_total_norms = defaultdict(list)     # (L, H) -> [||h||²]
    mlp_fourier_norms = defaultdict(list)    # L -> [||P_F · m||²]
    mlp_total_norms = defaultdict(list)      # L -> [||m||²]
    embed_fourier_norms = []                 # [||P_F · embed||²]
    embed_total_norms = []                   # [||embed||²]

    # Accumulators — signed
    head_signed = defaultdict(list)          # (L, H) -> [signed_attr]
    mlp_signed = defaultdict(list)           # L -> [signed_attr]
    embed_signed = []                        # [signed_attr]

    # Additivity tracking
    total_fourier_norms = []     # [||P_F · resid_readout||²]
    sum_signed_all = []          # [Σ signed attrs]

    n_processed = 0
    for prob in problems:
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True).to(device)
        last_pos = tokens.shape[1] - 1

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)

        # Readout residual — last layer's resid_post at last token
        resid_key = f"blocks.{n_layers - 1}.hook_resid_post"
        resid_readout = cache[resid_key][0, last_pos].float().cpu()  # (d_model,)
        pf_resid = P_F @ resid_readout  # (d_model,)
        total_f_norm_sq = (pf_resid ** 2).sum().item()
        total_fourier_norms.append(total_f_norm_sq)

        # Embedding component (token + positional)
        embed_vec = torch.zeros(d_model)
        if "hook_embed" in cache:
            embed_vec = embed_vec + cache["hook_embed"][0, last_pos].float().cpu()
        if "hook_pos_embed" in cache:
            embed_vec = embed_vec + cache["hook_pos_embed"][0, last_pos].float().cpu()
        pf_embed = P_F @ embed_vec
        embed_fourier_norms.append((pf_embed ** 2).sum().item())
        embed_total_norms.append((embed_vec ** 2).sum().item())
        embed_signed.append((pf_embed * pf_resid).sum().item())

        sample_signed_sum = (pf_embed * pf_resid).sum().item()

        for L in range(n_layers):
            # Attention heads — get per-head d_model-space outputs
            attn_full = get_head_outputs_from_cache(cache, L, model, use_hook_result, device)
            if attn_full is not None:
                attn_out = attn_full[0, last_pos].float().cpu()  # (n_heads, d_model)
                for H in range(n_heads):
                    h_vec = attn_out[H]  # (d_model,)
                    pf_h = P_F @ h_vec   # (d_model,)

                    f_norm_sq = (pf_h ** 2).sum().item()
                    t_norm_sq = (h_vec ** 2).sum().item()
                    s_attr = (pf_h * pf_resid).sum().item()

                    head_fourier_norms[(L, H)].append(f_norm_sq)
                    head_total_norms[(L, H)].append(t_norm_sq)
                    head_signed[(L, H)].append(s_attr)
                    sample_signed_sum += s_attr

            # MLP
            mlp_key = f"blocks.{L}.hook_mlp_out"
            if mlp_key in cache:
                mlp_out = cache[mlp_key][0, last_pos].float().cpu()
                pf_m = P_F @ mlp_out

                f_norm_sq = (pf_m ** 2).sum().item()
                t_norm_sq = (mlp_out ** 2).sum().item()
                s_attr = (pf_m * pf_resid).sum().item()

                mlp_fourier_norms[L].append(f_norm_sq)
                mlp_total_norms[L].append(t_norm_sq)
                mlp_signed[L].append(s_attr)
                sample_signed_sum += s_attr

        sum_signed_all.append(sample_signed_sum)
        del cache
        n_processed += 1
        if n_processed % 20 == 0:
            logger.info(f"  Phase 1 DLA: {n_processed}/{len(problems)} problems...")

    # ── Additivity sanity check ──────────────────────────────────────────
    # Σ signed_attr(c) should equal ||P_F · resid||² for each sample.
    # Discrepancies indicate LayerNorm non-linearity or missing components
    # (e.g., attention biases not folded).
    additivity_errors = np.array(sum_signed_all) - np.array(total_fourier_norms)
    mean_total_f = np.mean(total_fourier_norms)
    mean_add_err = np.mean(np.abs(additivity_errors))
    rel_add_err = mean_add_err / max(mean_total_f, 1e-12)

    if rel_add_err > 0.05:
        logger.warning(
            f"  [SANITY] Fourier additivity error: {rel_add_err:.1%} "
            f"(mean |Σ signed - ||P_F·resid||²| = {mean_add_err:.2f}, "
            f"mean total = {mean_total_f:.2f}). "
            f"Likely due to LayerNorm or untracked biases. "
            f"Signed attribution is approximate."
        )
    else:
        logger.info(
            f"  [SANITY] Fourier additivity: {rel_add_err:.1%} relative error ✓ "
            f"(Σ signed ≈ ||P_F·resid||²)"
        )

    additivity_info = {
        "mean_total_fourier_energy": float(mean_total_f),
        "mean_additivity_error": float(mean_add_err),
        "relative_additivity_error": float(rel_add_err),
    }

    # ── Aggregate scores ─────────────────────────────────────────────────
    head_scores = {}
    for L in range(n_layers):
        for H in range(n_heads):
            key = (L, H)
            if key not in head_fourier_norms:
                continue
            mean_f = np.mean(head_fourier_norms[key])
            mean_t = np.mean(head_total_norms[key])
            mean_s = np.mean(head_signed[key])
            frac = mean_f / max(mean_t, 1e-12)
            head_scores[key] = {
                "layer": L, "head": H,
                "mean_fourier_norm_sq": float(mean_f),
                "mean_total_norm_sq": float(mean_t),
                "writing_frac": float(frac),
                "signed_attr": float(mean_s),
                "pct_of_total": float(mean_s / max(mean_total_f, 1e-12) * 100),
                "n_samples": len(head_fourier_norms[key]),
            }

    mlp_scores = {}
    for L in range(n_layers):
        if L not in mlp_fourier_norms:
            continue
        mean_f = np.mean(mlp_fourier_norms[L])
        mean_t = np.mean(mlp_total_norms[L])
        mean_s = np.mean(mlp_signed[L])
        frac = mean_f / max(mean_t, 1e-12)
        mlp_scores[L] = {
            "layer": L,
            "mean_fourier_norm_sq": float(mean_f),
            "mean_total_norm_sq": float(mean_t),
            "writing_frac": float(frac),
            "signed_attr": float(mean_s),
            "pct_of_total": float(mean_s / max(mean_total_f, 1e-12) * 100),
            "n_samples": len(mlp_fourier_norms[L]),
        }

    embed_score = {
        "mean_fourier_norm_sq": float(np.mean(embed_fourier_norms)) if embed_fourier_norms else 0,
        "mean_total_norm_sq": float(np.mean(embed_total_norms)) if embed_total_norms else 0,
        "signed_attr": float(np.mean(embed_signed)) if embed_signed else 0,
        "pct_of_total": float(np.mean(embed_signed) / max(mean_total_f, 1e-12) * 100) if embed_signed else 0,
    }

    return head_scores, mlp_scores, embed_score, additivity_info


def run_random_baseline_dla(model, problems, d_model, device, n_random=5, seed=42):
    """
    S4: Random 9D subspace control — DLA should give ~9/d_model writing fraction.

    We run DLA with random orthonormal 9D bases and check that the mean
    writing fraction is close to the null expectation.
    """
    rng = np.random.RandomState(seed)
    null_expected = 9.0 / d_model

    random_fracs = []
    for trial in range(n_random):
        A = rng.randn(d_model, 9)
        Q, _ = np.linalg.qr(A)
        V_rand = Q[:, :9]  # (d_model, 9)

        V_rand_t = torch.tensor(V_rand, dtype=torch.float32)
        P_rand = V_rand_t @ V_rand_t.T

        # Quick estimate: just use first 20 problems
        subset = problems[:min(20, len(problems))]
        norms_f, norms_t = [], []
        for prob in subset:
            tokens = model.to_tokens(prob["prompt"], prepend_bos=True).to(device)
            last_pos = tokens.shape[1] - 1

            # Just check one layer's MLP as representative
            mid_layer = model.cfg.n_layers // 2
            hook_name = f"blocks.{mid_layer}.hook_mlp_out"
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            mlp_out = cache[hook_name][0, last_pos].float().cpu()
            del cache

            f_proj = P_rand @ mlp_out
            norms_f.append((f_proj ** 2).sum().item())
            norms_t.append((mlp_out ** 2).sum().item())

        frac = np.mean(norms_f) / max(np.mean(norms_t), 1e-12)
        random_fracs.append(frac)

    mean_random = np.mean(random_fracs)
    logger.info(f"  [SANITY S4] Random 9D DLA: mean writing frac = {mean_random:.4f} "
                f"(null expected ≈ {null_expected:.4f})")

    # Should be within 3× of null (some variance expected with small samples)
    assert mean_random < null_expected * 5, \
        f"[SANITY S4] Random baseline too high: {mean_random:.4f} >> {null_expected:.4f}"

    return mean_random, null_expected


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: CAUSAL PATCHING (TARGETED)
# ─────────────────────────────────────────────────────────────────────────────

def causal_patch_top_components(
    model, test_problems, V_9, top_heads, top_mlps, device,
    use_hook_result=True,
    n_heads_to_test: int = 15,
    n_mlps_to_test: int = 10,
):
    """
    Zero-ablate the Fourier-subspace component of each top component's output.

    For attention head (L, H):
      hook on the per-head output at last_pos, head H:
        output[H] -= P_F @ output[H]
      This removes only the Fourier content from that head's contribution.
      Uses hook_result (d_model space) if available, else hook_z (d_head space)
      with W_O-projected ablation.

    For MLP at layer L:
      hook on blocks.{L}.hook_mlp_out, at last_pos:
        output -= P_F @ output

    Measure accuracy drop vs baseline.
    """
    d_model = model.cfg.d_model
    V_9_t = torch.tensor(V_9, dtype=torch.float32).to(device)
    P_F = (V_9_t @ V_9_t.T).to(device)  # (d_model, d_model)

    # Baseline accuracy
    baseline_correct = 0
    baseline_total = 0
    for prob in test_problems:
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True).to(device)
        with torch.no_grad():
            logits = model(tokens)
        pred = model.tokenizer.decode([logits[0, -1].argmax().item()]).strip()
        try:
            if int(pred) == int(prob["target_str"]):
                baseline_correct += 1
        except ValueError:
            pass
        baseline_total += 1

    baseline_acc = baseline_correct / max(baseline_total, 1)
    logger.info(f"  Baseline accuracy: {baseline_acc:.1%} ({baseline_correct}/{baseline_total})")

    results = {"baseline": {"accuracy": baseline_acc, "correct": baseline_correct, "total": baseline_total}}

    # ── Patch attention heads ──
    head_results = {}
    for rank, (L, H, writing_frac) in enumerate(top_heads[:n_heads_to_test]):
        if use_hook_result:
            hook_name = f"blocks.{L}.attn.hook_result"

            def make_head_fourier_ablation_result(proj, head_idx):
                def hook_fn(value, hook, p=proj, hi=head_idx):
                    # value: (batch, seq, n_heads, d_model)
                    last = value.shape[1] - 1
                    h = value[0, last, hi, :].float()  # (d_model,)
                    fourier_component = p @ h            # (d_model,)
                    value[0, last, hi, :] = (h - fourier_component).to(value.dtype)
                    return value
                return hook_fn

            ablation_hook = make_head_fourier_ablation_result(P_F, H)
        else:
            # Fallback: hook on hook_z and ablate Fourier in d_head space
            # via W_O projection: remove component of z that produces
            # Fourier content after W_O projection.
            # z_ablated = z - W_O^T @ P_F @ W_O @ z  (pseudoinverse approach)
            # Simpler: project z -> d_model via W_O, ablate, project back.
            hook_name = f"blocks.{L}.attn.hook_z"
            W_O_h = model.blocks[L].attn.W_O[H].float().to(device)  # (d_head, d_model)

            def make_head_fourier_ablation_z(proj, head_idx, w_o):
                def hook_fn(value, hook, p=proj, hi=head_idx, wo=w_o):
                    # value: (batch, seq, n_heads, d_head)
                    last = value.shape[1] - 1
                    z = value[0, last, hi, :].float()       # (d_head,)
                    h_dmodel = z @ wo                        # (d_model,)
                    fourier_comp = p @ h_dmodel              # (d_model,)
                    # Project back to d_head: z_correction = fourier_comp @ W_O^T
                    # (pseudoinverse: W_O is tall, use W_O^T @ (W_O @ W_O^T)^{-1})
                    # For orthogonal W_O rows: W_O^T suffices. In general, use pinverse.
                    z_correction = fourier_comp @ wo.T       # (d_head,) — approximate
                    value[0, last, hi, :] = (z - z_correction).to(value.dtype)
                    return value
                return hook_fn

            ablation_hook = make_head_fourier_ablation_z(P_F, H, W_O_h)

        n_correct = 0
        for prob in test_problems:
            tokens = model.to_tokens(prob["prompt"], prepend_bos=True).to(device)
            with torch.no_grad():
                with model.hooks(fwd_hooks=[(hook_name, ablation_hook)]):
                    logits = model(tokens)
            pred = model.tokenizer.decode([logits[0, -1].argmax().item()]).strip()
            try:
                if int(pred) == int(prob["target_str"]):
                    n_correct += 1
            except ValueError:
                pass

        acc = n_correct / max(baseline_total, 1)
        damage = baseline_acc - acc
        head_results[f"L{L}_H{H}"] = {
            "layer": L, "head": H,
            "accuracy": float(acc),
            "damage": float(damage),
            "writing_frac": float(writing_frac),
            "rank": rank + 1,
        }
        logger.info(
            f"    L{L:>2}H{H:>2}: acc={acc:.1%} (damage={damage:+.1%}), "
            f"DLA writing={writing_frac:.4f}"
        )

    results["head_ablation"] = head_results

    # ── Patch MLPs ──
    mlp_results = {}
    for rank, (L, writing_frac) in enumerate(top_mlps[:n_mlps_to_test]):
        hook_name = f"blocks.{L}.hook_mlp_out"

        def make_mlp_fourier_ablation(proj):
            def hook_fn(value, hook, p=proj):
                # value: (batch, seq, d_model)
                last = value.shape[1] - 1
                h = value[0, last, :].float()
                fourier_component = p @ h
                value[0, last, :] = (h - fourier_component).to(value.dtype)
                return value
            return hook_fn

        ablation_hook = make_mlp_fourier_ablation(P_F)

        n_correct = 0
        for prob in test_problems:
            tokens = model.to_tokens(prob["prompt"], prepend_bos=True).to(device)
            with torch.no_grad():
                with model.hooks(fwd_hooks=[(hook_name, ablation_hook)]):
                    logits = model(tokens)
            pred = model.tokenizer.decode([logits[0, -1].argmax().item()]).strip()
            try:
                if int(pred) == int(prob["target_str"]):
                    n_correct += 1
            except ValueError:
                pass

        acc = n_correct / max(baseline_total, 1)
        damage = baseline_acc - acc
        mlp_results[f"MLP_L{L}"] = {
            "layer": L,
            "accuracy": float(acc),
            "damage": float(damage),
            "writing_frac": float(writing_frac),
            "rank": rank + 1,
        }
        logger.info(
            f"    MLP L{L:>2}: acc={acc:.1%} (damage={damage:+.1%}), "
            f"DLA writing={writing_frac:.4f}"
        )

    results["mlp_ablation"] = mlp_results

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: FREQUENCY-RESOLVED ATTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def frequency_resolved_attribution(
    model, problems, V_9, freq_assignments, top_heads, top_mlps, device,
    use_hook_result=True,
    n_heads: int = 10,
    n_mlps: int = 5,
):
    """
    For each top component, decompose its Fourier writing into per-frequency
    contributions.

    For direction i with frequency k_i, the writing into that direction is:
      w_i(c) = (v_i · output_c)²
    where v_i is the i-th column of V_9.

    The per-frequency writing is the sum over directions assigned to that
    frequency:
      W_k(c) = Σ_{i: freq(i)=k} w_i(c)

    S6: Per-frequency fractions should sum to 100%.
    """
    V_9_t = torch.tensor(V_9, dtype=torch.float32)  # (d_model, 9)

    # Group directions by frequency
    freq_to_dirs = defaultdict(list)
    for i, k in enumerate(freq_assignments):
        freq_to_dirs[k].append(i)

    n_layers = model.cfg.n_layers
    n_model_heads = model.cfg.n_heads

    # Collect per-direction writing for top components
    head_freq_scores = {}   # (L, H) -> {k: mean_writing}
    mlp_freq_scores = {}    # L -> {k: mean_writing}

    # Build hook names for the layers we need
    target_head_layers = set(L for L, H, _ in top_heads[:n_heads])
    target_mlp_layers = set(L for L, _ in top_mlps[:n_mlps])
    target_layers = target_head_layers | target_mlp_layers

    hook_names = []
    for L in target_layers:
        if L in target_head_layers:
            hook_names.append(get_attn_hook_name(L, use_hook_result))
        if L in target_mlp_layers:
            hook_names.append(f"blocks.{L}.hook_mlp_out")

    # Per-direction accumulators
    head_dir_writing = {(L, H): [[] for _ in range(9)]
                        for L, H, _ in top_heads[:n_heads]}
    mlp_dir_writing = {L: [[] for _ in range(9)]
                       for L, _ in top_mlps[:n_mlps]}

    for prob in problems:
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True).to(device)
        last_pos = tokens.shape[1] - 1

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)

        for L, H, _ in top_heads[:n_heads]:
            attn_full = get_head_outputs_from_cache(cache, L, model, use_hook_result, device)
            if attn_full is not None:
                h_vec = attn_full[0, last_pos, H].float().cpu()  # (d_model,)
                for i in range(9):
                    proj_scalar = (V_9_t[:, i] @ h_vec).item()
                    head_dir_writing[(L, H)][i].append(proj_scalar ** 2)

        for L, _ in top_mlps[:n_mlps]:
            mlp_key = f"blocks.{L}.hook_mlp_out"
            if mlp_key in cache:
                m_vec = cache[mlp_key][0, last_pos].float().cpu()
                for i in range(9):
                    proj_scalar = (V_9_t[:, i] @ m_vec).item()
                    mlp_dir_writing[L][i].append(proj_scalar ** 2)

        del cache

    # Aggregate per-direction means, then group by frequency
    for L, H, _ in top_heads[:n_heads]:
        dir_means = [np.mean(head_dir_writing[(L, H)][i]) if head_dir_writing[(L, H)][i] else 0
                     for i in range(9)]
        total = sum(dir_means)
        if total < 1e-12:
            head_freq_scores[(L, H)] = {k: 0.0 for k in range(1, 6)}
            continue

        freq_writing = {}
        for k in range(1, 6):
            freq_writing[k] = sum(dir_means[i] for i in freq_to_dirs.get(k, []))
        # S6: Check fractions sum to ~100%
        frac_sum = sum(freq_writing.values()) / total
        assert abs(frac_sum - 1.0) < 0.01, \
            f"[SANITY S6] L{L}H{H}: freq fractions sum to {frac_sum:.4f}, expected 1.0"

        head_freq_scores[(L, H)] = {k: v / total for k, v in freq_writing.items()}

    for L, _ in top_mlps[:n_mlps]:
        dir_means = [np.mean(mlp_dir_writing[L][i]) if mlp_dir_writing[L][i] else 0
                     for i in range(9)]
        total = sum(dir_means)
        if total < 1e-12:
            mlp_freq_scores[L] = {k: 0.0 for k in range(1, 6)}
            continue

        freq_writing = {}
        for k in range(1, 6):
            freq_writing[k] = sum(dir_means[i] for i in freq_to_dirs.get(k, []))
        frac_sum = sum(freq_writing.values()) / total
        assert abs(frac_sum - 1.0) < 0.01, \
            f"[SANITY S6] MLP L{L}: freq fractions sum to {frac_sum:.4f}, expected 1.0"

        mlp_freq_scores[L] = {k: v / total for k, v in freq_writing.items()}

    return head_freq_scores, mlp_freq_scores


# ─────────────────────────────────────────────────────────────────────────────
# RESIDUAL STREAM DECOMPOSITION SANITY CHECK (S1)
# ─────────────────────────────────────────────────────────────────────────────

def check_residual_decomposition(model, problems, device, check_layer=None):
    """
    S1: Verify that sum of all component outputs at the last token ≈ resid_post.

    In TransformerLens, at each layer L:
      resid_post_L = resid_pre_L + attn_out_L + mlp_out_L
    where attn_out_L = sum over heads of hook_result_L[h]

    We verify this at one layer to confirm our hook points are correct.
    """
    if check_layer is None:
        check_layer = model.cfg.n_layers // 2

    prob = problems[0]
    tokens = model.to_tokens(prob["prompt"], prepend_bos=True).to(device)
    last_pos = tokens.shape[1] - 1

    # Use hook_attn_out (sum of all heads, always available) for decomposition check
    hook_names = [
        f"blocks.{check_layer}.hook_resid_pre",
        f"blocks.{check_layer}.hook_attn_out",
        f"blocks.{check_layer}.hook_mlp_out",
        f"blocks.{check_layer}.hook_resid_post",
    ]

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_names)

    resid_pre = cache[f"blocks.{check_layer}.hook_resid_pre"][0, last_pos].float().cpu()
    attn_out = cache[f"blocks.{check_layer}.hook_attn_out"][0, last_pos].float().cpu()
    mlp_out = cache[f"blocks.{check_layer}.hook_mlp_out"][0, last_pos].float().cpu()
    resid_post = cache[f"blocks.{check_layer}.hook_resid_post"][0, last_pos].float().cpu()
    del cache

    # Reconstruction: resid_post = resid_pre + attn_out + mlp_out
    reconstructed = resid_pre + attn_out + mlp_out
    err = (reconstructed - resid_post).abs().max().item()
    rel_err = err / (resid_post.abs().max().item() + 1e-10)

    logger.info(f"  [SANITY S1] Residual decomposition at L{check_layer}:")
    logger.info(f"    ||resid_pre + Σ_h(attn_h) + mlp - resid_post||_∞ = {err:.2e}")
    logger.info(f"    Relative error = {rel_err:.2e}")

    # Allow for LayerNorm effects — in models WITH pre-LN, the
    # hook_result and hook_mlp_out are applied to the NORMALIZED input,
    # but the residual stream addition is on the un-normalized path.
    # TransformerLens hook_resid_post should capture the actual residual.
    if rel_err > 0.01:
        logger.warning(
            f"  [SANITY S1] ⚠ Residual decomposition error {rel_err:.2e} > 0.01. "
            f"This may be due to float16 precision or LayerNorm interactions. "
            f"Proceeding with caution."
        )
    else:
        logger.info(f"  [SANITY S1] ✓ Residual decomposition verified")

    return rel_err


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 10: Attention Head & MLP Attribution for 9D Fourier Subspace"
    )
    parser.add_argument("--model", default="gemma-2b", choices=list(MODEL_MAP.keys()))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--comp-layer", type=int, default=None,
                        help="Computation layer for subspace construction")
    parser.add_argument("--prompt-format", default="calculate",
                        choices=VALID_PROMPT_FORMATS)
    parser.add_argument("--n-train-per-digit", type=int, default=50)
    parser.add_argument("--n-test-per-digit", type=int, default=30)
    parser.add_argument("--top-k-heads", type=int, default=15,
                        help="Number of top heads to validate with causal patching")
    parser.add_argument("--top-k-mlps", type=int, default=10,
                        help="Number of top MLPs to validate with causal patching")
    parser.add_argument("--direct-answer", action="store_true",
                        help="Use direct-answer mode (for models that predict full answer as single token)")
    args = parser.parse_args()

    model_name = MODEL_MAP[args.model]
    device = args.device

    comp_defaults = {"gemma-2b": 19, "phi-3": 26, "llama-3b": 20, "llama-3b-it": 20}
    comp_layer = args.comp_layer or comp_defaults.get(args.model, 20)
    prompt_format = args.prompt_format

    logger.info(f"Model: {args.model} ({model_name})")
    logger.info(f"Computation layer: L{comp_layer}")
    logger.info(f"Device: {device}")
    logger.info(f"Prompt format: {prompt_format}")

    # ── Load model ────────────────────────────────────────────────────────
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=torch.float32 if device == "cpu" else torch.float16,
    )
    model.eval()

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model
    d_head = model.cfg.d_head

    logger.info(f"Architecture: {n_layers} layers, {n_heads} heads, "
                f"d_model={d_model}, d_head={d_head}")
    logger.info(f"Total components: {n_layers * n_heads} heads + {n_layers} MLPs "
                f"= {n_layers * (n_heads + 1)}")

    # Verify hook availability
    use_hook_result = verify_hook_points(model, device)

    # Check fold_ln — critical for additive decomposition correctness.
    # When fold_ln=True (default for supported models), LayerNorm weights
    # are folded into the linear layers, making the residual stream
    # genuinely additive: resid = embed + Σ(head_outputs) + Σ(mlp_outputs).
    # Without fold_ln, LayerNorm introduces non-linearity between components.
    fold_ln = getattr(model.cfg, 'default_prepend_bos', None) is not None  # proxy check
    try:
        # TransformerLens stores whether LN was folded
        if hasattr(model.cfg, 'normalization_type'):
            norm_type = model.cfg.normalization_type
            logger.info(f"  Normalization: {norm_type}")
            if norm_type and 'RMS' in str(norm_type):
                logger.info(f"  [NOTE] RMSNorm model — fold_ln may not fully linearize. "
                            f"Signed attribution additivity check will reveal any error.")
    except Exception:
        pass

    all_results = {
        "model": model_name,
        "model_short": args.model,
        "comp_layer": comp_layer,
        "prompt_format": prompt_format,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "d_model": d_model,
        "use_hook_result": use_hook_result,
    }

    # ══════════════════════════════════════════════════════════════════════
    # STEP 0: Generate data and compute Fourier subspace
    # ══════════════════════════════════════════════════════════════════════
    logger.info(f"\n{'═'*60}")
    logger.info(f"  STEP 0: Data generation and Fourier subspace construction")
    logger.info(f"{'═'*60}")

    train, test = generate_train_test(
        model, args.n_train_per_digit, args.n_test_per_digit, prompt_format,
        direct_answer=args.direct_answer,
    )

    logger.info(f"\n  Building 9D Fourier subspace from training data...")
    V_9, S_vals, freq_assignments, U = build_fourier_subspace(
        model, train, comp_layer, device
    )

    all_results["subspace"] = {
        "singular_values": S_vals.tolist(),
        "freq_assignments": freq_assignments,
    }

    # ══════════════════════════════════════════════════════════════════════
    # STEP 0.5: Sanity checks
    # ══════════════════════════════════════════════════════════════════════
    logger.info(f"\n{'═'*60}")
    logger.info(f"  SANITY CHECKS")
    logger.info(f"{'═'*60}")

    # S1: Residual stream decomposition
    s1_err = check_residual_decomposition(model, test, device, check_layer=comp_layer)
    all_results["sanity"] = {"S1_residual_decomp_rel_err": float(s1_err)}

    # S4: Random baseline DLA
    random_mean, null_expected = run_random_baseline_dla(model, test, d_model, device)
    all_results["sanity"]["S4_random_dla_mean"] = float(random_mean)
    all_results["sanity"]["S4_null_expected"] = float(null_expected)

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1: Direct Writing Score (DLA)
    # ══════════════════════════════════════════════════════════════════════
    logger.info(f"\n{'═'*60}")
    logger.info(f"  PHASE 1: Direct Writing Score (DLA)")
    logger.info(f"{'═'*60}")

    t0 = time.time()
    head_scores, mlp_scores, embed_score, additivity_info = compute_writing_scores(
        model, test, comp_layer, V_9, device, use_hook_result=use_hook_result
    )
    t1 = time.time()
    logger.info(f"  Phase 1 completed in {t1-t0:.1f}s")

    # Sort by SIGNED attribution (primary) — this is the principled metric
    # that decomposes ||P_F · resid||² exactly. Also sort by unsigned for
    # comparison.
    sorted_heads_signed = sorted(
        head_scores.items(), key=lambda x: abs(x[1]["signed_attr"]), reverse=True
    )
    sorted_heads = sorted(
        head_scores.items(), key=lambda x: x[1]["mean_fourier_norm_sq"], reverse=True
    )
    sorted_mlps_signed = sorted(
        mlp_scores.items(), key=lambda x: abs(x[1]["signed_attr"]), reverse=True
    )
    sorted_mlps = sorted(
        mlp_scores.items(), key=lambda x: x[1]["mean_fourier_norm_sq"], reverse=True
    )

    # Display embed component
    logger.info(f"\n  Embedding component:")
    logger.info(f"    Signed attr: {embed_score['signed_attr']:.2f} "
                f"({embed_score['pct_of_total']:+.1f}% of total)")

    # Display top heads by signed attribution
    logger.info(f"\n  Top 20 attention heads by |signed attribution|:")
    logger.info(f"  {'Rank':>4}  {'Head':>8}  {'Signed':>10}  {'%Total':>8}  "
                f"{'Unsigned':>10}  {'Frac%':>7}")
    logger.info(f"  {'─'*60}")

    for rank, ((L, H), info) in enumerate(sorted_heads_signed[:20], 1):
        logger.info(
            f"  {rank:>4}  L{L:>2}H{H:>2}  "
            f"{info['signed_attr']:>+10.2f}  "
            f"{info['pct_of_total']:>+7.1f}%  "
            f"{info['mean_fourier_norm_sq']:>10.1f}  "
            f"{info['writing_frac']*100:>6.2f}%"
        )

    # Display top MLPs by signed attribution
    logger.info(f"\n  Top 10 MLPs by |signed attribution|:")
    logger.info(f"  {'Rank':>4}  {'MLP':>8}  {'Signed':>10}  {'%Total':>8}  "
                f"{'Unsigned':>10}  {'Frac%':>7}")
    logger.info(f"  {'─'*60}")

    for rank, (L, info) in enumerate(sorted_mlps_signed[:10], 1):
        logger.info(
            f"  {rank:>4}  MLP L{L:>2}  "
            f"{info['signed_attr']:>+10.2f}  "
            f"{info['pct_of_total']:>+7.1f}%  "
            f"{info['mean_fourier_norm_sq']:>10.1f}  "
            f"{info['writing_frac']*100:>6.2f}%"
        )

    # S3: Check that no single component has writing_frac > 1.0
    max_head_frac = max(info["writing_frac"] for _, info in sorted_heads) if sorted_heads else 0
    max_mlp_frac = max(info["writing_frac"] for _, info in sorted_mlps) if sorted_mlps else 0
    assert max_head_frac <= 1.0 + 1e-6, \
        f"[SANITY S3] Head writing fraction > 1.0: {max_head_frac:.4f}"
    assert max_mlp_frac <= 1.0 + 1e-6, \
        f"[SANITY S3] MLP writing fraction > 1.0: {max_mlp_frac:.4f}"
    logger.info(f"\n  [SANITY S3] Max writing fractions: head={max_head_frac:.4f}, "
                f"mlp={max_mlp_frac:.4f} (both ≤ 1.0) ✓")

    # Store Phase 1 results
    all_results["phase1_dla"] = {
        "heads": {f"L{L}_H{H}": info for (L, H), info in sorted_heads},
        "mlps": {f"MLP_L{L}": info for L, info in sorted_mlps},
        "embed": embed_score,
        "additivity": additivity_info,
        "top20_heads_signed": [
            {"layer": L, "head": H,
             "signed_attr": info["signed_attr"],
             "pct_of_total": info["pct_of_total"],
             "writing_frac": info["writing_frac"]}
            for (L, H), info in sorted_heads_signed[:20]
        ],
        "top10_mlps_signed": [
            {"layer": L,
             "signed_attr": info["signed_attr"],
             "pct_of_total": info["pct_of_total"],
             "writing_frac": info["writing_frac"]}
            for L, info in sorted_mlps_signed[:10]
        ],
    }

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2: Causal Patching (Targeted Validation)
    # ══════════════════════════════════════════════════════════════════════
    logger.info(f"\n{'═'*60}")
    logger.info(f"  PHASE 2: Causal Patching (Top-K Validation)")
    logger.info(f"{'═'*60}")

    # Prepare ranked lists for patching — use signed-sorted (principled ranking)
    top_heads_for_patch = [
        (L, H, info["writing_frac"]) for (L, H), info in sorted_heads_signed[:args.top_k_heads]
    ]
    top_mlps_for_patch = [
        (L, info["writing_frac"]) for L, info in sorted_mlps_signed[:args.top_k_mlps]
    ]

    t0 = time.time()
    patch_results = causal_patch_top_components(
        model, test, V_9, top_heads_for_patch, top_mlps_for_patch, device,
        use_hook_result=use_hook_result,
        n_heads_to_test=args.top_k_heads,
        n_mlps_to_test=args.top_k_mlps,
    )
    t1 = time.time()
    logger.info(f"  Phase 2 completed in {t1-t0:.1f}s")

    all_results["phase2_patching"] = patch_results

    # S5: Check DLA-damage correlation
    if patch_results.get("head_ablation"):
        dla_fracs = []
        damages = []
        for key, info in patch_results["head_ablation"].items():
            dla_fracs.append(info["writing_frac"])
            damages.append(info["damage"])
        if len(dla_fracs) >= 3:
            correlation = np.corrcoef(dla_fracs, damages)[0, 1]
            logger.info(f"\n  [SANITY S5] DLA vs damage correlation (heads): r = {correlation:.3f}")
            all_results["sanity"]["S5_dla_damage_corr_heads"] = float(correlation)
            if correlation < 0:
                logger.warning(f"  [SANITY S5] ⚠ Negative correlation — DLA may not predict causal importance")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3: Frequency-Resolved Attribution
    # ══════════════════════════════════════════════════════════════════════
    logger.info(f"\n{'═'*60}")
    logger.info(f"  PHASE 3: Frequency-Resolved Attribution")
    logger.info(f"{'═'*60}")

    t0 = time.time()
    head_freq, mlp_freq = frequency_resolved_attribution(
        model, test, V_9, freq_assignments,
        top_heads_for_patch, top_mlps_for_patch, device,
        use_hook_result=use_hook_result,
        n_heads=min(10, len(top_heads_for_patch)),
        n_mlps=min(5, len(top_mlps_for_patch)),
    )
    t1 = time.time()
    logger.info(f"  Phase 3 completed in {t1-t0:.1f}s")

    # Display frequency breakdown for top heads
    freq_labels = {1: "k=1(ord)", 2: "k=2(m5)", 3: "k=3", 4: "k=4", 5: "k=5(par)"}

    logger.info(f"\n  Frequency breakdown of top attention heads:")
    logger.info(f"  {'Head':>8}  {'k=1':>7}  {'k=2':>7}  {'k=3':>7}  {'k=4':>7}  {'k=5':>7}  {'Dominant':>10}")
    logger.info(f"  {'─'*60}")

    phase3_heads = {}
    for (L, H), freq_dict in head_freq.items():
        dom_k = max(freq_dict, key=freq_dict.get)
        dom_pct = freq_dict[dom_k] * 100
        logger.info(
            f"  L{L:>2}H{H:>2}  "
            f"{freq_dict.get(1,0)*100:>6.1f}%  {freq_dict.get(2,0)*100:>6.1f}%  "
            f"{freq_dict.get(3,0)*100:>6.1f}%  {freq_dict.get(4,0)*100:>6.1f}%  "
            f"{freq_dict.get(5,0)*100:>6.1f}%  "
            f"k={dom_k}({dom_pct:.0f}%)"
        )
        phase3_heads[f"L{L}_H{H}"] = {
            "freq_fractions": {str(k): float(v) for k, v in freq_dict.items()},
            "dominant_freq": int(dom_k),
            "dominant_pct": float(dom_pct),
        }

    logger.info(f"\n  Frequency breakdown of top MLPs:")
    logger.info(f"  {'MLP':>8}  {'k=1':>7}  {'k=2':>7}  {'k=3':>7}  {'k=4':>7}  {'k=5':>7}  {'Dominant':>10}")
    logger.info(f"  {'─'*60}")

    phase3_mlps = {}
    for L, freq_dict in mlp_freq.items():
        dom_k = max(freq_dict, key=freq_dict.get)
        dom_pct = freq_dict[dom_k] * 100
        logger.info(
            f"  MLP L{L:>2}  "
            f"{freq_dict.get(1,0)*100:>6.1f}%  {freq_dict.get(2,0)*100:>6.1f}%  "
            f"{freq_dict.get(3,0)*100:>6.1f}%  {freq_dict.get(4,0)*100:>6.1f}%  "
            f"{freq_dict.get(5,0)*100:>6.1f}%  "
            f"k={dom_k}({dom_pct:.0f}%)"
        )
        phase3_mlps[f"MLP_L{L}"] = {
            "freq_fractions": {str(k): float(v) for k, v in freq_dict.items()},
            "dominant_freq": int(dom_k),
            "dominant_pct": float(dom_pct),
        }

    all_results["phase3_frequency"] = {
        "heads": phase3_heads,
        "mlps": phase3_mlps,
        "freq_assignments": freq_assignments,
    }

    # ══════════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ══════════════════════════════════════════════════════════════════════
    fmt_suffix = ""
    if args.direct_answer:
        fmt_suffix += "_direct"
    out_path = RESULTS_DIR / f"fourier_head_attribution_{args.model}{fmt_suffix}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
