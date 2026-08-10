#!/usr/bin/env python3
"""
Fourier Decomposition of Digit Subspace

Tests whether LLM arithmetic representations use Fourier features
(sin/cos at discrete frequencies mod 10).

Theory:
  For 10 digit classes, the DFT provides 9 non-trivial basis functions:
  - k=1,...,4: cos(2πkd/10) and sin(2πkd/10) — 8 directions
  - k=5: cos(πd) = (-1)^d — 1 direction (Nyquist)
  Total: 9 = 10 classes - 1

Tests:
  1. Energy spectrum: fraction of between-class variance at each frequency
  2. Per-neuron Fourier purity: do individual neurons encode specific frequencies?
  3. Frequency-resolved patching: does patching in Fourier-k subspace recover digits?
  4. Comparison: Fourier patching vs contrastive Fisher vs unembed patching

Usage:
    python fourier_decomposition.py --model llama-3b --layer 27 --device mps
    python fourier_decomposition.py --model gemma-2b --layer 21 --device mps
    python fourier_decomposition.py --model phi-3 --layer 28 --device mps
"""

import argparse
import json
import logging
import sys
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import shared utilities
from arithmetic_circuit_scan_updated import (
    MODEL_MAP,
    generate_teacher_forced_problems,
    generate_single_digit_problems,
    generate_direct_answer_problems,
    filter_correct_teacher_forced,
    filter_correct_direct_answer,
    get_context_target_tok,
    get_digit_token_ids,
    compute_unembed_basis,
    run_patching_experiment,
    check_orthonormal,
    log_section,
)


# ─────────────────────────────────────────────────────────────
# FOURIER BASIS CONSTRUCTION
# ─────────────────────────────────────────────────────────────

def build_fourier_basis_functions(N: int = 10) -> Tuple[np.ndarray, List[str]]:
    """
    Build the orthonormal Fourier basis functions for Z_N (digits 0..N-1).

    Returns:
        phi: (N, 9) matrix where phi[d, j] is the j-th basis function at digit d
        labels: list of human-readable labels for each basis function

    For N=10, the centered (non-DC) Fourier basis has 9 functions:
        cos(2π·1·d/10), sin(2π·1·d/10),  (frequency 1)
        cos(2π·2·d/10), sin(2π·2·d/10),  (frequency 2)
        cos(2π·3·d/10), sin(2π·3·d/10),  (frequency 3)
        cos(2π·4·d/10), sin(2π·4·d/10),  (frequency 4)
        cos(π·d) = (-1)^d                (frequency 5, Nyquist)

    Orthogonality relations (for k,l in {1,...,4}):
        Σ_d cos(2πkd/N)·cos(2πld/N) = (N/2)·δ_{kl}
        Σ_d sin(2πkd/N)·sin(2πld/N) = (N/2)·δ_{kl}
        Σ_d cos(2πkd/N)·sin(2πld/N) = 0
        Σ_d (-1)^d · (-1)^d = N
    """
    assert N == 10, "Currently only supports N=10 digit classes"
    d = np.arange(N)
    phi = np.zeros((N, 9))
    labels = []

    col = 0
    for k in range(1, 5):
        # Cosine at frequency k, normalized: sqrt(2/N) so that Σ_d phi^2 = 1
        phi[:, col] = np.sqrt(2.0 / N) * np.cos(2 * np.pi * k * d / N)
        labels.append(f"cos(2π·{k}d/10)")
        col += 1
        # Sine at frequency k
        phi[:, col] = np.sqrt(2.0 / N) * np.sin(2 * np.pi * k * d / N)
        labels.append(f"sin(2π·{k}d/10)")
        col += 1

    # Nyquist frequency k=5: cos(πd) = (-1)^d, sin(πd) = 0
    phi[:, col] = np.sqrt(1.0 / N) * np.cos(np.pi * d)
    labels.append("cos(πd)=(-1)^d")

    assert col == 8, f"Expected 8 as last column index, got {col}"
    assert len(labels) == 9

    # ── Sanity: verify orthonormality of phi ──
    G = phi.T @ phi  # should be I_9
    max_err = np.abs(G - np.eye(9)).max()
    assert max_err < 1e-10, (
        f"[SANITY] Fourier basis functions not orthonormal: max_err={max_err:.2e}"
    )
    logger.info(f"  [SANITY] Fourier basis functions orthonormal ✓ (max_err={max_err:.2e})")

    return phi, labels


def compute_per_digit_mean_activations(
    model,
    correct: List[dict],
    layer: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    """
    Compute per-digit mean activations at a given layer.

    Returns:
        mu: (10, d_model) — per-digit mean activations
        mu_centered: (10, d_model) — centered (grand mean subtracted)
        counts: dict mapping digit → number of problems
    """
    device = next(model.parameters()).device
    hook_name = f"blocks.{layer}.hook_resid_post"
    d_model = model.cfg.d_model

    # Group problems by digit
    by_digit = defaultdict(list)
    for prob in correct:
        by_digit[prob["ones_digit"]].append(prob)

    mu = np.zeros((10, d_model), dtype=np.float64)
    counts = {}

    for digit in range(10):
        probs = by_digit[digit]
        if len(probs) == 0:
            logger.warning(f"  No problems for digit {digit}!")
            counts[digit] = 0
            continue

        acts = []
        for prob in probs:
            tokens = prob["_tokens"]
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=hook_name)
                act = cache[hook_name][0, -1].detach().cpu().float().numpy()
                acts.append(act)
                del cache

        mu[digit] = np.mean(acts, axis=0)
        counts[digit] = len(acts)
        logger.info(f"  Digit {digit}: {len(acts)} problems, "
                     f"||μ||={np.linalg.norm(mu[digit]):.2f}")

    # Center: subtract grand mean
    grand_mean = mu.mean(axis=0)
    mu_centered = mu - grand_mean

    # Sanity: centered means sum to zero
    sum_norm = np.linalg.norm(mu_centered.sum(axis=0))
    assert sum_norm < 1e-6, (
        f"[SANITY] Centered means don't sum to zero: ||sum||={sum_norm:.2e}"
    )
    logger.info(f"  [SANITY] Centered means sum to zero ✓ (||sum||={sum_norm:.2e})")

    return mu, mu_centered, counts


def fourier_decomposition(
    mu_centered: np.ndarray,    # (10, d_model)
    phi: np.ndarray,            # (10, 9) — Fourier basis functions
    labels: List[str],
) -> Dict:
    """
    Decompose per-digit mean activations into Fourier components.

    For each basis function φ_j, the Fourier coefficient vector is:
        v_j = Σ_d φ_j(d) · μ̃_d  ∈ R^{d_model}

    The energy at frequency k (with cos and sin components):
        E_k = ||v_{cos,k}||² + ||v_{sin,k}||²

    Parseval identity:
        Σ_d ||μ̃_d||² = Σ_j ||v_j||²

    Returns dict with energy spectrum, coefficient vectors, and diagnostics.
    """
    N, d_model = mu_centered.shape
    assert N == 10
    n_basis = phi.shape[1]
    assert n_basis == 9

    # Compute Fourier coefficient vectors: V = phi^T @ mu_centered
    # V[j] = Σ_d phi[d,j] * mu_centered[d] → (9, d_model)
    V = phi.T @ mu_centered   # (9, d_model)

    # Energy per basis function
    energies = np.array([np.dot(V[j], V[j]) for j in range(9)])

    # Group into frequencies
    freq_energy = {}
    freq_labels = {}
    for k in range(1, 5):
        cos_idx = 2 * (k - 1)
        sin_idx = 2 * (k - 1) + 1
        freq_energy[k] = energies[cos_idx] + energies[sin_idx]
        freq_labels[k] = f"freq {k}"
    freq_energy[5] = energies[8]  # Nyquist
    freq_labels[5] = "freq 5 (Nyquist)"

    total_energy = sum(freq_energy.values())

    # ── Parseval sanity check ──
    lhs = np.sum([np.dot(mu_centered[d], mu_centered[d]) for d in range(10)])
    rhs = energies.sum()
    parseval_err = abs(lhs - rhs) / max(lhs, 1e-30)
    assert parseval_err < 1e-6, (
        f"[SANITY] Parseval identity FAILED: LHS={lhs:.6f}, RHS={rhs:.6f}, "
        f"rel_err={parseval_err:.2e}"
    )
    logger.info(f"  [SANITY] Parseval identity ✓ (rel_err={parseval_err:.2e})")

    # ── Reconstruction sanity check ──
    # mu_centered_reconstructed = phi @ V = phi @ (phi^T @ mu_centered)
    # Since phi has orthonormal columns spanning a 9D subspace of R^10,
    # and the centered means lie in a 9D subspace (they sum to 0),
    # the reconstruction should be exact.
    mu_recon = phi @ V  # (10, d_model)
    recon_err = np.linalg.norm(mu_centered - mu_recon) / np.linalg.norm(mu_centered)
    assert recon_err < 1e-6, (
        f"[SANITY] Fourier reconstruction FAILED: rel_err={recon_err:.2e}"
    )
    logger.info(f"  [SANITY] Fourier reconstruction exact ✓ (rel_err={recon_err:.2e})")

    # ── Log energy spectrum ──
    logger.info(f"\n  Fourier Energy Spectrum:")
    logger.info(f"  {'Frequency':>12}  {'Energy':>12}  {'Fraction':>10}  {'Bar'}")
    logger.info(f"  {'─'*55}")
    for k in range(1, 6):
        frac = freq_energy[k] / total_energy if total_energy > 0 else 0
        bar = '█' * int(frac * 40)
        logger.info(f"  {'k=' + str(k):>12}  {freq_energy[k]:>12.4f}  "
                     f"{frac:>9.1%}  {bar}")
    logger.info(f"  {'─'*55}")
    logger.info(f"  {'Total':>12}  {total_energy:>12.4f}  {'100.0%':>10}")

    # Concentration metric: fraction in top-2 frequencies
    sorted_freqs = sorted(freq_energy.items(), key=lambda x: x[1], reverse=True)
    top2_energy = sorted_freqs[0][1] + sorted_freqs[1][1]
    top2_frac = top2_energy / total_energy if total_energy > 0 else 0
    logger.info(f"\n  Top-2 frequencies: k={sorted_freqs[0][0]} and k={sorted_freqs[1][0]} "
                f"capture {top2_frac:.1%} of variance")

    # Interpretation
    if top2_frac > 0.7:
        interp = "STRONGLY FOURIER — 2 frequencies dominate"
    elif top2_frac > 0.5:
        interp = "MODERATELY FOURIER — some frequency concentration"
    else:
        interp = "NOT FOURIER — energy spread across frequencies (lookup-like)"
    logger.info(f"  Interpretation: {interp}")

    return {
        "fourier_coeff_vectors": V,            # (9, d_model)
        "energies_per_basis": energies,         # (9,)
        "freq_energy": freq_energy,             # {k: energy}
        "total_energy": total_energy,
        "freq_fractions": {k: freq_energy[k]/total_energy for k in range(1,6)},
        "top2_frac": top2_frac,
        "top2_freqs": (sorted_freqs[0][0], sorted_freqs[1][0]),
        "interpretation": interp,
        "parseval_rel_err": parseval_err,
        "recon_rel_err": recon_err,
    }


def per_neuron_fourier_analysis(
    mu_centered: np.ndarray,   # (10, d_model)
    phi: np.ndarray,           # (10, 9)
    top_k: int = 20,
) -> Dict:
    """
    For each activation dimension, compute Fourier purity.

    Fourier purity = max_k(E_j(k)) / E_j_total
    where E_j(k) = energy at frequency k for neuron j.

    Purity = 1.0 means all energy in one frequency (perfectly tuned).
    Purity ~ 0.2 means energy spread uniformly (5 frequencies → 1/5 = 0.2 baseline).
    """
    N, d_model = mu_centered.shape

    # Per-neuron Fourier coefficients
    # For each neuron j: x_j = [μ̃_0[j], ..., μ̃_9[j]] is a 10-element vector
    # Its Fourier coefficients: c_j = phi^T @ x_j → (9,) vector
    # c_j[i]^2 is the energy in basis function i for neuron j

    C = phi.T @ mu_centered  # (9, d_model) — same as V above

    # Energy per frequency per neuron
    freq_energy_per_neuron = np.zeros((5, d_model))  # freq 1..5
    for k in range(1, 5):
        cos_idx = 2 * (k - 1)
        sin_idx = 2 * (k - 1) + 1
        freq_energy_per_neuron[k-1] = C[cos_idx]**2 + C[sin_idx]**2
    freq_energy_per_neuron[4] = C[8]**2  # Nyquist

    total_per_neuron = freq_energy_per_neuron.sum(axis=0)  # (d_model,)

    # Purity: max frequency energy / total energy per neuron
    # Only compute for neurons with non-negligible total energy
    active_mask = total_per_neuron > 1e-10 * total_per_neuron.max()
    n_active = active_mask.sum()

    purity = np.zeros(d_model)
    peak_freq = np.zeros(d_model, dtype=int)
    for j in range(d_model):
        if active_mask[j]:
            purity[j] = freq_energy_per_neuron[:, j].max() / total_per_neuron[j]
            peak_freq[j] = freq_energy_per_neuron[:, j].argmax() + 1  # freq 1-indexed

    # Statistics
    active_purity = purity[active_mask]
    mean_purity = active_purity.mean()
    high_purity = (active_purity > 0.8).sum()
    med_purity = ((active_purity > 0.5) & (active_purity <= 0.8)).sum()
    low_purity = (active_purity <= 0.5).sum()

    logger.info(f"\n  Per-Neuron Fourier Purity ({n_active} active neurons):")
    logger.info(f"  Mean purity: {mean_purity:.3f}  (chance=0.200 for 5 freqs)")
    logger.info(f"  Single-freq (>0.8): {high_purity} ({100*high_purity/n_active:.1f}%)")
    logger.info(f"  Moderate (0.5-0.8): {med_purity} ({100*med_purity/n_active:.1f}%)")
    logger.info(f"  Distributed (<0.5): {low_purity} ({100*low_purity/n_active:.1f}%)")

    # Peak frequency distribution among high-purity neurons
    if high_purity > 0:
        high_mask = active_mask & (purity > 0.8)
        freq_hist = np.bincount(peak_freq[high_mask], minlength=6)[1:]  # freq 1-5
        logger.info(f"  High-purity neurons by peak frequency:")
        for k in range(1, 6):
            logger.info(f"    k={k}: {freq_hist[k-1]} neurons")

    # Top neurons by total energy
    top_idx = np.argsort(total_per_neuron)[::-1][:top_k]
    logger.info(f"\n  Top-{top_k} neurons by digit-discriminative energy:")
    logger.info(f"  {'Neuron':>8}  {'Energy':>10}  {'Purity':>8}  {'Peak freq':>10}")
    for j in top_idx:
        logger.info(f"  {j:>8}  {total_per_neuron[j]:>10.4f}  "
                     f"{purity[j]:>8.3f}  {'k='+str(peak_freq[j]):>10}")

    return {
        "mean_purity": float(mean_purity),
        "n_active": int(n_active),
        "high_purity_count": int(high_purity),
        "med_purity_count": int(med_purity),
        "low_purity_count": int(low_purity),
        "high_purity_frac": float(high_purity / n_active),
    }


def build_fourier_patching_basis(
    V: np.ndarray,              # (9, d_model) Fourier coefficient vectors
    freq_energy: Dict[int, float],
) -> Tuple[np.ndarray, List[Tuple[int, str]]]:
    """
    Build orthonormal basis for patching, ordered by frequency energy (highest first).

    Returns:
        basis: (d_model, 9) orthonormal matrix — columns are basis vectors
        ordering: list of (freq, label) for each column

    Strategy: sort frequencies by energy, orthonormalize within each frequency,
    then concatenate. This gives us a basis where the first 2 columns are the
    most energetic frequency's cos/sin, etc.
    """
    d_model = V.shape[1]

    # Sort frequencies by energy (descending)
    sorted_freqs = sorted(freq_energy.items(), key=lambda x: x[1], reverse=True)

    basis_cols = []
    ordering = []

    for freq_k, energy in sorted_freqs:
        if freq_k <= 4:
            cos_idx = 2 * (freq_k - 1)
            sin_idx = 2 * (freq_k - 1) + 1
            vecs = [V[cos_idx], V[sin_idx]]
            freq_labels = [f"cos(k={freq_k})", f"sin(k={freq_k})"]
        else:  # k=5 Nyquist
            vecs = [V[8]]
            freq_labels = [f"(-1)^d (k=5)"]

        # Orthonormalize these vectors against ALL previous basis columns
        for vec, lab in zip(vecs, freq_labels):
            v = vec.copy()
            # Project out all existing basis columns
            for prev in basis_cols:
                v -= np.dot(v, prev) * prev
            norm = np.linalg.norm(v)
            if norm < 1e-8:
                logger.warning(f"  Fourier direction {lab} has near-zero residual "
                               f"(norm={norm:.2e}) — skipping")
                continue
            v /= norm
            basis_cols.append(v)
            ordering.append((freq_k, lab))

    basis = np.column_stack(basis_cols)  # (d_model, n_cols)

    # Sanity: verify orthonormality
    G = basis.T @ basis
    max_err = np.abs(G - np.eye(basis.shape[1])).max()
    assert max_err < 1e-6, (
        f"[SANITY] Fourier patching basis not orthonormal: max_err={max_err:.2e}"
    )
    logger.info(f"  [SANITY] Fourier patching basis orthonormal ✓ "
                f"({basis.shape[1]} directions, max_err={max_err:.2e})")

    logger.info(f"  Fourier basis ordering (by energy):")
    for i, (freq_k, lab) in enumerate(ordering):
        logger.info(f"    Col {i}: {lab} (freq={freq_k})")

    return basis, ordering


def build_frequency_resolved_bases(
    V: np.ndarray,  # (9, d_model)
) -> Dict[int, np.ndarray]:
    """
    Build separate orthonormal basis for each frequency.

    Returns dict mapping freq_k → (d_model, n_dims) basis matrix.
    freq 1-4 each give 2D basis, freq 5 gives 1D basis.
    """
    bases = {}
    for k in range(1, 5):
        cos_idx = 2 * (k - 1)
        sin_idx = 2 * (k - 1) + 1
        vecs = np.column_stack([V[cos_idx], V[sin_idx]])  # (d_model, 2)
        # QR orthonormalization
        Q, R = np.linalg.qr(vecs)
        # Check both columns are non-degenerate
        if abs(R[1, 1]) < 1e-8:
            logger.warning(f"  Frequency k={k}: sin component degenerate")
            Q = Q[:, :1]
        bases[k] = Q

    # Nyquist
    v5 = V[8].copy()
    norm = np.linalg.norm(v5)
    if norm > 1e-8:
        bases[5] = (v5 / norm).reshape(-1, 1)
    else:
        logger.warning(f"  Frequency k=5: near-zero energy")

    return bases


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def balance_digit_counts(
    correct: List[dict],
    min_per_digit: int = 10,
) -> List[dict]:
    """
    Subsample problems so each digit has exactly min_per_digit observations.
    If any digit has fewer than min_per_digit, use whatever it has but warn.
    Returns balanced list.
    """
    import random
    by_digit = defaultdict(list)
    for p in correct:
        by_digit[p["ones_digit"]].append(p)

    actual_min = min(len(by_digit[d]) for d in range(10))
    target = min(min_per_digit, actual_min)

    if actual_min < min_per_digit:
        logger.warning(f"  Cannot balance to {min_per_digit}/digit — "
                       f"digit with fewest has {actual_min}. "
                       f"Using {target}/digit.")

    balanced = []
    for d in range(10):
        probs = by_digit[d]
        random.shuffle(probs)
        balanced.extend(probs[:target])

    logger.info(f"  Balanced to {target} per digit = {len(balanced)} total")
    return balanced


def run_fourier_at_layer(
    model, layer: int, correct: List[dict], phi: np.ndarray,
    basis_labels: List[str], skip_patching: bool = False,
) -> Dict:
    """
    Run full Fourier decomposition at a single layer.
    Returns results dict with energy spectrum, neuron analysis, and patching.
    """
    d_model = model.cfg.d_model

    # ── Per-digit mean activations ──
    mu, mu_centered, digit_counts = compute_per_digit_mean_activations(
        model, correct, layer,
    )

    # Verify all digits represented
    for d in range(10):
        assert digit_counts[d] >= 1, f"No problems for digit {d}"

    # ── Fourier decomposition ──
    log_section(f"FOURIER ENERGY SPECTRUM — L{layer}")
    decomp = fourier_decomposition(mu_centered, phi, basis_labels)
    V = decomp["fourier_coeff_vectors"]  # (9, d_model)

    # ── Per-neuron analysis ──
    log_section(f"PER-NEURON FOURIER PURITY — L{layer}")
    neuron_results = per_neuron_fourier_analysis(mu_centered, phi)

    # ── Build patching bases ──
    fourier_basis, fourier_ordering = build_fourier_patching_basis(
        V, decomp["freq_energy"],
    )
    freq_bases = build_frequency_resolved_bases(V)

    patching_results = {}
    if not skip_patching:
        # ── Frequency-resolved patching ──
        log_section(f"FREQUENCY-RESOLVED PATCHING AT L{layer}")
        for freq_k in sorted(freq_bases.keys()):
            basis_k = freq_bases[freq_k]
            n_dims = basis_k.shape[1]
            logger.info(f"\n  ── Frequency k={freq_k} ({n_dims}D) ──")
            results_k = run_patching_experiment(
                model, layer=layer, basis=basis_k,
                correct=correct, n_dims_list=[n_dims],
                label=f"Freq-{freq_k}",
                verify_partition_on_first=False,
            )
            patching_results[f"freq_{freq_k}"] = results_k

        # ── Cumulative Fourier patching ──
        log_section(f"CUMULATIVE FOURIER PATCHING (energy-ordered) AT L{layer}")
        patch_dims = sorted(set([2, 4, 5, 6, 8, 9]))
        patch_dims = [d for d in patch_dims if d <= fourier_basis.shape[1]]

        results_fourier = run_patching_experiment(
            model, layer=layer, basis=fourier_basis,
            correct=correct, n_dims_list=patch_dims,
            label="Fourier",
            verify_partition_on_first=True,
        )
        patching_results["cumulative"] = results_fourier

        # ── Random baseline ──
        log_section("RANDOM BASELINE")
        rng = np.random.RandomState(42)
        random_basis_raw = rng.randn(d_model, 9)
        random_basis, _ = np.linalg.qr(random_basis_raw)
        random_basis = random_basis[:, :9]

        results_random = run_patching_experiment(
            model, layer=layer, basis=random_basis,
            correct=correct, n_dims_list=[2, 5, 9],
            label="Random",
            verify_partition_on_first=False,
        )
        patching_results["random"] = results_random

        # ── Comparison summary ──
        log_section("COMPARISON SUMMARY")
        def _pct(results, dim_key, mode):
            if dim_key not in results:
                return "—"
            r = results[dim_key][mode]
            n = r["total"]
            return f"{100*r['transfer']/n:.1f}%" if n > 0 else "—"

        logger.info(f"  {'Method':<25} {'2D':>8} {'5D':>8} {'9D':>8}")
        logger.info(f"  {'─'*55}")
        logger.info(f"  {'Fourier (energy-ord)':<25} "
                     f"{_pct(results_fourier, '2D', 'sub'):>8} "
                     f"{_pct(results_fourier, '5D', 'sub'):>8} "
                     f"{_pct(results_fourier, '9D', 'sub'):>8}")
        logger.info(f"  {'Random':<25} "
                     f"{_pct(results_random, '2D', 'sub'):>8} "
                     f"{_pct(results_random, '5D', 'sub'):>8} "
                     f"{_pct(results_random, '9D', 'sub'):>8}")

    return {
        "layer": layer,
        "digit_counts": {str(k): v for k, v in digit_counts.items()},
        "freq_energy": {str(k): float(v) for k, v in decomp["freq_energy"].items()},
        "freq_fractions": {str(k): float(v) for k, v in decomp["freq_fractions"].items()},
        "total_energy": float(decomp["total_energy"]),
        "top2_frac": float(decomp["top2_frac"]),
        "top2_freqs": list(decomp["top2_freqs"]),
        "interpretation": decomp["interpretation"],
        "per_neuron": neuron_results,
        "fourier_ordering": [(int(k), lab) for k, lab in fourier_ordering],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fourier Decomposition of Digit Subspace"
    )
    parser.add_argument("--model", default="llama-3b",
                        help="Model key or HuggingFace path")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer to analyze (required unless --layer-sweep)")
    parser.add_argument("--layer-sweep", default="",
                        help="Comma-separated layer range, e.g. '15,16,17,18,19,20,27' "
                             "or 'L15-L27' for contiguous range")
    parser.add_argument("--device", default="mps",
                        help="Device (mps safe — no gradients needed)")
    parser.add_argument("--n-per-digit", type=int, default=100,
                        help="Problems per digit for teacher-forced generation")
    parser.add_argument("--n-test", type=int, default=0,
                        help="Max test problems (0 = no cap, use all correct)")
    parser.add_argument("--min-per-digit", type=int, default=15,
                        help="Minimum balanced observations per digit")
    parser.add_argument("--skip-patching", action="store_true",
                        help="Skip patching experiments (just do decomposition)")
    parser.add_argument("--direct-answer", action="store_true",
                        help="Use direct-answer mode (for models that predict full answer as single token)")
    args = parser.parse_args()

    # Parse layer arguments
    layers = []
    if args.layer_sweep:
        sweep = args.layer_sweep.strip()
        if '-' in sweep and sweep.replace('L','').replace('-','').replace(',','').isdigit() is False:
            # Handle 'L15-L27' format
            parts = sweep.replace('L','').split('-')
            layers = list(range(int(parts[0]), int(parts[1])+1))
        elif '-' in sweep and ',' not in sweep:
            parts = sweep.replace('L','').split('-')
            layers = list(range(int(parts[0]), int(parts[1])+1))
        else:
            layers = [int(x.strip().replace('L','')) for x in sweep.split(',')]
    elif args.layer is not None:
        layers = [args.layer]
    else:
        parser.error("Must specify --layer or --layer-sweep")

    model_name = MODEL_MAP.get(args.model, args.model)
    device = args.device

    log_section(f"FOURIER DECOMPOSITION — {model_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Layers: {layers}")

    # ── Load model ──
    import transformer_lens
    model = transformer_lens.HookedTransformer.from_pretrained(
        model_name, device=device,
    )
    model.eval()
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    logger.info(f"n_layers={n_layers}, d_model={d_model}")

    for L in layers:
        assert 0 <= L < n_layers, f"Layer {L} out of range [0, {n_layers})"

    # ── Generate and filter problems ──
    log_section("GENERATING TEST PROBLEMS")
    if args.direct_answer:
        logger.info("  Using direct-answer mode (full answer as single token)")
        test_flat, test_by_digit = generate_direct_answer_problems(
            n_per_digit=args.n_per_digit, operand_max=99,
        )
        max_n = args.n_test if args.n_test > 0 else len(test_flat)
        correct = filter_correct_direct_answer(model, test_flat, max_n=max_n)
    else:
        test_flat, test_by_digit = generate_teacher_forced_problems(
            n_per_digit=args.n_per_digit, operand_max=99,
        )
        max_n = args.n_test if args.n_test > 0 else len(test_flat)
        correct = filter_correct_teacher_forced(model, test_flat, max_n=max_n)

    if len(correct) < 20 and not args.direct_answer:
        logger.warning(f"Only {len(correct)} correct — falling back to single-digit")
        single_probs = generate_single_digit_problems(operand_max=9)
        max_n_sd = args.n_test if args.n_test > 0 else len(single_probs)
        correct = filter_correct_teacher_forced(model, single_probs, max_n=max_n_sd)

    assert len(correct) >= 10, (
        f"Only {len(correct)} correct problems — insufficient for analysis"
    )

    # ── Balance digit counts ──
    log_section("DATA BALANCE CHECK")
    by_digit = defaultdict(list)
    for p in correct:
        by_digit[p["ones_digit"]].append(p)
    for d in range(10):
        logger.info(f"  Digit {d}: {len(by_digit[d])} problems")
    min_count = min(len(by_digit[d]) for d in range(10))
    max_count = max(len(by_digit[d]) for d in range(10))
    imbalance = max_count / max(min_count, 1)
    logger.info(f"  Min={min_count}, Max={max_count}, Imbalance={imbalance:.1f}x")

    if min_count < args.min_per_digit:
        logger.warning(f"  ⚠ INSUFFICIENT DATA: digit with fewest has {min_count} obs "
                       f"(need {args.min_per_digit}). Per-digit means may be unreliable!")

    if imbalance > 2.0:
        logger.info(f"  Subsampling to balance digit counts...")
        correct = balance_digit_counts(correct, min_per_digit=args.min_per_digit)
    else:
        logger.info(f"  Data reasonably balanced (imbalance {imbalance:.1f}x ≤ 2.0)")

    # ── Fourier basis (shared across layers) ──
    phi, basis_labels = build_fourier_basis_functions(N=10)

    # ── Run per layer ──
    all_results = {}
    is_sweep = len(layers) > 1
    # For sweeps, skip patching by default (just spectrum) unless user explicitly wants it
    sweep_skip_patching = args.skip_patching or is_sweep

    for layer in layers:
        log_section(f"LAYER {layer}")
        result = run_fourier_at_layer(
            model, layer, correct, phi, basis_labels,
            skip_patching=sweep_skip_patching,
        )
        all_results[layer] = result

    # ── If sweep, run patching on the single most interesting layer ──
    if is_sweep and not args.skip_patching:
        # Pick layer with highest k=5 + k=2 energy fraction (CRT signature)
        best_layer = None
        best_crt_score = -1
        for L, res in all_results.items():
            k2_frac = res["freq_fractions"].get("2", 0)
            k5_frac = res["freq_fractions"].get("5", 0)
            crt_score = k2_frac + k5_frac
            if crt_score > best_crt_score:
                best_crt_score = crt_score
                best_layer = L
        logger.info(f"\n  Best CRT-signature layer: L{best_layer} "
                    f"(k=2+k=5 = {best_crt_score:.1%})")
        log_section(f"PATCHING AT BEST CRT LAYER L{best_layer}")
        patching_result = run_fourier_at_layer(
            model, best_layer, correct, phi, basis_labels,
            skip_patching=False,
        )
        all_results[f"{best_layer}_patching"] = patching_result

    # ── Layer sweep summary table ──
    if is_sweep:
        log_section("LAYER SWEEP SUMMARY")
        logger.info(f"  {'Layer':>6}  {'k=1':>8}  {'k=2':>8}  {'k=3':>8}  "
                     f"{'k=4':>8}  {'k=5':>8}  {'Top2%':>8}  {'CRT(k2+k5)':>12}")
        logger.info(f"  {'─'*80}")
        for L in layers:
            res = all_results[L]
            ff = res["freq_fractions"]
            crt = ff.get("2", 0) + ff.get("5", 0)
            logger.info(
                f"  L{L:>4}  {ff.get('1',0):>7.1%}  {ff.get('2',0):>7.1%}  "
                f"{ff.get('3',0):>7.1%}  {ff.get('4',0):>7.1%}  "
                f"{ff.get('5',0):>7.1%}  {res['top2_frac']:>7.1%}  "
                f"{crt:>11.1%}"
            )

    # ── Save results ──
    out_dir = Path("mathematical_toolkit_results")
    out_dir.mkdir(exist_ok=True)
    model_slug = args.model.replace("/", "_").replace("-", "_")

    if is_sweep:
        layer_tag = f"L{layers[0]}-L{layers[-1]}"
    else:
        layer_tag = f"L{layers[0]}"
    out_path = out_dir / f"fourier_decomposition_{model_slug}_{layer_tag}.json"

    save_data = {
        "model": model_name,
        "layers": layers,
        "device": device,
        "d_model": d_model,
        "n_correct": len(correct),
        "results": {str(k): v for k, v in all_results.items()},
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
