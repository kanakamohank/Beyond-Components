#!/usr/bin/env python3
"""
Neuron-Level Trigonometric Analysis
====================================

Analyzes whether the 9D Fourier subspace found in digit-mean activations
is concentrated in a few interpretable "trig neurons" or diffusely spread.

PART 1 — Residual Stream Dimension DFT:
  For each of d_model residual stream dimensions, DFT the 10-element
  digit-conditional mean vector.  Identify dimensions dominated by a
  single Fourier frequency.

PART 2 — MLP Neuron DFT:
  Same analysis on the d_mlp hidden neurons of the MLP at the computation
  layer.  These are the mechanistic equivalent of "clock neurons" from
  the grokking literature.

PART 3 — Phase Clustering:
  For the top neurons at each frequency, extract phase angles and test
  for non-uniform clustering.  Clustered phases imply a small number of
  canonical trig directions.

PART 4 — Component Fourier Attribution:
  For each attention head and MLP (layers 0..comp_layer), measure how
  much Fourier power they write into the residual stream.  Identifies
  which circuits produce the trig structure.

PART 5 — Sparse Dimension Steering:
  Apply Fourier rotation delta restricted to only the top-K residual
  stream dimensions by Fourier power.  Compare to full-subspace rotation.

Usage:
    python neuron_trig_analysis.py --model gemma-2b --layer 19 --device mps
    python neuron_trig_analysis.py --model phi-3-mini --layer 26 --device mps
    python neuron_trig_analysis.py --model llama-3b --layer 20 --device mps --direct-answer
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import logging
import argparse
import json
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

from experiments.fourier_phase_rotation import (
    compute_digit_fourier_basis, build_frequency_planes,
    compute_rotation_delta, generate_test_problems,
    filter_correct_problems, generate_single_digit_problems,
    filter_correct_single_digit, generate_single_digit_direct_answer_problems,
    MODEL_MAP, COMP_LAYERS,
)
from experiments.arithmetic_circuit_scan_updated import (
    generate_direct_answer_problems, filter_correct_direct_answer,
)

RESULTS_DIR = Path("mathematical_toolkit_results")
RESULTS_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# PART 1: RESIDUAL STREAM DIMENSION DFT
# ═══════════════════════════════════════════════════════════════════════

def collect_digit_means(model, problems, layer, device, n_problems=500):
    """Collect digit-conditional mean activations at the last token position.

    Returns:
        digit_means: (10, d_model) array — mean activation per digit
        digit_counts: (10,) array — number of problems per digit
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    d_model = model.cfg.d_model

    digit_sums = np.zeros((10, d_model), dtype=np.float64)
    digit_counts = np.zeros(10, dtype=int)

    with torch.no_grad():
        for i, prob in enumerate(problems[:n_problems]):
            tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
            act = cache[hook_name][0, -1].float().cpu().numpy()
            d = prob["ones_digit"]
            digit_sums[d] += act
            digit_counts[d] += 1
            del cache
            if (i + 1) % 100 == 0:
                logger.info(f"    Collecting digit means: {i+1}/{min(n_problems, len(problems))}")

    digit_means = np.zeros((10, d_model))
    for d in range(10):
        if digit_counts[d] > 0:
            digit_means[d] = digit_sums[d] / digit_counts[d]

    logger.info(f"  Digit counts: {digit_counts.tolist()} (min={digit_counts.min()})")
    if digit_counts.min() == 0:
        missing = [d for d in range(10) if digit_counts[d] == 0]
        logger.warning(f"  [WARN] Digits {missing} have 0 samples! Their means are zero vectors.")
        logger.warning(f"  This will distort DFT results. Increase --operand-range or n_problems.")
    assert digit_counts.min() >= 1, \
        f"[FATAL] Digit(s) with 0 samples: {[d for d in range(10) if digit_counts[d]==0]}. " \
        f"Cannot compute meaningful DFT. Use more problems or wider operand range."
    return digit_means, digit_counts


def compute_dimension_dft(digit_means):
    """Compute DFT power spectrum for each residual stream dimension.

    For each dimension i, takes the 10-element vector [mean_i(0)..mean_i(9)]
    and computes its DFT.  Returns power at each frequency k=1..5.

    Returns:
        power: (d_model, 5) — power at freq k=1..5 per dimension
        phases: (d_model, 5) — phase angle at freq k=1..5 per dimension
        purity: (d_model,) — max_k power(k) / sum_k power(k) per dimension
        dominant_freq: (d_model,) — frequency with highest power per dimension
        total_variance: (d_model,) — total between-digit variance per dimension
    """
    d_model = digit_means.shape[1]

    # Center across digits (remove DC component)
    centered = digit_means - digit_means.mean(axis=0, keepdims=True)  # (10, d_model)

    power = np.zeros((d_model, 5))
    phases = np.zeros((d_model, 5))

    for i in range(d_model):
        x = centered[:, i]  # 10-element vector
        dft = np.fft.fft(x)  # 10 complex coefficients

        for k_idx, k in enumerate(range(1, 6)):
            if k < 5:
                # Power from both positive and negative frequency
                # Divide by N^2 so sum(power) == variance == sum(x^2)/N (Parseval)
                power[i, k_idx] = (np.abs(dft[k])**2 + np.abs(dft[10 - k])**2) / 100
            else:
                # Nyquist (k=5): only one component
                power[i, k_idx] = np.abs(dft[5])**2 / 100
            phases[i, k_idx] = np.angle(dft[k])

    total_power = power.sum(axis=1)  # (d_model,)
    purity = np.zeros(d_model)
    dominant_freq = np.zeros(d_model, dtype=int)
    for i in range(d_model):
        if total_power[i] > 1e-12:
            purity[i] = power[i].max() / total_power[i]
            dominant_freq[i] = np.argmax(power[i]) + 1  # k=1..5
        else:
            purity[i] = 0
            dominant_freq[i] = 0

    total_variance = np.sum(centered**2, axis=0) / 10  # (d_model,)

    # Parseval sanity check: sum(power) should equal total_variance
    parseval_err = np.abs(total_power - total_variance)
    max_err = parseval_err.max()
    if max_err > 1e-8:
        logger.warning(f"  [WARN] Parseval check: max |sum(power) - variance| = {max_err:.2e}")
    else:
        logger.info(f"  [SANITY] Parseval check passed: max err = {max_err:.2e}")

    return power, phases, purity, dominant_freq, total_variance


def report_dimension_spectra(power, phases, purity, dominant_freq, total_variance,
                             top_n=20):
    """Report top residual stream dimensions by Fourier power."""
    d_model = power.shape[0]
    total_power = power.sum(axis=1)

    logger.info(f"\n{'='*80}")
    logger.info(f"PART 1: RESIDUAL STREAM DIMENSION DFT ({d_model} dimensions)")
    logger.info(f"{'='*80}")

    # Concentration analysis — answers "concentrated vs diffuse?"
    sorted_power = np.sort(total_power)[::-1]
    cumulative = np.cumsum(sorted_power) / max(sorted_power.sum(), 1e-12)
    for threshold in [0.5, 0.8, 0.9, 0.95, 0.99]:
        n_needed = int(np.searchsorted(cumulative, threshold)) + 1
        logger.info(f"  {threshold*100:.0f}% of Fourier power in top {n_needed}/{d_model} dims ({n_needed/d_model*100:.1f}%)")

    # Overall statistics
    n_high_purity = np.sum(purity > 0.8)
    n_medium_purity = np.sum((purity > 0.5) & (purity <= 0.8))
    logger.info(f"\n  Purity distribution:")
    logger.info(f"    >80% (single-freq dominated):  {n_high_purity} dims ({n_high_purity/d_model*100:.1f}%)")
    logger.info(f"    50-80% (moderate):              {n_medium_purity} dims ({n_medium_purity/d_model*100:.1f}%)")
    logger.info(f"    <50% (mixed/weak):              {d_model - n_high_purity - n_medium_purity} dims")

    # Frequency distribution among high-purity dimensions
    logger.info(f"\n  Frequency distribution (dims with purity > 50%):")
    for k in range(1, 6):
        n_k = np.sum((dominant_freq == k) & (purity > 0.5))
        logger.info(f"    k={k}: {n_k} dimensions")

    # Top dimensions by total Fourier power
    top_idx = np.argsort(-total_power)[:top_n]
    logger.info(f"\n  Top {top_n} dimensions by total Fourier power:")
    logger.info(f"  {'Dim':>6} | {'TotalPow':>10} | {'Purity':>7} | {'DomFreq':>7} | {'k=1':>8} | {'k=2':>8} | {'k=3':>8} | {'k=4':>8} | {'k=5':>8}")
    logger.info(f"  {'-'*6}-+-{'-'*10}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for idx in top_idx:
        logger.info(f"  {idx:6d} | {total_power[idx]:10.3f} | {purity[idx]:6.1%} | k={dominant_freq[idx]:5d} | "
                     f"{power[idx,0]:8.3f} | {power[idx,1]:8.3f} | {power[idx,2]:8.3f} | "
                     f"{power[idx,3]:8.3f} | {power[idx,4]:8.3f}")

    # Top dimensions per frequency
    for k in range(1, 6):
        k_idx = k - 1
        top_k = np.argsort(-power[:, k_idx])[:10]
        logger.info(f"\n  Top 10 dimensions for frequency k={k}:")
        logger.info(f"  {'Dim':>6} | {'Power(k)':>10} | {'Purity':>7} | {'Phase':>8}")
        logger.info(f"  {'-'*6}-+-{'-'*10}-+-{'-'*7}-+-{'-'*8}")
        for idx in top_k:
            logger.info(f"  {idx:6d} | {power[idx, k_idx]:10.3f} | {purity[idx]:6.1%} | {phases[idx, k_idx]:+8.3f}")

    return top_idx


# ═══════════════════════════════════════════════════════════════════════
# PART 2: MLP NEURON DFT
# ═══════════════════════════════════════════════════════════════════════

def collect_mlp_neuron_means(model, problems, layer, device, n_problems=500):
    """Collect digit-conditional mean MLP hidden activations (post nonlinearity).

    Returns:
        neuron_means: (10, d_mlp) array
        digit_counts: (10,) array
    """
    hook_name = f"blocks.{layer}.mlp.hook_post"
    d_mlp = model.cfg.d_mlp

    digit_sums = np.zeros((10, d_mlp), dtype=np.float64)
    digit_counts = np.zeros(10, dtype=int)

    with torch.no_grad():
        for i, prob in enumerate(problems[:n_problems]):
            tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
            act = cache[hook_name][0, -1].float().cpu().numpy()
            d = prob["ones_digit"]
            digit_sums[d] += act
            digit_counts[d] += 1
            del cache
            if (i + 1) % 100 == 0:
                logger.info(f"    Collecting MLP neurons: {i+1}/{min(n_problems, len(problems))}")

    neuron_means = np.zeros((10, d_mlp))
    for d in range(10):
        if digit_counts[d] > 0:
            neuron_means[d] = digit_sums[d] / digit_counts[d]

    return neuron_means, digit_counts


def report_mlp_neuron_spectra(power, phases, purity, dominant_freq, total_variance,
                               d_mlp, top_n=20):
    """Report top MLP neurons by Fourier power."""
    total_power = power.sum(axis=1)

    logger.info(f"\n{'='*80}")
    logger.info(f"PART 2: MLP NEURON DFT ({d_mlp} neurons)")
    logger.info(f"{'='*80}")

    # Overall statistics
    n_high_purity = np.sum(purity > 0.8)
    n_medium_purity = np.sum((purity > 0.5) & (purity <= 0.8))
    n_active = np.sum(total_power > 1e-6)
    logger.info(f"\n  Active neurons (any digit variance): {n_active}/{d_mlp}")
    logger.info(f"  Purity distribution (among active):")
    logger.info(f"    >80% (single-freq dominated):  {n_high_purity}")
    logger.info(f"    50-80% (moderate):              {n_medium_purity}")

    # Frequency distribution
    logger.info(f"\n  Frequency distribution (neurons with purity > 50%):")
    for k in range(1, 6):
        n_k = np.sum((dominant_freq == k) & (purity > 0.5))
        logger.info(f"    k={k}: {n_k} neurons")

    # Top MLP neurons by total Fourier power
    top_idx = np.argsort(-total_power)[:top_n]
    logger.info(f"\n  Top {top_n} MLP neurons by total Fourier power:")
    logger.info(f"  {'Neuron':>8} | {'TotalPow':>10} | {'Purity':>7} | {'DomFreq':>7} | {'k=1':>8} | {'k=2':>8} | {'k=3':>8} | {'k=4':>8} | {'k=5':>8}")
    logger.info(f"  {'-'*8}-+-{'-'*10}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for idx in top_idx:
        logger.info(f"  {idx:8d} | {total_power[idx]:10.4f} | {purity[idx]:6.1%} | k={dominant_freq[idx]:5d} | "
                     f"{power[idx,0]:8.4f} | {power[idx,1]:8.4f} | {power[idx,2]:8.4f} | "
                     f"{power[idx,3]:8.4f} | {power[idx,4]:8.4f}")

    # Top neurons per frequency
    for k in range(1, 6):
        k_idx = k - 1
        top_k = np.argsort(-power[:, k_idx])[:10]
        logger.info(f"\n  Top 10 MLP neurons for frequency k={k}:")
        logger.info(f"  {'Neuron':>8} | {'Power(k)':>10} | {'Purity':>7} | {'Phase':>8}")
        logger.info(f"  {'-'*8}-+-{'-'*10}-+-{'-'*7}-+-{'-'*8}")
        for idx in top_k:
            logger.info(f"  {idx:8d} | {power[idx, k_idx]:10.4f} | {purity[idx]:6.1%} | {phases[idx, k_idx]:+8.3f}")

    return top_idx


# ═══════════════════════════════════════════════════════════════════════
# PART 3: PHASE CLUSTERING
# ═══════════════════════════════════════════════════════════════════════

def analyze_phase_clustering(power, phases, label="dimensions", top_n=50):
    """Analyze phase distribution for top neurons at each frequency.

    Uses circular statistics: mean resultant length R (0=uniform, 1=perfectly
    clustered) and Rayleigh test for non-uniformity.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"PART 3: PHASE CLUSTERING ({label})")
    logger.info(f"{'='*80}")

    results = {}
    for k in range(1, 6):
        k_idx = k - 1
        top_idx = np.argsort(-power[:, k_idx])[:top_n]
        top_phases = phases[top_idx, k_idx]

        # Circular statistics
        # Mean resultant length R: how concentrated the phases are
        cos_sum = np.sum(np.cos(top_phases))
        sin_sum = np.sum(np.sin(top_phases))
        R = np.sqrt(cos_sum**2 + sin_sum**2) / top_n
        mean_phase = np.arctan2(sin_sum, cos_sum)

        # Rayleigh test: z = n * R^2, significant if z > ~3 for p<0.05
        z = top_n * R**2

        logger.info(f"\n  Frequency k={k} (top {top_n} {label}):")
        logger.info(f"    Mean resultant length R = {R:.3f} (0=uniform, 1=clustered)")
        logger.info(f"    Mean phase = {mean_phase:+.3f} rad ({np.degrees(mean_phase):+.1f}°)")
        logger.info(f"    Rayleigh z = {z:.1f} (>3 → significant clustering at p<0.05)")

        if k == 5:
            logger.info(f"    NOTE: k=5 (Nyquist) DFT coeff is always real → phases are 0 or π")
            logger.info(f"    Phase clustering for k=5 reflects sign distribution, not true phase")

        if R > 0.3:
            logger.info(f"    → CLUSTERED: phases are non-uniform ★")
        elif R > 0.15:
            logger.info(f"    → MODERATELY clustered")
        else:
            logger.info(f"    → DIFFUSE: phases are roughly uniform")

        # Check for bimodal clustering (opposite phases)
        doubled_phases = 2 * top_phases
        cos2 = np.sum(np.cos(doubled_phases))
        sin2 = np.sum(np.sin(doubled_phases))
        R2 = np.sqrt(cos2**2 + sin2**2) / top_n
        if R2 > R and R2 > 0.3:
            logger.info(f"    → BIMODAL clustering detected (R2={R2:.3f} > R={R:.3f})")
            logger.info(f"      cos/sin pairs tend to be ~180° apart")

        results[k] = {
            'R': float(R), 'mean_phase': float(mean_phase),
            'rayleigh_z': float(z), 'R2_bimodal': float(R2),
        }

    return results


# ═══════════════════════════════════════════════════════════════════════
# PART 4: COMPONENT FOURIER ATTRIBUTION
# ═══════════════════════════════════════════════════════════════════════

def compute_component_attribution(model, problems, layer, fourier_basis, device,
                                  n_problems=200):
    """Measure how much Fourier power each component writes.

    For each attention head and MLP at layers 0..comp_layer, projects their
    output onto the Fourier subspace and measures power.

    Returns:
        attn_power: dict (L, H) -> mean Fourier power
        mlp_power: dict L -> mean Fourier power
    """
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model

    # Fourier projector
    V = torch.tensor(fourier_basis, dtype=torch.float32)  # (9, d_model)
    # V should be (d_model, 9) — let's check and transpose if needed
    if V.shape[0] == 9:
        V = V.T  # now (d_model, 9)
    P_F = V @ V.T  # (d_model, d_model)

    # Build hook names for all components up to comp_layer
    hook_names = []
    for L in range(layer + 1):
        hook_names.append(f"blocks.{L}.hook_attn_out")
        hook_names.append(f"blocks.{L}.hook_mlp_out")

    # Accumulators
    attn_fourier = defaultdict(float)  # L -> sum of ||P_F · attn_out||²
    mlp_fourier = defaultdict(float)   # L -> sum of ||P_F · mlp_out||²
    n_samples = 0

    logger.info(f"\n{'='*80}")
    logger.info(f"PART 4: COMPONENT FOURIER ATTRIBUTION (layers 0..{layer})")
    logger.info(f"{'='*80}")

    with torch.no_grad():
        for i, prob in enumerate(problems[:n_problems]):
            tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)

            for L in range(layer + 1):
                # Attention output
                attn_key = f"blocks.{L}.hook_attn_out"
                if attn_key in cache:
                    attn_out = cache[attn_key][0, -1].float().cpu()  # (d_model,)
                    pf_attn = P_F @ attn_out
                    attn_fourier[L] += (pf_attn ** 2).sum().item()

                # MLP output
                mlp_key = f"blocks.{L}.hook_mlp_out"
                if mlp_key in cache:
                    mlp_out = cache[mlp_key][0, -1].float().cpu()  # (d_model,)
                    pf_mlp = P_F @ mlp_out
                    mlp_fourier[L] += (pf_mlp ** 2).sum().item()

            del cache
            n_samples += 1
            if (i + 1) % 50 == 0:
                logger.info(f"    Attribution pass: {i+1}/{min(n_problems, len(problems))}")

    # Normalize
    for L in range(layer + 1):
        attn_fourier[L] /= max(n_samples, 1)
        mlp_fourier[L] /= max(n_samples, 1)

    # Report
    logger.info(f"\n  Component Fourier Power (averaged over {n_samples} problems):")
    logger.info(f"  {'Component':>15} | {'Fourier Power':>15}")
    logger.info(f"  {'-'*15}-+-{'-'*15}")

    # Sort by power
    all_components = []
    for L in range(layer + 1):
        all_components.append((f"L{L}_attn", attn_fourier[L]))
        all_components.append((f"L{L}_mlp", mlp_fourier[L]))
    all_components.sort(key=lambda x: -x[1])

    for name, pwr in all_components[:20]:
        bar = "█" * int(pwr / max(all_components[0][1], 1e-8) * 30)
        logger.info(f"  {name:>15} | {pwr:15.3f} {bar}")

    # Summary: top 5 components
    logger.info(f"\n  Top 5 Fourier-writing components:")
    for i, (name, pwr) in enumerate(all_components[:5]):
        logger.info(f"    #{i+1}: {name} (power={pwr:.3f})")

    return dict(attn_fourier), dict(mlp_fourier)


# ═══════════════════════════════════════════════════════════════════════
# PART 5: SPARSE DIMENSION STEERING
# ═══════════════════════════════════════════════════════════════════════

def sparse_steering_test(model, test_problems, layer, basis_problems, dim_power,
                         device, k_values=None):
    """Test steering using only top-K residual stream dimensions.

    Restricts the Fourier rotation delta to the K dimensions with highest
    total Fourier power.  Compares to full-subspace rotation.
    """
    if k_values is None:
        d_model = model.cfg.d_model
        k_values = [10, 25, 50, 100, 250, 500, d_model]

    logger.info(f"\n{'='*80}")
    logger.info(f"PART 5: SPARSE DIMENSION STEERING")
    logger.info(f"{'='*80}")

    # Compute basis and frequency planes
    basis, freq_assignments, svals, digit_scores, freq_purities = \
        compute_digit_fourier_basis(model, basis_problems, layer, n_problems=500)
    freq_planes = build_frequency_planes(basis, freq_assignments, svals)

    hook_name = f"blocks.{layer}.hook_resid_post"

    freq_planes_torch = {}
    for k, plane in freq_planes.items():
        freq_planes_torch[k] = {
            'vecs': torch.tensor(plane['vecs'], dtype=torch.float32, device=device),
            'svals': torch.tensor(plane['svals'], dtype=torch.float32, device=device),
        }

    # Get digit token IDs for logit lens
    digit_token_ids = []
    for d in range(10):
        toks = model.to_tokens(str(d), prepend_bos=False)[0]
        digit_token_ids.append(toks[-1].item())
    digit_ids_t = torch.tensor(digit_token_ids, device=device)

    # Cache clean activations
    single_digit = [p for p in test_problems if p.get('n_digits', 1) == 1]
    clean_data = []
    with torch.no_grad():
        for prob in single_digit:
            tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
            act = cache[hook_name][0, -1].clone()
            clean_data.append((act, prob["ones_digit"]))
            del cache

    n = len(clean_data)
    logger.info(f"  Test problems: {n}")

    # Rank dimensions by total Fourier power
    total_power = dim_power.sum(axis=1)  # (d_model,)
    dim_ranking = np.argsort(-total_power)

    # Test each K
    results = []
    for K in k_values:
        K = min(K, len(dim_ranking))
        # Create mask: 1 for top-K dimensions, 0 elsewhere
        mask = torch.zeros(model.cfg.d_model, dtype=torch.float32, device=device)
        mask[dim_ranking[:K]] = 1.0

        n_exact = 0
        n_total = 0
        rank_sum = 0.0
        prob_sum = 0.0

        for act, d_orig in clean_data:
            for j in range(1, 10):
                target = (d_orig + j) % 10
                delta = compute_rotation_delta(
                    act.float(), freq_planes_torch, j, mode="coherent"
                )
                # Mask: keep only top-K dimensions
                delta_sparse = delta * mask

                new_act = act.float() + delta_sparse

                # Logit lens
                if hasattr(model, 'ln_final'):
                    normed = model.ln_final(new_act.unsqueeze(0)).squeeze(0)
                else:
                    normed = new_act
                logits = normed @ model.W_U
                digit_logits = logits[digit_ids_t]
                probs = torch.softmax(digit_logits.float(), dim=0)

                pred = digit_logits.argmax().item()
                rank_val = (digit_logits > digit_logits[target]).sum().item()

                if pred == target:
                    n_exact += 1
                n_total += 1
                rank_sum += rank_val
                prob_sum += probs[target].item()

        exact_pct = n_exact / max(n_total, 1) * 100
        mean_rank = rank_sum / max(n_total, 1)
        mean_prob = prob_sum / max(n_total, 1)

        results.append({
            'K': K, 'exact_pct': exact_pct,
            'mean_rank': mean_rank, 'mean_prob': mean_prob,
        })
        pct_dims = K / model.cfg.d_model * 100
        logger.info(f"  K={K:5d} ({pct_dims:5.1f}% dims) | exact={exact_pct:5.1f}% | rank={mean_rank:.2f} | prob={mean_prob:.3f}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from transformer_lens import HookedTransformer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma-2b")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--operand-range", type=int, default=30)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--direct-answer", action="store_true")
    parser.add_argument("--skip-mlp", action="store_true",
                        help="Skip MLP neuron analysis (saves time/memory)")
    parser.add_argument("--skip-attribution", action="store_true",
                        help="Skip component attribution (saves time)")
    parser.add_argument("--skip-steering", action="store_true",
                        help="Skip sparse steering test")
    args = parser.parse_args()

    if args.layer is None:
        args.layer = COMP_LAYERS.get(args.model, 20)

    model_name = MODEL_MAP[args.model]
    logger.info(f"Loading {model_name} on {args.device}...")
    model = HookedTransformer.from_pretrained(model_name, device=args.device)
    model.eval()
    logger.info(f"  d_model={model.cfg.d_model}, d_mlp={model.cfg.d_mlp}, "
                f"n_heads={model.cfg.n_heads}, n_layers={model.cfg.n_layers}")

    # Generate problems
    if args.direct_answer:
        da_problems, _ = generate_direct_answer_problems(
            n_per_digit=100, operand_max=args.operand_range)
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
        sd_da = generate_single_digit_direct_answer_problems()
        test_problems = filter_correct_single_digit(model, sd_da)
    else:
        all_problems = generate_test_problems(max_operand=args.operand_range)
        basis_problems = filter_correct_problems(model, all_problems, n_test=len(all_problems))
        sd_problems = generate_single_digit_problems()
        test_problems = filter_correct_single_digit(model, sd_problems)

    logger.info(f"\nBasis problems: {len(basis_problems)}")
    logger.info(f"Test problems: {len(test_problems)} single-digit")

    # ── PART 1: Residual stream dimension DFT ──
    logger.info(f"\n  Collecting residual stream digit means at layer {args.layer}...")
    digit_means, digit_counts = collect_digit_means(
        model, basis_problems, args.layer, args.device, n_problems=500)

    dim_power, dim_phases, dim_purity, dim_dom_freq, dim_variance = \
        compute_dimension_dft(digit_means)

    top_dims = report_dimension_spectra(
        dim_power, dim_phases, dim_purity, dim_dom_freq, dim_variance)

    # ── PART 2: MLP neuron DFT ──
    if not args.skip_mlp:
        logger.info(f"\n  Collecting MLP neuron digit means at layer {args.layer}...")
        mlp_means, _ = collect_mlp_neuron_means(
            model, basis_problems, args.layer, args.device, n_problems=500)

        mlp_power, mlp_phases, mlp_purity, mlp_dom_freq, mlp_variance = \
            compute_dimension_dft(mlp_means)

        top_mlp = report_mlp_neuron_spectra(
            mlp_power, mlp_phases, mlp_purity, mlp_dom_freq, mlp_variance,
            d_mlp=model.cfg.d_mlp)

    # ── PART 3: Phase clustering ──
    logger.info("")
    dim_phase_results = analyze_phase_clustering(
        dim_power, dim_phases, label="resid dimensions", top_n=50)

    if not args.skip_mlp:
        mlp_phase_results = analyze_phase_clustering(
            mlp_power, mlp_phases, label="MLP neurons", top_n=50)

    # ── PART 4: Component attribution ──
    if not args.skip_attribution:
        # Derive 9D SVD basis directly from digit_means (no extra forward passes)
        centroid = digit_means.mean(axis=0, keepdims=True)
        M_centered = digit_means - centroid
        _, _, Vt = np.linalg.svd(M_centered, full_matrices=False)
        basis_9d = Vt[:9].T  # (d_model, 9)

        attn_power, mlp_layer_power = compute_component_attribution(
            model, basis_problems, args.layer, basis_9d,
            args.device, n_problems=200)

    # ── PART 5: Sparse steering ──
    if not args.skip_steering:
        d_model = model.cfg.d_model
        k_values = [10, 25, 50, 100, 250, 500, min(1000, d_model), d_model]
        steering_results = sparse_steering_test(
            model, test_problems, args.layer, basis_problems, dim_power,
            args.device, k_values=k_values)

    logger.info(f"\n{'='*80}")
    logger.info(f"ANALYSIS COMPLETE")
    logger.info(f"{'='*80}")
