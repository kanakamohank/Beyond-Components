#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 4: Weight-Level Fourier Circuit Analysis.

After Phase 3 identifies *which* SVD directions are periodic and at which
frequencies, this script digs into *how* those directions implement Fourier
features at the weight level.

Phase 4a — SVD Direction Decomposition:
    For each surviving periodic direction k at frequency f:
    1. Extract input singular vector Vh[k, :] (d_model) and output U[:, k].
    2. Project residual stream activations onto Vh[k, :] → scalar signal p(n).
    3. Fit p(n) = A·cos(2πfn/N) + B·sin(2πfn/N) + C  →  measure R².
    4. Report amplitude, phase, and fit quality for each direction.
    5. For directions at the *same* frequency, check if they form orthogonal
       cos/sin pairs (phase difference ≈ π/2).

Phase 4b — MLP Neuron Trig Identity Analysis:
    For MLP L23 (the strongest surviving MLP component):
    1. Collect MLP hidden activations for an (a, b) grid of addition prompts.
    2. For each neuron, compute 2-D DFT of activation(a, b).
    3. Test separability: does neuron(a,b) ≈ f(a)·g(b)?
    4. Check trig identity: sin(f(a+b)) = sin(fa)cos(fb) + cos(fa)sin(fb).

Usage:
    python experiments/analyze_fourier_circuits.py \\
        --config configs/arithmetic_pythia_config.yaml \\
        --checkpoint svd_logs/.../model_final.pt \\
        --phase3_results phase3_results/svd_direction_fourier_....json \\
        [--operand_range_end 100] \\
        [--output_dir phase4_results]

Requires: A completed Phase 3 run with saved JSON results.
"""

import argparse
import datetime
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.fourier_discovery import (
    compute_fourier_power_spectrum,
    identify_dominant_frequency,
)
from src.data.arithmetic_dataset import (
    ArithmeticPromptGenerator,
)
from src.models.masked_transformer_circuit import MaskedTransformerCircuit, mask_fn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class DirectionFitResult:
    """Cos/sin fit result for a single SVD direction.

    Attributes:
        layer: Layer index.
        head: Head index (None for MLP).
        component: 'OV', 'MLP_in', or 'MLP_out'.
        direction_idx: SVD direction index.
        mask_value: Learned mask value.
        singular_value: Singular value σ_k.
        effective_strength: σ_k × mask.
        dominant_frequency: Best frequency from Phase 3.
        power_ratio: Power ratio from Phase 3.
        cos_coeff: Fitted cosine coefficient A.
        sin_coeff: Fitted sinusoidal coefficient B.
        dc_offset: Fitted constant offset C.
        amplitude: sqrt(A² + B²).
        phase: atan2(B, A) in radians.
        r_squared: R² of single-frequency cos/sin + DC fit.
        r_squared_multi: R² of multi-frequency fit (primary + all secondaries).
        residual_std: Std of residuals after single-frequency fit.
        secondary_frequencies: List of (freq, power_ratio) for secondary peaks.
    """
    layer: int
    head: Optional[int]
    component: str
    direction_idx: int
    mask_value: float
    singular_value: float
    effective_strength: float
    dominant_frequency: int
    power_ratio: float
    cos_coeff: float
    sin_coeff: float
    dc_offset: float
    amplitude: float
    phase: float
    r_squared: float
    r_squared_multi: float
    residual_std: float
    secondary_frequencies: List[Tuple[int, float]] = field(default_factory=list)


@dataclass
class FrequencyGroupResult:
    """Analysis of directions sharing the same frequency.

    Attributes:
        frequency: The shared Fourier frequency.
        n_directions: Number of directions at this frequency.
        directions: List of DirectionFitResult at this frequency.
        phase_diffs: List of pairwise phase differences (radians).
        has_cos_sin_pair: Whether two directions are ≈π/2 apart in phase.
        subspace_dimension: Effective dimensionality of the Fourier subspace.
        input_cosine_sims: Pairwise cosine similarities of Vh vectors.
    """
    frequency: int
    n_directions: int
    directions: List[DirectionFitResult] = field(default_factory=list)
    phase_diffs: List[float] = field(default_factory=list)
    has_cos_sin_pair: bool = False
    subspace_dimension: float = 0.0
    input_cosine_sims: List[float] = field(default_factory=list)


@dataclass
class NeuronTrigResult:
    """Trig identity analysis for a single MLP neuron.

    Attributes:
        layer: MLP layer index.
        neuron_idx: Neuron index within the MLP hidden layer.
        dominant_freq_2d: Dominant 2-D DFT frequency (f_a, f_b).
        power_ratio_2d: Power ratio for the dominant 2-D frequency.
        separability_score: How well neuron(a,b) ≈ f(a)·g(b). 1.0 = perfect.
        trig_identity_r2: R² for sin(f(a+b)) trig identity fit.
        sum_frequency: The frequency f in the trig identity sin(f·(a+b)/N).
        activation_std: Standard deviation of neuron activations over the grid.
    """
    layer: int
    neuron_idx: int
    dominant_freq_2d: Tuple[int, int]
    power_ratio_2d: float
    separability_score: float
    trig_identity_r2: float
    sum_frequency: int
    activation_std: float


@dataclass
class MLPTrigSummary:
    """Summary of MLP trig identity analysis.

    Attributes:
        layer: MLP layer index.
        n_neurons_total: Total neurons in the MLP.
        n_neurons_periodic: Neurons with significant 2-D periodicity.
        n_neurons_separable: Neurons with separability_score > threshold.
        n_neurons_trig_identity: Neurons matching trig identity pattern.
        top_trig_neurons: Top neurons by trig_identity_r2.
        frequency_histogram: Dict mapping sum_frequency → count of neurons.
    """
    layer: int
    n_neurons_total: int
    n_neurons_periodic: int = 0
    n_neurons_separable: int = 0
    n_neurons_trig_identity: int = 0
    top_trig_neurons: List[NeuronTrigResult] = field(default_factory=list)
    frequency_histogram: Dict[int, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Phase 4a: Direction-level cos/sin fitting
# ---------------------------------------------------------------------------

def fit_cosine_sine(
    signal: np.ndarray,
    frequency: int,
) -> Tuple[float, float, float, float, float, float, float]:
    """Fit signal p(n) = A·cos(2πfn/N) + B·sin(2πfn/N) + C.

    Args:
        signal: 1-D array of length N (scalar projections indexed by number).
        frequency: The frequency f to fit.

    Returns:
        (A, B, C, amplitude, phase, r_squared, residual_std)
    """
    N = len(signal)
    n = np.arange(N, dtype=np.float64)
    theta = 2.0 * np.pi * frequency * n / N

    # Design matrix: [cos(θ), sin(θ), 1]
    X = np.column_stack([np.cos(theta), np.sin(theta), np.ones(N)])

    # Least squares fit
    coeffs, residuals, _, _ = np.linalg.lstsq(X, signal, rcond=None)
    A, B, C = coeffs

    amplitude = math.sqrt(A * A + B * B)
    phase = math.atan2(B, A)

    # R² calculation
    fitted = X @ coeffs
    ss_res = np.sum((signal - fitted) ** 2)
    ss_tot = np.sum((signal - np.mean(signal)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    residual_std = float(np.std(signal - fitted))

    return float(A), float(B), float(C), amplitude, phase, r_squared, residual_std


def fit_multi_frequency(
    signal: np.ndarray,
    frequencies: List[int],
) -> float:
    """Fit signal to multiple cos/sin frequencies simultaneously and return R².

    Fits: p(n) = Σ_f [A_f·cos(2πfn/N) + B_f·sin(2πfn/N)] + C

    Args:
        signal: 1-D array of length N.
        frequencies: List of frequency indices to include in the fit.

    Returns:
        R² of the multi-frequency fit.
    """
    if not frequencies:
        return 0.0

    N = len(signal)
    n = np.arange(N, dtype=np.float64)

    # Build design matrix: [cos(θ_f1), sin(θ_f1), cos(θ_f2), sin(θ_f2), ..., 1]
    cols = []
    for f in frequencies:
        theta = 2.0 * np.pi * f * n / N
        cols.append(np.cos(theta))
        cols.append(np.sin(theta))
    cols.append(np.ones(N))

    X = np.column_stack(cols)
    coeffs, _, _, _ = np.linalg.lstsq(X, signal, rcond=None)
    fitted = X @ coeffs

    ss_res = np.sum((signal - fitted) ** 2)
    ss_tot = np.sum((signal - np.mean(signal)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def find_secondary_frequencies(
    signal: np.ndarray,
    primary_freq: int,
    min_ratio: float = 2.0,
    max_secondary: int = 3,
) -> List[Tuple[int, float]]:
    """Find secondary frequency peaks in the signal after removing the primary.

    Args:
        signal: 1-D array of length N.
        primary_freq: The dominant frequency to exclude.
        min_ratio: Minimum power ratio for a secondary peak.
        max_secondary: Maximum number of secondary peaks to return.

    Returns:
        List of (frequency, power_ratio) tuples.
    """
    N = len(signal)
    dft = np.fft.fft(signal)
    n_freqs = N // 2 + 1
    power = np.abs(dft[:n_freqs]) ** 2

    # Zero out DC and primary
    power_masked = power.copy()
    power_masked[0] = 0.0
    if primary_freq < n_freqs:
        power_masked[primary_freq] = 0.0

    results = []
    for _ in range(max_secondary):
        if len(power_masked) <= 1:
            break
        idx = int(np.argmax(power_masked))
        if idx == 0:
            break
        peak_power = power_masked[idx]
        # Baseline: mean of remaining non-zero, non-peak freqs
        others = np.concatenate([power_masked[1:idx], power_masked[idx + 1:]])
        others = others[others > 0]
        baseline = float(np.mean(others)) if len(others) > 0 else 0.0
        ratio = peak_power / baseline if baseline > 1e-12 else 0.0
        if ratio < min_ratio:
            break
        results.append((idx, float(ratio)))
        power_masked[idx] = 0.0

    return results


def analyze_direction_fit(
    circuit: MaskedTransformerCircuit,
    activations: np.ndarray,
    layer: int,
    head: Optional[int],
    component: str,
    direction_idx: int,
    mask_value: float,
    singular_value: float,
    dominant_frequency: int,
    power_ratio: float,
) -> Optional[DirectionFitResult]:
    """Perform cos/sin fit analysis on a single SVD direction.

    Args:
        circuit: MaskedTransformerCircuit with cached SVD.
        activations: Shape (N, d_model) — residual stream activations.
        layer: Layer index.
        head: Head index (None for MLP).
        component: 'OV', 'MLP_in', or 'MLP_out'.
        direction_idx: SVD direction index k.
        mask_value: Learned mask value for direction k.
        singular_value: Singular value σ_k.
        dominant_frequency: Frequency from Phase 3.
        power_ratio: Power ratio from Phase 3.

    Returns:
        DirectionFitResult or None if direction cannot be analyzed.
    """
    k = direction_idx

    # Resolve cache key and extract SVD components
    if component == "OV":
        head_key = f"differential_head_{layer}_{head}"
        cache_key = f"{head_key}_ov"
    elif component == "QK":
        head_key = f"differential_head_{layer}_{head}"
        cache_key = f"{head_key}_qk"
    elif component in ("MLP_in", "MLP_out"):
        suffix = "in" if component == "MLP_in" else "out"
        cache_key = f"mlp_{layer}_{suffix}"
    else:
        logger.warning(f"Unknown component type: {component}")
        return None

    # Handle memory-optimized SVD cache (3-tuples) and MLP lazy-loading from disk
    svd_data = circuit.svd_cache.get(cache_key)
    if svd_data is None and component in ("MLP_in", "MLP_out"):
        suffix = "in" if component == "MLP_in" else "out"
        svd_data = circuit._load_svd_components(layer, -1, f'mlp_{suffix}')
    if svd_data is None:
        logger.warning(f"SVD cache missing for {cache_key}")
        return None

    if len(svd_data) == 4:
        U, S, Vh, _ = svd_data
    else:
        U, S, Vh = svd_data

    # Get the projection vector in d_model residual-stream space.
    # Must match Phase 3 (analyze_svd_directions.py) for consistency:
    #   OV/QK: Vh[k, :] (d_model) — the output direction this SVD component writes
    #   MLP_in: U[:, k] (d_model+1, strip bias) — input direction from residual stream
    #   MLP_out: Vh[k, :] (d_model) — output direction into residual stream
    if component in ("OV", "QK"):
        # W_OV is (d_model+1, d_model), SVD → U (d_model+1, r), Vh (r, d_model)
        # Vh[k, :] is in d_model output space — matches residual stream
        vec = Vh[k, :].cpu().numpy()
    elif component == "MLP_in":
        # W_in_aug is (d_model+1, d_mlp), SVD → U (d_model+1, r), Vh (r, d_mlp)
        # U[:, k] is in d_model+1 input space (strip bias row → d_model)
        vec = U[:, k].cpu().numpy()
    elif component == "MLP_out":
        # W_out_aug is (d_mlp+1, d_model), SVD → U (d_mlp+1, r), Vh (r, d_model)
        # Vh[k, :] is in d_model output space — matches residual stream
        vec = Vh[k, :].cpu().numpy()
    else:
        return None

    # Handle augmented dimensions: strip bias row if present
    if vec.shape[0] == activations.shape[1] + 1:
        vec = vec[1:]
    elif vec.shape[0] != activations.shape[1]:
        logger.warning(
            f"Dimension mismatch for {cache_key} dir[{k}]: "
            f"vec={vec.shape[0]}, act={activations.shape[1]}"
        )
        return None

    # Project activations onto this direction → scalar signal p(n)
    # Ensure float32 for numpy linalg compatibility
    activations = activations.astype(np.float32) if activations.dtype != np.float32 else activations
    vec = vec.astype(np.float32) if vec.dtype != np.float32 else vec
    projections = activations @ vec  # (N,)

    # Fit cos/sin at the dominant frequency
    A, B, C, amplitude, phase, r_sq, res_std = fit_cosine_sine(
        projections, dominant_frequency
    )

    # Find secondary frequency peaks
    secondary = find_secondary_frequencies(
        projections, dominant_frequency, min_ratio=2.0, max_secondary=5
    )

    # Multi-frequency R²: fit primary + all secondary frequencies together
    all_freqs = [dominant_frequency] + [f for f, _ in secondary]
    r_sq_multi = fit_multi_frequency(projections, all_freqs)

    return DirectionFitResult(
        layer=layer,
        head=head,
        component=component,
        direction_idx=k,
        mask_value=mask_value,
        singular_value=singular_value,
        effective_strength=singular_value * mask_value,
        dominant_frequency=dominant_frequency,
        power_ratio=power_ratio,
        cos_coeff=A,
        sin_coeff=B,
        dc_offset=C,
        amplitude=amplitude,
        phase=phase,
        r_squared=r_sq,
        r_squared_multi=r_sq_multi,
        residual_std=res_std,
        secondary_frequencies=secondary,
    )


# ---------------------------------------------------------------------------
# Phase 4a: Cross-direction frequency group analysis
# ---------------------------------------------------------------------------

def analyze_frequency_groups(
    fit_results: List[DirectionFitResult],
    circuit: MaskedTransformerCircuit,
    phase_tolerance: float = 0.3,
) -> List[FrequencyGroupResult]:
    """Group directions by frequency and check for cos/sin pair structure.

    For directions sharing the same frequency f, checks:
    - Phase differences: are any pair ≈ π/2 apart (cos/sin pair)?
    - Input-space cosine similarity of Vh vectors.

    Args:
        fit_results: List of DirectionFitResult from Phase 4a.
        circuit: MaskedTransformerCircuit with cached SVD.
        phase_tolerance: Max deviation from π/2 to flag as cos/sin pair (radians).

    Returns:
        List of FrequencyGroupResult, one per frequency.
    """
    # Group by dominant frequency
    freq_groups: Dict[int, List[DirectionFitResult]] = {}
    for r in fit_results:
        freq_groups.setdefault(r.dominant_frequency, []).append(r)

    results = []
    for freq, directions in sorted(freq_groups.items()):
        group = FrequencyGroupResult(
            frequency=freq,
            n_directions=len(directions),
            directions=directions,
        )

        # Compute pairwise phase differences
        phases = [d.phase for d in directions]
        phase_diffs = []
        has_pair = False

        for i in range(len(phases)):
            for j in range(i + 1, len(phases)):
                diff = abs(phases[i] - phases[j])
                # Normalize to [0, π]
                diff = diff % (2 * math.pi)
                if diff > math.pi:
                    diff = 2 * math.pi - diff
                phase_diffs.append(diff)

                # Check if ≈ π/2 (cos/sin pair)
                if abs(diff - math.pi / 2) < phase_tolerance:
                    has_pair = True

        group.phase_diffs = phase_diffs
        group.has_cos_sin_pair = has_pair

        # Compute input-space cosine similarities between Vh vectors
        input_vecs = []
        for d in directions:
            vec = _extract_input_vector(circuit, d)
            if vec is not None:
                input_vecs.append(vec)

        cosine_sims = []
        for i in range(len(input_vecs)):
            for j in range(i + 1, len(input_vecs)):
                sim = float(np.dot(input_vecs[i], input_vecs[j]) / (
                    np.linalg.norm(input_vecs[i]) * np.linalg.norm(input_vecs[j]) + 1e-12
                ))
                cosine_sims.append(sim)

        group.input_cosine_sims = cosine_sims

        # Effective subspace dimension: count of directions with multi-freq R² > 0.5
        group.subspace_dimension = sum(1 for d in directions if d.r_squared_multi > 0.5)

        results.append(group)

    return results


def _extract_input_vector(
    circuit: MaskedTransformerCircuit,
    result: DirectionFitResult,
) -> Optional[np.ndarray]:
    """Extract the d_model projection vector for a direction (matches Phase 3)."""
    k = result.direction_idx
    component = result.component
    layer = result.layer
    head = result.head

    if component == "OV":
        cache_key = f"differential_head_{layer}_{head}_ov"
    elif component == "MLP_in":
        cache_key = f"mlp_{layer}_in"
    elif component == "MLP_out":
        cache_key = f"mlp_{layer}_out"
    else:
        return None

    svd_data = circuit.svd_cache.get(cache_key)
    if svd_data is None and component in ("MLP_in", "MLP_out"):
        suffix = "in" if component == "MLP_in" else "out"
        svd_data = circuit._load_svd_components(result.layer, -1, f'mlp_{suffix}')
    if svd_data is None:
        return None

    if len(svd_data) == 4:
        U, S, Vh, _ = svd_data
    else:
        U, S, Vh = svd_data

    # Match Phase 3 projection vectors:
    #   OV: Vh[k, :] (d_model output space)
    #   MLP_in: U[:, k] (d_model+1 input space, strip bias)
    #   MLP_out: Vh[k, :] (d_model output space)
    if component == "MLP_in":
        vec = U[:, k].cpu().numpy()
    else:  # OV, MLP_out
        vec = Vh[k, :].cpu().numpy()

    # Strip bias row if augmented (d_model+1 → d_model)
    d_model = circuit.d_model
    if vec.shape[0] == d_model + 1:
        vec = vec[1:]
    return vec


# ---------------------------------------------------------------------------
# Phase 4b: MLP neuron trig identity analysis
# ---------------------------------------------------------------------------

def collect_mlp_hidden_activations(
    model,
    prompts_grid: List[List],
    layer: int,
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    """Collect MLP hidden-layer activations for a 2-D grid of prompts.

    Args:
        model: HookedTransformer model.
        prompts_grid: 2-D list of prompts indexed by [a][b].
        layer: MLP layer to hook.
        device: Torch device.
        batch_size: Batch size for forward passes.

    Returns:
        Array of shape (N_a, N_b, d_mlp) — hidden activations.
    """
    hook_name = f"blocks.{layer}.mlp.hook_post"
    N_a = len(prompts_grid)
    N_b = len(prompts_grid[0])

    # Flatten the grid
    flat_prompts = []
    for a_row in prompts_grid:
        for sample in a_row:
            prompt_str = sample.prompt if hasattr(sample, "prompt") else str(sample)
            flat_prompts.append(prompt_str)

    total = len(flat_prompts)
    all_activations = []

    for start in tqdm(range(0, total, batch_size), desc="MLP activations"):
        batch = flat_prompts[start : start + batch_size]
        tokens = model.to_tokens(batch, prepend_bos=True)
        tokens = tokens.to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=[hook_name],
            )

        # Extract last-token activations
        act = cache[hook_name][:, -1, :].cpu().float().numpy()  # (batch, d_mlp)
        all_activations.append(act)

    all_activations = np.concatenate(all_activations, axis=0)  # (N_a*N_b, d_mlp)
    return all_activations.reshape(N_a, N_b, -1)


def analyze_neuron_trig_identity(
    activations_2d: np.ndarray,
    neuron_idx: int,
    layer: int,
    max_freq: int = 10,
) -> NeuronTrigResult:
    """Analyze a single MLP neuron for trig identity structure.

    Tests whether neuron(a, b) ≈ sin(f·(a+b)/N) via trig identity:
        sin(f(a+b)/N) = sin(fa/N)cos(fb/N) + cos(fa/N)sin(fb/N)

    Args:
        activations_2d: Shape (N_a, N_b) — neuron activations over (a, b) grid.
        neuron_idx: Index of this neuron.
        layer: MLP layer index.
        max_freq: Maximum frequency to search for trig identity fit.

    Returns:
        NeuronTrigResult with fit statistics.
    """
    N_a, N_b = activations_2d.shape
    act_std = float(np.std(activations_2d))

    # 2-D DFT
    dft_2d = np.fft.fft2(activations_2d)
    power_2d = np.abs(dft_2d) ** 2
    # Zero DC
    power_2d[0, 0] = 0.0

    # Find dominant 2-D frequency
    n_fa = N_a // 2 + 1
    n_fb = N_b // 2 + 1
    power_half = power_2d[:n_fa, :n_fb]
    peak_idx = np.unravel_index(np.argmax(power_half), power_half.shape)
    dom_fa, dom_fb = int(peak_idx[0]), int(peak_idx[1])
    peak_power = float(power_half[dom_fa, dom_fb])

    # Power ratio: peak vs mean of non-DC, non-peak
    power_flat = power_half.flatten()
    power_flat[0] = 0.0  # DC
    power_flat[dom_fa * n_fb + dom_fb] = 0.0  # peak
    non_peak = power_flat[power_flat > 0]
    baseline = float(np.mean(non_peak)) if len(non_peak) > 0 else 0.0
    ratio_2d = peak_power / baseline if baseline > 1e-12 else 0.0

    # Separability test: SVD of activations_2d
    # A perfectly separable function f(a)·g(b) has rank 1
    U_sep, S_sep, _ = np.linalg.svd(activations_2d - np.mean(activations_2d))
    total_var = np.sum(S_sep ** 2)
    sep_score = float(S_sep[0] ** 2 / total_var) if total_var > 1e-12 else 0.0

    # Trig identity fit: test sin(f·(a+b)/N) for various f
    best_r2 = -1.0
    best_f = 0
    N = max(N_a, N_b)
    a_grid = np.arange(N_a, dtype=np.float64)
    b_grid = np.arange(N_b, dtype=np.float64)
    A, B_grid = np.meshgrid(a_grid, b_grid, indexing="ij")
    ab_sum = A + B_grid  # (N_a, N_b)

    act_flat = activations_2d.flatten()
    ss_tot = np.sum((act_flat - np.mean(act_flat)) ** 2)

    # Cap at N//2 - 1 to avoid Nyquist frequency artifact
    for f in range(1, min(max_freq + 1, N // 2)):
        theta = 2.0 * np.pi * f * ab_sum / N
        # Design matrix: [cos(f·(a+b)), sin(f·(a+b)), 1]
        X = np.column_stack([
            np.cos(theta).flatten(),
            np.sin(theta).flatten(),
            np.ones(N_a * N_b),
        ])
        coeffs, _, _, _ = np.linalg.lstsq(X, act_flat, rcond=None)
        fitted = X @ coeffs
        ss_res = np.sum((act_flat - fitted) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        if r2 > best_r2:
            best_r2 = r2
            best_f = f

    return NeuronTrigResult(
        layer=layer,
        neuron_idx=neuron_idx,
        dominant_freq_2d=(dom_fa, dom_fb),
        power_ratio_2d=ratio_2d,
        separability_score=sep_score,
        trig_identity_r2=float(best_r2),
        sum_frequency=best_f,
        activation_std=act_std,
    )


def analyze_mlp_trig(
    model,
    circuit: MaskedTransformerCircuit,
    prompt_gen: ArithmeticPromptGenerator,
    layer: int,
    operand_range: range,
    device: torch.device,
    batch_size: int = 16,
    trig_r2_threshold: float = 0.3,
    max_freq: int = 10,
    top_k_neurons: int = 50,
) -> MLPTrigSummary:
    """Run Phase 4b trig identity analysis on one MLP layer.

    Args:
        model: HookedTransformer model.
        circuit: MaskedTransformerCircuit with masks loaded.
        prompt_gen: ArithmeticPromptGenerator for the operand range.
        layer: MLP layer to analyze.
        operand_range: Range of operand values.
        device: Torch device.
        batch_size: Batch size for activation collection.
        trig_r2_threshold: Min R² to flag a neuron as trig-identity-like.
        max_freq: Maximum frequency to test in trig identity fit.
        top_k_neurons: Number of top neurons to include in results.

    Returns:
        MLPTrigSummary with neuron-level analysis.
    """
    # Build the (a, b) prompt grid
    op_list = list(operand_range)
    N = len(op_list)
    logger.info(f"Building {N}x{N} prompt grid for MLP L{layer} trig analysis")

    prompts_grid = []
    for a in op_list:
        row = []
        for b in op_list:
            sample = prompt_gen.get_by_operands(a, b)
            if sample is None:
                raise ValueError(f"Missing prompt for ({a}, {b})")
            row.append(sample)
        prompts_grid.append(row)

    # Collect MLP hidden activations
    logger.info(f"Collecting MLP L{layer} hidden activations ({N}x{N} grid)...")
    hidden_acts = collect_mlp_hidden_activations(
        model, prompts_grid, layer, device, batch_size
    )
    d_mlp = hidden_acts.shape[2]
    logger.info(f"Hidden activations shape: {hidden_acts.shape} (d_mlp={d_mlp})")

    # Pre-filter: only analyze neurons with significant variance
    neuron_stds = np.std(hidden_acts, axis=(0, 1))
    std_threshold = np.percentile(neuron_stds, 50)  # top 50% by variance
    candidate_neurons = np.where(neuron_stds >= std_threshold)[0]
    logger.info(
        f"Analyzing {len(candidate_neurons)}/{d_mlp} neurons "
        f"(std >= {std_threshold:.4f})"
    )

    # Analyze each candidate neuron
    neuron_results = []
    n_periodic = 0
    n_separable = 0
    n_trig = 0
    freq_hist: Dict[int, int] = {}

    for idx in tqdm(candidate_neurons, desc="Analyzing neurons"):
        act_2d = hidden_acts[:, :, idx]  # (N_a, N_b)
        result = analyze_neuron_trig_identity(act_2d, int(idx), layer, max_freq)
        neuron_results.append(result)

        if result.power_ratio_2d >= 3.0:
            n_periodic += 1
        if result.separability_score >= 0.5:
            n_separable += 1
        if result.trig_identity_r2 >= trig_r2_threshold:
            n_trig += 1
            freq_hist[result.sum_frequency] = freq_hist.get(result.sum_frequency, 0) + 1

    # Sort by trig identity R² and take top-k
    neuron_results.sort(key=lambda r: -r.trig_identity_r2)
    top_neurons = neuron_results[:top_k_neurons]

    return MLPTrigSummary(
        layer=layer,
        n_neurons_total=d_mlp,
        n_neurons_periodic=n_periodic,
        n_neurons_separable=n_separable,
        n_neurons_trig_identity=n_trig,
        top_trig_neurons=top_neurons,
        frequency_histogram=freq_hist,
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def results_to_dict(
    fit_results: List[DirectionFitResult],
    freq_groups: List[FrequencyGroupResult],
    mlp_summaries: List[MLPTrigSummary],
) -> dict:
    """Convert all Phase 4 results to a JSON-serializable dict."""
    return {
        "direction_fits": [
            {
                "layer": r.layer,
                "head": r.head,
                "component": r.component,
                "direction_idx": r.direction_idx,
                "mask_value": r.mask_value,
                "singular_value": r.singular_value,
                "effective_strength": r.effective_strength,
                "dominant_frequency": r.dominant_frequency,
                "power_ratio": r.power_ratio,
                "cos_coeff": r.cos_coeff,
                "sin_coeff": r.sin_coeff,
                "dc_offset": r.dc_offset,
                "amplitude": r.amplitude,
                "phase": r.phase,
                "r_squared": r.r_squared,
                "r_squared_multi": r.r_squared_multi,
                "residual_std": r.residual_std,
                "secondary_frequencies": [
                    {"freq": f, "ratio": rat} for f, rat in r.secondary_frequencies
                ],
            }
            for r in fit_results
        ],
        "frequency_groups": [
            {
                "frequency": g.frequency,
                "n_directions": g.n_directions,
                "phase_diffs": g.phase_diffs,
                "has_cos_sin_pair": g.has_cos_sin_pair,
                "subspace_dimension": g.subspace_dimension,
                "input_cosine_sims": g.input_cosine_sims,
                "direction_labels": [
                    f"L{d.layer}H{d.head}_{d.component}_dir{d.direction_idx}"
                    if d.head is not None
                    else f"MLP_L{d.layer}_{d.component}_dir{d.direction_idx}"
                    for d in g.directions
                ],
            }
            for g in freq_groups
        ],
        "mlp_trig_summaries": [
            {
                "layer": m.layer,
                "n_neurons_total": m.n_neurons_total,
                "n_neurons_periodic": m.n_neurons_periodic,
                "n_neurons_separable": m.n_neurons_separable,
                "n_neurons_trig_identity": m.n_neurons_trig_identity,
                "frequency_histogram": {
                    str(k): v for k, v in m.frequency_histogram.items()
                },
                "top_trig_neurons": [
                    {
                        "neuron_idx": n.neuron_idx,
                        "dominant_freq_2d": list(n.dominant_freq_2d),
                        "power_ratio_2d": n.power_ratio_2d,
                        "separability_score": n.separability_score,
                        "trig_identity_r2": n.trig_identity_r2,
                        "sum_frequency": n.sum_frequency,
                        "activation_std": n.activation_std,
                    }
                    for n in m.top_trig_neurons
                ],
            }
            for m in mlp_summaries
        ],
    }


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------

def print_phase4_summary(
    fit_results: List[DirectionFitResult],
    freq_groups: List[FrequencyGroupResult],
    mlp_summaries: List[MLPTrigSummary],
) -> str:
    """Generate a human-readable Phase 4 summary."""
    lines = []
    lines.append("=" * 80)
    lines.append("PHASE 4: WEIGHT-LEVEL FOURIER CIRCUIT ANALYSIS")
    lines.append("=" * 80)
    lines.append("")

    # Phase 4a: Direction fits
    lines.append("--- Phase 4a: Cos/Sin Fits ---")
    lines.append("")

    # Sort by effective strength
    sorted_fits = sorted(fit_results, key=lambda r: -r.effective_strength)
    header = "{:>12s}  {:>5s}  {:>6s}  {:>5s}  {:>7s}  {:>7s}  {:>6s}  {:>7s}  {:>7s}  {}".format(
        "Location", "Dir", "Mask", "Freq", "R²(1)", "R²(M)", "Amp", "Phase", "Str", "Secondary"
    )
    lines.append(header)
    lines.append("-" * 100)

    for r in sorted_fits:
        label = "L{}H{}".format(r.layer, r.head) if r.head is not None else "MLP_L{}".format(r.layer)
        loc = "{} {}".format(label, r.component)
        sec_str = ", ".join(
            "f{}({:.1f}x)".format(f, rat) for f, rat in r.secondary_frequencies
        )
        lines.append(
            "{:>12s}  {:>5d}  {:>6.3f}  {:>5d}  {:>7.4f}  {:>7.4f}  {:>6.3f}  {:>7.3f}  {:>7.2f}  {}".format(
                loc, r.direction_idx, r.mask_value, r.dominant_frequency,
                r.r_squared, r.r_squared_multi, r.amplitude, r.phase,
                r.effective_strength, sec_str,
            )
        )

    lines.append("")

    # Phase 4a: Frequency groups
    lines.append("--- Phase 4a: Frequency Groups ---")
    lines.append("")
    for g in sorted(freq_groups, key=lambda g: g.frequency):
        pair_str = "YES ✓" if g.has_cos_sin_pair else "no"
        lines.append(
            "  freq={}: {} directions, cos/sin pair: {}, subspace_dim: {:.0f}".format(
                g.frequency, g.n_directions, pair_str, g.subspace_dimension,
            )
        )
        if g.phase_diffs:
            lines.append(
                "    phase diffs: {}".format(
                    ", ".join("{:.3f}".format(d) for d in g.phase_diffs)
                )
            )
        if g.input_cosine_sims:
            lines.append(
                "    Vh cosine sims: {}".format(
                    ", ".join("{:.3f}".format(s) for s in g.input_cosine_sims)
                )
            )

    lines.append("")

    # Phase 4b: MLP trig analysis
    if mlp_summaries:
        lines.append("--- Phase 4b: MLP Neuron Trig Identity Analysis ---")
        lines.append("")
        for m in mlp_summaries:
            lines.append("  MLP L{}: {}/{} neurons analyzed".format(
                m.layer, m.n_neurons_periodic + m.n_neurons_separable, m.n_neurons_total
            ))
            lines.append("    Periodic (2D DFT): {}".format(m.n_neurons_periodic))
            lines.append("    Separable f(a)·g(b): {}".format(m.n_neurons_separable))
            lines.append("    Trig identity sin(f(a+b)): {}".format(m.n_neurons_trig_identity))
            if m.frequency_histogram:
                lines.append("    Frequency histogram: {}".format(
                    ", ".join(
                        "f{}:{}".format(k, v) for k, v in sorted(m.frequency_histogram.items())
                    )
                ))
            lines.append("")
            lines.append("    Top trig-identity neurons:")
            for n in m.top_trig_neurons[:15]:
                lines.append(
                    "      neuron[{:4d}]: R²={:.4f}  sum_freq={}  "
                    "sep={:.3f}  2d_peak=({},{})  2d_ratio={:.1f}x".format(
                        n.neuron_idx, n.trig_identity_r2, n.sum_frequency,
                        n.separability_score, n.dominant_freq_2d[0],
                        n.dominant_freq_2d[1], n.power_ratio_2d,
                    )
                )

    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def load_phase3_results(path: str, mask_threshold: float = 0.3) -> List[dict]:
    """Load Phase 3 JSON results and extract surviving directions."""
    with open(path) as f:
        data = json.load(f)

    survivors = []
    for r in data.get("results", []):
        for d in r.get("directions", []):
            if d["mask_value"] > mask_threshold:
                survivors.append({
                    "layer": r["layer"],
                    "head": r["head"],
                    "component": r["component"],
                    "direction_idx": d["direction_idx"],
                    "mask_value": d["mask_value"],
                    "singular_value": d["singular_value"],
                    "effective_strength": d["effective_strength"],
                    "dominant_frequency": d["dominant_frequency"],
                    "power_ratio": d["power_ratio"],
                })
    return survivors


def run_analysis(
    config: dict,
    checkpoint_path: str,
    phase3_path: str,
    operand_range_end: Optional[int] = None,
    device_str: Optional[str] = None,
    output_dir: str = "phase4_results",
    trig_r2_threshold: float = 0.3,
    max_freq: int = 10,
    batch_size: int = 16,
    mask_threshold: float = 0.3,
) -> Tuple[List[DirectionFitResult], List[FrequencyGroupResult], List[MLPTrigSummary]]:
    """Run the full Phase 4 analysis.

    Args:
        config: Loaded YAML config dict.
        checkpoint_path: Path to Phase 2 checkpoint.
        phase3_path: Path to Phase 3 results JSON.
        operand_range_end: Override operand range end from config.
        device_str: Device override.
        output_dir: Where to save results.
        trig_r2_threshold: Min R² for trig identity neurons.
        max_freq: Max frequency for trig identity search.
        batch_size: Batch size for forward passes.

    Returns:
        Tuple of (fit_results, freq_groups, mlp_summaries).
    """
    from transformer_lens import HookedTransformer

    # Resolve device
    if device_str is None:
        device_str = config.get("training", {}).get("device", None)
    if device_str is None:
        if torch.cuda.is_available():
            device_str = "cuda"
        elif torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    # Load Phase 3 results
    logger.info(f"Loading Phase 3 results from {phase3_path}")
    survivors = load_phase3_results(phase3_path, mask_threshold=mask_threshold)
    logger.info(f"Found {len(survivors)} surviving directions (mask > {mask_threshold})")

    # Load model
    logger.info(f"Loading model: {config['model']['name']}")
    dtype_str = config.get("model", {}).get("dtype", None)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map.get(dtype_str, None)
    cache_dir = os.path.expanduser(config["model"]["pretrained_cache_dir"])
    model = HookedTransformer.from_pretrained(
        config["model"]["name"],
        cache_dir=cache_dir,
        dtype=dtype,
    )
    model = model.to(device)
    model.eval()

    # Parse train_masks
    train_masks = config.get("train_masks", None)

    # Initialize circuit with masks
    logger.info("Initializing MaskedTransformerCircuit...")
    circuit = MaskedTransformerCircuit(
        model=model,
        device=device,
        cache_svd=config["masking"]["cache_svd"],
        mask_init_value=config["masking"]["mask_init_value"],
        train_masks=train_masks,
    )

    # Load trained masks from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    for key in checkpoint.get("ov_masks", {}):
        if key in circuit.ov_masks:
            circuit.ov_masks[key].data = checkpoint["ov_masks"][key].to(device)
    for key in checkpoint.get("qk_masks", {}):
        if key in circuit.qk_masks:
            circuit.qk_masks[key].data = checkpoint["qk_masks"][key].to(device)
    if circuit.mask_mlp:
        for key in checkpoint.get("mlp_in_masks", {}):
            if key in circuit.mlp_in_masks:
                circuit.mlp_in_masks[key].data = checkpoint["mlp_in_masks"][key].to(device)
        for key in checkpoint.get("mlp_out_masks", {}):
            if key in circuit.mlp_out_masks:
                circuit.mlp_out_masks[key].data = checkpoint["mlp_out_masks"][key].to(device)
    logger.info("Masks loaded successfully")

    # Set up operand range and prompts
    arith_cfg = config["arithmetic"]
    range_end = operand_range_end or arith_cfg.get("operand_range_end", 10)
    op_range = range(arith_cfg.get("operand_range_start", 0), range_end)
    prompt_gen = ArithmeticPromptGenerator(
        operand_range=op_range,
        operation=arith_cfg.get("operation", "add"),
        prompt_template=arith_cfg.get("prompt_template", "{a} + {b} ="),
        shuffle=False,
    )

    # Collect activations for n+0 prompts (same as Phase 3)
    fourier_prompts = []
    for n in op_range:
        sample = prompt_gen.get_by_operands(n, 0)
        if sample is None:
            raise ValueError(f"Missing prompt for ({n}, 0)")
        fourier_prompts.append(sample)

    active_layers = sorted(set(s["layer"] for s in survivors))
    logger.info(f"Collecting activations for layers: {active_layers}")

    # Collect residual stream activations
    prompt_strings = [
        s.prompt if hasattr(s, "prompt") else str(s) for s in fourier_prompts
    ]
    all_activations: Dict[str, np.ndarray] = {}

    for start in tqdm(
        range(0, len(prompt_strings), batch_size), desc="Collecting activations"
    ):
        batch = prompt_strings[start : start + batch_size]
        tokens = model.to_tokens(batch, prepend_bos=True)
        tokens = tokens.to(device)

        hooks = [f"blocks.{l}.hook_resid_pre" for l in active_layers]
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hooks)

        for hook in hooks:
            act = cache[hook][:, -1, :].cpu()  # (batch, d_model)
            if hook not in all_activations:
                all_activations[hook] = act
            else:
                all_activations[hook] = torch.cat(
                    [all_activations[hook], act], dim=0
                )

    # Convert to numpy
    for key in all_activations:
        if isinstance(all_activations[key], torch.Tensor):
            all_activations[key] = all_activations[key].numpy()

    # ---------------------------------------------------------------
    # Phase 4a: Cos/sin fits for each surviving direction
    # ---------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("PHASE 4a: COS/SIN DIRECTION FITS")
    logger.info("=" * 70)

    fit_results: List[DirectionFitResult] = []

    for surv in tqdm(survivors, desc="Fitting directions"):
        hook = f"blocks.{surv['layer']}.hook_resid_pre"
        if hook not in all_activations:
            continue

        act_np = all_activations[hook]
        result = analyze_direction_fit(
            circuit=circuit,
            activations=act_np,
            layer=surv["layer"],
            head=surv["head"],
            component=surv["component"],
            direction_idx=surv["direction_idx"],
            mask_value=surv["mask_value"],
            singular_value=surv["singular_value"],
            dominant_frequency=surv["dominant_frequency"],
            power_ratio=surv["power_ratio"],
        )
        if result is not None:
            fit_results.append(result)

    logger.info(f"Fitted {len(fit_results)} directions")

    # Frequency group analysis
    logger.info("Analyzing frequency groups...")
    freq_groups = analyze_frequency_groups(fit_results, circuit)

    # ---------------------------------------------------------------
    # Phase 4b: MLP trig identity analysis
    # ---------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("PHASE 4b: MLP NEURON TRIG IDENTITY ANALYSIS")
    logger.info("=" * 70)

    mlp_summaries: List[MLPTrigSummary] = []

    # Find surviving MLP layers
    mlp_layers = sorted(set(
        s["layer"] for s in survivors
        if s["component"] in ("MLP_in", "MLP_out")
    ))

    # Use a smaller operand range for 4b (N²  prompts → can be expensive)
    mlp_op_range_end = min(range_end, 20)
    mlp_op_range = range(arith_cfg.get("operand_range_start", 0), mlp_op_range_end)
    mlp_prompt_gen = ArithmeticPromptGenerator(
        operand_range=mlp_op_range,
        operation=arith_cfg.get("operation", "add"),
        prompt_template=arith_cfg.get("prompt_template", "{a} + {b} ="),
        shuffle=False,
    )
    logger.info(f"MLP trig analysis operand range: 0-{mlp_op_range_end - 1} ({len(mlp_op_range)}x{len(mlp_op_range)} grid)")

    for layer in mlp_layers:
        logger.info(f"Analyzing MLP L{layer}...")
        summary = analyze_mlp_trig(
            model=model,
            circuit=circuit,
            prompt_gen=mlp_prompt_gen,
            layer=layer,
            operand_range=mlp_op_range,
            device=device,
            batch_size=batch_size,
            trig_r2_threshold=trig_r2_threshold,
            max_freq=max_freq,
        )
        mlp_summaries.append(summary)

    # ---------------------------------------------------------------
    # Print and save results
    # ---------------------------------------------------------------
    summary_text = print_phase4_summary(fit_results, freq_groups, mlp_summaries)
    logger.info("\n" + summary_text)

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        output_dir, f"fourier_circuit_analysis_{timestamp}.json"
    )

    output_data = {
        "metadata": {
            "phase": "4",
            "timestamp": timestamp,
            "checkpoint": checkpoint_path,
            "phase3_results": phase3_path,
            "operand_range_end": range_end,
            "mlp_operand_range_end": mlp_op_range_end,
            "n_survivors": len(survivors),
            "n_fitted": len(fit_results),
        },
        "results": results_to_dict(fit_results, freq_groups, mlp_summaries),
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    return fit_results, freq_groups, mlp_summaries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 4: Weight-Level Fourier Circuit Analysis"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to Phase 2 checkpoint (.pt file)",
    )
    parser.add_argument(
        "--phase3_results", type=str, required=True,
        help="Path to Phase 3 results JSON file",
    )
    parser.add_argument(
        "--operand_range_end", type=int, default=None,
        help="Override operand_range_end from config (e.g. 100 for range 0-99)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device override (e.g. 'mps', 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--output_dir", type=str, default="phase4_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--trig_r2_threshold", type=float, default=0.3,
        help="Min R² for a neuron to be flagged as trig-identity (default: 0.3)",
    )
    parser.add_argument(
        "--max_freq", type=int, default=10,
        help="Max frequency to test in MLP trig identity fit (default: 10)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for forward passes (default: 16)",
    )
    parser.add_argument(
        "--mask_threshold", type=float, default=0.3,
        help="Min mask value for a direction to be included (default: 0.3)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    logger.info("=" * 70)
    logger.info("PHASE 4: WEIGHT-LEVEL FOURIER CIRCUIT ANALYSIS")
    logger.info("=" * 70)

    fit_results, freq_groups, mlp_summaries = run_analysis(
        config=config,
        checkpoint_path=args.checkpoint,
        phase3_path=args.phase3_results,
        operand_range_end=args.operand_range_end,
        device_str=args.device,
        output_dir=args.output_dir,
        trig_r2_threshold=args.trig_r2_threshold,
        max_freq=args.max_freq,
        batch_size=args.batch_size,
        mask_threshold=args.mask_threshold,
    )

    # Final summary
    n_good_fit = sum(1 for r in fit_results if r.r_squared_multi >= 0.5)
    n_cos_sin_pairs = sum(1 for g in freq_groups if g.has_cos_sin_pair)
    n_trig_neurons = sum(m.n_neurons_trig_identity for m in mlp_summaries)

    logger.info("=" * 70)
    logger.info(
        f"PHASE 4 COMPLETE: {n_good_fit}/{len(fit_results)} directions with multi-freq R²≥0.5, "
        f"{n_cos_sin_pairs} cos/sin pairs, {n_trig_neurons} trig-identity neurons"
    )
    logger.info("=" * 70)

    return fit_results, freq_groups, mlp_summaries


if __name__ == "__main__":
    main()
