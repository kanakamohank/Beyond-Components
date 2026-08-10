#!/usr/bin/env python
"""
Phase 5: Causal Validation via Direction-Level Scalar Swapping.

For each important SVD direction from Phase 3/4, this script:
  1. Runs clean prompt (a + b =) through the model
  2. Runs corrupted prompt (a' + b = or a + b' =) through the model
  3. Patches the scalar component along the direction from clean → corrupted
  4. Measures how much the model recovers the correct (clean) answer

Experiments:
  A) Individual direction patching — per-direction logit recovery
  B) Cumulative top-K patching — top 5/10/15/all directions together
  C) Per-frequency group patching
  D) A-corruption vs B-corruption comparison
  E) Subspace patching — patch the full periodic subspace at once

Success criteria (from plan):
  - Scalar swapping of top-5 directions recovers >50% of logit difference

Usage:
    python experiments/causal_validation.py \\
        --config configs/arithmetic_pythia_config.yaml \\
        --checkpoint svd_logs/.../model_final.pt \\
        --phase3_results phase3_results/svd_direction_fourier_....json \\
        --output_dir phase5_results
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.arithmetic_dataset import ArithmeticPromptGenerator, ArithmeticSample
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
class DirectionSpec:
    """Specification for one SVD direction to patch.

    Attributes:
        layer: Transformer layer index.
        head: Attention head index (None for MLP).
        component: 'OV', 'MLP_in', or 'MLP_out'.
        direction_idx: SVD direction index k.
        mask_value: Learned mask value.
        singular_value: Singular value σ_k.
        effective_strength: σ_k × mask.
        dominant_frequency: Best Fourier frequency from Phase 3.
        hook_name: TransformerLens hook point for patching.
        vec: d_model projection vector (Vh[k,:] for OV/MLP_out, U[:,k] stripped for MLP_in).
    """
    layer: int
    head: Optional[int]
    component: str
    direction_idx: int
    mask_value: float
    singular_value: float
    effective_strength: float
    dominant_frequency: int
    hook_name: str
    vec: np.ndarray  # (d_model,) unit direction in residual stream


@dataclass
class PairResult:
    """Result of patching one clean/corrupted pair for one direction set.

    Attributes:
        clean_a: Clean operand a.
        clean_b: Clean operand b.
        corrupt_a: Corrupted operand a.
        corrupt_b: Corrupted operand b.
        clean_answer: Correct answer (a + b).
        corrupt_answer: Corrupted answer (a' + b or a + b').
        logit_clean: Logit of clean answer under clean run.
        logit_corrupt: Logit of clean answer under corrupted run.
        logit_patched: Logit of clean answer under patched run.
        logit_recovery: Fraction of logit diff recovered.
        clean_rank: Rank of clean answer token in clean logits.
        corrupt_rank: Rank of clean answer token in corrupted logits.
        patched_rank: Rank of clean answer token in patched logits.
    """
    clean_a: int
    clean_b: int
    corrupt_a: int
    corrupt_b: int
    clean_answer: int
    corrupt_answer: int
    logit_clean: float
    logit_corrupt: float
    logit_patched: float
    logit_recovery: float
    clean_rank: int
    corrupt_rank: int
    patched_rank: int


@dataclass
class DirectionPatchResult:
    """Aggregate result of patching a single direction across all pairs.

    Attributes:
        direction_label: Human-readable label (e.g. 'L5H5_OV_dir0').
        layer: Layer index.
        head: Head index (None for MLP).
        component: Component type.
        direction_idx: SVD direction index.
        effective_strength: σ_k × mask.
        dominant_frequency: Phase 3 frequency.
        mean_logit_recovery: Mean logit recovery across pairs.
        median_logit_recovery: Median logit recovery.
        std_logit_recovery: Std of logit recovery.
        mean_rank_improvement: Mean improvement in answer rank.
        n_pairs: Number of valid pairs tested.
    """
    direction_label: str
    layer: int
    head: Optional[int]
    component: str
    direction_idx: int
    effective_strength: float
    dominant_frequency: int
    mean_logit_recovery: float
    median_logit_recovery: float
    std_logit_recovery: float
    mean_rank_improvement: float
    n_pairs: int


@dataclass
class CumulativePatchResult:
    """Result of patching top-K directions cumulatively.

    Attributes:
        k: Number of directions patched.
        direction_labels: Labels of directions included.
        mean_logit_recovery: Mean logit recovery across pairs.
        median_logit_recovery: Median logit recovery.
        std_logit_recovery: Std.
        mean_rank_improvement: Mean rank improvement.
        n_pairs: Number of valid pairs.
    """
    k: int
    direction_labels: List[str]
    mean_logit_recovery: float
    median_logit_recovery: float
    std_logit_recovery: float
    mean_rank_improvement: float
    n_pairs: int


# ---------------------------------------------------------------------------
# Clean/corrupted pair generation
# ---------------------------------------------------------------------------

def generate_corruption_pairs(
    operand_range: range,
    corruption_type: str = "A",
    n_pairs: int = 100,
    seed: int = 42,
) -> List[Tuple[ArithmeticSample, ArithmeticSample]]:
    """Generate clean/corrupted arithmetic prompt pairs.

    Args:
        operand_range: Range of operand values.
        corruption_type: 'A' (change first operand) or 'B' (change second).
        n_pairs: Maximum number of pairs to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of (clean_sample, corrupted_sample) tuples where the answer differs.
    """
    rng = np.random.RandomState(seed)
    ops = list(operand_range)

    gen = ArithmeticPromptGenerator(
        operand_range=operand_range, operation="add",
        prompt_template="{a} + {b} =", shuffle=False,
    )

    pairs = []
    attempts = 0
    max_attempts = n_pairs * 10

    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        a = rng.choice(ops)
        b = rng.choice(ops)

        if corruption_type == "A":
            # Change operand a → a'
            candidates = [x for x in ops if x != a and (x + b) != (a + b)]
            if not candidates:
                continue
            a_prime = rng.choice(candidates)
            b_prime = b
        else:
            # Change operand b → b'
            candidates = [x for x in ops if x != b and (a + x) != (a + b)]
            if not candidates:
                continue
            a_prime = a
            b_prime = rng.choice(candidates)

        clean = gen.get_by_operands(a, b)
        corrupt = gen.get_by_operands(a_prime, b_prime)

        if clean is not None and corrupt is not None and clean.answer != corrupt.answer:
            pairs.append((clean, corrupt))

    return pairs


# ---------------------------------------------------------------------------
# Direction extraction
# ---------------------------------------------------------------------------

def extract_direction_specs(
    circuit: MaskedTransformerCircuit,
    survivors: List[dict],
) -> List[DirectionSpec]:
    """Convert Phase 3 survivor dicts into DirectionSpec objects with vectors.

    Args:
        circuit: Circuit with loaded SVD cache.
        survivors: List of Phase 3 survivor dicts.

    Returns:
        List of DirectionSpec objects, sorted by effective_strength descending.
    """
    specs = []

    for s in survivors:
        layer = s["layer"]
        head = s.get("head")
        component = s["component"]
        k = s["direction_idx"]

        # Resolve SVD cache key
        if component == "OV":
            cache_key = f"differential_head_{layer}_{head}_ov"
        elif component == "MLP_in":
            cache_key = f"mlp_{layer}_in"
        elif component == "MLP_out":
            cache_key = f"mlp_{layer}_out"
        else:
            continue

        # Handle memory-optimized SVD cache (3-tuples) and MLP lazy-loading
        svd_data = circuit.svd_cache.get(cache_key)
        if svd_data is None and component in ("MLP_in", "MLP_out"):
            suffix = "in" if component == "MLP_in" else "out"
            svd_data = circuit._load_svd_components(layer, -1, f'mlp_{suffix}')
        if svd_data is None:
            logger.warning(f"SVD cache missing for {cache_key}, skipping")
            continue

        if len(svd_data) == 4:
            U, S, Vh, _ = svd_data
        else:
            U, S, Vh = svd_data

        # Extract projection vector (matches Phase 3/4 conventions)
        if component in ("OV", "MLP_out"):
            vec = Vh[k, :].cpu().numpy()  # d_model output space
        elif component == "MLP_in":
            vec_raw = U[:, k].cpu().numpy()  # d_model+1 input space
            vec = vec_raw[1:]  # strip bias row
        else:
            continue

        # Determine hook point for patching
        # hook_attn_out: attention output in d_model space (batch, seq, d_model)
        # hook_mlp_out: MLP output in d_model space (batch, seq, d_model)
        # Note: hook_attn_out is the sum of all heads.  Since each head's
        # Vh[k,:] is unique (Vh cosine sims ≈ 0), patching the sum along
        # Vh[k,:] is approximately equivalent to patching just that head.
        if component == "OV":
            hook_name = f"blocks.{layer}.hook_attn_out"
        else:  # MLP_in or MLP_out
            hook_name = f"blocks.{layer}.hook_mlp_out"

        mask_value = s["mask_value"]
        sv = float(S[k].cpu())

        label = (
            f"L{layer}H{head}_{component}_dir{k}"
            if head is not None
            else f"MLP_L{layer}_{component}_dir{k}"
        )

        specs.append(DirectionSpec(
            layer=layer,
            head=head,
            component=component,
            direction_idx=k,
            mask_value=mask_value,
            singular_value=sv,
            effective_strength=sv * mask_value,
            dominant_frequency=s.get("dominant_frequency", 0),
            hook_name=hook_name,
            vec=vec,
        ))

    # Sort by effective strength descending
    specs.sort(key=lambda d: -d.effective_strength)
    return specs


# ---------------------------------------------------------------------------
# Core patching logic
# ---------------------------------------------------------------------------

def run_patching_experiment(
    model: HookedTransformer,
    pairs: List[Tuple[ArithmeticSample, ArithmeticSample]],
    specs_to_patch: List[DirectionSpec],
    device: torch.device,
    batch_size: int = 16,
) -> List[PairResult]:
    """Run scalar swapping for a set of directions across all pairs.

    For each pair, patches ALL specified directions simultaneously,
    then measures logit recovery.  OV directions are patched per-head
    via ``attn.hook_result``; MLP directions via ``hook_mlp_out``.

    Args:
        model: HookedTransformer model.
        pairs: List of (clean, corrupted) ArithmeticSample pairs.
        specs_to_patch: List of DirectionSpec objects to patch.
        device: Torch device.
        batch_size: Batch size for model runs.

    Returns:
        List of PairResult objects.
    """
    if not specs_to_patch:
        return []

    results = []

    # Collect all hook names we need caches for
    all_hooks = sorted(set(s.hook_name for s in specs_to_patch))

    # Group specs by hook
    hook_specs: Dict[str, List[DirectionSpec]] = {}
    for s in specs_to_patch:
        hook_specs.setdefault(s.hook_name, []).append(s)

    for start_idx in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start_idx : start_idx + batch_size]
        bs = len(batch_pairs)

        clean_prompts = [p[0].prompt for p in batch_pairs]
        corrupt_prompts = [p[1].prompt for p in batch_pairs]

        # Tokenize
        clean_tokens = model.to_tokens(clean_prompts, prepend_bos=True).to(device)
        corrupt_tokens = model.to_tokens(corrupt_prompts, prepend_bos=True).to(device)

        # Get answer token IDs — use first token of encoded answer
        # (to_single_token fails for some tokenizers that split " 7" into multiple tokens)
        def _answer_token_id(answer_str):
            toks = model.tokenizer.encode(answer_str, add_special_tokens=False)
            return toks[0]

        clean_answer_strs = [p[0].answer_str for p in batch_pairs]
        clean_answer_ids = torch.tensor(
            [_answer_token_id(s) for s in clean_answer_strs],
            device=device,
        )

        # Run clean and corrupted to get caches
        with torch.no_grad():
            _, clean_cache = model.run_with_cache(
                clean_tokens, names_filter=all_hooks
            )

        # Last token position (all prompts have same length for arithmetic)
        seq_pos = clean_tokens.shape[1] - 1

        # Compute clean and corrupted logits at last position
        with torch.no_grad():
            clean_logits = model(clean_tokens)[:, seq_pos, :]  # (bs, vocab)
            corrupt_logits = model(corrupt_tokens)[:, seq_pos, :]  # (bs, vocab)

        # Build patching hooks
        def make_patch_hook(hook_name, hook_spec_list):
            """Create hook that replaces direction components with clean values.

            Both hook_attn_out and hook_mlp_out have shape (batch, seq, d_model).
            Uses original corrupt activation for all projections (no interference).
            """
            # Pre-compute normalized direction tensors
            vecs_t = []
            for sp in hook_spec_list:
                v = torch.tensor(sp.vec, dtype=torch.float32, device=device)
                v = v / (torch.norm(v) + 1e-12)
                vecs_t.append(v)

            clean_act_full = clean_cache[hook_name]  # (bs, seq, d_model)

            def hook_fn(value, hook):
                # value: (batch, seq, d_model) — may be float16 on MPS
                act_dtype = value.dtype
                original = value[:, seq_pos, :].clone().float()
                clean_act = clean_act_full[:, seq_pos, :].float()
                for v_norm in vecs_t:
                    s_clean = (clean_act @ v_norm).unsqueeze(-1)  # (bs, 1)
                    s_corrupt = (original @ v_norm).unsqueeze(-1)  # (bs, 1)
                    delta = (s_clean - s_corrupt) * v_norm.unsqueeze(0)  # (bs, d_model)
                    value[:, seq_pos, :] = value[:, seq_pos, :] + delta.to(act_dtype)
                return value

            return hook_fn

        hooks_list = [
            (hook_name, make_patch_hook(hook_name, hook_specs[hook_name]))
            for hook_name in all_hooks
        ]

        with torch.no_grad():
            patched_logits = model.run_with_hooks(
                corrupt_tokens,
                fwd_hooks=hooks_list,
            )[:, seq_pos, :]  # (bs, vocab)

        # Compute per-pair metrics
        batch_indices = torch.arange(bs, device=device)

        logit_clean_vals = clean_logits[batch_indices, clean_answer_ids].detach().cpu().numpy()
        logit_corrupt_vals = corrupt_logits[batch_indices, clean_answer_ids].detach().cpu().numpy()
        logit_patched_vals = patched_logits[batch_indices, clean_answer_ids].detach().cpu().numpy()

        # Ranks (0-indexed, lower is better)
        clean_ranks = (clean_logits > clean_logits[batch_indices, clean_answer_ids].unsqueeze(-1)).sum(-1).detach().cpu().numpy()
        corrupt_ranks = (corrupt_logits > corrupt_logits[batch_indices, clean_answer_ids].unsqueeze(-1)).sum(-1).detach().cpu().numpy()
        patched_ranks = (patched_logits > patched_logits[batch_indices, clean_answer_ids].unsqueeze(-1)).sum(-1).detach().cpu().numpy()

        for i, (clean_s, corrupt_s) in enumerate(batch_pairs):
            denom = logit_clean_vals[i] - logit_corrupt_vals[i]
            recovery = (
                (logit_patched_vals[i] - logit_corrupt_vals[i]) / denom
                if abs(denom) > 1e-6
                else 0.0
            )

            results.append(PairResult(
                clean_a=clean_s.operand_a,
                clean_b=clean_s.operand_b,
                corrupt_a=corrupt_s.operand_a,
                corrupt_b=corrupt_s.operand_b,
                clean_answer=clean_s.answer,
                corrupt_answer=corrupt_s.answer,
                logit_clean=float(logit_clean_vals[i]),
                logit_corrupt=float(logit_corrupt_vals[i]),
                logit_patched=float(logit_patched_vals[i]),
                logit_recovery=float(recovery),
                clean_rank=int(clean_ranks[i]),
                corrupt_rank=int(corrupt_ranks[i]),
                patched_rank=int(patched_ranks[i]),
            ))

    return results


def aggregate_pair_results(pair_results: List[PairResult], min_logit_diff: float = 0.1) -> dict:
    """Compute aggregate statistics from a list of PairResult.

    Args:
        pair_results: Results from patching experiment.
        min_logit_diff: Filter pairs where |logit_clean - logit_corrupt| < this.
            Prevents degenerate denominators from dominating the mean.
    """
    if not pair_results:
        return {
            "mean_logit_recovery": 0.0,
            "median_logit_recovery": 0.0,
            "std_logit_recovery": 0.0,
            "trimmed_mean_recovery": 0.0,
            "mean_rank_improvement": 0.0,
            "n_pairs": 0,
            "n_pairs_filtered": 0,
        }

    # Filter degenerate pairs
    valid = [r for r in pair_results
             if abs(r.logit_clean - r.logit_corrupt) >= min_logit_diff]

    if not valid:
        valid = pair_results  # fallback: use all if none pass filter

    recoveries = [r.logit_recovery for r in valid]
    rank_improvements = [r.corrupt_rank - r.patched_rank for r in valid]

    # Trimmed mean: drop top/bottom 10%
    sorted_rec = sorted(recoveries)
    trim = max(1, len(sorted_rec) // 10)
    trimmed = sorted_rec[trim:-trim] if len(sorted_rec) > 2 * trim else sorted_rec

    return {
        "mean_logit_recovery": float(np.mean(recoveries)),
        "median_logit_recovery": float(np.median(recoveries)),
        "std_logit_recovery": float(np.std(recoveries)),
        "trimmed_mean_recovery": float(np.mean(trimmed)),
        "mean_rank_improvement": float(np.mean(rank_improvements)),
        "n_pairs": len(pair_results),
        "n_pairs_filtered": len(valid),
    }


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_individual_direction_experiment(
    model: HookedTransformer,
    pairs: List[Tuple[ArithmeticSample, ArithmeticSample]],
    specs: List[DirectionSpec],
    device: torch.device,
    batch_size: int = 16,
) -> List[DirectionPatchResult]:
    """Experiment A: Patch each direction individually, measure its isolated effect.

    Args:
        model: HookedTransformer.
        pairs: Clean/corrupted pairs.
        specs: Direction specifications sorted by effective_strength.
        device: Torch device.
        batch_size: Batch size.

    Returns:
        List of DirectionPatchResult, one per direction.
    """
    results = []

    for spec in tqdm(specs, desc="Individual direction patching"):
        label = (
            f"L{spec.layer}H{spec.head}_{spec.component}_dir{spec.direction_idx}"
            if spec.head is not None
            else f"MLP_L{spec.layer}_{spec.component}_dir{spec.direction_idx}"
        )

        pair_results = run_patching_experiment(
            model, pairs, [spec], device, batch_size
        )
        agg = aggregate_pair_results(pair_results)

        results.append(DirectionPatchResult(
            direction_label=label,
            layer=spec.layer,
            head=spec.head,
            component=spec.component,
            direction_idx=spec.direction_idx,
            effective_strength=spec.effective_strength,
            dominant_frequency=spec.dominant_frequency,
            mean_logit_recovery=agg["mean_logit_recovery"],
            median_logit_recovery=agg["median_logit_recovery"],
            std_logit_recovery=agg["std_logit_recovery"],
            mean_rank_improvement=agg["mean_rank_improvement"],
            n_pairs=agg["n_pairs"],
        ))

    return results


def run_cumulative_experiment(
    model: HookedTransformer,
    pairs: List[Tuple[ArithmeticSample, ArithmeticSample]],
    specs: List[DirectionSpec],
    device: torch.device,
    k_values: List[int] = None,
    batch_size: int = 16,
) -> List[CumulativePatchResult]:
    """Experiment B: Patch top-K directions cumulatively.

    Args:
        model: HookedTransformer.
        pairs: Clean/corrupted pairs.
        specs: Direction specifications sorted by effective_strength.
        device: Torch device.
        k_values: List of K values to test (default: [1, 3, 5, 10, 15, 20, all]).
        batch_size: Batch size.

    Returns:
        List of CumulativePatchResult.
    """
    if k_values is None:
        k_values = sorted(set(
            [1, 3, 5, 10, 15, 20, len(specs)]
        ))
        k_values = [k for k in k_values if k <= len(specs)]

    results = []

    for k in tqdm(k_values, desc="Cumulative top-K patching"):
        top_k_specs = specs[:k]

        labels = []
        for spec in top_k_specs:
            label = (
                f"L{spec.layer}H{spec.head}_{spec.component}_dir{spec.direction_idx}"
                if spec.head is not None
                else f"MLP_L{spec.layer}_{spec.component}_dir{spec.direction_idx}"
            )
            labels.append(label)

        pair_results = run_patching_experiment(
            model, pairs, top_k_specs, device, batch_size
        )
        agg = aggregate_pair_results(pair_results)

        results.append(CumulativePatchResult(
            k=k,
            direction_labels=labels,
            mean_logit_recovery=agg["mean_logit_recovery"],
            median_logit_recovery=agg["median_logit_recovery"],
            std_logit_recovery=agg["std_logit_recovery"],
            mean_rank_improvement=agg["mean_rank_improvement"],
            n_pairs=agg["n_pairs"],
        ))

    return results


def run_frequency_group_experiment(
    model: HookedTransformer,
    pairs: List[Tuple[ArithmeticSample, ArithmeticSample]],
    specs: List[DirectionSpec],
    device: torch.device,
    batch_size: int = 16,
) -> Dict[int, dict]:
    """Experiment C: Patch all directions at each frequency simultaneously.

    Args:
        model: HookedTransformer.
        pairs: Clean/corrupted pairs.
        specs: Direction specifications.
        device: Torch device.
        batch_size: Batch size.

    Returns:
        Dict mapping frequency → aggregate result dict.
    """
    # Group by dominant frequency
    freq_groups: Dict[int, List[DirectionSpec]] = {}
    for spec in specs:
        f = spec.dominant_frequency
        if f not in freq_groups:
            freq_groups[f] = []
        freq_groups[f].append(spec)

    results = {}
    for freq in sorted(freq_groups.keys()):
        group_specs = freq_groups[freq]
        logger.info(f"  Frequency group f={freq}: {len(group_specs)} directions")

        pair_results = run_patching_experiment(
            model, pairs, group_specs, device, batch_size
        )
        agg = aggregate_pair_results(pair_results)
        agg["n_directions"] = len(group_specs)
        agg["direction_labels"] = [
            f"L{s.layer}H{s.head}_{s.component}_dir{s.direction_idx}"
            if s.head is not None
            else f"MLP_L{s.layer}_{s.component}_dir{s.direction_idx}"
            for s in group_specs
        ]
        results[freq] = agg

    return results


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def save_results(
    output_dir: str,
    individual_results_A: List[DirectionPatchResult],
    individual_results_B: List[DirectionPatchResult],
    cumulative_results_A: List[CumulativePatchResult],
    cumulative_results_B: List[CumulativePatchResult],
    freq_group_results_A: Dict[int, dict],
    freq_group_results_B: Dict[int, dict],
    n_pairs_A: int,
    n_pairs_B: int,
) -> str:
    """Serialize all results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"causal_validation_{timestamp}.json")

    def dir_result_to_dict(r: DirectionPatchResult) -> dict:
        return {
            "direction_label": r.direction_label,
            "layer": r.layer,
            "head": r.head,
            "component": r.component,
            "direction_idx": r.direction_idx,
            "effective_strength": r.effective_strength,
            "dominant_frequency": r.dominant_frequency,
            "mean_logit_recovery": r.mean_logit_recovery,
            "median_logit_recovery": r.median_logit_recovery,
            "std_logit_recovery": r.std_logit_recovery,
            "mean_rank_improvement": r.mean_rank_improvement,
            "n_pairs": r.n_pairs,
        }

    def cum_result_to_dict(r: CumulativePatchResult) -> dict:
        return {
            "k": r.k,
            "direction_labels": r.direction_labels,
            "mean_logit_recovery": r.mean_logit_recovery,
            "median_logit_recovery": r.median_logit_recovery,
            "std_logit_recovery": r.std_logit_recovery,
            "mean_rank_improvement": r.mean_rank_improvement,
            "n_pairs": r.n_pairs,
        }

    data = {
        "timestamp": timestamp,
        "n_pairs_A_corruption": n_pairs_A,
        "n_pairs_B_corruption": n_pairs_B,
        "A_corruption": {
            "individual": [dir_result_to_dict(r) for r in individual_results_A],
            "cumulative": [cum_result_to_dict(r) for r in cumulative_results_A],
            "frequency_groups": {
                str(f): v for f, v in freq_group_results_A.items()
            },
        },
        "B_corruption": {
            "individual": [dir_result_to_dict(r) for r in individual_results_B],
            "cumulative": [cum_result_to_dict(r) for r in cumulative_results_B],
            "frequency_groups": {
                str(f): v for f, v in freq_group_results_B.items()
            },
        },
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Results saved to {path}")
    return path


def print_summary(
    individual_results_A: List[DirectionPatchResult],
    individual_results_B: List[DirectionPatchResult],
    cumulative_results_A: List[CumulativePatchResult],
    cumulative_results_B: List[CumulativePatchResult],
    freq_group_results_A: Dict[int, dict],
    freq_group_results_B: Dict[int, dict],
    ben_cumul_A: List[CumulativePatchResult] = None,
    ben_cumul_B: List[CumulativePatchResult] = None,
):
    """Print human-readable summary of results."""
    ben_cumul_A = ben_cumul_A or []
    ben_cumul_B = ben_cumul_B or []

    lines = []
    lines.append("=" * 90)
    lines.append("PHASE 5: CAUSAL VALIDATION — DIRECTION-LEVEL SCALAR SWAPPING")
    lines.append("=" * 90)

    for corruption, indiv, cumul, freq_g, ben_cumul in [
        ("A-corruption (change a)", individual_results_A, cumulative_results_A, freq_group_results_A, ben_cumul_A),
        ("B-corruption (change b)", individual_results_B, cumulative_results_B, freq_group_results_B, ben_cumul_B),
    ]:
        lines.append("")
        lines.append(f"--- {corruption} ---")
        lines.append("")

        # Individual direction table
        lines.append("  Individual Direction Patching (sorted by median logit recovery):")
        lines.append(
            "  {:>30s}  {:>7s}  {:>5s}  {:>8s}  {:>8s}  {:>8s}".format(
                "Direction", "Str", "Freq", "Mean", "Median", "RankΔ"
            )
        )
        lines.append("  " + "-" * 75)
        sorted_indiv = sorted(indiv, key=lambda r: -r.median_logit_recovery)
        for r in sorted_indiv:
            lines.append(
                "  {:>30s}  {:>7.2f}  {:>5d}  {:>+8.4f}  {:>+8.4f}  {:>+8.1f}".format(
                    r.direction_label, r.effective_strength,
                    r.dominant_frequency, r.mean_logit_recovery,
                    r.median_logit_recovery, r.mean_rank_improvement,
                )
            )

        # Cumulative (by effective strength)
        lines.append("")
        lines.append("  Cumulative Top-K (by effective strength):")
        lines.append(
            "  {:>5s}  {:>8s}  {:>8s}  {:>8s}".format(
                "K", "Mean", "Median", "RankΔ"
            )
        )
        lines.append("  " + "-" * 35)
        for r in cumul:
            marker = " ← target" if r.k == 5 else ""
            lines.append(
                "  {:>5d}  {:>+8.4f}  {:>+8.4f}  {:>+8.1f}{}".format(
                    r.k, r.mean_logit_recovery,
                    r.median_logit_recovery, r.mean_rank_improvement,
                    marker,
                )
            )

        # Beneficial-only cumulative
        if ben_cumul:
            lines.append("")
            lines.append("  Beneficial-Only Cumulative (dirs with positive individual median):")
            lines.append(
                "  {:>5s}  {:>8s}  {:>8s}  {:>8s}".format(
                    "K", "Mean", "Median", "RankΔ"
                )
            )
            lines.append("  " + "-" * 35)
            for r in ben_cumul:
                marker = " ★" if r.k == 5 else ""
                lines.append(
                    "  {:>5d}  {:>+8.4f}  {:>+8.4f}  {:>+8.1f}{}".format(
                        r.k, r.mean_logit_recovery,
                        r.median_logit_recovery, r.mean_rank_improvement,
                        marker,
                    )
                )

        # Frequency groups
        lines.append("")
        lines.append("  Frequency Group Patching:")
        for freq in sorted(freq_g.keys()):
            g = freq_g[freq]
            lines.append(
                "  freq={:>2d}: {:>2d} dirs, mean_recovery={:>+.4f}, median={:>+.4f}".format(
                    freq, g["n_directions"],
                    g["mean_logit_recovery"], g["median_logit_recovery"],
                )
            )

    summary = "\n".join(lines)
    print(summary)
    return summary


# ---------------------------------------------------------------------------
# Phase 3 results loader (reused from analyze_fourier_circuits.py)
# ---------------------------------------------------------------------------

def load_phase3_results(path: str, mask_threshold: float = 0.3) -> List[dict]:
    """Load Phase 3 survivor directions from JSON.

    Phase 3 JSON has nested structure:
        results[i] = {layer, head, component, directions: [{direction_idx, ...}]}
    We flatten and filter by mask_value > mask_threshold.
    """
    with open(path) as f:
        data = json.load(f)

    survivors = []
    for comp in data.get("results", []):
        layer = comp["layer"]
        head = comp.get("head")
        component = comp["component"]
        for d in comp.get("directions", []):
            if d.get("mask_value", 0) > mask_threshold:
                survivors.append({
                    "layer": layer,
                    "head": head,
                    "component": component,
                    **d,
                })

    return survivors


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 5: Causal validation via direction-level scalar swapping"
    )
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--phase3_results", required=True, help="Path to Phase 3 JSON results")
    parser.add_argument("--output_dir", default="phase5_results", help="Output directory")
    parser.add_argument("--operand_range_end", type=int, default=None)
    parser.add_argument("--n_pairs", type=int, default=100, help="Number of clean/corrupted pairs per corruption type")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for model runs")
    parser.add_argument("--device", default=None, help="Device (auto-detect if not specified)")
    parser.add_argument("--mask_threshold", type=float, default=0.3,
                        help="Min mask value for a direction to be included (default: 0.3)")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Load Phase 3 results
    logger.info(f"Loading Phase 3 results from {args.phase3_results}")
    survivors = load_phase3_results(args.phase3_results, mask_threshold=args.mask_threshold)
    logger.info(f"Found {len(survivors)} surviving directions (mask > {args.mask_threshold})")

    # Load model
    logger.info(f"Loading model: {config['model']['name']}")
    import yaml as _yaml  # already imported above via argparse block
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

    # Initialize circuit
    train_masks = config.get("train_masks", None)
    logger.info("Initializing MaskedTransformerCircuit...")
    circuit = MaskedTransformerCircuit(
        model=model,
        device=device,
        cache_svd=config["masking"]["cache_svd"],
        mask_init_value=config["masking"]["mask_init_value"],
        train_masks=train_masks,
    )

    # Load trained masks
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    logger.info(f"Loaded checkpoint from {args.checkpoint}")

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

    # Extract direction specs
    specs = extract_direction_specs(circuit, survivors)
    logger.info(f"Extracted {len(specs)} direction specs, sorted by effective strength")
    for i, s in enumerate(specs[:5]):
        label = (
            f"L{s.layer}H{s.head}_{s.component}_dir{s.direction_idx}"
            if s.head is not None
            else f"MLP_L{s.layer}_{s.component}_dir{s.direction_idx}"
        )
        logger.info(f"  #{i+1}: {label}  str={s.effective_strength:.2f}  freq={s.dominant_frequency}")

    # Generate corruption pairs
    arith_cfg = config["arithmetic"]
    range_end = args.operand_range_end or arith_cfg.get("operand_range_end", 10)
    op_range = range(arith_cfg.get("operand_range_start", 0), range_end)

    logger.info(f"Generating corruption pairs (operand range 0-{range_end - 1})...")
    pairs_A = generate_corruption_pairs(op_range, "A", n_pairs=args.n_pairs, seed=42)
    pairs_B = generate_corruption_pairs(op_range, "B", n_pairs=args.n_pairs, seed=43)
    logger.info(f"  A-corruption pairs: {len(pairs_A)}")
    logger.info(f"  B-corruption pairs: {len(pairs_B)}")

    # ---------------------------------------------------------------
    # Experiment A: Individual direction patching
    # ---------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("EXPERIMENT A: INDIVIDUAL DIRECTION PATCHING")
    logger.info("=" * 70)

    logger.info("  Running A-corruption...")
    indiv_A = run_individual_direction_experiment(
        model, pairs_A, specs, device, args.batch_size
    )

    logger.info("  Running B-corruption...")
    indiv_B = run_individual_direction_experiment(
        model, pairs_B, specs, device, args.batch_size
    )

    # ---------------------------------------------------------------
    # Experiment B: Cumulative top-K patching
    # ---------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("EXPERIMENT B: CUMULATIVE TOP-K PATCHING")
    logger.info("=" * 70)

    logger.info("  Running A-corruption...")
    cumul_A = run_cumulative_experiment(
        model, pairs_A, specs, device, batch_size=args.batch_size
    )

    logger.info("  Running B-corruption...")
    cumul_B = run_cumulative_experiment(
        model, pairs_B, specs, device, batch_size=args.batch_size
    )

    # ---------------------------------------------------------------
    # Experiment C: Frequency group patching
    # ---------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("EXPERIMENT C: FREQUENCY GROUP PATCHING")
    logger.info("=" * 70)

    logger.info("  Running A-corruption...")
    freq_A = run_frequency_group_experiment(
        model, pairs_A, specs, device, args.batch_size
    )

    logger.info("  Running B-corruption...")
    freq_B = run_frequency_group_experiment(
        model, pairs_B, specs, device, args.batch_size
    )

    # ---------------------------------------------------------------
    # Experiment D: Beneficial-only cumulative patching
    # ---------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("EXPERIMENT D: BENEFICIAL-ONLY CUMULATIVE PATCHING")
    logger.info("=" * 70)

    # For each corruption type, select only directions with positive
    # individual median recovery, then run cumulative patching.
    def get_beneficial_specs(indiv_results, all_specs):
        """Return specs sorted by individual median recovery (descending),
        keeping only those with positive median recovery."""
        label_to_spec = {}
        for s in all_specs:
            label = (
                f"L{s.layer}H{s.head}_{s.component}_dir{s.direction_idx}"
                if s.head is not None
                else f"MLP_L{s.layer}_{s.component}_dir{s.direction_idx}"
            )
            label_to_spec[label] = s

        beneficial = []
        for r in sorted(indiv_results, key=lambda x: -x.median_logit_recovery):
            if r.median_logit_recovery > 0.0 and r.direction_label in label_to_spec:
                beneficial.append(label_to_spec[r.direction_label])
        return beneficial

    ben_specs_A = get_beneficial_specs(indiv_A, specs)
    ben_specs_B = get_beneficial_specs(indiv_B, specs)
    logger.info(f"  A-corruption: {len(ben_specs_A)} beneficial directions")
    logger.info(f"  B-corruption: {len(ben_specs_B)} beneficial directions")

    logger.info("  Running A-corruption beneficial cumulative...")
    ben_cumul_A = run_cumulative_experiment(
        model, pairs_A, ben_specs_A, device, batch_size=args.batch_size
    ) if ben_specs_A else []

    logger.info("  Running B-corruption beneficial cumulative...")
    ben_cumul_B = run_cumulative_experiment(
        model, pairs_B, ben_specs_B, device, batch_size=args.batch_size
    ) if ben_specs_B else []

    # ---------------------------------------------------------------
    # Save and summarize
    # ---------------------------------------------------------------
    result_path = save_results(
        args.output_dir,
        indiv_A, indiv_B,
        cumul_A, cumul_B,
        freq_A, freq_B,
        len(pairs_A), len(pairs_B),
    )

    summary_text = print_summary(
        indiv_A, indiv_B,
        cumul_A, cumul_B,
        freq_A, freq_B,
        ben_cumul_A, ben_cumul_B,
    )

    # Save summary text
    summary_path = result_path.replace(".json", "_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)

    # Final verdict — use beneficial cumulative top-5 (median, robust)
    def best_top5_median(cumul_list):
        for r in cumul_list:
            if r.k >= 5:
                return r.median_logit_recovery
        if cumul_list:
            return cumul_list[-1].median_logit_recovery
        return 0.0

    ben5_A = best_top5_median(ben_cumul_A)
    ben5_B = best_top5_median(ben_cumul_B)
    avg_ben5 = (ben5_A + ben5_B) / 2

    logger.info("=" * 70)
    logger.info(
        f"PHASE 5 VERDICT (beneficial-only, median):  "
        f"A-corr top-5={ben5_A:+.4f}  B-corr top-5={ben5_B:+.4f}  "
        f"avg={avg_ben5:+.4f}  {'✓ PASS' if avg_ben5 > 0.5 else '✗ BELOW 0.5 TARGET'}"
    )
    logger.info("=" * 70)

    return indiv_A, indiv_B, cumul_A, cumul_B, freq_A, freq_B


if __name__ == "__main__":
    main()
