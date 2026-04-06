#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 3: Fourier Analysis of SVD Directions.

After Phase 2 identifies the arithmetic circuit via SVD mask learning, this
script analyzes the *surviving* SVD directions to find which specific linear
subspaces carry periodic/trigonometric structure.

Algorithm:
    1. Load the trained mask checkpoint from Phase 2.
    2. For each head/MLP with a high mask value (active in the circuit):
       a. Extract the top-k SVD directions (U, S, Vh) from the cached SVD.
       b. Collect activations for all arithmetic prompts.
       c. Project activations onto each SVD direction.
       d. Run 1-D DFT on the projected activations (indexed by number).
       e. Identify which directions encode periodic structure.
    3. Output a report mapping: (layer, head, svd_direction_k) → (frequency, power_ratio).

Usage:
    python experiments/analyze_svd_directions.py \\
        --config configs/arithmetic_pythia_config.yaml \\
        --checkpoint logs/arithmetic_circuit_discovery_best.pt \\
        [--mask_threshold 0.3] \\
        [--top_k_directions 10] \\
        [--device mps]

Requires: A completed Phase 2 training run with a saved checkpoint.
"""

import argparse
import datetime
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
    generate_arithmetic_prompts,
)
from src.models.masked_transformer_circuit import MaskedTransformerCircuit, mask_fn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class SVDDirectionFourierResult:
    """Fourier analysis result for a single SVD direction.

    Attributes:
        layer: Layer index.
        head: Head index (None for MLP).
        component: 'OV', 'MLP_in', or 'MLP_out'.
        direction_idx: SVD direction index (0 = highest singular value).
        singular_value: The singular value for this direction.
        mask_value: The learned mask value for this direction.
        effective_strength: singular_value * mask_value.
        dominant_frequency: Strongest periodic frequency in projections.
        power_ratio: Signal-to-noise ratio of the dominant frequency.
        power_spectrum: Full power spectrum array.
    """
    layer: int
    head: Optional[int]
    component: str
    direction_idx: int
    singular_value: float
    mask_value: float
    effective_strength: float
    dominant_frequency: int
    power_ratio: float
    power_spectrum: Optional[np.ndarray] = None


@dataclass
class HeadCircuitSummary:
    """Summary of periodic structure found in one attention head or MLP layer.

    Attributes:
        layer: Layer index.
        head: Head index (None for MLP).
        component: 'OV', 'MLP_in', or 'MLP_out'.
        avg_mask: Average mask value across all directions.
        n_active_directions: Number of directions above mask threshold.
        n_periodic_directions: Number of active directions with periodic signal.
        top_directions: List of SVDDirectionFourierResult for periodic directions.
    """
    layer: int
    head: Optional[int]
    component: str
    avg_mask: float
    n_active_directions: int
    n_periodic_directions: int
    top_directions: List[SVDDirectionFourierResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_path: str, device: str = "cpu") -> dict:
    """Load a Phase 2 training checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Device to load tensors onto.

    Returns:
        Checkpoint dict with keys: qk_masks, ov_masks, mlp_in_masks,
        mlp_out_masks, sparsity_stats, config, etc.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    torch.serialization.add_safe_globals([torch.nn.modules.container.ParameterDict])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    step = checkpoint.get('step', '?')
    val_loss = checkpoint.get('val_loss', None)
    val_str = f", val_loss={val_loss:.4f}" if isinstance(val_loss, (int, float)) else ""
    logger.info(
        f"Loaded checkpoint from {checkpoint_path} "
        f"(step={step}{val_str})"
    )
    return checkpoint


def get_active_heads(
    circuit: MaskedTransformerCircuit,
    mask_threshold: float = 0.3,
    skip_frozen_qk: bool = True,
) -> List[Tuple[int, int, str, float]]:
    """Identify heads/MLPs with at least one mask direction above threshold.

    Uses **max** mask value per component (not average), since a head with
    1 surviving direction out of 65 is still active even though avg ≈ 0.05.

    Args:
        circuit: Initialized MaskedTransformerCircuit with loaded masks.
        mask_threshold: Minimum max-mask value for a component to be analyzed.
        skip_frozen_qk: If True, skip QK masks that are uniformly ~1.0
            (frozen for RoPE models). Default True.

    Returns:
        List of (layer, head, component, max_mask) tuples, sorted by max_mask
        descending. For MLP components, head is None.
    """
    active = []

    for layer in range(circuit.n_layers):
        for head in range(circuit.n_heads):
            head_key = f"differential_head_{layer}_{head}"

            # OV mask — use max, not mean
            if head_key in circuit.ov_masks:
                ov_mask = mask_fn(circuit.ov_masks[head_key]).detach().cpu()
                max_ov = ov_mask.max().item()
                if max_ov >= mask_threshold:
                    active.append((layer, head, "OV", max_ov))

            # QK mask
            if head_key in circuit.qk_masks:
                qk_mask = mask_fn(circuit.qk_masks[head_key]).detach().cpu()
                max_qk = qk_mask.max().item()
                # Skip frozen QK (all values uniformly near 1.0)
                if skip_frozen_qk and qk_mask.min().item() > 0.99:
                    continue
                if max_qk >= mask_threshold:
                    active.append((layer, head, "QK", max_qk))

        # MLP masks — use max, not mean
        mlp_key = f"mlp_{layer}"
        if circuit.mask_mlp:
            if mlp_key in circuit.mlp_in_masks:
                mlp_in = mask_fn(circuit.mlp_in_masks[mlp_key]).detach().cpu()
                max_in = mlp_in.max().item()
                if max_in >= mask_threshold:
                    active.append((layer, None, "MLP_in", max_in))

            if mlp_key in circuit.mlp_out_masks:
                mlp_out = mask_fn(circuit.mlp_out_masks[mlp_key]).detach().cpu()
                max_out = mlp_out.max().item()
                if max_out >= mask_threshold:
                    active.append((layer, None, "MLP_out", max_out))

    active.sort(key=lambda x: -x[3])
    logger.info(f"Found {len(active)} active components above threshold {mask_threshold}")
    return active


def collect_last_token_activations(
    model,
    prompts: List,
    layers: List[int],
    device: torch.device,
    batch_size: int = 16,
) -> Dict[str, torch.Tensor]:
    """Collect residual stream activations at last token for all prompts.

    Args:
        model: HookedTransformer model.
        prompts: List of ArithmeticSample or strings.
        layers: Layer indices to collect.
        device: Torch device.
        batch_size: Processing batch size.

    Returns:
        Dict mapping hook_name -> tensor of shape (n_prompts, d_model).
    """
    hook_names = []
    for l in layers:
        hook_names.append(f"blocks.{l}.hook_resid_pre")
        hook_names.append(f"blocks.{l}.ln1.hook_normalized")

    prompt_texts = [p.prompt if hasattr(p, "prompt") else str(p) for p in prompts]
    n_prompts = len(prompt_texts)

    result = {h: [] for h in hook_names}

    for batch_start in tqdm(
        range(0, n_prompts, batch_size),
        desc="Collecting activations",
        disable=n_prompts <= batch_size,
    ):
        batch_texts = prompt_texts[batch_start : batch_start + batch_size]
        tokens = model.to_tokens(batch_texts).to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)

        # Find last non-pad token positions
        if model.tokenizer.pad_token_id is not None:
            pad_id = model.tokenizer.pad_token_id
            lengths = (tokens != pad_id).sum(dim=1)
            last_pos = lengths - 1
        else:
            last_pos = torch.full(
                (tokens.shape[0],), tokens.shape[1] - 1, dtype=torch.long
            )

        batch_indices = torch.arange(tokens.shape[0])

        for hook in hook_names:
            if hook not in cache:
                continue
            tensor = cache[hook]  # (batch, seq, d_model)
            for i in range(len(batch_texts)):
                act = tensor[i, last_pos[i], :].cpu()
                result[hook].append(act)

        del cache
        if device.type == "mps":
            torch.mps.empty_cache()

    # Stack
    stacked = {}
    for hook, arrays in result.items():
        if arrays:
            stacked[hook] = torch.stack(arrays, dim=0)  # (n_prompts, d_model)

    return stacked


def analyze_svd_direction(
    activations: np.ndarray,
    svd_vector: np.ndarray,
) -> Tuple[np.ndarray, int, float]:
    """Project activations onto an SVD direction and run Fourier analysis.

    Args:
        activations: Shape (N, d_model) — one row per number.
        svd_vector: Shape (d_model,) — the SVD direction to project onto.

    Returns:
        Tuple of (projections, dominant_frequency, power_ratio).
        projections has shape (N,).
    """
    # Project: scalar projection of each activation onto the direction
    projections = activations @ svd_vector  # (N,)

    # For DFT we need shape (N, 1)
    proj_2d = projections.reshape(-1, 1)
    power, _ = compute_fourier_power_spectrum(proj_2d)
    dom_freq, ratio = identify_dominant_frequency(power)

    return projections, dom_freq, ratio


def analyze_head_directions(
    circuit: MaskedTransformerCircuit,
    activations: np.ndarray,
    layer: int,
    head: int,
    component: str,
    top_k: int = 10,
    direction_power_threshold: float = 3.0,
) -> HeadCircuitSummary:
    """Analyze SVD directions for one attention head.

    Args:
        circuit: MaskedTransformerCircuit with cached SVD.
        activations: Shape (N, d_model) — residual stream at this layer.
        layer: Layer index.
        head: Head index.
        component: 'OV' or 'QK'.
        top_k: Number of top SVD directions to analyze.
        direction_power_threshold: Minimum power ratio for a direction
            to be considered "periodic".

    Returns:
        HeadCircuitSummary with analysis results.
    """
    head_key = f"differential_head_{layer}_{head}"

    # Get SVD components and mask from cache
    if component == "OV":
        cache_key = f"{head_key}_ov"
        masks = circuit.ov_masks
    else:
        cache_key = f"{head_key}_qk"
        masks = circuit.qk_masks

    if cache_key not in circuit.svd_cache:
        logger.warning(f"SVD cache missing for {cache_key}, skipping")
        return HeadCircuitSummary(
            layer=layer, head=head, component=component,
            avg_mask=0.0, n_active_directions=0, n_periodic_directions=0,
        )

    U, S, Vh, _ = circuit.svd_cache[cache_key]
    mask_values = mask_fn(masks[head_key]).detach().cpu().numpy()

    # Scan ALL directions (not just top_k) since survivors can be at high indices
    n_directions = min(len(S), len(mask_values))
    avg_mask = float(mask_values.mean())

    all_results = []
    n_active = 0
    n_periodic = 0

    for k in range(n_directions):
        mv = float(mask_values[k])
        sv = float(S[k].cpu())
        strength = sv * mv

        if mv < 0.01:  # Skip effectively pruned directions
            continue
        n_active += 1

        # Get the right singular vector (input-space direction)
        # Vh has shape (rank, d_model) or (rank, d_model+1) for augmented
        vh_k = Vh[k, :].cpu().numpy()

        # Handle augmented dimensions: strip the bias column if present
        if vh_k.shape[0] == activations.shape[1] + 1:
            vh_k = vh_k[1:]  # Drop the bias (first) column
        elif vh_k.shape[0] != activations.shape[1]:
            logger.warning(
                f"Dimension mismatch: Vh[{k}] has {vh_k.shape[0]} dims, "
                f"activations have {activations.shape[1]}. Skipping."
            )
            continue

        # Analyze this direction
        _, dom_freq, ratio = analyze_svd_direction(activations, vh_k)

        result = SVDDirectionFourierResult(
            layer=layer,
            head=head,
            component=component,
            direction_idx=k,
            singular_value=sv,
            mask_value=mv,
            effective_strength=strength,
            dominant_frequency=dom_freq,
            power_ratio=ratio,
        )

        if ratio >= direction_power_threshold:
            n_periodic += 1

        # Save ALL directions with significant mask (survivors), plus periodic ones
        if mv >= 0.3 or ratio >= direction_power_threshold:
            all_results.append(result)

    # Sort: survivors first (by mask desc), then periodic (by ratio desc)
    all_results.sort(key=lambda r: (-r.mask_value, -r.power_ratio))

    return HeadCircuitSummary(
        layer=layer,
        head=head,
        component=component,
        avg_mask=avg_mask,
        n_active_directions=n_active,
        n_periodic_directions=n_periodic,
        top_directions=all_results,
    )


def analyze_mlp_directions(
    circuit: MaskedTransformerCircuit,
    activations: np.ndarray,
    layer: int,
    component: str,
    top_k: int = 10,
    direction_power_threshold: float = 3.0,
) -> HeadCircuitSummary:
    """Analyze SVD directions for one MLP layer.

    Args:
        circuit: MaskedTransformerCircuit with cached SVD.
        activations: Shape (N, d_model) — residual stream at this layer.
        layer: Layer index.
        component: 'MLP_in' or 'MLP_out'.
        top_k: Number of top SVD directions to analyze.
        direction_power_threshold: Minimum power ratio to flag periodic.

    Returns:
        HeadCircuitSummary with analysis results.
    """
    mlp_key = f"mlp_{layer}"
    comp_lower = component.lower()

    cache_key = f"mlp_{layer}_{'in' if comp_lower == 'mlp_in' else 'out'}"
    masks = circuit.mlp_in_masks if comp_lower == "mlp_in" else circuit.mlp_out_masks

    if cache_key not in circuit.svd_cache:
        logger.warning(f"SVD cache missing for {cache_key}, skipping")
        return HeadCircuitSummary(
            layer=layer, head=None, component=component,
            avg_mask=0.0, n_active_directions=0, n_periodic_directions=0,
        )

    U, S, Vh, _ = circuit.svd_cache[cache_key]
    mask_values = mask_fn(masks[mlp_key]).detach().cpu().numpy()

    # Scan ALL directions (not just top_k) since survivors can be at high indices
    n_directions = min(len(S), len(mask_values))
    avg_mask = float(mask_values.mean())

    all_results = []
    n_active = 0
    n_periodic = 0

    for k in range(n_directions):
        mv = float(mask_values[k])
        sv = float(S[k].cpu())
        strength = sv * mv

        if mv < 0.01:
            continue
        n_active += 1

        # MLP_in:  W_in_aug is (d_model+1, d_mlp), SVD → U (d_model+1, r), Vh (r, d_mlp)
        #   Input-space direction = U[:, k] in d_model+1 space (strip bias row)
        # MLP_out: W_out_aug is (d_mlp+1, d_model), SVD → U (d_mlp+1, r), Vh (r, d_model)
        #   Output-space direction in residual stream = Vh[k, :] in d_model space
        if comp_lower == "mlp_in":
            vec = U[:, k].cpu().numpy()
        else:
            vec = Vh[k, :].cpu().numpy()

        # Handle augmented dimensions
        if vec.shape[0] == activations.shape[1] + 1:
            vec = vec[1:]
        elif vec.shape[0] != activations.shape[1]:
            logger.warning(
                f"MLP dimension mismatch: vec has {vec.shape[0]} dims, "
                f"activations have {activations.shape[1]}. Skipping."
            )
            continue

        _, dom_freq, ratio = analyze_svd_direction(activations, vec)

        result = SVDDirectionFourierResult(
            layer=layer,
            head=None,
            component=component,
            direction_idx=k,
            singular_value=sv,
            mask_value=mv,
            effective_strength=strength,
            dominant_frequency=dom_freq,
            power_ratio=ratio,
        )

        if ratio >= direction_power_threshold:
            n_periodic += 1

        # Save ALL directions with significant mask (survivors), plus periodic ones
        if mv >= 0.3 or ratio >= direction_power_threshold:
            all_results.append(result)

    all_results.sort(key=lambda r: (-r.mask_value, -r.power_ratio))

    return HeadCircuitSummary(
        layer=layer,
        head=None,
        component=component,
        avg_mask=avg_mask,
        n_active_directions=n_active,
        n_periodic_directions=n_periodic,
        top_directions=all_results,
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def results_to_dict(summaries: List[HeadCircuitSummary]) -> dict:
    """Convert analysis results to JSON-serializable dict."""
    entries = []
    for s in summaries:
        entry = {
            "layer": s.layer,
            "head": s.head,
            "component": s.component,
            "avg_mask": s.avg_mask,
            "n_active_directions": s.n_active_directions,
            "n_periodic_directions": s.n_periodic_directions,
            "directions": [],
        }
        for d in s.top_directions:
            entry["directions"].append({
                "direction_idx": d.direction_idx,
                "singular_value": d.singular_value,
                "mask_value": d.mask_value,
                "effective_strength": d.effective_strength,
                "dominant_frequency": d.dominant_frequency,
                "power_ratio": d.power_ratio,
            })
        entries.append(entry)
    return entries


def print_summary(summaries: List[HeadCircuitSummary]) -> str:
    """Generate a human-readable summary."""
    lines = ["=" * 70]
    lines.append("PHASE 3: SVD DIRECTION FOURIER ANALYSIS")
    lines.append("=" * 70)

    # Count stats
    total_active = sum(s.n_active_directions for s in summaries)
    total_periodic = sum(s.n_periodic_directions for s in summaries)
    lines.append(f"Components analyzed: {len(summaries)}")
    lines.append(f"Active SVD directions: {total_active}")
    lines.append(f"Periodic SVD directions: {total_periodic}")
    lines.append("")

    # Group by component type
    for comp_type in ["OV", "QK", "MLP_in", "MLP_out"]:
        comp_summaries = [s for s in summaries if s.component == comp_type]
        if not comp_summaries:
            continue

        lines.append(f"--- {comp_type} ---")
        # Sort by number of periodic directions, then avg_mask
        comp_summaries.sort(key=lambda s: (-s.n_periodic_directions, -s.avg_mask))

        for s in comp_summaries[:20]:  # Top 20 per component
            label = f"L{s.layer}H{s.head}" if s.head is not None else f"L{s.layer}"
            lines.append(
                f"  {label:8s} {s.component:7s}  mask={s.avg_mask:.4f}  "
                f"active={s.n_active_directions:3d}  periodic={s.n_periodic_directions:3d}"
            )
            for d in s.top_directions[:5]:
                lines.append(
                    f"    dir[{d.direction_idx:2d}]: sv={d.singular_value:.2f}  "
                    f"mask={d.mask_value:.4f}  freq={d.dominant_frequency}  "
                    f"ratio={d.power_ratio:.1f}x  strength={d.effective_strength:.2f}"
                )
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_analysis(
    config: dict,
    checkpoint_path: str,
    mask_threshold: float = 0.3,
    top_k_directions: int = 10,
    direction_power_threshold: float = 3.0,
    device_str: Optional[str] = None,
    output_dir: str = "phase3_results",
    operand_range_end: Optional[int] = None,
) -> List[HeadCircuitSummary]:
    """Run the full Phase 3 analysis.

    Args:
        config: Loaded YAML config dict.
        checkpoint_path: Path to Phase 2 checkpoint.
        mask_threshold: Min avg mask to consider a component active.
        top_k_directions: Number of top SVD directions per component.
        direction_power_threshold: Min power ratio for periodic flag.
        device_str: Device override.
        output_dir: Where to save results.

    Returns:
        List of HeadCircuitSummary objects.
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

    # Load model
    logger.info(f"Loading model: {config['model']['name']}")
    model = HookedTransformer.from_pretrained(
        config["model"]["name"],
        cache_dir=config["model"]["pretrained_cache_dir"],
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
    checkpoint = load_checkpoint(checkpoint_path, device=str(device))
    logger.info("Loading trained masks from checkpoint...")

    # Restore mask parameters
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

    # Identify active components
    active_components = get_active_heads(circuit, mask_threshold=mask_threshold)
    if not active_components:
        logger.warning(
            f"No components above mask threshold {mask_threshold}. "
            "Try lowering --mask_threshold."
        )
        return []

    # Determine which layers need activations
    active_layers = sorted(set(layer for layer, _, _, _ in active_components))
    logger.info(f"Active layers: {active_layers}")

    # Generate arithmetic prompts ordered by number value
    arith_cfg = config["arithmetic"]
    range_end = operand_range_end or arith_cfg.get("operand_range_end", 10)
    op_range = range(
        arith_cfg.get("operand_range_start", 0),
        range_end,
    )
    prompt_gen = ArithmeticPromptGenerator(
        operand_range=op_range,
        operation=arith_cfg.get("operation", "add"),
        prompt_template=arith_cfg.get("prompt_template", "{a} + {b} ="),
        shuffle=False,
    )

    # Create ordered prompts: n+0 for each n (measures number representation)
    fourier_prompts = []
    for n in op_range:
        sample = prompt_gen.get_by_operands(n, 0)
        if sample is None:
            raise ValueError(f"Missing prompt for ({n}, 0)")
        fourier_prompts.append(sample)

    logger.info(f"Fourier prompts: {len(fourier_prompts)} ordered by number value")

    # Collect activations
    logger.info("Collecting residual stream activations...")
    all_activations = collect_last_token_activations(
        model=model,
        prompts=fourier_prompts,
        layers=active_layers,
        device=device,
        batch_size=config.get("fourier", {}).get("batch_size", 16),
    )

    # Analyze each active component
    summaries: List[HeadCircuitSummary] = []

    for layer, head, component, avg_mask in tqdm(
        active_components, desc="Analyzing SVD directions"
    ):
        hook = f"blocks.{layer}.hook_resid_pre"
        if hook not in all_activations:
            logger.warning(f"No activations for {hook}, skipping")
            continue

        act_np = all_activations[hook].numpy()  # (N, d_model)

        if component in ("OV", "QK") and head is not None:
            summary = analyze_head_directions(
                circuit=circuit,
                activations=act_np,
                layer=layer,
                head=head,
                component=component,
                top_k=top_k_directions,
                direction_power_threshold=direction_power_threshold,
            )
        elif component in ("MLP_in", "MLP_out"):
            summary = analyze_mlp_directions(
                circuit=circuit,
                activations=act_np,
                layer=layer,
                component=component,
                top_k=top_k_directions,
                direction_power_threshold=direction_power_threshold,
            )
        else:
            continue

        summaries.append(summary)

    # Print and save results
    summary_text = print_summary(summaries)
    print("\n" + summary_text)

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        output_dir, f"svd_direction_fourier_{timestamp}.json"
    )

    output_data = {
        "timestamp": timestamp,
        "config": {
            "mask_threshold": mask_threshold,
            "top_k_directions": top_k_directions,
            "direction_power_threshold": direction_power_threshold,
            "checkpoint": checkpoint_path,
            "model": config["model"]["name"],
        },
        "summary": {
            "n_components_analyzed": len(summaries),
            "n_active_directions": sum(s.n_active_directions for s in summaries),
            "n_periodic_directions": sum(s.n_periodic_directions for s in summaries),
        },
        "results": results_to_dict(summaries),
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    return summaries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 3: Fourier Analysis of SVD Directions"
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
        "--mask_threshold", type=float, default=0.3,
        help="Min avg mask value for a component to be analyzed (default: 0.3)",
    )
    parser.add_argument(
        "--top_k_directions", type=int, default=10,
        help="Number of top SVD directions to analyze per component (default: 10)",
    )
    parser.add_argument(
        "--direction_power_threshold", type=float, default=3.0,
        help="Min power ratio for a direction to be flagged periodic (default: 3.0)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device override (e.g. 'mps', 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--output_dir", type=str, default="phase3_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--operand_range_end", type=int, default=None,
        help="Override operand_range_end from config (e.g. 10 for range 0-9)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    logger.info("=" * 70)
    logger.info("PHASE 3: FOURIER ANALYSIS OF SVD DIRECTIONS")
    logger.info("=" * 70)

    summaries = run_analysis(
        config=config,
        checkpoint_path=args.checkpoint,
        mask_threshold=args.mask_threshold,
        top_k_directions=args.top_k_directions,
        direction_power_threshold=args.direction_power_threshold,
        device_str=args.device,
        output_dir=args.output_dir,
        operand_range_end=args.operand_range_end,
    )

    # Final summary
    total_periodic = sum(s.n_periodic_directions for s in summaries)
    logger.info("=" * 70)
    logger.info(f"PHASE 3 COMPLETE: Found {total_periodic} periodic SVD directions")
    logger.info("=" * 70)

    return summaries


if __name__ == "__main__":
    main()
