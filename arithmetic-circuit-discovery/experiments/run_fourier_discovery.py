#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run Fourier Discovery on Arithmetic Prompts.

Phase 1 of the Geometric SVD Framework: bottom-up identification of periodic
number representations in transformer residual streams using DFT analysis.

Usage:
    # Default: Pythia-1.4B on MPS
    python experiments/run_fourier_discovery.py --config configs/arithmetic_pythia_config.yaml

    # Override model at command line
    python experiments/run_fourier_discovery.py --config configs/arithmetic_pythia_config.yaml --model_key gpt2-small

    # Analyze specific layers only
    python experiments/run_fourier_discovery.py --config configs/arithmetic_pythia_config.yaml --layers 6,12,18

    # Also scan individual attention heads
    python experiments/run_fourier_discovery.py --config configs/arithmetic_pythia_config.yaml --scan_heads
"""

import argparse
import json
import logging
import os
import sys
import datetime
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.fourier_discovery import FourierDiscovery, LayerFourierResult
from src.analysis.fourier_plots import plot_all
from src.analysis.experiment_history import append_experiment_result
from src.data.arithmetic_dataset import (
    ArithmeticPromptGenerator,
    generate_arithmetic_prompts,
)
from src.utils.model_registry import (
    get_model_spec,
    list_available_models,
    load_model,
    verify_single_token_numbers,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _numpy_to_python(obj):
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: _numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_numpy_to_python(i) for i in obj]
    return obj


def save_results(
    layer_results: list,
    head_results: dict,
    config: dict,
    output_dir: str,
    model_key: str,
):
    """Save Fourier discovery results to JSON and print summary.

    Args:
        layer_results: List of LayerFourierResult.
        head_results: Dict of (layer, head) -> FourierResult.
        config: Full run config.
        output_dir: Directory to save results.
        model_key: Model registry key.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Serialize layer results
    layer_data = []
    for lr in layer_results:
        entry = {
            "layer": lr.layer,
            "dominant_frequency": int(lr.dominant_frequency),
            "dominant_frequency_power_ratio": float(lr.dominant_frequency_power_ratio),
            "resid_pre": {
                "power_spectrum": lr.resid_pre.power_spectrum.tolist(),
                "dominant_frequency": int(lr.resid_pre.dominant_frequency),
                "power_ratio": float(lr.resid_pre.dominant_frequency_power_ratio),
            },
        }
        if lr.resid_post is not None:
            entry["resid_post"] = {
                "power_spectrum": lr.resid_post.power_spectrum.tolist(),
                "dominant_frequency": int(lr.resid_post.dominant_frequency),
                "power_ratio": float(lr.resid_post.dominant_frequency_power_ratio),
            }
        layer_data.append(entry)

    # Serialize head results
    head_data = {}
    for (layer, head), result in sorted(head_results.items()):
        key = f"L{layer}H{head}"
        head_data[key] = {
            "layer": layer,
            "head": head,
            "dominant_frequency": int(result.dominant_frequency),
            "power_ratio": float(result.dominant_frequency_power_ratio),
            "power_spectrum": result.power_spectrum.tolist(),
        }

    # Build summary
    summary = {
        "timestamp": timestamp,
        "model_key": model_key,
        "n_layers_analyzed": len(layer_results),
        "n_significant_heads": len(head_results),
        "layer_results": layer_data,
        "head_results": head_data,
        "config": _numpy_to_python(config),
    }

    # Save
    output_path = os.path.join(output_dir, f"fourier_results_{model_key}_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Fourier Discovery for Arithmetic Circuit Analysis"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g. configs/arithmetic_pythia_config.yaml)",
    )
    parser.add_argument(
        "--model_key", type=str, default=None,
        help=f"Override model key. Available: {list_available_models()}",
    )
    parser.add_argument(
        "--layers", type=str, default=None,
        help="Comma-separated layer indices (e.g. '6,12,18'). Default: all layers.",
    )
    parser.add_argument(
        "--scan_heads", action="store_true",
        help="Also run per-head Fourier analysis (slower but more detailed).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device override (e.g. 'mps', 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Override output directory.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load and validate YAML config."""
    import yaml

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Sanity checks
    assert "model" in config, "Config must have a 'model' section"
    assert "fourier" in config, "Config must have a 'fourier' section"
    assert "arithmetic" in config, "Config must have an 'arithmetic' section"

    return config


def main():
    """Main entry point for Fourier discovery."""
    args = parse_args()
    config = load_config(args.config)

    # Resolve model key
    model_key = args.model_key or config["model"].get("key", config["model"]["name"])
    logger.info(f"Model key: {model_key}")

    # Resolve device
    device_str = args.device or config.get("training", {}).get("device", None)

    # Resolve output directory
    output_dir = args.output_dir or config["fourier"].get("output_dir", "fourier_results")

    # Resolve layers
    layers = None
    if args.layers:
        layers = [int(x.strip()) for x in args.layers.split(",")]
    elif config["fourier"].get("layers") is not None:
        layers = config["fourier"]["layers"]

    # ---- Load model ----
    logger.info("=" * 60)
    logger.info("PHASE 1: FOURIER DISCOVERY")
    logger.info("=" * 60)

    model, spec = load_model(model_key, device=device_str)
    device = next(model.parameters()).device
    logger.info(f"Model loaded: {spec.transformer_lens_name} on {device}")
    logger.info(f"Architecture: {spec.n_layers}L / {spec.n_heads}H / d={spec.d_model}")

    # ---- Verify single-token numbers ----
    logger.info("Verifying single-token number encoding...")
    token_check = verify_single_token_numbers(model)
    all_single = all(token_check.values())
    if all_single:
        logger.info("All digits 0-9 are single tokens.")
    else:
        multi_token = [n for n, ok in token_check.items() if not ok]
        logger.warning(f"Multi-token digits: {multi_token}. Results may need adjustment.")

    # ---- Generate prompts ----
    arith_cfg = config["arithmetic"]
    op_range = range(
        arith_cfg.get("operand_range_start", 0),
        arith_cfg.get("operand_range_end", 10),
    )
    prompt_gen = ArithmeticPromptGenerator(
        operand_range=op_range,
        operation=arith_cfg.get("operation", "add"),
        prompt_template=arith_cfg.get("prompt_template", "{a} + {b} ="),
        shuffle=False,  # Order matters: prompt i → number i for DFT
    )
    logger.info(f"Generated {len(prompt_gen)} arithmetic prompts")

    # For Fourier analysis we need prompts ordered by the "number value" they
    # represent.  For single-digit addition with range(0,10), we use one
    # operand as the varying number.  Strategy: fix b=0 so prompt i represents
    # number i (a = i, b = 0, answer = i).
    #
    # NOTE: This measures "does the representation of n+0 vary periodically
    # with n?" — correct for Phase 1 (finding helical encoding of individual
    # numbers).  Phase 4a/4b (sum representation periodicity) will need
    # prompts where both operands vary.
    fourier_prompts = []
    for n in op_range:
        sample = prompt_gen.get_by_operands(n, 0)
        if sample is None:
            raise ValueError(
                f"Could not find prompt for operands ({n}, 0). "
                "Ensure operand_range includes 0."
            )
        fourier_prompts.append(sample)

    logger.info(
        f"Fourier analysis prompts: {len(fourier_prompts)} "
        f"('{fourier_prompts[0].prompt}' → '{fourier_prompts[-1].prompt}')"
    )

    # ---- Run Fourier discovery ----
    fourier_cfg = config["fourier"]
    discovery = FourierDiscovery(model, device=device)

    logger.info("Running layer-wise Fourier analysis...")
    layer_results = discovery.run_all_layers(
        prompts=fourier_prompts,
        layers=layers,
        position=fourier_cfg.get("position", "last"),
        include_resid_post=fourier_cfg.get("include_resid_post", False),
        store_activations=fourier_cfg.get("store_activations", False),
        store_dft=fourier_cfg.get("store_dft", False),
        batch_size=fourier_cfg.get("batch_size", 16),
    )

    # Print summary
    summary_text = FourierDiscovery.summarize(layer_results)
    print("\n" + summary_text + "\n")

    # ---- Optional: per-head analysis ----
    head_results = {}
    if args.scan_heads:
        logger.info("Running per-head Fourier analysis...")
        threshold = fourier_cfg.get("head_power_ratio_threshold", 3.0)
        head_results = discovery.analyze_attention_heads(
            prompts=fourier_prompts,
            layers=layers,
            position=fourier_cfg.get("position", "last"),
            batch_size=fourier_cfg.get("batch_size", 16),
            power_ratio_threshold=threshold,
        )
        logger.info(f"Found {len(head_results)} significant heads")

        # Print head summary
        if head_results:
            print("\nSIGNIFICANT ATTENTION HEADS:")
            print("-" * 50)
            for (layer, head), result in sorted(
                head_results.items(),
                key=lambda x: x[1].dominant_frequency_power_ratio,
                reverse=True,
            ):
                print(
                    f"  L{layer}H{head}: freq={result.dominant_frequency}, "
                    f"ratio={result.dominant_frequency_power_ratio:.1f}x"
                )

    # ---- Save results ----
    output_path = save_results(
        layer_results=layer_results,
        head_results=head_results,
        config=config,
        output_dir=output_dir,
        model_key=model_key,
    )

    # ---- Generate plots ----
    logger.info("Generating visualizations...")
    figs = plot_all(
        layer_results=layer_results,
        head_results=head_results if head_results else None,
        n_layers=spec.n_layers,
        n_heads=spec.n_heads,
        output_dir=output_dir,
        model_key=model_key,
        save=True,
    )
    logger.info(f"Generated {len(figs)} plots in {output_dir}/plots/")

    # ---- Append to experiment history (persistent, append-only) ----
    best = max(layer_results, key=lambda r: r.dominant_frequency_power_ratio)
    history_summary = {
        "n_layers_analyzed": len(layer_results),
        "best_layer": best.layer,
        "best_freq": int(best.dominant_frequency),
        "best_ratio": float(best.dominant_frequency_power_ratio),
        "n_significant_heads": len(head_results),
        "all_single_token": all(token_check.values()),
    }
    append_experiment_result(
        experiment_type="fourier_discovery",
        model_key=model_key,
        results_summary=history_summary,
        config=config,
        output_path=output_path,
        notes=f"Phase 1 Fourier discovery on {model_key}",
    )

    logger.info("=" * 60)
    logger.info("FOURIER DISCOVERY COMPLETE")
    logger.info(f"Results: {output_path}")
    logger.info(f"Plots:   {output_dir}/plots/")
    logger.info(f"History: experiment_history.jsonl")
    logger.info("=" * 60)

    return layer_results, head_results


if __name__ == "__main__":
    main()
