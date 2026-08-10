#!/usr/bin/env python3
"""
Multi-Model Analysis Runner
============================
Runs the full mathematical toolkit + Fisher causal intervention suite
on multiple models sequentially. Each model is loaded, analyzed, and
unloaded before proceeding to the next to manage memory.

Usage:
    python experiments/run_multi_model_suite.py
    python experiments/run_multi_model_suite.py --models pythia-6.9b,llama-3b
    python experiments/run_multi_model_suite.py --skip-fisher  # toolkit only
    python experiments/run_multi_model_suite.py --skip-toolkit  # fisher only
"""

import os
import sys
import gc
import time
import json
import subprocess
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model configurations: (key, layers_for_fisher, operand_range, n_fisher_problems)
# Layers chosen at ~60-80% depth where arithmetic computation is concentrated
MODEL_CONFIGS = {
    "pythia-6.9b": {
        "fisher_layers": "20,24",
        "operand_range": 20,       # smaller grid for 6.9B memory
        "n_fisher_problems": 100,  # fewer problems for memory
        "n_test_problems": 80,
        "batch_size": 2,
        "notes": "32 layers, d=4096. ~14GB in bfloat16.",
    },
    "gemma-7b": {
        "fisher_layers": "18,22",
        "operand_range": 20,
        "n_fisher_problems": 100,
        "n_test_problems": 80,
        "batch_size": 2,
        "notes": "28 layers, d=3072. ~14GB in bfloat16.",
    },
    "llama-3b": {
        "fisher_layers": "18,22",
        "operand_range": 25,       # can afford slightly more for 3B
        "n_fisher_problems": 150,
        "n_test_problems": 100,
        "batch_size": 4,
        "notes": "28 layers, d=3072. ~6GB in bfloat16.",
    },
}

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PYTHON = sys.executable


def run_mathematical_toolkit(model_key: str, config: dict) -> bool:
    """Run the full 5-analysis mathematical toolkit on one model."""
    logger.info(f"\n{'='*70}")
    logger.info(f"MATHEMATICAL TOOLKIT: {model_key}")
    logger.info(f"{'='*70}")
    logger.info(f"Config: {config['notes']}")

    cmd = [
        PYTHON, str(SCRIPT_DIR / "mathematical_toolkit.py"),
        "--model", model_key,
        "--operand-range", str(config["operand_range"]),
        "--batch-size", str(config["batch_size"]),
        "--analysis", "all",
    ]

    logger.info(f"Command: {' '.join(cmd)}")
    start = time.time()

    try:
        result = subprocess.run(
            cmd, cwd=str(PROJECT_DIR),
            capture_output=False,  # stream output to console
            timeout=7200,  # 2 hour timeout per model
        )
        elapsed = time.time() - start
        if result.returncode == 0:
            logger.info(f"  Mathematical toolkit SUCCEEDED for {model_key} ({elapsed:.0f}s)")
            return True
        else:
            logger.error(f"  Mathematical toolkit FAILED for {model_key} (exit code {result.returncode}, {elapsed:.0f}s)")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"  Mathematical toolkit TIMED OUT for {model_key}")
        return False
    except Exception as e:
        logger.error(f"  Mathematical toolkit ERROR for {model_key}: {e}")
        return False


def run_fisher_intervention(model_key: str, config: dict) -> bool:
    """Run the Fisher causal intervention suite on one model."""
    logger.info(f"\n{'='*70}")
    logger.info(f"FISHER INTERVENTION SUITE: {model_key}")
    logger.info(f"{'='*70}")

    cmd = [
        PYTHON, str(SCRIPT_DIR / "fisher_phase_shift.py"),
        "--model", model_key,
        "--layers", config["fisher_layers"],
        "--operand-range", str(config["operand_range"]),
        "--n-fisher-problems", str(config["n_fisher_problems"]),
        "--n-test-problems", str(config["n_test_problems"]),
    ]

    logger.info(f"Command: {' '.join(cmd)}")
    start = time.time()

    try:
        result = subprocess.run(
            cmd, cwd=str(PROJECT_DIR),
            capture_output=False,
            timeout=7200,
        )
        elapsed = time.time() - start
        if result.returncode == 0:
            logger.info(f"  Fisher intervention SUCCEEDED for {model_key} ({elapsed:.0f}s)")
            return True
        else:
            logger.error(f"  Fisher intervention FAILED for {model_key} (exit code {result.returncode}, {elapsed:.0f}s)")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"  Fisher intervention TIMED OUT for {model_key}")
        return False
    except Exception as e:
        logger.error(f"  Fisher intervention ERROR for {model_key}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Multi-Model Analysis Runner")
    parser.add_argument("--models", type=str, default="pythia-6.9b,gemma-7b,llama-3b",
                        help="Comma-separated model keys to run")
    parser.add_argument("--skip-toolkit", action="store_true",
                        help="Skip mathematical toolkit (run Fisher only)")
    parser.add_argument("--skip-fisher", action="store_true",
                        help="Skip Fisher intervention (run toolkit only)")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]

    # Validate model keys
    for m in models:
        if m not in MODEL_CONFIGS:
            logger.error(f"Unknown model key: {m}. Available: {list(MODEL_CONFIGS.keys())}")
            sys.exit(1)

    logger.info(f"\n{'#'*70}")
    logger.info(f"# MULTI-MODEL ANALYSIS SUITE")
    logger.info(f"# Models: {models}")
    logger.info(f"# Toolkit: {'YES' if not args.skip_toolkit else 'SKIP'}")
    logger.info(f"# Fisher:  {'YES' if not args.skip_fisher else 'SKIP'}")
    logger.info(f"{'#'*70}")

    results = {}
    total_start = time.time()

    for model_key in models:
        config = MODEL_CONFIGS[model_key]
        model_results = {"model": model_key}

        logger.info(f"\n\n{'*'*70}")
        logger.info(f"*** STARTING: {model_key} ***")
        logger.info(f"*** {config['notes']} ***")
        logger.info(f"{'*'*70}")

        # Run mathematical toolkit
        if not args.skip_toolkit:
            model_results["toolkit"] = run_mathematical_toolkit(model_key, config)
        else:
            model_results["toolkit"] = "skipped"

        # Force garbage collection between runs
        gc.collect()

        # Run Fisher intervention suite
        if not args.skip_fisher:
            model_results["fisher"] = run_fisher_intervention(model_key, config)
        else:
            model_results["fisher"] = "skipped"

        # Force garbage collection before next model
        gc.collect()

        results[model_key] = model_results
        logger.info(f"\n  {model_key} complete: toolkit={model_results['toolkit']}, fisher={model_results['fisher']}")

    total_elapsed = time.time() - total_start

    # Final summary
    logger.info(f"\n\n{'='*70}")
    logger.info(f"MULTI-MODEL SUITE COMPLETE ({total_elapsed:.0f}s total)")
    logger.info(f"{'='*70}")
    for model_key, res in results.items():
        toolkit_status = "PASS" if res["toolkit"] is True else ("SKIP" if res["toolkit"] == "skipped" else "FAIL")
        fisher_status = "PASS" if res["fisher"] is True else ("SKIP" if res["fisher"] == "skipped" else "FAIL")
        logger.info(f"  {model_key}: toolkit={toolkit_status}, fisher={fisher_status}")

    # Save summary
    output_dir = Path("mathematical_toolkit_results")
    output_dir.mkdir(exist_ok=True)
    summary_file = output_dir / f"multi_model_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "models": models,
            "results": {k: {kk: str(vv) for kk, vv in v.items()} for k, v in results.items()},
            "total_seconds": total_elapsed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)
    logger.info(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()
