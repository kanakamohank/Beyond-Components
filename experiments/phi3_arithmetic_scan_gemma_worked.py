#!/usr/bin/env python3
"""
Phi-3 Arithmetic Layer Scan + Unembed-Aligned Patching

Two experiments:
1. Full-layer scan: For each layer 0-31, full activation patch to find which
   layer(s) contain the arithmetic result (highest transfer ceiling).
2. Unembed-aligned patching: At the best layer(s), use SVD of W_U[:, digit_tokens]
   as patching basis instead of Fisher. Tests Explanation A: the arithmetic circuit
   stores results in unembed-aligned directions that Fisher misses.
"""

import argparse
import logging
import json
import sys
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from fisher_patching import (
    generate_teacher_forced_problems,
    filter_correct_teacher_forced,
    get_model_prediction,
    MODEL_MAP,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_digit_token_ids(model):
    """Get token IDs for digits 0-9, handling tokenizers that prepend space."""
    digit_ids = []
    for d in range(10):
        toks = model.to_tokens(str(d), prepend_bos=False)
        digit_ids.append(toks[0, -1].item())  # Use last token (handles space prefix)
    return digit_ids


def compute_unembed_basis(model, digit_token_ids):
    """Compute SVD of W_U restricted to digit token columns.

    W_U has shape (d_model, vocab_size). Extract columns for digit tokens
    to get W_digits of shape (d_model, 10). SVD gives:
        W_digits = U @ S @ V^T
    Top left singular vectors (columns of U) are the directions in residual
    stream space that most affect digit logits.

    Returns:
        U: (d_model, 10) orthonormal basis
        S: (10,) singular values
    """
    W_U = model.W_U.detach().float().cpu()  # (d_model, vocab_size)
    W_digits = W_U[:, digit_token_ids]  # (d_model, 10)

    # Center: subtract mean column to focus on digit-discriminative directions
    W_centered = W_digits - W_digits.mean(dim=1, keepdim=True)

    U, S, Vt = torch.linalg.svd(W_centered, full_matrices=False)
    # U: (d_model, 10), S: (10,), Vt: (10, 10)

    logger.info(f"  Unembed SVD singular values: {S.numpy()}")
    logger.info(f"  Top-1/Top-2 ratio: {S[0]/S[1]:.2f}x")
    logger.info(f"  Variance explained by top-2: {(S[:2]**2).sum()/(S**2).sum():.1%}")
    logger.info(f"  Variance explained by top-5: {(S[:5]**2).sum()/(S**2).sum():.1%}")

    return U.numpy(), S.numpy()


def run_full_patch_at_layer(model, layer, correct_problems):
    """Run ONLY full activation patching at a single layer. Returns transfer stats."""
    hook_name = f"blocks.{layer}.hook_resid_post"

    # Cache clean activations
    for prob in correct_problems:
        tokens = prob["_tokens"]
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
            prob[f"_act_L{layer}"] = cache[hook_name][0, -1].clone()
            del cache

    # Group by ones digit
    by_digit = defaultdict(list)
    for p in correct_problems:
        by_digit[p["ones_digit"]].append(p)

    digits_available = sorted(by_digit.keys())
    results = {"total": 0, "transfer": 0, "changed": 0, "stayed": 0}
    n_pairs_per_combo = 3

    for clean_digit in digits_available:
        for corrupt_digit in digits_available:
            if clean_digit == corrupt_digit:
                continue

            clean_probs = by_digit[clean_digit][:n_pairs_per_combo]
            corrupt_probs = by_digit[corrupt_digit][:n_pairs_per_combo]

            for ci, clean_prob in enumerate(clean_probs):
                if ci >= len(corrupt_probs):
                    break
                corrupt_prob = corrupt_probs[ci]

                clean_act = clean_prob[f"_act_L{layer}"]
                corrupt_act = corrupt_prob[f"_act_L{layer}"]
                corrupt_tokens = corrupt_prob["_tokens"]
                delta = clean_act - corrupt_act

                def hook_fn(act, hook, d=delta):
                    act[:, -1, :] = act[:, -1, :] + d.unsqueeze(0)
                    return act

                with torch.no_grad():
                    with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
                        logits = model(corrupt_tokens)

                pred_tok = logits[0, -1].argmax(dim=-1).item()
                pred_str = model.tokenizer.decode([pred_tok]).strip()

                try:
                    pred_digit = int(pred_str)
                    results["total"] += 1
                    if pred_digit == clean_digit:
                        results["transfer"] += 1
                    if pred_digit != corrupt_digit:
                        results["changed"] += 1
                    if pred_digit == corrupt_digit:
                        results["stayed"] += 1
                except ValueError:
                    results["total"] += 1

    # Clean up cached activations to save memory
    for prob in correct_problems:
        prob.pop(f"_act_L{layer}", None)

    return results


def run_subspace_patch_at_layer(model, layer, basis, correct_problems,
                                 n_dims_list, label="Unembed"):
    """Run full + subspace + ortho patching at a layer with given basis.

    basis: numpy array (d_model, n_directions) — orthonormal columns
    """
    device = next(model.parameters()).device
    hook_name = f"blocks.{layer}.hook_resid_post"

    # Cache clean activations
    logger.info(f"  Caching clean activations at L{layer}...")
    for prob in correct_problems:
        tokens = prob["_tokens"]
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
            prob["_clean_act"] = cache[hook_name][0, -1].clone()
            del cache

    # Group by ones digit
    by_digit = defaultdict(list)
    for p in correct_problems:
        by_digit[p["ones_digit"]].append(p)

    digits_available = sorted(by_digit.keys())
    n_pairs_per_combo = 3

    all_results = {}

    for n_dims in n_dims_list:
        if n_dims > basis.shape[1]:
            logger.info(f"  Skipping {n_dims}D: only {basis.shape[1]} directions available")
            continue

        sub_basis = torch.tensor(
            basis[:, :n_dims].T,  # (n_dims, d_model)
            dtype=torch.float32, device=device
        )

        dim_results = {}
        for patch_mode in ["full", "subspace", "ortho"]:
            res = {"total": 0, "transfer": 0, "changed": 0, "stayed": 0}

            for clean_digit in digits_available:
                for corrupt_digit in digits_available:
                    if clean_digit == corrupt_digit:
                        continue

                    clean_probs = by_digit[clean_digit][:n_pairs_per_combo]
                    corrupt_probs = by_digit[corrupt_digit][:n_pairs_per_combo]

                    for ci, clean_prob in enumerate(clean_probs):
                        if ci >= len(corrupt_probs):
                            break
                        corrupt_prob = corrupt_probs[ci]

                        clean_act = clean_prob["_clean_act"]
                        corrupt_act = corrupt_prob["_clean_act"]
                        corrupt_tokens = corrupt_prob["_tokens"]

                        if patch_mode == "full":
                            delta = clean_act - corrupt_act
                        elif patch_mode == "subspace":
                            clean_proj = sub_basis @ clean_act.float()
                            corrupt_proj = sub_basis @ corrupt_act.float()
                            delta_sub = clean_proj - corrupt_proj
                            delta = (sub_basis.T @ delta_sub).to(clean_act.dtype)
                        elif patch_mode == "ortho":
                            full_delta = clean_act - corrupt_act
                            clean_proj = sub_basis @ clean_act.float()
                            corrupt_proj = sub_basis @ corrupt_act.float()
                            sub_delta = (sub_basis.T @ (clean_proj - corrupt_proj)).to(clean_act.dtype)
                            delta = full_delta - sub_delta

                        def hook_fn(act, hook, d=delta):
                            act[:, -1, :] = act[:, -1, :] + d.unsqueeze(0)
                            return act

                        with torch.no_grad():
                            with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
                                logits = model(corrupt_tokens)

                        pred_tok = logits[0, -1].argmax(dim=-1).item()
                        pred_str = model.tokenizer.decode([pred_tok]).strip()

                        try:
                            pred_digit = int(pred_str)
                            res["total"] += 1
                            if pred_digit == clean_digit:
                                res["transfer"] += 1
                            if pred_digit != corrupt_digit:
                                res["changed"] += 1
                            if pred_digit == corrupt_digit:
                                res["stayed"] += 1
                        except ValueError:
                            res["total"] += 1

            dim_results[patch_mode] = res

        # Log summary
        logger.info(f"\n{'='*60}")
        logger.info(f"{label} PATCHING SUMMARY (Layer {layer}, {n_dims}D)")
        logger.info(f"{'='*60}")
        for name, mode in [("FULL", "full"), (f"{label.upper()}", "subspace"), ("ORTHO", "ortho")]:
            r = dim_results[mode]
            n = r["total"]
            if n > 0:
                logger.info(f"  {name:12s}: transfer={r['transfer']}/{n} ({100*r['transfer']/n:.1f}%), "
                             f"changed={r['changed']}/{n} ({100*r['changed']/n:.1f}%), "
                             f"stayed={r['stayed']}/{n} ({100*r['stayed']/n:.1f}%)")

        all_results[f"{n_dims}D"] = dim_results

    # Cleanup
    for prob in correct_problems:
        prob.pop("_clean_act", None)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Phi-3 Layer Scan + Unembed Patching")
    parser.add_argument("--device", default="cpu", help="Device (cpu recommended)")
    parser.add_argument("--skip-scan", action="store_true", help="Skip full-layer scan")
    parser.add_argument("--unembed-layers", default="24",
                        help="Comma-separated layers for unembed patching")
    args = parser.parse_args()

    from transformer_lens import HookedTransformer

    device = args.device
    # MPS is safe here: no gradient computation, only forward passes + SVD of weights

    model_name = "microsoft/Phi-3-mini-4k-instruct"
    logger.info("=" * 60)
    logger.info("PHI-3 ARITHMETIC LAYER SCAN + UNEMBED PATCHING")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")

    logger.info("Loading Phi-3...")
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=torch.float32)
    n_layers = model.cfg.n_layers
    logger.info(f"  n_layers={n_layers}, d_model={model.cfg.d_model}")

    # Generate and filter test problems (teacher-forced)
    test_problems, _ = generate_teacher_forced_problems(n_per_digit=15, operand_max=99)
    correct = filter_correct_teacher_forced(model, test_problems, max_n=150)
    by_digit_counts = defaultdict(int)
    for p in correct:
        by_digit_counts[p["ones_digit"]] += 1
    logger.info(f"Found {len(correct)} correct teacher-forced test problems")
    logger.info(f"  By digit: {dict(sorted(by_digit_counts.items()))}")

    all_results = {"model": model_name, "device": device, "n_test_problems": len(correct)}

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT 1: Full-layer scan
    # ═══════════════════════════════════════════════════════════
    if not args.skip_scan:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# EXPERIMENT 1: FULL-LAYER SCAN (layers 0-{n_layers-1})")
        logger.info(f"{'#'*60}")

        scan_results = {}
        for layer in range(n_layers):
            logger.info(f"\n  Layer {layer}...")
            res = run_full_patch_at_layer(model, layer, correct)
            n = res["total"]
            if n > 0:
                transfer_pct = 100 * res["transfer"] / n
                changed_pct = 100 * res["changed"] / n
                logger.info(f"    Full patch: transfer={res['transfer']}/{n} ({transfer_pct:.1f}%), "
                             f"changed={res['changed']}/{n} ({changed_pct:.1f}%)")
            else:
                transfer_pct = 0
                changed_pct = 0
                logger.info(f"    Full patch: no valid pairs")

            scan_results[f"layer_{layer}"] = {
                "transfer": res["transfer"],
                "changed": res["changed"],
                "stayed": res["stayed"],
                "total": n,
                "transfer_pct": round(transfer_pct, 1),
                "changed_pct": round(changed_pct, 1),
            }

        all_results["layer_scan"] = scan_results

        # Print summary table
        logger.info(f"\n{'='*60}")
        logger.info("LAYER SCAN SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"{'Layer':>6} {'Transfer%':>10} {'Changed%':>10}")
        logger.info("-" * 30)
        for layer in range(n_layers):
            r = scan_results[f"layer_{layer}"]
            marker = " <<<" if r["transfer_pct"] > 50 else ""
            logger.info(f"  L{layer:>3}: {r['transfer_pct']:>8.1f}%  {r['changed_pct']:>8.1f}%{marker}")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT 2: Unembed-aligned patching
    # ═══════════════════════════════════════════════════════════
    unembed_layers = [int(l) for l in args.unembed_layers.split(",")]

    logger.info(f"\n{'#'*60}")
    logger.info(f"# EXPERIMENT 2: UNEMBED-ALIGNED PATCHING")
    logger.info(f"{'#'*60}")

    digit_token_ids = get_digit_token_ids(model)
    logger.info(f"Digit token IDs: {digit_token_ids}")

    unembed_basis, unembed_svals = compute_unembed_basis(model, digit_token_ids)
    all_results["unembed_singular_values"] = unembed_svals.tolist()

    for layer in unembed_layers:
        logger.info(f"\n{'='*60}")
        logger.info(f"UNEMBED PATCHING AT LAYER {layer}")
        logger.info(f"{'='*60}")

        unembed_results = run_subspace_patch_at_layer(
            model, layer, unembed_basis, correct,
            n_dims_list=[2, 5, 9], label="Unembed"
        )

        # Store results
        layer_key = f"unembed_L{layer}"
        all_results[layer_key] = {}
        for dim_key, modes in unembed_results.items():
            all_results[layer_key][dim_key] = {
                mode: {k: v for k, v in res.items()}
                for mode, res in modes.items()
            }

    # Save
    out_dir = Path("mathematical_toolkit_results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "phi3_layer_scan_unembed.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
