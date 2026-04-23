#!/usr/bin/env python3
"""
Generalization Tests: Subspace Substitution, Subtraction, Multi-digit

Three quick experiments to test whether the Fourier encoding generalizes:
  1. Subspace substitution — swap Fourier projections between problems
     with different answer digits → does the output change to the donor digit?
  2. Subtraction — does the ones-digit of a-b use the same Fourier basis?
  3. Multi-digit (tens digit) — does the tens digit also have Fourier structure?

All tests run on Gemma 2B (best-characterized model, teacher-forced).

Usage:
    python generalization_tests.py --device mps
    python generalization_tests.py --device mps --test substitution
    python generalization_tests.py --device mps --test subtraction
    python generalization_tests.py --device mps --test multidigit
"""

import argparse
import json
import logging
import sys
import random
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from arithmetic_circuit_scan_updated import (
    MODEL_MAP,
    generate_teacher_forced_problems,
    filter_correct_teacher_forced,
    get_context_target_tok,
    get_digit_token_ids,
    log_section,
    serialise,
)
from eigenvector_dft import dft_of_scores, analyze_layer, collect_per_digit_means

RESULTS_DIR = Path("mathematical_toolkit_results")
RESULTS_DIR.mkdir(exist_ok=True)

random.seed(42)
np.random.seed(42)


# ═══════════════════════════════════════════════════════════════
# TEST 1: SUBSPACE SUBSTITUTION
# ═══════════════════════════════════════════════════════════════

def run_substitution_test(model, device, comp_layer=19, readout_layer=25, addition_basis=None, correct_problems=None):
    """
    Swap the 9D Fourier projection between problems with different answer digits.
    If the model outputs the DONOR's digit, the Fourier subspace carries digit identity.
    """
    log_section("TEST 1: SUBSPACE SUBSTITUTION (Fourier Identity Transfer)")
    logger.info(f"  Comp layer: L{comp_layer}, Readout layer: L{readout_layer}")

    # Use pre-built correct problems if provided, otherwise generate
    if correct_problems is not None:
        correct = correct_problems
    else:
        flat, by_digit = generate_teacher_forced_problems(n_per_digit=50, operand_max=30)
        correct = filter_correct_teacher_forced(model, flat, max_n=300)

    # Group correct problems by ones digit
    by_ones = defaultdict(list)
    for p in correct:
        by_ones[p["ones_digit"]].append(p)

    logger.info(f"  Correct problems by digit: {dict((d, len(ps)) for d, ps in sorted(by_ones.items()))}")

    # Use pre-built Fourier basis or build one
    if addition_basis is not None:
        basis = addition_basis
        logger.info(f"  Using pre-built addition Fourier basis: {basis.shape}")
    else:
        logger.info(f"\n  Building Fourier basis at L{comp_layer}...")
        means = collect_per_digit_means(model, correct, comp_layer, device)
        grand_mean = means.mean(axis=0, keepdims=True)
        centered = means - grand_mean
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        basis = Vt[:9].T  # (d_model, 9)
        logger.info(f"  Basis shape: {basis.shape}, top singular values: {S[:9].round(2)}")

    # Also build a random control basis
    d_model = basis.shape[0]
    rng = np.random.RandomState(99)
    rand_basis = np.linalg.qr(rng.randn(d_model, 9))[0]  # (d_model, 9) orthonormal

    hook_name = f"blocks.{comp_layer}.hook_resid_post"
    basis_t = torch.tensor(basis, dtype=torch.float32).to(device)
    rand_basis_t = torch.tensor(rand_basis, dtype=torch.float32).to(device)

    results = {"transfer_to_donor": 0, "transfer_to_other": 0,
               "stays_recipient": 0, "total": 0}
    rand_results = {"transfer_to_donor": 0, "transfer_to_other": 0,
                    "stays_recipient": 0, "total": 0}
    per_shift = defaultdict(lambda: {"correct": 0, "total": 0})

    n_test = 0
    max_test = 200

    digits_available = [d for d in range(10) if len(by_ones[d]) >= 2]

    for recip_digit in digits_available:
        for donor_digit in digits_available:
            if recip_digit == donor_digit:
                continue
            if n_test >= max_test:
                break

            # Cycle through available problems (not just the first)
            recip_idx = n_test % len(by_ones[recip_digit])
            donor_idx = (n_test // 10) % len(by_ones[donor_digit])
            recip_prob = by_ones[recip_digit][recip_idx]
            donor_prob = by_ones[donor_digit][donor_idx]

            recip_tokens = recip_prob["_tokens"].to(device)
            donor_tokens = donor_prob["_tokens"].to(device)

            # Get donor's activation at comp layer
            donor_holder = {}
            def capture_donor(act, hook, h=donor_holder):
                h["act"] = act.detach().clone()
                return act

            with torch.no_grad():
                with model.hooks(fwd_hooks=[(hook_name, capture_donor)]):
                    model(donor_tokens)
            donor_act = donor_holder["act"][0, -1]  # (d_model,)

            # Run recipient with substituted Fourier projection
            def substitute_hook(act, hook):
                a = act.clone()
                orig = a[0, -1]  # (d_model,)
                # Project out recipient's Fourier component
                proj_recip = basis_t @ (basis_t.T @ orig.float())
                # Project donor's Fourier component
                proj_donor = basis_t @ (basis_t.T @ donor_act.float())
                # Substitute: remove recipient's, add donor's
                a[0, -1] = (orig.float() - proj_recip + proj_donor).to(a.dtype)
                return a

            with torch.no_grad():
                with model.hooks(fwd_hooks=[(hook_name, substitute_hook)]):
                    logits = model(recip_tokens)

            pred_tok = logits[0, -1].argmax().item()
            pred_str = model.to_string([pred_tok]).strip()

            # Classify result
            results["total"] += 1
            shift = (donor_digit - recip_digit) % 10

            if pred_str.isdigit():
                pred_digit = int(pred_str)
                if pred_digit == donor_digit:
                    results["transfer_to_donor"] += 1
                    per_shift[shift]["correct"] += 1
                elif pred_digit == recip_digit:
                    results["stays_recipient"] += 1
                else:
                    results["transfer_to_other"] += 1
            else:
                results["transfer_to_other"] += 1

            per_shift[shift]["total"] += 1

            # Random control: swap random 9D projection
            def substitute_random_hook(act, hook):
                a = act.clone()
                orig = a[0, -1]
                proj_recip_r = rand_basis_t @ (rand_basis_t.T @ orig.float())
                proj_donor_r = rand_basis_t @ (rand_basis_t.T @ donor_act.float())
                a[0, -1] = (orig.float() - proj_recip_r + proj_donor_r).to(a.dtype)
                return a

            with torch.no_grad():
                with model.hooks(fwd_hooks=[(hook_name, substitute_random_hook)]):
                    rand_logits = model(recip_tokens)

            rand_pred_tok = rand_logits[0, -1].argmax().item()
            rand_pred_str = model.to_string([rand_pred_tok]).strip()
            rand_results["total"] += 1
            if rand_pred_str.isdigit():
                rand_pred_digit = int(rand_pred_str)
                if rand_pred_digit == donor_digit:
                    rand_results["transfer_to_donor"] += 1
                elif rand_pred_digit == recip_digit:
                    rand_results["stays_recipient"] += 1
                else:
                    rand_results["transfer_to_other"] += 1
            else:
                rand_results["transfer_to_other"] += 1

            n_test += 1

        if n_test >= max_test:
            break

    # Report
    total = results["total"]
    logger.info(f"\n{'='*60}")
    logger.info(f"  SUBSTITUTION RESULTS (N={total})")
    logger.info(f"{'='*60}")
    logger.info(f"  Transfer to DONOR digit:    {results['transfer_to_donor']}/{total} = {100*results['transfer_to_donor']/total:.1f}%")
    logger.info(f"  Stays at RECIPIENT digit:   {results['stays_recipient']}/{total} = {100*results['stays_recipient']/total:.1f}%")
    logger.info(f"  Other digit:                {results['transfer_to_other']}/{total} = {100*results['transfer_to_other']/total:.1f}%")
    logger.info(f"  Chance (random digit):      10.0%")
    rt = rand_results["total"]
    if rt > 0:
        logger.info(f"\n  RANDOM CONTROL (swap random 9D):")
        logger.info(f"    Transfer to donor:  {rand_results['transfer_to_donor']}/{rt} = {100*rand_results['transfer_to_donor']/rt:.1f}%")
        logger.info(f"    Stays at recipient: {rand_results['stays_recipient']}/{rt} = {100*rand_results['stays_recipient']/rt:.1f}%")

    logger.info(f"\n  Per-shift breakdown:")
    for shift in sorted(per_shift.keys()):
        s = per_shift[shift]
        rate = 100 * s["correct"] / s["total"] if s["total"] > 0 else 0
        logger.info(f"    Shift {shift:+d}: {s['correct']}/{s['total']} = {rate:.1f}%")

    return {
        "test": "substitution",
        "comp_layer": comp_layer,
        "results": results,
        "per_shift": {str(k): v for k, v in per_shift.items()},
        "transfer_rate": results["transfer_to_donor"] / total if total > 0 else 0,
        "random_control": rand_results,
    }


# ═══════════════════════════════════════════════════════════════
# TEST 2: SUBTRACTION
# ═══════════════════════════════════════════════════════════════

def generate_subtraction_problems(n_per_digit=50, operand_max=30):
    """Generate teacher-forced subtraction problems: a - b = c where a >= b."""
    by_digit = defaultdict(list)

    for a in range(operand_max + 1):
        for b in range(a + 1):  # a >= b so result >= 0
            answer = a - b
            ones = answer % 10
            answer_str = str(answer)

            # Teacher-forced: model sees everything up to ones digit
            if len(answer_str) == 1:
                prompt = f"Calculate:\n{a} - {b} = "
            else:
                prompt = f"Calculate:\n{a} - {b} = {answer_str[:-1]}"

            by_digit[ones].append({
                "prompt": prompt,
                "answer": answer,
                "ones_digit": ones,
                "target_str": answer_str[-1],
                "a": a, "b": b,
                "has_carry": int((a % 10) < (b % 10)),  # borrow
            })

    flat = []
    for d in range(10):
        pool = list(by_digit[d])
        random.shuffle(pool)
        flat.extend(pool[:n_per_digit])
    random.shuffle(flat)

    logger.info(f"  Generated {len(flat)} subtraction problems")
    for d in range(10):
        logger.info(f"    Digit {d}: {len(by_digit[d])} available, sampled {min(n_per_digit, len(by_digit[d]))}")
    return flat, by_digit


def run_subtraction_test(model, device, comp_layer=19, addition_basis=None):
    """
    Check if subtraction ones-digit uses the same Fourier encoding as addition.

    Two tests:
    A) Compute per-digit mean activations for subtraction → SVD → DFT analysis
    B) Compare subtraction Fourier basis with addition Fourier basis (subspace alignment)
    """
    log_section("TEST 2: SUBTRACTION FOURIER STRUCTURE")

    # Generate subtraction problems
    flat, by_digit = generate_subtraction_problems(n_per_digit=50, operand_max=30)

    # Filter for correct predictions
    correct = filter_correct_teacher_forced(model, flat, max_n=300)

    if len(correct) < 30:
        logger.warning(f"  Only {len(correct)} correct subtraction problems — model may not support subtraction well")
        if len(correct) < 10:
            logger.error("  Too few correct problems, skipping subtraction test")
            return {"test": "subtraction", "status": "insufficient_data", "n_correct": len(correct)}

    # Collect per-digit means at computation layer
    logger.info(f"\n  Collecting per-digit means at L{comp_layer}...")
    means = collect_per_digit_means(model, correct, comp_layer, device)

    # SVD
    grand_mean = means.mean(axis=0, keepdims=True)
    centered = means - grand_mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # DFT analysis
    sub_results, is_perfect = analyze_layer(
        f"Subtraction ones-digit (L{comp_layer})", U.T, S
    )

    # Compare with addition basis if provided
    alignment = None
    if addition_basis is not None:
        sub_basis = Vt[:9].T  # (d_model, 9)
        # Principal angles between subspaces
        M = addition_basis.T @ sub_basis  # (9, 9)
        _, cos_angles, _ = np.linalg.svd(M)
        cos_angles = np.clip(cos_angles, 0, 1)
        alignment = cos_angles.tolist()
        logger.info(f"\n  Subspace alignment (addition ↔ subtraction):")
        logger.info(f"    Principal cosines: {[f'{c:.3f}' for c in cos_angles]}")
        logger.info(f"    Mean cosine: {np.mean(cos_angles):.3f}")
        if np.mean(cos_angles) > 0.9:
            logger.info(f"    ★ HIGH ALIGNMENT: subtraction uses ~same subspace as addition!")
        elif np.mean(cos_angles) > 0.7:
            logger.info(f"    ◆ MODERATE ALIGNMENT: partial overlap")
        else:
            logger.info(f"    ✗ LOW ALIGNMENT: different subspaces")

    return {
        "test": "subtraction",
        "comp_layer": comp_layer,
        "n_correct": len(correct),
        "is_perfect_fourier": is_perfect,
        "directions": sub_results,
        "singular_values": S[:9].tolist(),
        "alignment_with_addition": alignment,
    }


# ═══════════════════════════════════════════════════════════════
# TEST 3: MULTI-DIGIT (TENS DIGIT)
# ═══════════════════════════════════════════════════════════════

def generate_tens_digit_problems(n_per_digit=50, operand_max=99):
    """
    Generate teacher-forced problems where the TARGET is the TENS digit.
    Format: "Calculate:\n45 + 37 = " → model predicts "8" (tens digit of 82)
    Only for answers >= 10 (otherwise no tens digit).
    """
    by_tens = defaultdict(list)

    for a in range(operand_max + 1):
        for b in range(operand_max + 1):
            answer = a + b
            if answer < 10:
                continue  # no tens digit
            tens = (answer // 10) % 10
            answer_str = str(answer)
            # Target: the tens digit (first digit of answer for 2-digit, second-to-last for 3-digit)
            target_str = str(tens)

            # For teacher-forced tens digit prediction:
            # The model sees "a + b = " and must predict the first digit of the answer
            # (For 3-digit answers like 198, it sees "a + b = 1" and predicts "9")
            if answer >= 100:
                hundreds = answer_str[0]
                prompt = f"Calculate:\n{a} + {b} = {hundreds}"
            else:
                prompt = f"Calculate:\n{a} + {b} = "

            by_tens[tens].append({
                "prompt": prompt,
                "answer": answer,
                "ones_digit": tens,  # reuse key for compatibility with collect_per_digit_means
                "target_str": target_str,
                "a": a, "b": b,
                "actual_tens": tens,
                "actual_ones": answer % 10,
            })

    flat = []
    for d in range(10):
        pool = list(by_tens[d])
        random.shuffle(pool)
        flat.extend(pool[:n_per_digit])
    random.shuffle(flat)

    logger.info(f"  Generated {len(flat)} tens-digit problems")
    for d in range(10):
        logger.info(f"    Tens={d}: {len(by_tens[d])} available, sampled {min(n_per_digit, len(by_tens[d]))}")
    return flat, by_tens


def run_multidigit_test(model, device, comp_layer=19, addition_basis=None):
    """
    Check if the tens digit also has Fourier structure at the computation layer.

    The key question: does ℤ/10ℤ Fourier structure extend to other digit positions,
    or is it specific to the ones digit?
    """
    log_section("TEST 3: MULTI-DIGIT (TENS DIGIT FOURIER STRUCTURE)")

    flat, by_tens = generate_tens_digit_problems(n_per_digit=50, operand_max=50)

    # Filter for correct predictions
    correct = filter_correct_teacher_forced(model, flat, max_n=300)

    if len(correct) < 30:
        logger.warning(f"  Only {len(correct)} correct tens-digit problems")
        if len(correct) < 10:
            logger.error("  Too few correct problems, skipping multi-digit test")
            return {"test": "multidigit", "status": "insufficient_data", "n_correct": len(correct)}

    # Collect per-digit means (grouped by tens digit)
    logger.info(f"\n  Collecting per-tens-digit means at L{comp_layer}...")
    means = collect_per_digit_means(model, correct, comp_layer, device)

    # SVD
    grand_mean = means.mean(axis=0, keepdims=True)
    centered = means - grand_mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # DFT analysis
    tens_results, is_perfect = analyze_layer(
        f"Tens digit (L{comp_layer})", U.T, S
    )

    # Compare with addition (ones digit) basis
    alignment = None
    if addition_basis is not None:
        tens_basis = Vt[:9].T  # (d_model, 9)
        M = addition_basis.T @ tens_basis
        _, cos_angles, _ = np.linalg.svd(M)
        cos_angles = np.clip(cos_angles, 0, 1)
        alignment = cos_angles.tolist()
        logger.info(f"\n  Subspace alignment (ones-digit addition ↔ tens-digit):")
        logger.info(f"    Principal cosines: {[f'{c:.3f}' for c in cos_angles]}")
        logger.info(f"    Mean cosine: {np.mean(cos_angles):.3f}")
        if np.mean(cos_angles) > 0.9:
            logger.info(f"    ★ HIGH ALIGNMENT: tens digit uses ~same subspace!")
        elif np.mean(cos_angles) > 0.7:
            logger.info(f"    ◆ MODERATE ALIGNMENT: partial overlap")
        else:
            logger.info(f"    ✗ LOW ALIGNMENT: different subspaces (expected — different ℤ/10ℤ)")

    return {
        "test": "multidigit",
        "comp_layer": comp_layer,
        "n_correct": len(correct),
        "is_perfect_fourier": is_perfect,
        "directions": tens_results,
        "singular_values": S[:9].tolist(),
        "alignment_with_ones_addition": alignment,
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generalization Tests")
    parser.add_argument("--model", default="gemma-2b", choices=list(MODEL_MAP.keys()))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--comp-layer", type=int, default=None)
    parser.add_argument("--test", default="all",
                        choices=["all", "substitution", "subtraction", "multidigit"])
    args = parser.parse_args()

    comp_defaults = {"gemma-2b": 19, "phi-3": 26, "llama-3b": 20}
    readout_defaults = {"gemma-2b": 25, "phi-3": 31, "llama-3b": 27}
    comp_layer = args.comp_layer or comp_defaults.get(args.model, 19)
    readout_layer = readout_defaults.get(args.model, 25)

    model_name = MODEL_MAP[args.model]
    device = args.device

    logger.info(f"Model: {args.model} ({model_name})")
    logger.info(f"Computation layer: L{comp_layer}")
    logger.info(f"Device: {device}")
    logger.info(f"Test: {args.test}")

    # Load model
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=torch.float32 if device == "cpu" else torch.float16,
    )
    model.eval()

    all_results = {}

    # First, build the addition Fourier basis (shared reference)
    log_section("BUILDING ADDITION FOURIER BASIS (reference)")
    add_flat, add_by_digit = generate_teacher_forced_problems(n_per_digit=80, operand_max=50)
    add_correct = filter_correct_teacher_forced(model, add_flat, max_n=500)

    add_means = collect_per_digit_means(model, add_correct, comp_layer, device)
    add_centered = add_means - add_means.mean(axis=0, keepdims=True)
    add_U, add_S, add_Vt = np.linalg.svd(add_centered, full_matrices=False)
    addition_basis = add_Vt[:9].T  # (d_model, 9)
    logger.info(f"  Addition basis built: {addition_basis.shape}")
    analyze_layer(f"Addition ones-digit reference (L{comp_layer})", add_U.T, add_S)

    # Run requested tests
    if args.test in ("all", "substitution"):
        all_results["substitution"] = run_substitution_test(
            model, device, comp_layer, readout_layer,
            addition_basis=addition_basis, correct_problems=add_correct
        )

    if args.test in ("all", "subtraction"):
        all_results["subtraction"] = run_subtraction_test(
            model, device, comp_layer, addition_basis
        )

    if args.test in ("all", "multidigit"):
        all_results["multidigit"] = run_multidigit_test(
            model, device, comp_layer, addition_basis
        )

    # Save results
    out_path = RESULTS_DIR / f"generalization_tests_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump(serialise(all_results), f, indent=2)
    logger.info(f"\n{'='*60}")
    logger.info(f"  Results saved to {out_path}")
    logger.info(f"{'='*60}")

    # Summary
    log_section("SUMMARY")
    if "substitution" in all_results:
        r = all_results["substitution"]
        rate = r["transfer_rate"] * 100
        logger.info(f"  Substitution: {rate:.1f}% transfer to donor digit (chance=10%)")

    if "subtraction" in all_results:
        r = all_results["subtraction"]
        if r.get("status") == "insufficient_data":
            logger.info(f"  Subtraction: SKIPPED (only {r['n_correct']} correct problems)")
        else:
            fourier = "★ PERFECT" if r["is_perfect_fourier"] else "NOT perfect"
            align = f", alignment={np.mean(r['alignment_with_addition']):.3f}" if r.get("alignment_with_addition") else ""
            logger.info(f"  Subtraction: {fourier} Fourier basis{align}")

    if "multidigit" in all_results:
        r = all_results["multidigit"]
        if r.get("status") == "insufficient_data":
            logger.info(f"  Multi-digit: SKIPPED (only {r['n_correct']} correct problems)")
        else:
            fourier = "★ PERFECT" if r["is_perfect_fourier"] else "NOT perfect"
            align = f", alignment={np.mean(r['alignment_with_ones_addition']):.3f}" if r.get("alignment_with_ones_addition") else ""
            logger.info(f"  Tens digit: {fourier} Fourier basis{align}")


if __name__ == "__main__":
    main()
