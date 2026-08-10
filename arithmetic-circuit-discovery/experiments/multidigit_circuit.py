#!/usr/bin/env python3
"""
Multi-Digit Arithmetic Circuit Discovery

Four experiments to map how transformers handle multi-digit addition:
  A. Carry-conditioned tens-digit Fourier analysis
  D. Carry signal causal intervention (flip carry → tens digit ±1?)
  C. Carry Router head identification (attention patterns)
  B. Operand digit subspace decomposition (per-position Fourier)

Usage:
    python multidigit_circuit.py --device mps                    # all experiments
    python multidigit_circuit.py --device mps --test A           # just Exp A
    python multidigit_circuit.py --device mps --test A D C B     # specific order
"""

import argparse
import json
import logging
import sys
import random
import time
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path

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
    filter_correct_direct_answer,
    log_section,
    serialise,
)
from eigenvector_dft import analyze_layer, collect_per_digit_means
from carry_stratification import (
    compute_basis_from_means,
    analyze_group_dft,
    principal_angles,
    analyze_carry_directions,
    MIN_SAMPLES_PER_DIGIT,
)

RESULTS_DIR = Path("mathematical_toolkit_results")
RESULTS_DIR.mkdir(exist_ok=True)

random.seed(42)
np.random.seed(42)


# ═══════════════════════════════════════════════════════════════
# SHARED: Problem Generation for Multi-Digit
# ═══════════════════════════════════════════════════════════════

def generate_tens_digit_problems_with_carry(n_per_digit=80, operand_max=50):
    """
    Generate teacher-forced problems targeting the TENS digit.
    Tags each problem with carry status from ones-digit addition.

    For answer 10-99: "Calculate:\na + b = " → predicts tens digit
    For answer 100+:  "Calculate:\na + b = {hundreds}" → predicts tens digit
    """
    by_tens = defaultdict(list)

    for a in range(operand_max + 1):
        for b in range(operand_max + 1):
            answer = a + b
            if answer < 10:
                continue
            tens = (answer // 10) % 10
            ones = answer % 10
            answer_str = str(answer)
            target_str = str(tens)

            # Carry from ones position
            has_carry = int((a % 10) + (b % 10) >= 10)

            if answer >= 100:
                prompt = f"Calculate:\n{a} + {b} = {answer_str[0]}"
            else:
                prompt = f"Calculate:\n{a} + {b} = "

            by_tens[tens].append({
                "prompt": prompt,
                "answer": answer,
                "ones_digit": tens,  # compatibility with collect_per_digit_means
                "target_str": target_str,
                "a": a, "b": b,
                "actual_tens": tens,
                "actual_ones": ones,
                "has_carry": has_carry,
                "a_ones": a % 10, "a_tens": a // 10,
                "b_ones": b % 10, "b_tens": b // 10,
            })

    flat = []
    for d in range(10):
        pool = list(by_tens[d])
        random.shuffle(pool)
        flat.extend(pool[:n_per_digit])
    random.shuffle(flat)

    logger.info(f"  Generated {len(flat)} tens-digit problems")
    for d in range(10):
        n_avail = len(by_tens[d])
        n_sampled = min(n_per_digit, n_avail)
        logger.info(f"    Tens={d}: {n_avail} available, sampled {n_sampled}")
    return flat, by_tens


def generate_tens_digit_problems_fullanswer(n_per_digit=80, operand_max=50):
    """
    Full-answer variant for models whose tokenizer encodes multi-digit numbers
    as single tokens (e.g. LLaMA: "68" → 1 token, not "6"+"8").

    Prompt = "Calculate:\\n{a} + {b} = " (no teacher-forced prefix)
    target_str = full answer string (e.g. "68")
    The tens digit is extracted post-hoc from the predicted number.
    """
    by_tens = defaultdict(list)

    for a in range(operand_max + 1):
        for b in range(operand_max + 1):
            answer = a + b
            if answer < 10:
                continue
            tens = (answer // 10) % 10
            ones = answer % 10
            answer_str = str(answer)
            has_carry = int((a % 10) + (b % 10) >= 10)

            prompt = f"Calculate:\n{a} + {b} = "

            by_tens[tens].append({
                "prompt": prompt,
                "answer": answer,
                "ones_digit": tens,
                "target_str": answer_str,
                "actual_tens": tens,
                "actual_ones": ones,
                "has_carry": has_carry,
                "a": a, "b": b,
                "a_ones": a % 10, "a_tens": a // 10,
                "b_ones": b % 10, "b_tens": b // 10,
            })

    flat = []
    for d in range(10):
        pool = list(by_tens[d])
        random.shuffle(pool)
        flat.extend(pool[:n_per_digit])
    random.shuffle(flat)

    logger.info(f"  Generated {len(flat)} tens-digit problems (full-answer mode)")
    for d in range(10):
        n_avail = len(by_tens[d])
        n_sampled = min(n_per_digit, n_avail)
        logger.info(f"    Tens={d}: {n_avail} available, sampled {n_sampled}")
    return flat, by_tens


def filter_correct_fullanswer_tens(model, problems, max_n=400):
    """
    Filter for full-answer models: accept if model's top prediction is the
    correct full answer as a single token.
    """
    correct = []
    for prob in problems:
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
        target_toks = model.to_tokens(prob["target_str"], prepend_bos=False)
        if target_toks.shape[1] != 1:
            continue
        target_tok = target_toks[0, 0].item()
        with torch.no_grad():
            logits = model(tokens)
        pred = logits[0, -1].argmax().item()
        if pred == target_tok:
            prob["_tokens"] = tokens
            prob["_target_tok"] = target_tok
            correct.append(prob)
        if len(correct) >= max_n:
            break

    counts = defaultdict(int)
    for p in correct:
        counts[p["ones_digit"]] += 1
    logger.info(f"  Correct problems: {len(correct)}/{max_n} attempted")
    logger.info(f"  By digit: {dict(sorted(counts.items()))}")
    return correct


# ═══════════════════════════════════════════════════════════════
# EXP A: Carry-Conditioned Tens-Digit Fourier Analysis
# ═══════════════════════════════════════════════════════════════

def run_exp_a(model, device, comp_layer, d_model):
    """
    Split tens-digit problems by carry=0 vs carry=1.
    Run SVD+DFT on each group separately.
    If carry-conditioning recovers perfect Fourier → carry contamination was the cause.
    """
    log_section("EXP A: CARRY-CONDITIONED TENS-DIGIT FOURIER ANALYSIS")

    flat, _ = generate_tens_digit_problems_with_carry(n_per_digit=80, operand_max=50)
    correct = filter_correct_teacher_forced(model, flat, max_n=500)

    if len(correct) < 30:
        logger.error(f"  Only {len(correct)} correct — too few")
        return {"status": "insufficient_data", "n_correct": len(correct)}

    # Split by carry
    carry_probs = [p for p in correct if p["has_carry"] == 1]
    nocarry_probs = [p for p in correct if p["has_carry"] == 0]
    logger.info(f"  Carry: {len(carry_probs)}, No-carry: {len(nocarry_probs)}")

    # Collect activations
    logger.info(f"  Collecting activations at L{comp_layer}...")
    t0 = time.time()
    hook_name = f"blocks.{comp_layer}.hook_resid_post"

    # Per-tens-digit means for each group
    means_all = np.zeros((10, d_model))
    means_carry = np.zeros((10, d_model))
    means_nocarry = np.zeros((10, d_model))
    counts_all = np.zeros(10, dtype=int)
    counts_carry = np.zeros(10, dtype=int)
    counts_nocarry = np.zeros(10, dtype=int)

    digit_acts_carry = defaultdict(list)
    digit_acts_nocarry = defaultdict(list)

    for prob in correct:
        d = prob["ones_digit"]  # = tens digit of answer
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True).to(device)

        holder = {}
        def capture(act, hook, h=holder):
            h["act"] = act.detach()
            return act

        with torch.no_grad():
            with model.hooks(fwd_hooks=[(hook_name, capture)]):
                model(tokens)

        if "act" in holder:
            act_vec = holder["act"][0, -1].cpu().float().numpy()
            if prob["has_carry"] == 1:
                digit_acts_carry[d].append(act_vec)
            else:
                digit_acts_nocarry[d].append(act_vec)

    for d in range(10):
        nc = len(digit_acts_carry[d])
        nn = len(digit_acts_nocarry[d])
        counts_carry[d] = nc
        counts_nocarry[d] = nn
        counts_all[d] = nc + nn

        if nc > 0:
            means_carry[d] = np.mean(digit_acts_carry[d], axis=0)
        if nn > 0:
            means_nocarry[d] = np.mean(digit_acts_nocarry[d], axis=0)
        if nc + nn > 0:
            all_acts = digit_acts_carry[d] + digit_acts_nocarry[d]
            means_all[d] = np.mean(all_acts, axis=0)

    logger.info(f"  Activations collected in {time.time()-t0:.1f}s")

    logger.info(f"\n  Per-tens-digit distribution:")
    for d in range(10):
        logger.info(f"    Tens={d}: carry={counts_carry[d]}, no-carry={counts_nocarry[d]}")

    # Compute bases
    V_all, S_all, U_all, cent_all, digits_all = compute_basis_from_means(
        means_all, counts_all, "ALL-TENS")
    V_carry, S_carry, U_carry, cent_carry, digits_carry = compute_basis_from_means(
        means_carry, counts_carry, "CARRY-TENS")
    V_nocarry, S_nocarry, U_nocarry, cent_nocarry, digits_nocarry = compute_basis_from_means(
        means_nocarry, counts_nocarry, "NOCARRY-TENS")

    # DFT analysis
    results_sections = {}

    if V_all is not None:
        dft_all, perf_all, pur_all = analyze_group_dft(
            f"ALL tens-digit ({len(correct)} problems)", U_all, S_all, digits_all)
        results_sections["all"] = {
            "perfect_fourier": perf_all, "mean_purity": pur_all,
            "singular_values": S_all.tolist(), "dft": dft_all,
            "digits_used": digits_all,
        }

    if V_carry is not None:
        dft_carry, perf_carry, pur_carry = analyze_group_dft(
            f"CARRY tens-digit ({len(carry_probs)} problems)", U_carry, S_carry, digits_carry)
        results_sections["carry"] = {
            "perfect_fourier": perf_carry, "mean_purity": pur_carry,
            "singular_values": S_carry.tolist(), "dft": dft_carry,
            "digits_used": digits_carry,
        }

    if V_nocarry is not None:
        dft_nocarry, perf_nocarry, pur_nocarry = analyze_group_dft(
            f"NO-CARRY tens-digit ({len(nocarry_probs)} problems)", U_nocarry, S_nocarry, digits_nocarry)
        results_sections["nocarry"] = {
            "perfect_fourier": perf_nocarry, "mean_purity": pur_nocarry,
            "singular_values": S_nocarry.tolist(), "dft": dft_nocarry,
            "digits_used": digits_nocarry,
        }

    # Subspace alignment
    alignment = {}
    if V_carry is not None and V_nocarry is not None:
        k_min = min(V_carry.shape[1], V_nocarry.shape[1])
        angles_cn, cos_cn = principal_angles(V_carry[:, :k_min], V_nocarry[:, :k_min])
        mean_angle = float(np.degrees(angles_cn.mean()))
        logger.info(f"\n  Carry vs No-carry tens subspace: mean angle = {mean_angle:.1f}°")
        alignment["carry_vs_nocarry_mean_angle_deg"] = mean_angle
        alignment["carry_vs_nocarry_angles_deg"] = np.degrees(angles_cn).tolist()

    # Carry direction for tens digit
    carry_dir = None
    if V_all is not None:
        logger.info(f"\n  Carry direction geometry (tens digit):")
        carry_dir = analyze_carry_directions(
            means_all, means_carry, means_nocarry,
            counts_carry, counts_nocarry, V_all, cent_all)

    # Summary
    log_section("EXP A SUMMARY")
    for group in ["all", "carry", "nocarry"]:
        if group in results_sections:
            r = results_sections[group]
            status = "★ PERFECT" if r["perfect_fourier"] else "NOT perfect"
            logger.info(f"  {group.upper()}: {status} Fourier basis, purity={r['mean_purity']:.1f}%")

    return {
        "test": "A_carry_conditioned_tens",
        "comp_layer": comp_layer,
        "n_correct": len(correct),
        "n_carry": len(carry_probs),
        "n_nocarry": len(nocarry_probs),
        **results_sections,
        "alignment": alignment,
        "carry_direction": {str(k): v for k, v in carry_dir.items()} if carry_dir else None,
    }


# ═══════════════════════════════════════════════════════════════
# EXP D: Carry Signal Causal Intervention
# ═══════════════════════════════════════════════════════════════

def run_exp_d(model, device, comp_layer, d_model):
    """
    Find the carry direction (mean_carry - mean_nocarry), flip it at comp_layer,
    check if tens digit shifts by ±1.
    """
    log_section("EXP D: CARRY SIGNAL CAUSAL INTERVENTION")

    # Generate ONES-DIGIT addition problems to build carry direction
    # (carry direction was characterized for ones-digit problems in carry_stratification)
    logger.info("  Step 1: Build carry direction from ones-digit addition problems...")
    ones_flat, _ = generate_teacher_forced_problems(n_per_digit=80, operand_max=50)
    ones_correct = filter_correct_teacher_forced(model, ones_flat, max_n=500)

    hook_name = f"blocks.{comp_layer}.hook_resid_post"

    # Collect per-carry-status activations
    carry_acts = []
    nocarry_acts = []
    for prob in ones_correct:
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True).to(device)
        holder = {}
        def capture(act, hook, h=holder):
            h["act"] = act.detach()
            return act
        with torch.no_grad():
            with model.hooks(fwd_hooks=[(hook_name, capture)]):
                model(tokens)
        if "act" in holder:
            v = holder["act"][0, -1].cpu().float().numpy()
            if prob.get("has_carry", 0) == 1:
                carry_acts.append(v)
            else:
                nocarry_acts.append(v)

    logger.info(f"  Carry: {len(carry_acts)} acts, No-carry: {len(nocarry_acts)} acts")

    if len(carry_acts) < 20 or len(nocarry_acts) < 20:
        logger.error("  Not enough carry/nocarry samples")
        return {"status": "insufficient_data"}

    carry_mean = np.mean(carry_acts, axis=0)
    nocarry_mean = np.mean(nocarry_acts, axis=0)
    carry_dir = carry_mean - nocarry_mean  # points FROM nocarry TO carry
    carry_dir_norm = carry_dir / (np.linalg.norm(carry_dir) + 1e-10)

    logger.info(f"  Carry direction norm: {np.linalg.norm(carry_dir):.2f}")

    # Step 2: Generate tens-digit problems and test carry intervention
    logger.info("\n  Step 2: Test carry intervention on tens-digit problems...")
    tens_flat, _ = generate_tens_digit_problems_with_carry(n_per_digit=50, operand_max=50)
    tens_correct = filter_correct_teacher_forced(model, tens_flat, max_n=300)

    if len(tens_correct) < 20:
        logger.error(f"  Only {len(tens_correct)} correct tens-digit problems")
        return {"status": "insufficient_data"}

    carry_dir_t = torch.tensor(carry_dir, dtype=torch.float32).to(device)
    carry_dir_norm_t = torch.tensor(carry_dir_norm, dtype=torch.float32).to(device)

    results = {
        "flip_carry": {"changed": 0, "shifted_pm1": 0, "total": 0},
        "add_carry": {"changed": 0, "shifted_plus1": 0, "total": 0},
        "remove_carry": {"changed": 0, "shifted_minus1": 0, "total": 0},
    }

    for prob in tens_correct:
        tokens = prob["_tokens"].to(device)
        target_tok = prob["_target_tok"]
        original_tens = prob["actual_tens"]
        has_carry = prob["has_carry"]

        # Intervention 1: FLIP carry direction (negate the carry component)
        def flip_hook(act, hook):
            a = act.clone()
            orig = a[0, -1].float()
            proj = (orig @ carry_dir_norm_t) * carry_dir_norm_t
            a[0, -1] = (orig - 2 * proj).to(a.dtype)  # flip = subtract 2× projection
            return a

        with torch.no_grad():
            with model.hooks(fwd_hooks=[(hook_name, flip_hook)]):
                logits = model(tokens)
        pred_tok = logits[0, -1].argmax().item()
        pred_str = model.to_string([pred_tok]).strip()

        results["flip_carry"]["total"] += 1
        if pred_str.isdigit():
            pred_digit = int(pred_str)
            if pred_digit != original_tens:
                results["flip_carry"]["changed"] += 1
            expected_shift = (original_tens - 1) % 10 if has_carry else (original_tens + 1) % 10
            if pred_digit == expected_shift:
                results["flip_carry"]["shifted_pm1"] += 1

        # Intervention 2: ADD carry (for no-carry problems, add carry direction)
        if not has_carry:
            def add_carry_hook(act, hook):
                a = act.clone()
                a[0, -1] = (a[0, -1].float() + carry_dir_t).to(a.dtype)
                return a

            with torch.no_grad():
                with model.hooks(fwd_hooks=[(hook_name, add_carry_hook)]):
                    logits = model(tokens)
            pred_tok = logits[0, -1].argmax().item()
            pred_str = model.to_string([pred_tok]).strip()

            results["add_carry"]["total"] += 1
            if pred_str.isdigit():
                pred_digit = int(pred_str)
                if pred_digit != original_tens:
                    results["add_carry"]["changed"] += 1
                if pred_digit == (original_tens + 1) % 10:
                    results["add_carry"]["shifted_plus1"] += 1

        # Intervention 3: REMOVE carry (for carry problems, subtract carry direction)
        if has_carry:
            def remove_carry_hook(act, hook):
                a = act.clone()
                a[0, -1] = (a[0, -1].float() - carry_dir_t).to(a.dtype)
                return a

            with torch.no_grad():
                with model.hooks(fwd_hooks=[(hook_name, remove_carry_hook)]):
                    logits = model(tokens)
            pred_tok = logits[0, -1].argmax().item()
            pred_str = model.to_string([pred_tok]).strip()

            results["remove_carry"]["total"] += 1
            if pred_str.isdigit():
                pred_digit = int(pred_str)
                if pred_digit != original_tens:
                    results["remove_carry"]["changed"] += 1
                if pred_digit == (original_tens - 1) % 10:
                    results["remove_carry"]["shifted_minus1"] += 1

    # Report
    log_section("EXP D RESULTS")
    for name, r in results.items():
        t = r["total"]
        if t == 0:
            continue
        changed = r["changed"]
        specific_key = [k for k in r if k not in ("changed", "total")][0]
        specific = r[specific_key]
        logger.info(f"  {name}: changed {changed}/{t} ({100*changed/t:.1f}%), "
                    f"target shift {specific}/{t} ({100*specific/t:.1f}%)")

    return {
        "test": "D_carry_intervention",
        "comp_layer": comp_layer,
        "carry_dir_norm": float(np.linalg.norm(carry_dir)),
        "n_ones_correct": len(ones_correct),
        "n_tens_correct": len(tens_correct),
        "results": results,
    }


# ═══════════════════════════════════════════════════════════════
# EXP C: Carry Router Head Identification
# ═══════════════════════════════════════════════════════════════

def run_exp_c(model, device, comp_layer, d_model):
    """
    For tens-digit problems, extract attention patterns.
    Find heads that attend from the output position to ones-digit operand tokens.
    """
    log_section("EXP C: CARRY ROUTER HEAD IDENTIFICATION")

    flat, _ = generate_tens_digit_problems_with_carry(n_per_digit=30, operand_max=50)
    correct = filter_correct_teacher_forced(model, flat, max_n=150)

    if len(correct) < 20:
        logger.error(f"  Only {len(correct)} correct problems")
        return {"status": "insufficient_data"}

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # For each problem, identify token positions of operands
    # Prompt: "Calculate:\n45 + 37 = " or "Calculate:\n45 + 37 = 8"
    # Tokens: [BOS, Calculate, :, \n, 4, 5, +, 3, 7, =, ...]
    # We need to identify which tokens are ones-digits vs tens-digits of operands

    # Accumulate attention from last position to each semantic role
    attn_to_a_ones = np.zeros((n_layers, n_heads))
    attn_to_a_tens = np.zeros((n_layers, n_heads))
    attn_to_b_ones = np.zeros((n_layers, n_heads))
    attn_to_b_tens = np.zeros((n_layers, n_heads))
    attn_to_operator = np.zeros((n_layers, n_heads))  # + sign
    attn_to_equals = np.zeros((n_layers, n_heads))   # = sign
    n_counted = 0

    for prob in correct:
        a, b = prob["a"], prob["b"]
        prompt = prob["prompt"]
        tokens = model.to_tokens(prompt, prepend_bos=True).to(device)
        seq_len = tokens.shape[1]

        # Decode each token to find positions
        token_strs = [model.to_string([tokens[0, i].item()]).strip() for i in range(seq_len)]

        # Find operand digit positions by matching the prompt structure
        # "Calculate:\n{a} + {b} = ..."
        a_str = str(a)
        b_str = str(b)

        # Search for the pattern in token strings
        # Simple approach: find "+" token, operand a is before it, operand b is after
        plus_pos = None
        eq_pos = None
        for i, ts in enumerate(token_strs):
            if '+' in ts and plus_pos is None:
                plus_pos = i
            if '=' in ts:
                eq_pos = i

        if plus_pos is None or eq_pos is None:
            continue

        # Identify a's digit positions (before +)
        # and b's digit positions (between + and =)
        a_digit_positions = []
        b_digit_positions = []

        for i in range(1, plus_pos):
            if any(c.isdigit() for c in token_strs[i]):
                a_digit_positions.append(i)
        for i in range(plus_pos + 1, eq_pos):
            if any(c.isdigit() for c in token_strs[i]):
                b_digit_positions.append(i)

        if len(a_digit_positions) == 0 or len(b_digit_positions) == 0:
            continue

        # Last digit position = ones digit, second-to-last = tens digit
        a_ones_pos = a_digit_positions[-1]
        a_tens_pos = a_digit_positions[-2] if len(a_digit_positions) >= 2 else None
        b_ones_pos = b_digit_positions[-1]
        b_tens_pos = b_digit_positions[-2] if len(b_digit_positions) >= 2 else None

        # Run model and collect ALL attention patterns
        # Only cache attention patterns to save memory
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens, return_type="logits",
                names_filter=lambda name: "attn.hook_pattern" in name,
            )

        last_pos = seq_len - 1

        for layer in range(n_layers):
            # attn shape: (1, n_heads, seq_len, seq_len)
            attn = cache[f"blocks.{layer}.attn.hook_pattern"][0]  # (n_heads, seq, seq)
            attn_from_last = attn[:, last_pos, :]  # (n_heads, seq_len)

            attn_np = attn_from_last.cpu().float().numpy()

            attn_to_a_ones[layer] += attn_np[:, a_ones_pos]
            if a_tens_pos is not None:
                attn_to_a_tens[layer] += attn_np[:, a_tens_pos]
            attn_to_b_ones[layer] += attn_np[:, b_ones_pos]
            if b_tens_pos is not None:
                attn_to_b_tens[layer] += attn_np[:, b_tens_pos]
            attn_to_operator[layer] += attn_np[:, plus_pos]
            attn_to_equals[layer] += attn_np[:, eq_pos]

        n_counted += 1
        del cache
        if device == "mps":
            torch.mps.empty_cache()

    if n_counted == 0:
        logger.error("  No problems successfully analyzed")
        return {"status": "no_data"}

    # Normalize
    attn_to_a_ones /= n_counted
    attn_to_a_tens /= n_counted
    attn_to_b_ones /= n_counted
    attn_to_b_tens /= n_counted
    attn_to_operator /= n_counted
    attn_to_equals /= n_counted

    # The "carry-relevant" attention = attention to ones-digit positions
    # because carry depends on a_ones + b_ones
    carry_relevant_attn = attn_to_a_ones + attn_to_b_ones  # (n_layers, n_heads)

    # Find top carry-router heads
    flat_idx = np.argsort(carry_relevant_attn.ravel())[::-1]
    top_heads = []
    logger.info(f"\n  Top 15 heads attending to ones-digit operands (from {n_counted} problems):")
    logger.info(f"  {'Rank':>4}  {'Head':>10}  {'→a₀':>8}  {'→b₀':>8}  {'→a₁':>8}  {'→b₁':>8}  {'Σ(ones)':>8}  {'→op':>8}")
    logger.info(f"  {'─'*70}")

    for rank, idx in enumerate(flat_idx[:15]):
        layer = idx // n_heads
        head = idx % n_heads
        top_heads.append({
            "layer": int(layer), "head": int(head),
            "attn_a_ones": float(attn_to_a_ones[layer, head]),
            "attn_b_ones": float(attn_to_b_ones[layer, head]),
            "attn_a_tens": float(attn_to_a_tens[layer, head]),
            "attn_b_tens": float(attn_to_b_tens[layer, head]),
            "carry_relevant": float(carry_relevant_attn[layer, head]),
        })
        logger.info(
            f"  {rank+1:>4}  L{layer}H{head:>2}     "
            f"{attn_to_a_ones[layer,head]:>7.3f}  {attn_to_b_ones[layer,head]:>7.3f}  "
            f"{attn_to_a_tens[layer,head]:>7.3f}  {attn_to_b_tens[layer,head]:>7.3f}  "
            f"{carry_relevant_attn[layer,head]:>7.3f}  {attn_to_operator[layer,head]:>7.3f}"
        )

    # Layer-level aggregation
    logger.info(f"\n  Per-layer mean attention to ones-digit operands:")
    layer_carry_attn = carry_relevant_attn.mean(axis=1)  # (n_layers,)
    for layer in range(n_layers):
        bar = "█" * int(layer_carry_attn[layer] * 200)
        logger.info(f"    L{layer:>2}: {layer_carry_attn[layer]:.4f} {bar}")

    return {
        "test": "C_carry_router_heads",
        "n_problems": n_counted,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "top_heads": top_heads,
        "layer_carry_attn_mean": layer_carry_attn.tolist(),
    }


# ═══════════════════════════════════════════════════════════════
# EXP B: Operand Digit Subspace Decomposition
# ═══════════════════════════════════════════════════════════════

def run_exp_b(model, device, comp_layer, d_model):
    """
    Group activations by each input digit position (a₀, a₁, b₀, b₁).
    Check for independent Fourier subspaces per position.
    """
    log_section("EXP B: OPERAND DIGIT SUBSPACE DECOMPOSITION")

    # Use multi-digit addition problems
    flat, _ = generate_teacher_forced_problems(n_per_digit=80, operand_max=50)
    correct = filter_correct_teacher_forced(model, flat, max_n=500)

    if len(correct) < 50:
        logger.error(f"  Only {len(correct)} correct problems")
        return {"status": "insufficient_data"}

    # Collect activations
    logger.info(f"  Collecting activations at L{comp_layer}...")
    hook_name = f"blocks.{comp_layer}.hook_resid_post"

    # Group by each operand digit position
    acts_by_a0 = defaultdict(list)  # a ones digit
    acts_by_a1 = defaultdict(list)  # a tens digit
    acts_by_b0 = defaultdict(list)  # b ones digit
    acts_by_b1 = defaultdict(list)  # b tens digit

    for prob in correct:
        a, b = prob["a"], prob["b"]
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True).to(device)

        holder = {}
        def capture(act, hook, h=holder):
            h["act"] = act.detach()
            return act

        with torch.no_grad():
            with model.hooks(fwd_hooks=[(hook_name, capture)]):
                model(tokens)

        if "act" in holder:
            v = holder["act"][0, -1].cpu().float().numpy()
            acts_by_a0[a % 10].append(v)
            acts_by_a1[a // 10].append(v)
            acts_by_b0[b % 10].append(v)
            acts_by_b1[b // 10].append(v)

    # Compute per-digit means and SVD+DFT for each position
    positions = {
        "a_ones (a₀)": acts_by_a0,
        "a_tens (a₁)": acts_by_a1,
        "b_ones (b₀)": acts_by_b0,
        "b_tens (b₁)": acts_by_b1,
    }

    bases = {}  # store (d_model, 9) basis for each position
    position_results = {}

    for pos_name, acts_by_digit in positions.items():
        means = np.zeros((10, d_model))
        counts = np.zeros(10, dtype=int)
        for d in range(10):
            counts[d] = len(acts_by_digit[d])
            if counts[d] > 0:
                means[d] = np.mean(acts_by_digit[d], axis=0)

        logger.info(f"\n  {pos_name}: counts = {dict(enumerate(counts.astype(int).tolist()))}")

        V, S, U, centroid, used_digits = compute_basis_from_means(means, counts, pos_name)
        if V is None:
            position_results[pos_name] = {"status": "insufficient_data"}
            continue

        dft_results, is_perfect, mean_purity = analyze_group_dft(
            f"{pos_name} (L{comp_layer})", U, S, used_digits)

        bases[pos_name] = V
        position_results[pos_name] = {
            "perfect_fourier": is_perfect,
            "mean_purity": mean_purity,
            "singular_values": S.tolist(),
            "n_dirs": V.shape[1],
            "digits_used": used_digits,
            "dft": dft_results,
        }

    # Pairwise subspace alignment
    logger.info(f"\n  Pairwise subspace alignment:")
    alignment = {}
    pos_names = [n for n in positions.keys() if n in bases]
    for i, n1 in enumerate(pos_names):
        for n2 in pos_names[i+1:]:
            V1, V2 = bases[n1], bases[n2]
            k = min(V1.shape[1], V2.shape[1])
            angles, cos_angles = principal_angles(V1[:, :k], V2[:, :k])
            mean_angle = float(np.degrees(angles.mean()))
            key = f"{n1} ↔ {n2}"
            alignment[key] = {"mean_angle_deg": mean_angle, "cos_angles": cos_angles.tolist()}
            logger.info(f"    {key}: mean angle = {mean_angle:.1f}°")

    # Summary
    log_section("EXP B SUMMARY")
    for pos_name, r in position_results.items():
        if r.get("status") == "insufficient_data":
            logger.info(f"  {pos_name}: INSUFFICIENT DATA")
        else:
            status = "★ PERFECT" if r["perfect_fourier"] else "NOT perfect"
            logger.info(f"  {pos_name}: {status} Fourier, purity={r['mean_purity']:.1f}%, dims={r['n_dirs']}")

    return {
        "test": "B_operand_decomposition",
        "comp_layer": comp_layer,
        "n_correct": len(correct),
        "positions": position_results,
        "alignment": alignment,
    }


# ═══════════════════════════════════════════════════════════════
# EXP F: Carry Router Head Ablation (Causal)
# ═══════════════════════════════════════════════════════════════

def _discover_carry_router_heads(model, correct_problems, device, top_n=2):
    """
    Quick attention analysis to find carry-router heads for ANY model.
    Returns list of (layer, head) tuples sorted by attention to ones-digit operands.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    attn_to_ones = np.zeros((n_layers, n_heads))  # attention to a₀ + b₀
    n_counted = 0

    sample = correct_problems[:min(80, len(correct_problems))]

    for prob in sample:
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True).to(device)
        seq_len = tokens.shape[1]
        token_strs = [model.to_string([t]) for t in tokens[0]]

        plus_pos = eq_pos = None
        for i, ts in enumerate(token_strs):
            if '+' in ts and plus_pos is None:
                plus_pos = i
            if '=' in ts:
                eq_pos = i
        if plus_pos is None or eq_pos is None:
            continue

        a_digit_positions = [i for i in range(1, plus_pos) if any(c.isdigit() for c in token_strs[i])]
        b_digit_positions = [i for i in range(plus_pos + 1, eq_pos) if any(c.isdigit() for c in token_strs[i])]
        if not a_digit_positions or not b_digit_positions:
            continue

        a_ones_pos = a_digit_positions[-1]
        b_ones_pos = b_digit_positions[-1]

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens, return_type="logits",
                names_filter=lambda name: "attn.hook_pattern" in name,
            )

        last_pos = seq_len - 1
        for layer in range(n_layers):
            attn = cache[f"blocks.{layer}.attn.hook_pattern"][0]  # (n_heads, seq, seq)
            attn_from_last = attn[:, last_pos, :].cpu().float().numpy()
            attn_to_ones[layer] += attn_from_last[:, a_ones_pos] + attn_from_last[:, b_ones_pos]

        n_counted += 1
        del cache
        if device == "mps":
            torch.mps.empty_cache()

    if n_counted > 0:
        attn_to_ones /= n_counted

    # Rank all (layer, head) by ones-digit attention
    ranked = []
    for layer in range(n_layers):
        for head in range(n_heads):
            ranked.append((layer, head, attn_to_ones[layer, head]))
    ranked.sort(key=lambda x: -x[2])

    top_heads = [(l, h) for l, h, _ in ranked[:top_n]]
    logger.info(f"  Auto-discovered carry router heads (top {top_n}):")
    for l, h, score in ranked[:top_n]:
        logger.info(f"    L{l}H{h}: ones-attn = {score:.3f}")

    return top_heads, ranked


def _eval_tens_digit_accuracy(model, problems, device, fwd_hooks=None,
                              full_answer=False):
    """Evaluate tens-digit prediction accuracy with optional hooks.

    If full_answer=True, the model predicts the full answer as one token
    (e.g. "68") and we extract the tens digit from the predicted number.
    Otherwise, the model predicts the tens digit directly (teacher-forced).
    """
    correct = 0
    total = 0
    for prob in problems:
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True).to(device)
        with torch.no_grad():
            if fwd_hooks:
                with model.hooks(fwd_hooks=fwd_hooks):
                    logits = model(tokens)
            else:
                logits = model(tokens)
        pred_tok = logits[0, -1].argmax().item()
        pred_str = model.to_string([pred_tok]).strip()

        if full_answer:
            # Parse predicted number, extract tens digit
            try:
                pred_num = int(pred_str)
                pred_tens = str((pred_num // 10) % 10)
            except ValueError:
                pred_tens = ""
            expected = str(prob["actual_tens"])
            if pred_tens == expected:
                correct += 1
        else:
            expected = prob["target_str"]
            if pred_str == expected:
                correct += 1
        total += 1
    return correct, total


def run_exp_f(model, device, comp_layer, d_model, full_answer=False):
    """
    Ablate carry-router heads (L16H5, L16H6) and measure:
    - Accuracy on carry problems vs no-carry problems
    - Compare to random-head ablation control
    If these heads are causal carry routers, ablating them should
    hurt carry-group accuracy MORE than no-carry-group.
    """
    log_section("EXP F: CARRY ROUTER HEAD ABLATION (CAUSAL)")

    if full_answer:
        flat, _ = generate_tens_digit_problems_fullanswer(n_per_digit=60, operand_max=50)
        correct = filter_correct_fullanswer_tens(model, flat, max_n=400)
    else:
        flat, _ = generate_tens_digit_problems_with_carry(n_per_digit=60, operand_max=50)
        correct = filter_correct_teacher_forced(model, flat, max_n=400)

    carry_probs = [p for p in correct if p["has_carry"] == 1]
    nocarry_probs = [p for p in correct if p["has_carry"] == 0]
    logger.info(f"  Problems: {len(correct)} total, {len(carry_probs)} carry, {len(nocarry_probs)} no-carry")

    n_heads = model.cfg.n_heads

    # Baseline
    logger.info("\n  Baseline (no ablation):")
    c_carry, t_carry = _eval_tens_digit_accuracy(model, carry_probs, device, full_answer=full_answer)
    c_nocarry, t_nocarry = _eval_tens_digit_accuracy(model, nocarry_probs, device, full_answer=full_answer)
    logger.info(f"    Carry:    {c_carry}/{t_carry} = {100*c_carry/max(t_carry,1):.1f}%")
    logger.info(f"    No-carry: {c_nocarry}/{t_nocarry} = {100*c_nocarry/max(t_nocarry,1):.1f}%")

    baseline = {
        "carry": {"correct": c_carry, "total": t_carry},
        "nocarry": {"correct": c_nocarry, "total": t_nocarry},
    }

    # Auto-discover carry router heads
    logger.info("\n  Discovering carry router heads...")
    top2_heads, ranked = _discover_carry_router_heads(model, correct, device, top_n=2)
    # Also get next 2 ranked as secondary, and 2 from ~rank 5-6
    secondary_heads = [(l, h) for l, h, _ in ranked[2:4]]
    tertiary_heads = [(l, h) for l, h, _ in ranked[4:6]]

    n_layers = model.cfg.n_layers
    top_label = "+".join(f"L{l}H{h}" for l, h in top2_heads)
    sec_label = "+".join(f"L{l}H{h}" for l, h in secondary_heads)
    tert_label = "+".join(f"L{l}H{h}" for l, h in tertiary_heads)

    # Define ablation targets
    targets = {
        f"{top_label} (carry routers)": top2_heads,
        f"{sec_label} (secondary)": secondary_heads,
        f"{tert_label} (tertiary)": tertiary_heads,
        f"L{comp_layer}H0 (comp layer)": [(comp_layer, 0)],
        "L2H0+H1 (random early control)": [(2, 0), (2, 1)],
        f"L{n_layers-4}H0+H1 (random late control)": [(n_layers - 4, 0), (n_layers - 4, 1)],
    }

    ablation_results = {}

    for name, heads in targets.items():
        logger.info(f"\n  Ablating {name}:")

        def make_ablation_hook(target_heads):
            def hook(act, hook):
                # act shape: (batch, seq, n_heads, d_head) for hook_z
                a = act.clone()
                for (_, h) in target_heads:
                    a[0, :, h, :] = 0.0
                return a
            return hook

        # Build hook list — ablate at hook_z (attention output before projection)
        hook_list = []
        layers_involved = set(l for l, h in heads)
        for layer in layers_involved:
            layer_heads = [(l, h) for l, h in heads if l == layer]
            hook_list.append(
                (f"blocks.{layer}.attn.hook_z", make_ablation_hook(layer_heads))
            )

        c_carry_abl, t_carry_abl = _eval_tens_digit_accuracy(
            model, carry_probs, device, fwd_hooks=hook_list, full_answer=full_answer)
        c_nocarry_abl, t_nocarry_abl = _eval_tens_digit_accuracy(
            model, nocarry_probs, device, fwd_hooks=hook_list, full_answer=full_answer)

        acc_carry = 100 * c_carry_abl / max(t_carry_abl, 1)
        acc_nocarry = 100 * c_nocarry_abl / max(t_nocarry_abl, 1)
        base_carry = 100 * c_carry / max(t_carry, 1)
        base_nocarry = 100 * c_nocarry / max(t_nocarry, 1)
        dmg_carry = base_carry - acc_carry
        dmg_nocarry = base_nocarry - acc_nocarry

        logger.info(f"    Carry:    {c_carry_abl}/{t_carry_abl} = {acc_carry:.1f}% (Δ = {-dmg_carry:+.1f}%)")
        logger.info(f"    No-carry: {c_nocarry_abl}/{t_nocarry_abl} = {acc_nocarry:.1f}% (Δ = {-dmg_nocarry:+.1f}%)")
        logger.info(f"    Carry-specific damage: {dmg_carry - dmg_nocarry:+.1f}% ← KEY METRIC")

        ablation_results[name] = {
            "heads": [(l, h) for l, h in heads],
            "carry": {"correct": c_carry_abl, "total": t_carry_abl, "acc": acc_carry, "damage": dmg_carry},
            "nocarry": {"correct": c_nocarry_abl, "total": t_nocarry_abl, "acc": acc_nocarry, "damage": dmg_nocarry},
            "carry_specific_damage": dmg_carry - dmg_nocarry,
        }

    # Summary
    log_section("EXP F SUMMARY")
    logger.info(f"  {'Target':<35} {'Carry dmg':>10} {'No-carry dmg':>12} {'Δ (carry-specific)':>18}")
    logger.info(f"  {'─'*78}")
    for name, r in ablation_results.items():
        logger.info(f"  {name:<35} {r['carry']['damage']:>9.1f}% {r['nocarry']['damage']:>11.1f}% {r['carry_specific_damage']:>+17.1f}%")

    return {
        "test": "F_carry_router_ablation",
        "baseline": baseline,
        "ablations": ablation_results,
    }


# ═══════════════════════════════════════════════════════════════
# EXP G: Tens-Digit-Native Carry Direction Intervention
# ═══════════════════════════════════════════════════════════════

def run_exp_g(model, device, comp_layer, d_model, full_answer=False):
    """
    Compute carry direction FROM tens-digit activations directly
    (not from ones-digit problems). Then intervene.
    Also try multiple intervention layers (L14-L19).
    """
    log_section("EXP G: TENS-DIGIT-NATIVE CARRY DIRECTION")

    if full_answer:
        flat, _ = generate_tens_digit_problems_fullanswer(n_per_digit=80, operand_max=50)
        correct = filter_correct_fullanswer_tens(model, flat, max_n=500)
    else:
        flat, _ = generate_tens_digit_problems_with_carry(n_per_digit=80, operand_max=50)
        correct = filter_correct_teacher_forced(model, flat, max_n=500)

    carry_probs = [p for p in correct if p["has_carry"] == 1]
    nocarry_probs = [p for p in correct if p["has_carry"] == 0]
    logger.info(f"  {len(correct)} correct: {len(carry_probs)} carry, {len(nocarry_probs)} no-carry")

    # Try multiple layers relative to comp_layer
    test_layers = sorted(set([
        max(0, comp_layer - 5),
        max(0, comp_layer - 3),
        max(0, comp_layer - 2),
        comp_layer,
    ]))
    results = {}

    for layer in test_layers:
        logger.info(f"\n  ── Layer L{layer} ──")
        hook_name = f"blocks.{layer}.hook_resid_post"

        # Collect per-carry-status activations at this layer
        carry_acts = []
        nocarry_acts = []
        for prob in correct:
            tokens = model.to_tokens(prob["prompt"], prepend_bos=True).to(device)
            holder = {}
            def capture(act, hook, h=holder):
                h["act"] = act.detach()
                return act
            with torch.no_grad():
                with model.hooks(fwd_hooks=[(hook_name, capture)]):
                    model(tokens)
            if "act" in holder:
                v = holder["act"][0, -1].cpu().float().numpy()
                if prob["has_carry"] == 1:
                    carry_acts.append(v)
                else:
                    nocarry_acts.append(v)

        carry_mean = np.mean(carry_acts, axis=0)
        nocarry_mean = np.mean(nocarry_acts, axis=0)
        carry_dir = carry_mean - nocarry_mean
        carry_dir_unit = carry_dir / (np.linalg.norm(carry_dir) + 1e-10)
        carry_dir_t = torch.tensor(carry_dir, dtype=torch.float32).to(device)
        carry_dir_unit_t = torch.tensor(carry_dir_unit, dtype=torch.float32).to(device)

        logger.info(f"    Carry direction norm: {np.linalg.norm(carry_dir):.2f}")

        # Test: remove carry from carry problems, add carry to no-carry problems
        remove_changed = 0
        remove_shifted = 0
        remove_total = 0
        add_changed = 0
        add_shifted = 0
        add_total = 0

        for prob in correct:
            tokens = model.to_tokens(prob["prompt"], prepend_bos=True).to(device)
            tens = prob["actual_tens"]
            has_carry = prob["has_carry"]

            if has_carry:
                # Remove carry
                def remove_hook(act, hook, cd=carry_dir_t):
                    a = act.clone()
                    a[0, -1] = (a[0, -1].float() - cd).to(a.dtype)
                    return a

                with torch.no_grad():
                    with model.hooks(fwd_hooks=[(hook_name, remove_hook)]):
                        logits = model(tokens)
                pred_tok = logits[0, -1].argmax().item()
                pred_str = model.to_string([pred_tok]).strip()
                remove_total += 1
                if full_answer:
                    try:
                        pred_d = (int(pred_str) // 10) % 10
                    except ValueError:
                        pred_d = -1
                elif pred_str.isdigit():
                    pred_d = int(pred_str)
                else:
                    pred_d = -1
                if pred_d >= 0:
                    if pred_d != tens:
                        remove_changed += 1
                    if pred_d == (tens - 1) % 10:
                        remove_shifted += 1
            else:
                # Add carry
                def add_hook(act, hook, cd=carry_dir_t):
                    a = act.clone()
                    a[0, -1] = (a[0, -1].float() + cd).to(a.dtype)
                    return a

                with torch.no_grad():
                    with model.hooks(fwd_hooks=[(hook_name, add_hook)]):
                        logits = model(tokens)
                pred_tok = logits[0, -1].argmax().item()
                pred_str = model.to_string([pred_tok]).strip()
                add_total += 1
                if full_answer:
                    try:
                        pred_d = (int(pred_str) // 10) % 10
                    except ValueError:
                        pred_d = -1
                elif pred_str.isdigit():
                    pred_d = int(pred_str)
                else:
                    pred_d = -1
                if pred_d >= 0:
                    if pred_d != tens:
                        add_changed += 1
                    if pred_d == (tens + 1) % 10:
                        add_shifted += 1

        logger.info(f"    Remove carry: {remove_changed}/{remove_total} changed ({100*remove_changed/max(remove_total,1):.1f}%), "
                    f"{remove_shifted}/{remove_total} shifted -1 ({100*remove_shifted/max(remove_total,1):.1f}%)")
        logger.info(f"    Add carry:    {add_changed}/{add_total} changed ({100*add_changed/max(add_total,1):.1f}%), "
                    f"{add_shifted}/{add_total} shifted +1 ({100*add_shifted/max(add_total,1):.1f}%)")

        results[f"L{layer}"] = {
            "carry_dir_norm": float(np.linalg.norm(carry_dir)),
            "remove_carry": {"changed": remove_changed, "shifted": remove_shifted, "total": remove_total},
            "add_carry": {"changed": add_changed, "shifted": add_shifted, "total": add_total},
        }

    # Summary
    log_section("EXP G SUMMARY")
    logger.info(f"  {'Layer':<8} {'||dir||':>8} {'Remove Δ':>10} {'Remove ±1':>10} {'Add Δ':>10} {'Add ±1':>10}")
    logger.info(f"  {'─'*58}")
    for layer_name, r in results.items():
        rm = r["remove_carry"]
        ad = r["add_carry"]
        logger.info(f"  {layer_name:<8} {r['carry_dir_norm']:>8.1f} "
                    f"{100*rm['changed']/max(rm['total'],1):>9.1f}% "
                    f"{100*rm['shifted']/max(rm['total'],1):>9.1f}% "
                    f"{100*ad['changed']/max(ad['total'],1):>9.1f}% "
                    f"{100*ad['shifted']/max(ad['total'],1):>9.1f}%")

    return {"test": "G_tens_native_carry_dir", "layer_results": results}


# ═══════════════════════════════════════════════════════════════
# EXP H: End-to-End Causal Chain
# ═══════════════════════════════════════════════════════════════

def run_exp_h(model, device, comp_layer, d_model, full_answer=False):
    """
    Ablate components along the hypothesized chain and measure
    tens-digit accuracy (carry vs no-carry):
      1. Attention routing (auto-discovered carry router heads)
      2. MLP computation (MLP at comp_layer)
      3. MLP readout (last 2 MLP layers)
      4. Combined: routing + computation
      5. Full chain: routing + computation + readout
    """
    log_section("EXP H: END-TO-END CAUSAL CHAIN")

    if full_answer:
        flat, _ = generate_tens_digit_problems_fullanswer(n_per_digit=60, operand_max=50)
        correct = filter_correct_fullanswer_tens(model, flat, max_n=400)
    else:
        flat, _ = generate_tens_digit_problems_with_carry(n_per_digit=60, operand_max=50)
        correct = filter_correct_teacher_forced(model, flat, max_n=400)

    carry_probs = [p for p in correct if p["has_carry"] == 1]
    nocarry_probs = [p for p in correct if p["has_carry"] == 0]
    all_probs = correct
    logger.info(f"  {len(correct)} correct: {len(carry_probs)} carry, {len(nocarry_probs)} no-carry")

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    readout_l1 = n_layers - 2  # second-to-last layer
    readout_l2 = n_layers - 1  # last layer

    # Auto-discover carry router heads
    logger.info("\n  Discovering carry router heads for chain...")
    top2_heads, _ = _discover_carry_router_heads(model, correct, device, top_n=2)

    # Build routing hooks — group heads by layer
    routing_hooks = []
    heads_by_layer = defaultdict(list)
    for l, h in top2_heads:
        heads_by_layer[l].append(h)

    def zero_head_hook(target_heads):
        def hook(act, hook):
            a = act.clone()
            for h in target_heads:
                a[0, :, h, :] = 0.0
            return a
        return hook

    for layer, heads in heads_by_layer.items():
        routing_hooks.append(
            (f"blocks.{layer}.attn.hook_z", zero_head_hook(heads))
        )

    def zero_mlp_hook(act, hook):
        a = act.clone()
        a[0, -1, :] = 0.0  # zero the MLP output at the last position
        return a

    router_label = "+".join(f"L{l}H{h}" for l, h in top2_heads)

    # Define ablation conditions
    conditions = {
        "Baseline (no ablation)": [],
        f"Attn routing ({router_label})": routing_hooks,
        f"MLP computation (MLP L{comp_layer})": [
            (f"blocks.{comp_layer}.hook_mlp_out", zero_mlp_hook),
        ],
        f"MLP readout (MLP L{readout_l1}+L{readout_l2})": [
            (f"blocks.{readout_l1}.hook_mlp_out", zero_mlp_hook),
            (f"blocks.{readout_l2}.hook_mlp_out", zero_mlp_hook),
        ],
        "Routing + Computation": routing_hooks + [
            (f"blocks.{comp_layer}.hook_mlp_out", zero_mlp_hook),
        ],
        "Full chain (R+C+Readout)": routing_hooks + [
            (f"blocks.{comp_layer}.hook_mlp_out", zero_mlp_hook),
            (f"blocks.{readout_l1}.hook_mlp_out", zero_mlp_hook),
            (f"blocks.{readout_l2}.hook_mlp_out", zero_mlp_hook),
        ],
    }

    results = {}

    for name, hooks in conditions.items():
        logger.info(f"\n  {name}:")
        c_all, t_all = _eval_tens_digit_accuracy(
            model, all_probs, device, fwd_hooks=hooks if hooks else None, full_answer=full_answer)
        c_carry, t_carry = _eval_tens_digit_accuracy(
            model, carry_probs, device, fwd_hooks=hooks if hooks else None, full_answer=full_answer)
        c_nocarry, t_nocarry = _eval_tens_digit_accuracy(
            model, nocarry_probs, device, fwd_hooks=hooks if hooks else None, full_answer=full_answer)

        acc_all = 100 * c_all / max(t_all, 1)
        acc_carry = 100 * c_carry / max(t_carry, 1)
        acc_nocarry = 100 * c_nocarry / max(t_nocarry, 1)

        logger.info(f"    All:      {c_all}/{t_all} = {acc_all:.1f}%")
        logger.info(f"    Carry:    {c_carry}/{t_carry} = {acc_carry:.1f}%")
        logger.info(f"    No-carry: {c_nocarry}/{t_nocarry} = {acc_nocarry:.1f}%")

        results[name] = {
            "all": {"correct": c_all, "total": t_all, "acc": acc_all},
            "carry": {"correct": c_carry, "total": t_carry, "acc": acc_carry},
            "nocarry": {"correct": c_nocarry, "total": t_nocarry, "acc": acc_nocarry},
        }

    # Summary
    log_section("EXP H SUMMARY — END-TO-END CHAIN")
    base_all = results["Baseline (no ablation)"]["all"]["acc"]
    base_carry = results["Baseline (no ablation)"]["carry"]["acc"]
    base_nocarry = results["Baseline (no ablation)"]["nocarry"]["acc"]
    logger.info(f"  {'Condition':<35} {'All':>8} {'Carry':>8} {'No-carry':>8} {'Δ All':>8} {'Δ Carry':>8}")
    logger.info(f"  {'─'*80}")
    for name, r in results.items():
        d_all = r["all"]["acc"] - base_all
        d_carry = r["carry"]["acc"] - base_carry
        d_nocarry = r["nocarry"]["acc"] - base_nocarry
        logger.info(f"  {name:<35} {r['all']['acc']:>7.1f}% {r['carry']['acc']:>7.1f}% "
                    f"{r['nocarry']['acc']:>7.1f}% {d_all:>+7.1f}% {d_carry:>+7.1f}%")

    return {"test": "H_end_to_end_chain", "conditions": results}


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Multi-Digit Circuit Discovery")
    parser.add_argument("--model", default="gemma-2b", choices=list(MODEL_MAP.keys()))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--comp-layer", type=int, default=None)
    parser.add_argument("--test", nargs="+", default=["A", "D", "C", "B"],
                        choices=["A", "D", "C", "B", "F", "G", "H"])
    parser.add_argument("--full-answer", action="store_true",
                        help="Full-answer mode for models that predict multi-digit "
                             "numbers as single tokens (e.g. LLaMA)")
    args = parser.parse_args()

    comp_defaults = {"gemma-2b": 19, "phi-3": 26, "llama-3b": 20}
    comp_layer = args.comp_layer or comp_defaults.get(args.model, 19)

    model_name = MODEL_MAP[args.model]
    device = args.device

    full_answer = args.full_answer
    logger.info(f"Model: {args.model} ({model_name})")
    logger.info(f"Computation layer: L{comp_layer}")
    logger.info(f"Device: {device}")
    logger.info(f"Tests: {args.test}")
    if full_answer:
        logger.info(f"Full-answer mode: ON (single-token number predictions)")

    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=torch.float32 if device == "cpu" else torch.float16,
    )
    model.eval()
    d_model = model.cfg.d_model

    all_results = {"model": args.model, "comp_layer": comp_layer}

    if "A" in args.test:
        all_results["exp_a"] = run_exp_a(model, device, comp_layer, d_model)

    if "D" in args.test:
        all_results["exp_d"] = run_exp_d(model, device, comp_layer, d_model)

    if "C" in args.test:
        all_results["exp_c"] = run_exp_c(model, device, comp_layer, d_model)

    if "B" in args.test:
        all_results["exp_b"] = run_exp_b(model, device, comp_layer, d_model)

    if "F" in args.test:
        all_results["exp_f"] = run_exp_f(model, device, comp_layer, d_model, full_answer=full_answer)

    if "G" in args.test:
        all_results["exp_g"] = run_exp_g(model, device, comp_layer, d_model, full_answer=full_answer)

    if "H" in args.test:
        all_results["exp_h"] = run_exp_h(model, device, comp_layer, d_model, full_answer=full_answer)

    # Save
    out_path = RESULTS_DIR / f"multidigit_circuit_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump(serialise(all_results), f, indent=2)

    log_section("ALL RESULTS SAVED")
    logger.info(f"  {out_path}")


if __name__ == "__main__":
    main()
