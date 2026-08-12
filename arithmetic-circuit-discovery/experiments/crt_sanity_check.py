#!/usr/bin/env python3
"""
CRT Sanity Check: Validate that Fisher subspace has freq-2 and freq-5 structure,
and that CRT-aware rotations produce predicted digit shifts.

Theory:
  - If models use CRT (mod-2 × mod-5) for mod-10 arithmetic:
    - Freq-2 (period 5) encodes mod-5 residue
    - Freq-5 (period 2) encodes mod-2 residue (even/odd)
  - Rotating in the freq-2 plane by 2π/5 should shift digit by +6 mod 10
  - Rotating in the freq-5 plane by π should shift digit by +5 mod 10

This script:
  1. Loads a model, computes Fisher eigenvectors
  2. Projects activations into Fisher subspace
  3. DFT analysis to find freq-2 and freq-5 sub-planes
  4. Tests CRT-aware rotations vs old mod-10 rotations
"""

import argparse
import logging
import json
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Any, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# MODEL SETUP (reuse from fisher_phase_shift.py)
# ═══════════════════════════════════════════════════════════════

MODEL_MAP = {
    "gemma-2b": "google/gemma-2-2b",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "pythia-6.9b": "EleutherAI/pythia-6.9b",
}


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def generate_problems(operand_range=20, single_digit_only=False):
    """Generate addition problems with known answers."""
    problems = []
    for a in range(operand_range):
        for b in range(operand_range):
            answer = a + b
            if single_digit_only and answer >= 10:
                continue
            ones_digit = answer % 10
            n_digits = len(str(answer))
            first_digit = int(str(answer)[0])
            prompt = f"Calculate:\n{a} + {b} = "
            problems.append({
                "prompt": prompt,
                "answer": answer,
                "ones_digit": ones_digit,
                "first_digit": first_digit,
                "n_digits": n_digits,
                "a": a, "b": b,
            })
    np.random.shuffle(problems)
    return problems


# ═══════════════════════════════════════════════════════════════
# STEP 1: Compute Fisher eigenvectors
# ═══════════════════════════════════════════════════════════════

def compute_fisher_eigenvectors(model, problems, layer, n_problems=200):
    """Compute Fisher Information Matrix eigenvectors at given layer."""
    logger.info(f"  Computing Fisher eigenvectors at layer {layer} ({n_problems} problems)...")
    device = next(model.parameters()).device
    hook_name = f"blocks.{layer}.hook_resid_post"
    d_model = model.cfg.d_model

    fisher_matrix = np.zeros((d_model, d_model))
    n_valid = 0

    for i, prob in enumerate(problems[:n_problems]):
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
        answer_str = str(prob["ones_digit"])
        answer_tok = model.to_single_token(answer_str)

        activation_holder = {}

        def capture_hook(act, hook, holder=activation_holder):
            holder['act'] = act
            act.requires_grad_(True)
            act.retain_grad()
            return act

        try:
            with model.hooks(fwd_hooks=[(hook_name, capture_hook)]):
                logits = model(tokens)

            log_probs = F.log_softmax(logits[0, -1].float(), dim=-1)
            log_p = log_probs[answer_tok]
            log_p.backward(retain_graph=False)

            if 'act' in activation_holder and activation_holder['act'].grad is not None:
                g = activation_holder['act'].grad[0, -1].detach().cpu().float().numpy()
                fisher_matrix += np.outer(g, g)
                n_valid += 1
        except Exception as e:
            logger.debug(f"  Problem {i} failed: {e}")
            continue
        finally:
            model.zero_grad()

        if (i + 1) % 50 == 0:
            logger.info(f"    Processed {i+1}/{n_problems} ({n_valid} valid)")

    if n_valid < 10:
        raise ValueError(f"Only {n_valid} valid gradients — not enough")

    fisher_matrix /= n_valid

    eigenvalues, eigenvectors = np.linalg.eigh(fisher_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Effective dimension
    total = eigenvalues.sum()
    if total > 0:
        p = eigenvalues / total
        p = p[p > 1e-10]
        eff_dim = np.exp(-np.sum(p * np.log(p)))
    else:
        eff_dim = 0

    logger.info(f"  Fisher: eff_dim={eff_dim:.2f}, λ₁/λ₂={eigenvalues[0]/eigenvalues[1]:.1f}×, valid={n_valid}")
    return eigenvalues, eigenvectors, eff_dim


# ═══════════════════════════════════════════════════════════════
# STEP 2: Collect activations and group by digit
# ═══════════════════════════════════════════════════════════════

def collect_digit_activations(model, problems, layer, n_problems=200):
    """Collect activations grouped by ones digit."""
    logger.info(f"  Collecting activations at layer {layer} grouped by digit...")
    device = next(model.parameters()).device
    hook_name = f"blocks.{layer}.hook_resid_post"

    digit_acts = {d: [] for d in range(10)}

    for i, prob in enumerate(problems[:n_problems]):
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
            act = cache[hook_name][0, -1].cpu().float().numpy()
            del cache
        digit_acts[prob["ones_digit"]].append(act)

        if (i + 1) % 50 == 0:
            logger.info(f"    Collected {i+1}/{n_problems}")

    # Compute means
    digit_means = {}
    for d in range(10):
        if len(digit_acts[d]) > 0:
            digit_means[d] = np.mean(digit_acts[d], axis=0)
            logger.info(f"    Digit {d}: {len(digit_acts[d])} samples")

    return digit_means, digit_acts


# ═══════════════════════════════════════════════════════════════
# STEP 3: DFT analysis to find CRT planes in Fisher subspace
# ═══════════════════════════════════════════════════════════════

def find_crt_planes(digit_means, eigenvectors, n_fisher=10):
    """
    Find CRT planes using TWO approaches:
    A) DFT in the FULL activation space (no Fisher projection)
    B) DFT in the Fisher subspace
    Then validate both with circularity checks.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"CRT PLANE IDENTIFICATION")
    logger.info(f"{'='*60}")

    digit_array = np.array([digit_means[d] for d in range(10)])  # (10, d_model)
    d_model = digit_array.shape[1]

    # ── Approach A: DFT in FULL activation space ──
    logger.info(f"\n  APPROACH A: DFT in full d_model={d_model} space")
    centered = digit_array - digit_array.mean(axis=0)
    fft_full = np.fft.fft(centered, axis=0)  # (10, d_model)

    power_full = np.sum(np.abs(fft_full) ** 2, axis=1)  # power per frequency
    logger.info(f"  DFT Power by frequency (full space):")
    for freq in range(6):
        pct = 100 * power_full[freq] / power_full[1:6].sum() if power_full[1:6].sum() > 0 else 0
        logger.info(f"    Freq {freq} (T={'DC' if freq==0 else f'{10/freq:.1f}'}): "
                     f"power={power_full[freq]:.1f} ({pct:.1f}%)")

    # Extract freq-2 and freq-5 directions in full space
    f2_re_raw = fft_full[2].real.copy()
    f2_im_raw = fft_full[2].imag.copy()
    f5_re_raw = fft_full[5].real.copy()

    f2r_norm = np.linalg.norm(f2_re_raw)
    f2i_norm = np.linalg.norm(f2_im_raw)
    f5r_norm = np.linalg.norm(f5_re_raw)
    f5i_norm = np.linalg.norm(fft_full[5].imag)

    logger.info(f"\n  Freq-2 (mod-5): Re_norm={f2r_norm:.2f}, Im_norm={f2i_norm:.2f}")
    logger.info(f"  Freq-5 (mod-2): Re_norm={f5r_norm:.2f}, Im_norm={f5i_norm:.2f} (Im should≈0)")

    # Normalize and orthogonalize freq-2 plane
    v2_re = f2_re_raw / f2r_norm if f2r_norm > 1e-10 else f2_re_raw
    v2_im = f2_im_raw.copy()
    v2_im -= np.dot(v2_im, v2_re) * v2_re  # Gram-Schmidt
    v2_im_orth_norm = np.linalg.norm(v2_im)
    v2_im = v2_im / v2_im_orth_norm if v2_im_orth_norm > 1e-10 else v2_im

    v5_re = f5_re_raw / f5r_norm if f5r_norm > 1e-10 else f5_re_raw

    # ── CIRCULARITY CHECK: Are digit means on a circle in freq-2 plane? ──
    logger.info(f"\n  CIRCULARITY CHECK (freq-2 plane, full space):")
    coords_2d = []
    for d in range(10):
        c_re = np.dot(digit_means[d], v2_re)
        c_im = np.dot(digit_means[d], v2_im)
        mag = np.sqrt(c_re**2 + c_im**2)
        phase = np.arctan2(c_im, c_re) * 180 / np.pi
        coords_2d.append((c_re, c_im, mag, phase))
        logger.info(f"    Digit {d} (mod5={d%5}): re={c_re:+8.2f}, im={c_im:+8.2f}, "
                     f"mag={mag:7.2f}, phase={phase:+7.1f}°")

    mags = [c[2] for c in coords_2d]
    mag_mean = np.mean(mags)
    mag_std = np.std(mags)
    circularity = 1.0 - (mag_std / mag_mean) if mag_mean > 0 else 0
    logger.info(f"    Mag mean={mag_mean:.2f}, std={mag_std:.2f}, "
                 f"circularity={circularity:.3f} (1=circle, 0=degenerate)")

    # Check phase separation by mod-5
    logger.info(f"\n  Phase clustering by mod-5 (pairs with same mod5 should be similar):")
    phases_by_mod5 = {}
    for d in range(10):
        r = d % 5
        if r not in phases_by_mod5:
            phases_by_mod5[r] = []
        phases_by_mod5[r].append(coords_2d[d][3])

    for r in range(5):
        p = phases_by_mod5[r]
        if len(p) == 2:
            diff = abs(p[0] - p[1])
            if diff > 180:
                diff = 360 - diff
            logger.info(f"    mod5={r}: phases=[{p[0]:+.1f}°, {p[1]:+.1f}°], pair_diff={diff:.1f}°")

    # Check if phases are evenly spaced (5 clusters at ~72° apart)
    mean_phases = []
    for r in range(5):
        p = phases_by_mod5[r]
        # Use circular mean
        mean_phase = np.arctan2(np.mean([np.sin(x * np.pi/180) for x in p]),
                                 np.mean([np.cos(x * np.pi/180) for x in p])) * 180 / np.pi
        mean_phases.append(mean_phase)

    sorted_phases = sorted(mean_phases)
    diffs = [sorted_phases[i+1] - sorted_phases[i] for i in range(4)]
    diffs.append(360 + sorted_phases[0] - sorted_phases[4])  # wrap
    logger.info(f"    Mean phases by mod5: {[f'{p:.1f}°' for p in mean_phases]}")
    logger.info(f"    Angular gaps: {[f'{d:.1f}°' for d in diffs]} (ideal: all 72°)")

    # ── Freq-5 (mod-2) validation ──
    logger.info(f"\n  Freq-5 (mod-2) separation check:")
    even_projs = []
    odd_projs = []
    for d in range(10):
        proj = np.dot(digit_means[d], v5_re)
        if d % 2 == 0:
            even_projs.append(proj)
        else:
            odd_projs.append(proj)
        logger.info(f"    Digit {d} (mod2={d%2}): proj={proj:+.4f}")

    even_mean = np.mean(even_projs)
    odd_mean = np.mean(odd_projs)
    separation = abs(even_mean - odd_mean)
    pooled_std = np.sqrt((np.var(even_projs) + np.var(odd_projs)) / 2)
    snr = separation / pooled_std if pooled_std > 0 else 0
    logger.info(f"    Even mean={even_mean:+.4f}, Odd mean={odd_mean:+.4f}, "
                 f"separation={separation:.4f}, SNR={snr:.2f}")

    # ── Approach B: DFT in Fisher subspace ──
    logger.info(f"\n  APPROACH B: DFT in Fisher subspace (top-{n_fisher} dims)")
    fisher_basis = eigenvectors[:, :n_fisher]
    digit_fisher = digit_array @ fisher_basis
    centered_fisher = digit_fisher - digit_fisher.mean(axis=0)
    fft_fisher = np.fft.fft(centered_fisher, axis=0)

    power_fisher = np.sum(np.abs(fft_fisher) ** 2, axis=1)
    logger.info(f"  DFT Power by frequency (Fisher subspace):")
    for freq in range(6):
        pct = 100 * power_fisher[freq] / power_fisher[1:6].sum() if power_fisher[1:6].sum() > 0 else 0
        logger.info(f"    Freq {freq} (T={'DC' if freq==0 else f'{10/freq:.1f}'}): "
                     f"power={power_fisher[freq]:.1f} ({pct:.1f}%)")

    crt_power = power_fisher[2] + power_fisher[5]
    mod10_power = power_fisher[1]
    other_power = power_fisher[3] + power_fisher[4]
    logger.info(f"\n  CRT (f2+f5)={crt_power:.1f} vs Mod-10 (f1)={mod10_power:.1f} "
                 f"vs Other (f3+f4)={other_power:.1f}")
    logger.info(f"  CRT > Mod-10? {'YES' if crt_power > mod10_power else 'NO'}")

    return {
        "freq2_v1": v2_re,
        "freq2_v2": v2_im,
        "freq5_v1": v5_re,
        "circularity": circularity,
        "mod2_snr": snr,
        "power_full": power_full.tolist(),
        "power_fisher": power_fisher.tolist(),
    }


# ═══════════════════════════════════════════════════════════════
# STEP 4: CRT-aware rotation test
# ═══════════════════════════════════════════════════════════════

def crt_reconstruct(r2, r5):
    """Reconstruct digit from CRT residues."""
    for d in range(10):
        if d % 2 == r2 and d % 5 == r5:
            return d
    return -1


def test_crt_rotation(model, problems, layer, crt_planes, n_test=50):
    """
    Test CRT-aware rotations:
    1. Mod-5 rotation: rotate in freq-2 plane by k × 2π/5 → digit shifts by k×6 mod 10
    2. Mod-2 flip: negate freq-5 direction → digit shifts by +5 mod 10
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"CRT-AWARE ROTATION TEST at Layer {layer}")
    logger.info(f"{'='*60}")

    device = next(model.parameters()).device
    hook_name = f"blocks.{layer}.hook_resid_post"

    v1 = torch.tensor(crt_planes["freq2_v1"], dtype=torch.float32, device=device)
    v2 = torch.tensor(crt_planes["freq2_v2"], dtype=torch.float32, device=device)
    v_mod2 = torch.tensor(crt_planes["freq5_v1"], dtype=torch.float32, device=device)

    # Filter to single-digit problems the model gets correct
    correct_problems = []
    for prob in problems[:min(len(problems), n_test * 5)]:
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
        with torch.no_grad():
            logits = model(tokens)
        pred_tok = logits[0, -1].argmax(dim=-1).item()
        pred_str = model.tokenizer.decode([pred_tok]).strip()
        try:
            pred_digit = int(pred_str)
            if pred_digit == prob["first_digit"] and prob["n_digits"] == 1:
                correct_problems.append(prob)
        except ValueError:
            pass
        if len(correct_problems) >= n_test:
            break

    # Count digit distribution
    digit_counts = {}
    for p in correct_problems:
        d = p["ones_digit"]
        digit_counts[d] = digit_counts.get(d, 0) + 1
    logger.info(f"  Using {len(correct_problems)} correct single-digit problems")
    logger.info(f"  Digit distribution: {dict(sorted(digit_counts.items()))}")

    # ── Test 1: Mod-5 rotation (freq-2 plane, rotate by k × 2π/5) ──
    logger.info(f"\n  TEST 1: Mod-5 rotation (freq-2 plane, 2π/5 steps)")
    logger.info(f"  Expected: digit → (digit + 6k) mod 10")

    mod5_results = {k: {"exact_crt": 0, "changed": 0, "total": 0} for k in range(1, 5)}

    for prob in correct_problems:
        if prob["n_digits"] != 1:
            continue  # Only single-digit for exact validation

        tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
        original_digit = prob["ones_digit"]

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
            clean_act = cache[hook_name][0, -1].clone()
            del cache

        c1 = torch.dot(clean_act.float(), v1)
        c2 = torch.dot(clean_act.float(), v2)

        for k in range(1, 5):
            theta = k * 2 * np.pi / 5  # CRT-aware: 2π/5, not 2π/10!
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            new_c1 = c1 * cos_t - c2 * sin_t
            new_c2 = c1 * sin_t + c2 * cos_t
            delta = (new_c1 - c1) * v1 + (new_c2 - c2) * v2

            def rot_hook(act, hook, d=delta):
                act[:, -1, :] = act[:, -1, :] + d.to(act.dtype).unsqueeze(0)
                return act

            with torch.no_grad():
                with model.hooks(fwd_hooks=[(hook_name, rot_hook)]):
                    logits = model(tokens)

            pred_tok = logits[0, -1].argmax(dim=-1).item()
            pred_str = model.tokenizer.decode([pred_tok]).strip()

            try:
                pred_digit = int(pred_str)
                expected = (original_digit + 6 * k) % 10
                mod5_results[k]["total"] += 1
                if pred_digit != original_digit:
                    mod5_results[k]["changed"] += 1
                if pred_digit == expected:
                    mod5_results[k]["exact_crt"] += 1
            except ValueError:
                mod5_results[k]["total"] += 1

    for k in range(1, 5):
        r = mod5_results[k]
        if r["total"] > 0:
            logger.info(f"    k={k} (θ={k*72}°): changed={r['changed']}/{r['total']} ({100*r['changed']/r['total']:.1f}%), "
                        f"exact_crt={r['exact_crt']}/{r['total']} ({100*r['exact_crt']/r['total']:.1f}%)")

    # ── Test 2: Mod-2 flip (negate freq-5 direction) ──
    logger.info(f"\n  TEST 2: Mod-2 flip (negate freq-5 projection)")
    logger.info(f"  Expected: digit → (digit + 5) mod 10")

    mod2_results = {"exact_crt": 0, "changed": 0, "total": 0}

    for prob in correct_problems:
        if prob["n_digits"] != 1:
            continue

        tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
        original_digit = prob["ones_digit"]

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
            clean_act = cache[hook_name][0, -1].clone()
            del cache

        # Project onto mod-2 direction and negate it
        proj = torch.dot(clean_act.float(), v_mod2)
        delta = -2 * proj * v_mod2  # negate = subtract 2× projection

        def mod2_hook(act, hook, d=delta):
            act[:, -1, :] = act[:, -1, :] + d.to(act.dtype).unsqueeze(0)
            return act

        with torch.no_grad():
            with model.hooks(fwd_hooks=[(hook_name, mod2_hook)]):
                logits = model(tokens)

        pred_tok = logits[0, -1].argmax(dim=-1).item()
        pred_str = model.tokenizer.decode([pred_tok]).strip()

        try:
            pred_digit = int(pred_str)
            expected = (original_digit + 5) % 10
            mod2_results["total"] += 1
            if pred_digit != original_digit:
                mod2_results["changed"] += 1
            if pred_digit == expected:
                mod2_results["exact_crt"] += 1
        except ValueError:
            mod2_results["total"] += 1

    r = mod2_results
    if r["total"] > 0:
        logger.info(f"    Mod-2 flip: changed={r['changed']}/{r['total']} ({100*r['changed']/r['total']:.1f}%), "
                    f"exact_crt={r['exact_crt']}/{r['total']} ({100*r['exact_crt']/r['total']:.1f}%)")

    # ── Test 3: Old mod-10 rotation for comparison ──
    logger.info(f"\n  TEST 3 (CONTROL): Old mod-10 rotation (2π/10 steps) for comparison")

    mod10_results = {k: {"exact_mod10": 0, "changed": 0, "total": 0} for k in [1, 3, 5]}

    for prob in correct_problems:
        if prob["n_digits"] != 1:
            continue

        tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
        original_digit = prob["ones_digit"]

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
            clean_act = cache[hook_name][0, -1].clone()
            del cache

        c1 = torch.dot(clean_act.float(), v1)
        c2 = torch.dot(clean_act.float(), v2)

        for k in [1, 3, 5]:
            theta = k * 2 * np.pi / 10  # OLD: mod-10 assumption
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            new_c1 = c1 * cos_t - c2 * sin_t
            new_c2 = c1 * sin_t + c2 * cos_t
            delta = (new_c1 - c1) * v1 + (new_c2 - c2) * v2

            def rot_hook(act, hook, d=delta):
                act[:, -1, :] = act[:, -1, :] + d.to(act.dtype).unsqueeze(0)
                return act

            with torch.no_grad():
                with model.hooks(fwd_hooks=[(hook_name, rot_hook)]):
                    logits = model(tokens)

            pred_tok = logits[0, -1].argmax(dim=-1).item()
            pred_str = model.tokenizer.decode([pred_tok]).strip()

            try:
                pred_digit = int(pred_str)
                expected = (original_digit + k) % 10
                mod10_results[k]["total"] += 1
                if pred_digit != original_digit:
                    mod10_results[k]["changed"] += 1
                if pred_digit == expected:
                    mod10_results[k]["exact_mod10"] += 1
            except ValueError:
                mod10_results[k]["total"] += 1

    for k in [1, 3, 5]:
        r = mod10_results[k]
        if r["total"] > 0:
            logger.info(f"    k={k} (θ={k*36}°): changed={r['changed']}/{r['total']} ({100*r['changed']/r['total']:.1f}%), "
                        f"exact_mod10={r['exact_mod10']}/{r['total']} ({100*r['exact_mod10']/r['total']:.1f}%)")

    return {
        "mod5_rotation": mod5_results,
        "mod2_flip": mod2_results,
        "old_mod10": mod10_results,
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CRT Sanity Check")
    parser.add_argument("--model", default="gemma-2b", help="Model key")
    parser.add_argument("--layer", type=int, default=22, help="Layer to test")
    parser.add_argument("--operand-range", type=int, default=30, help="For activation collection")
    parser.add_argument("--n-fisher", type=int, default=200, help="Problems for Fisher computation")
    parser.add_argument("--n-test", type=int, default=80, help="Problems for rotation test")
    args = parser.parse_args()

    from transformer_lens import HookedTransformer

    model_name = MODEL_MAP.get(args.model, args.model)
    device = get_device()
    dtype = torch.float32

    logger.info("=" * 60)
    logger.info("CRT SANITY CHECK (v2 — with circularity validation)")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}, Layer: {args.layer}")

    logger.info(f"Loading {model_name} on {device}...")
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=dtype)

    # Use broad range for activation collection (covers all 10 digits well)
    all_problems = generate_problems(args.operand_range)
    logger.info(f"Generated {len(all_problems)} problems (operand_range={args.operand_range})")

    # Use single-digit problems for rotation test (exact CRT validation)
    single_digit_problems = generate_problems(10, single_digit_only=True)
    logger.info(f"Generated {len(single_digit_problems)} single-digit problems for rotation test")

    # Step 1: Fisher eigenvectors (use broad range)
    eigenvalues, eigenvectors, eff_dim = compute_fisher_eigenvectors(
        model, all_problems, args.layer, args.n_fisher
    )

    # Step 2: Collect activations grouped by digit (use broad range for 10+ per digit)
    digit_means, digit_acts = collect_digit_activations(
        model, all_problems, args.layer, n_problems=300
    )

    # Step 3: Find CRT planes via DFT
    n_fisher_dims = max(10, int(eff_dim * 2))
    crt_planes = find_crt_planes(digit_means, eigenvectors, n_fisher=n_fisher_dims)

    # Step 4: Test CRT-aware rotations using SINGLE-DIGIT problems only
    results = test_crt_rotation(model, single_digit_problems, args.layer, crt_planes, n_test=args.n_test)

    # Save results
    out_dir = Path("mathematical_toolkit_results")
    out_dir.mkdir(exist_ok=True)
    save_results = {
        "model": model_name,
        "layer": args.layer,
        "eff_dim": eff_dim,
        "mod5_rotation": {str(k): v for k, v in results["mod5_rotation"].items()},
        "mod2_flip": results["mod2_flip"],
        "old_mod10": {str(k): v for k, v in results["old_mod10"].items()},
    }
    out_path = out_dir / f"crt_sanity_{args.model}_L{args.layer}.json"
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
