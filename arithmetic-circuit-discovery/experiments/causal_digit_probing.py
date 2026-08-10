#!/usr/bin/env python3
"""
Causal Digit Probing: Three approaches to understand how LLMs encode ones digits.

Approach 1: Mod-2 Focused Steering
  - Use the freq-5 (even/odd) direction confirmed by DFT analysis (SNR=12.94)
  - Flip even↔odd by negating freq-5 projection
  - Verify: does digit shift by +5 mod 10?

Approach 2: Sign-based Causal Probing
  - Negate projection along INDIVIDUAL Fisher eigenvectors
  - Map which dims encode which digit features (even/odd, <5/≥5, specific digits)
  - Build a "causal feature map"

Approach 3: Linear Probe Steering
  - Train a linear probe to decode ones digit from activations
  - Use probe weight differences as intervention directions
  - No geometric assumptions needed

Usage:
  python causal_digit_probing.py --model gemma-2b --layer 22
"""

import argparse
import logging
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_MAP = {
    "gemma-2b": "google/gemma-2-2b",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "pythia-6.9b": "EleutherAI/pythia-6.9b",
    "llama-3b": "meta-llama/Llama-3.2-3B",
}


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def generate_problems(operand_range=30, single_digit_only=False):
    """Generate addition problems."""
    problems = []
    for a in range(operand_range):
        for b in range(operand_range):
            answer = a + b
            if single_digit_only and answer >= 10:
                continue
            ones_digit = answer % 10
            n_digits = len(str(answer))
            first_digit = int(str(answer)[0])
            problems.append({
                "prompt": f"Calculate:\n{a} + {b} = ",
                "answer": answer,
                "ones_digit": ones_digit,
                "first_digit": first_digit,
                "n_digits": n_digits,
                "a": a, "b": b,
            })
    np.random.shuffle(problems)
    return problems


def get_model_prediction(model, prompt):
    """Get model's predicted first digit."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    with torch.no_grad():
        logits = model(tokens)
    pred_tok = logits[0, -1].argmax(dim=-1).item()
    pred_str = model.tokenizer.decode([pred_tok]).strip()
    try:
        return int(pred_str), True
    except ValueError:
        return -1, False


def filter_correct_problems(model, problems, max_n=100):
    """Filter to problems where model predicts correctly."""
    correct = []
    for prob in problems:
        pred, is_num = get_model_prediction(model, prob["prompt"])
        if is_num and pred == prob["first_digit"]:
            correct.append(prob)
        if len(correct) >= max_n:
            break
    return correct


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
            continue
        finally:
            model.zero_grad()

        if (i + 1) % 50 == 0:
            logger.info(f"    Processed {i+1}/{n_problems} ({n_valid} valid)")

    if n_valid < 10:
        raise ValueError(f"Only {n_valid} valid gradients")

    fisher_matrix /= n_valid
    eigenvalues, eigenvectors = np.linalg.eigh(fisher_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    total = eigenvalues.sum()
    if total > 0:
        p = eigenvalues / total
        p = p[p > 1e-10]
        eff_dim = np.exp(-np.sum(p * np.log(p)))
    else:
        eff_dim = 0

    logger.info(f"  Fisher: eff_dim={eff_dim:.2f}, top eigenvalues={eigenvalues[:5].round(4)}, valid={n_valid}")
    return eigenvalues, eigenvectors, eff_dim


def collect_activations(model, problems, layer, n_problems=300):
    """Collect activations at a specific layer."""
    logger.info(f"  Collecting activations at layer {layer}...")
    hook_name = f"blocks.{layer}.hook_resid_post"
    activations = []
    labels = []

    for i, prob in enumerate(problems[:n_problems]):
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
            act = cache[hook_name][0, -1].cpu().float().numpy()
            del cache
        activations.append(act)
        labels.append(prob["ones_digit"])

        if (i + 1) % 100 == 0:
            logger.info(f"    Collected {i+1}/{n_problems}")

    return np.array(activations), np.array(labels)


def compute_freq5_direction(activations, labels):
    """Compute the freq-5 (even/odd) direction via DFT of digit means."""
    digit_means = {}
    for d in range(10):
        mask = labels == d
        if mask.sum() > 0:
            digit_means[d] = activations[mask].mean(axis=0)

    if len(digit_means) < 10:
        logger.warning(f"  Only {len(digit_means)} digits represented — need 10")
        return None, None

    digit_array = np.array([digit_means[d] for d in range(10)])
    centered = digit_array - digit_array.mean(axis=0)
    fft_full = np.fft.fft(centered, axis=0)

    # Freq-5 = period 2 = even/odd
    f5_direction = fft_full[5].real.copy()
    f5_norm = np.linalg.norm(f5_direction)
    if f5_norm > 1e-10:
        f5_direction /= f5_norm

    # Validate
    even_projs = [np.dot(digit_means[d], f5_direction) for d in range(0, 10, 2)]
    odd_projs = [np.dot(digit_means[d], f5_direction) for d in range(1, 10, 2)]
    snr = abs(np.mean(even_projs) - np.mean(odd_projs)) / np.sqrt((np.var(even_projs) + np.var(odd_projs)) / 2)

    logger.info(f"  Freq-5 direction: norm={f5_norm:.2f}, even/odd SNR={snr:.2f}")
    return f5_direction, digit_means


# ═══════════════════════════════════════════════════════════════
# APPROACH 1: Mod-2 Focused Steering
# ═══════════════════════════════════════════════════════════════

def approach1_mod2_steering(model, problems, layer, f5_direction):
    """
    Flip even↔odd by negating the freq-5 (even/odd) projection.
    Prediction: digit d → (d + 5) mod 10
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"APPROACH 1: MOD-2 FOCUSED STEERING (Layer {layer})")
    logger.info(f"{'='*60}")
    logger.info(f"  Theory: negating freq-5 direction should flip even↔odd")
    logger.info(f"  Predicted shift: d → (d + 5) mod 10")

    device = next(model.parameters()).device
    hook_name = f"blocks.{layer}.hook_resid_post"
    v_mod2 = torch.tensor(f5_direction, dtype=torch.float32, device=device)

    correct = filter_correct_problems(model, problems, max_n=80)
    logger.info(f"  Testing {len(correct)} correct single-digit problems")

    # Track per-digit results
    results = {"total": 0, "changed": 0, "exact_plus5": 0, "details": []}
    digit_results = defaultdict(lambda: {"total": 0, "changed": 0, "exact_plus5": 0})

    for prob in correct:
        if prob["n_digits"] != 1:
            continue

        tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
        original = prob["ones_digit"]

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
            clean_act = cache[hook_name][0, -1].clone()
            del cache

        # Negate the freq-5 projection: subtract 2× the projection
        proj = torch.dot(clean_act.float(), v_mod2)
        delta = -2 * proj * v_mod2

        def hook_fn(act, hook, d=delta):
            act[:, -1, :] = act[:, -1, :] + d.to(act.dtype).unsqueeze(0)
            return act

        with torch.no_grad():
            with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
                logits = model(tokens)

        pred_tok = logits[0, -1].argmax(dim=-1).item()
        pred_str = model.tokenizer.decode([pred_tok]).strip()

        try:
            pred_digit = int(pred_str)
            expected = (original + 5) % 10
            changed = pred_digit != original
            exact = pred_digit == expected

            results["total"] += 1
            results["changed"] += int(changed)
            results["exact_plus5"] += int(exact)
            digit_results[original]["total"] += 1
            digit_results[original]["changed"] += int(changed)
            digit_results[original]["exact_plus5"] += int(exact)

            results["details"].append({
                "original": original, "predicted": pred_digit,
                "expected": expected, "changed": changed, "exact": exact,
            })
        except ValueError:
            results["total"] += 1

    # Also test with scaling (amplify the even/odd signal instead of negating)
    logger.info(f"\n  --- Negate Results ---")
    n = results["total"]
    if n > 0:
        logger.info(f"  Total: {n}, Changed: {results['changed']} ({100*results['changed']/n:.1f}%), "
                     f"Exact +5: {results['exact_plus5']} ({100*results['exact_plus5']/n:.1f}%)")

    logger.info(f"\n  Per-digit breakdown:")
    for d in sorted(digit_results.keys()):
        r = digit_results[d]
        if r["total"] > 0:
            logger.info(f"    d={d} → expected {(d+5)%10}: "
                         f"changed={r['changed']}/{r['total']}, exact={r['exact_plus5']}/{r['total']}")

    # Show the actual shift histogram
    shift_counts = defaultdict(int)
    for det in results["details"]:
        shift = (det["predicted"] - det["original"]) % 10
        shift_counts[shift] += 1

    logger.info(f"\n  Shift histogram (actual Δ observed):")
    for shift in sorted(shift_counts.keys()):
        count = shift_counts[shift]
        bar = "█" * int(count * 30 / max(shift_counts.values())) if shift_counts else ""
        logger.info(f"    Δ={shift}: {count:3d} ({100*count/n:.1f}%) {bar}")

    return results


# ═══════════════════════════════════════════════════════════════
# APPROACH 2: Sign-based Causal Probing
# ═══════════════════════════════════════════════════════════════

def approach2_sign_probing(model, problems, layer, eigenvectors, eigenvalues, n_top=10):
    """
    Negate individual Fisher eigenvector projections and measure the effect
    on digit predictions. Map which dims control which digit features.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"APPROACH 2: SIGN-BASED CAUSAL PROBING (Layer {layer})")
    logger.info(f"{'='*60}")
    logger.info(f"  Negating top-{n_top} individual Fisher directions, one at a time")

    device = next(model.parameters()).device
    hook_name = f"blocks.{layer}.hook_resid_post"

    correct = filter_correct_problems(model, problems, max_n=60)
    single_digit = [p for p in correct if p["n_digits"] == 1]
    logger.info(f"  Using {len(single_digit)} correct single-digit problems")

    causal_map = []

    for dim_idx in range(n_top):
        evec = eigenvectors[:, dim_idx]
        eval_ = eigenvalues[dim_idx]
        v = torch.tensor(evec, dtype=torch.float32, device=device)

        dim_results = {
            "dim": dim_idx, "eigenvalue": float(eval_),
            "total": 0, "changed": 0,
            "even_to_odd": 0, "odd_to_even": 0,
            "shift_hist": defaultdict(int),
        }

        for prob in single_digit:
            tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
            original = prob["ones_digit"]

            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=hook_name)
                clean_act = cache[hook_name][0, -1].clone()
                del cache

            proj = torch.dot(clean_act.float(), v)
            delta = -2 * proj * v

            def hook_fn(act, hook, d=delta):
                act[:, -1, :] = act[:, -1, :] + d.to(act.dtype).unsqueeze(0)
                return act

            with torch.no_grad():
                with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
                    logits = model(tokens)

            pred_tok = logits[0, -1].argmax(dim=-1).item()
            pred_str = model.tokenizer.decode([pred_tok]).strip()

            try:
                pred_digit = int(pred_str)
                dim_results["total"] += 1
                if pred_digit != original:
                    dim_results["changed"] += 1
                    shift = (pred_digit - original) % 10
                    dim_results["shift_hist"][shift] += 1
                    # Track parity flips
                    if original % 2 == 0 and pred_digit % 2 == 1:
                        dim_results["even_to_odd"] += 1
                    elif original % 2 == 1 and pred_digit % 2 == 0:
                        dim_results["odd_to_even"] += 1
                else:
                    dim_results["shift_hist"][0] += 1
            except ValueError:
                dim_results["total"] += 1

        n = dim_results["total"]
        parity_flips = dim_results["even_to_odd"] + dim_results["odd_to_even"]
        ch = dim_results["changed"]

        # Identify what this dimension encodes
        features = []
        if n > 0 and ch > 0:
            parity_rate = parity_flips / ch if ch > 0 else 0
            if parity_rate > 0.6:
                features.append("PARITY (even/odd)")
            # Check for +5 shift dominance
            shift5_count = dim_results["shift_hist"].get(5, 0)
            if ch > 0 and shift5_count / ch > 0.3:
                features.append("+5_SHIFT")
            # Check for +1/-1 neighbor shifts
            n1 = dim_results["shift_hist"].get(1, 0) + dim_results["shift_hist"].get(9, 0)
            if ch > 0 and n1 / ch > 0.5:
                features.append("NEIGHBOR (±1)")

        feature_str = ", ".join(features) if features else "unclear"

        if n > 0:
            logger.info(f"  Dim {dim_idx} (λ={eval_:.6f}): changed={ch}/{n} ({100*ch/n:.1f}%), "
                         f"parity_flips={parity_flips}, features=[{feature_str}]")
            # Show top shifts
            top_shifts = sorted(dim_results["shift_hist"].items(), key=lambda x: -x[1])[:3]
            shift_str = ", ".join([f"Δ={s}:{c}" for s, c in top_shifts if s != 0])
            if shift_str:
                logger.info(f"          Top shifts: {shift_str}")

        dim_results["shift_hist"] = dict(dim_results["shift_hist"])
        dim_results["features"] = feature_str
        causal_map.append(dim_results)

    return causal_map


# ═══════════════════════════════════════════════════════════════
# APPROACH 3: Linear Probe Steering
# ═══════════════════════════════════════════════════════════════

def approach3_linear_probe(model, all_problems, single_problems, layer, activations, labels):
    """
    Train a linear probe to decode ones digit, then use probe weights
    as intervention directions.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"APPROACH 3: LINEAR PROBE STEERING (Layer {layer})")
    logger.info(f"{'='*60}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    # Train probe
    logger.info(f"  Training linear probe on {len(activations)} samples...")
    scaler = StandardScaler()
    X = scaler.fit_transform(activations)
    y = labels

    probe = LogisticRegression(max_iter=1000, C=1.0, multi_class='multinomial',
                                solver='lbfgs', random_state=42)

    # Cross-validate first
    cv_scores = cross_val_score(probe, X, y, cv=5, scoring='accuracy')
    logger.info(f"  Probe 5-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Fit on all data
    probe.fit(X, y)
    train_acc = probe.score(X, y)
    logger.info(f"  Probe training accuracy: {train_acc:.3f}")

    # The probe weights (10, d_model) — each row is the direction for one digit
    # We need to un-scale them: w_original = w_scaled / scale
    probe_weights = probe.coef_ / scaler.scale_[np.newaxis, :]  # (10, d_model)
    probe_bias = probe.intercept_ - (probe.coef_ @ (scaler.mean_ / scaler.scale_))

    logger.info(f"  Probe weight norms per digit: {[f'{np.linalg.norm(probe_weights[d]):.2f}' for d in range(10)]}")

    # Compute "steering vectors" between digit pairs
    # To shift from digit d1 to digit d2: add (w[d2] - w[d1]) to activation
    device = next(model.parameters()).device
    hook_name = f"blocks.{layer}.hook_resid_post"

    correct = filter_correct_problems(model, single_problems, max_n=60)
    single_digit = [p for p in correct if p["n_digits"] == 1]
    logger.info(f"  Testing on {len(single_digit)} correct single-digit problems")

    # Test 1: Steer each digit to (digit + 5) mod 10 (even↔odd)
    logger.info(f"\n  TEST A: Probe steering d → (d+5)%10")
    plus5_results = {"total": 0, "exact": 0, "changed": 0}
    shift_hist = defaultdict(int)

    for prob in single_digit:
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
        original = prob["ones_digit"]
        target = (original + 5) % 10

        # Steering vector: w[target] - w[original]
        steer = probe_weights[target] - probe_weights[original]
        steer_torch = torch.tensor(steer, dtype=torch.float32, device=device)

        # Scale the steering vector — start with magnitude matching the projection difference
        # Try multiple scales to find what works
        best_scale = None
        best_pred = None

        for scale in [0.5, 1.0, 2.0, 5.0, 10.0]:
            delta = scale * steer_torch

            def hook_fn(act, hook, d=delta):
                act[:, -1, :] = act[:, -1, :] + d.to(act.dtype).unsqueeze(0)
                return act

            with torch.no_grad():
                with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
                    logits = model(tokens)

            pred_tok = logits[0, -1].argmax(dim=-1).item()
            pred_str = model.tokenizer.decode([pred_tok]).strip()

            try:
                pred_digit = int(pred_str)
                if pred_digit == target:
                    best_scale = scale
                    best_pred = pred_digit
                    break
                elif best_pred is None:
                    best_pred = pred_digit
                    best_scale = scale
            except ValueError:
                pass

        if best_pred is not None:
            plus5_results["total"] += 1
            if best_pred != original:
                plus5_results["changed"] += 1
            if best_pred == target:
                plus5_results["exact"] += 1
            shift = (best_pred - original) % 10
            shift_hist[shift] += 1

    n = plus5_results["total"]
    if n > 0:
        logger.info(f"    Total={n}, Changed={plus5_results['changed']} ({100*plus5_results['changed']/n:.1f}%), "
                     f"Exact +5={plus5_results['exact']} ({100*plus5_results['exact']/n:.1f}%)")

    # Test 2: For each scale independently, what's the success rate?
    logger.info(f"\n  TEST B: Probe steering d → (d+5)%10 at fixed scales")
    for scale in [1.0, 2.0, 5.0, 10.0, 20.0]:
        scale_results = {"total": 0, "exact": 0, "changed": 0}

        for prob in single_digit:
            tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
            original = prob["ones_digit"]
            target = (original + 5) % 10

            steer = probe_weights[target] - probe_weights[original]
            delta = scale * torch.tensor(steer, dtype=torch.float32, device=device)

            def hook_fn(act, hook, d=delta):
                act[:, -1, :] = act[:, -1, :] + d.to(act.dtype).unsqueeze(0)
                return act

            with torch.no_grad():
                with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
                    logits = model(tokens)

            pred_tok = logits[0, -1].argmax(dim=-1).item()
            pred_str = model.tokenizer.decode([pred_tok]).strip()
            try:
                pred_digit = int(pred_str)
                scale_results["total"] += 1
                if pred_digit != original:
                    scale_results["changed"] += 1
                if pred_digit == target:
                    scale_results["exact"] += 1
            except ValueError:
                scale_results["total"] += 1

        n = scale_results["total"]
        if n > 0:
            logger.info(f"    scale={scale:5.1f}: changed={scale_results['changed']}/{n} ({100*scale_results['changed']/n:.1f}%), "
                         f"exact={scale_results['exact']}/{n} ({100*scale_results['exact']/n:.1f}%)")

    # Test 3: Steer to arbitrary target digits (not just +5)
    logger.info(f"\n  TEST C: Probe steering d → (d+k)%10 for k=1..9 at best scale")
    for k in range(1, 10):
        k_results = {"total": 0, "exact": 0, "changed": 0}

        for prob in single_digit:
            tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
            original = prob["ones_digit"]
            target = (original + k) % 10

            steer = probe_weights[target] - probe_weights[original]
            delta = 10.0 * torch.tensor(steer, dtype=torch.float32, device=device)

            def hook_fn(act, hook, d=delta):
                act[:, -1, :] = act[:, -1, :] + d.to(act.dtype).unsqueeze(0)
                return act

            with torch.no_grad():
                with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
                    logits = model(tokens)

            pred_tok = logits[0, -1].argmax(dim=-1).item()
            pred_str = model.tokenizer.decode([pred_tok]).strip()
            try:
                pred_digit = int(pred_str)
                k_results["total"] += 1
                if pred_digit != original:
                    k_results["changed"] += 1
                if pred_digit == target:
                    k_results["exact"] += 1
            except ValueError:
                k_results["total"] += 1

        n = k_results["total"]
        if n > 0:
            logger.info(f"    k={k} (d→d+{k}): exact={k_results['exact']}/{n} ({100*k_results['exact']/n:.1f}%), "
                         f"changed={k_results['changed']}/{n} ({100*k_results['changed']/n:.1f}%)")

    return {
        "probe_cv_accuracy": float(cv_scores.mean()),
        "probe_train_accuracy": float(train_acc),
        "plus5_results": plus5_results,
        "shift_hist": dict(shift_hist),
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Causal Digit Probing")
    parser.add_argument("--model", default="gemma-2b")
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--operand-range", type=int, default=30)
    parser.add_argument("--n-fisher", type=int, default=200)
    args = parser.parse_args()

    from transformer_lens import HookedTransformer

    model_name = MODEL_MAP.get(args.model, args.model)
    device = get_device()
    dtype = torch.float32

    logger.info("=" * 60)
    logger.info("CAUSAL DIGIT PROBING")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}, Layer: {args.layer}")

    logger.info(f"Loading {model_name} on {device}...")
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=dtype)

    # Generate problems
    all_problems = generate_problems(args.operand_range)
    single_problems = generate_problems(10, single_digit_only=True)
    logger.info(f"Generated {len(all_problems)} total, {len(single_problems)} single-digit problems")

    # Compute Fisher eigenvectors
    eigenvalues, eigenvectors, eff_dim = compute_fisher_eigenvectors(
        model, all_problems, args.layer, args.n_fisher
    )

    # Collect activations for probe training
    activations, labels = collect_activations(model, all_problems, args.layer, n_problems=300)

    # Compute freq-5 direction
    f5_direction, digit_means = compute_freq5_direction(activations, labels)

    all_results = {
        "model": model_name, "layer": args.layer, "eff_dim": eff_dim,
    }

    # ── Approach 1: Mod-2 Focused Steering ──
    if f5_direction is not None:
        mod2_results = approach1_mod2_steering(model, single_problems, args.layer, f5_direction)
        all_results["approach1_mod2"] = {
            "total": mod2_results["total"],
            "changed": mod2_results["changed"],
            "exact_plus5": mod2_results["exact_plus5"],
        }
    else:
        logger.warning("  Skipping Approach 1: freq-5 direction not found")

    # ── Approach 2: Sign-based Causal Probing ──
    causal_map = approach2_sign_probing(
        model, single_problems, args.layer, eigenvectors, eigenvalues, n_top=10
    )
    all_results["approach2_causal_map"] = [
        {"dim": r["dim"], "eigenvalue": r["eigenvalue"], "total": r["total"],
         "changed": r["changed"], "features": r["features"],
         "shift_hist": r["shift_hist"]}
        for r in causal_map
    ]

    # ── Approach 3: Linear Probe Steering ──
    probe_results = approach3_linear_probe(
        model, all_problems, single_problems, args.layer, activations, labels
    )
    all_results["approach3_probe"] = probe_results

    # Save
    out_dir = Path("mathematical_toolkit_results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"causal_probing_{args.model}_L{args.layer}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\n{'='*60}")
    logger.info(f"All results saved to {out_path}")


if __name__ == "__main__":
    main()
