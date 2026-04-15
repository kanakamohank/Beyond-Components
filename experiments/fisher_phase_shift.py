#!/usr/bin/env python3
"""
Fisher Causal Phase-Shift Intervention
=======================================

THE DEFINITIVE TEST: If the Fisher eigenvectors define the true causal plane
for arithmetic, then rotating activations within that 2D plane should
predictably shift the model's predicted ones digit.

Previous phase-shift tests in the SVD plane got 0% success.
This test rotates in the FISHER plane — the directions of maximum
∂ log p(correct) / ∂ activation, not maximum variance.

Protocol:
  1. Compute Fisher Information Matrix at target layers → top-2 eigenvectors
  2. Compute PCA/SVD top-2 directions as a control
  3. For each test problem where model is correct:
     a. Hook at target layer
     b. Project activation onto 2D Fisher plane
     c. Rotate by θ = k × 2π/10 (k=1..9)
     d. Inject rotated activation, check if ones digit shifts by k
  4. Compare Fisher-plane vs SVD-plane success rates

Usage:
    python experiments/fisher_phase_shift.py --model phi-3-mini --layers 24,26
    python experiments/fisher_phase_shift.py --model gemma-2b --layers 20,22,25
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import gc
import argparse
import logging
import warnings
from typing import Dict, List, Tuple, Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from arithmetic_circuit_scan_updated import (
    generate_direct_answer_problems,
    filter_correct_direct_answer,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════
# UTILITIES
# ═════════════════════════════════════════════════════════════

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_model_name(key: str) -> str:
    MODEL_MAP = {
        "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
        "gemma-2b": "google/gemma-2-2b",
        "gemma-7b": "google/gemma-7b",
        "gpt2-small": "gpt2",
        "gpt2-medium": "gpt2-medium",
        "pythia-1.4b": "EleutherAI/pythia-1.4b",
        "pythia-6.9b": "EleutherAI/pythia-6.9b",
        "llama-3b": "meta-llama/Llama-3.2-3B",
    }
    return MODEL_MAP.get(key, key)


def get_first_digit_token_id(model, answer: int) -> int:
    """Get the token ID for the FIRST DIGIT of the answer.
    
    For single-digit answers (0-9), this is the digit itself.
    For multi-digit answers (10+), this is the tens digit.
    
    NOTE: We use prompts with trailing space, so the model directly
    predicts the digit, not a space token.
    """
    digit_str = str(answer)
    first_char = digit_str[0]  # "8" for 8, "3" for 37
    tokens = model.to_tokens(first_char, prepend_bos=False)
    return tokens[0, 0].item()


def generate_test_problems(max_operand: int = 30, few_shot: bool = True):
    """Generate arithmetic problems for testing."""
    few_shot_prefix = ""
    if few_shot:
        few_shot_prefix = "Calculate:\n12 + 7 = 19\n34 + 15 = 49\n"

    problems = []
    for a in range(max_operand):
        for b in range(max_operand):
            answer = a + b
            # CRITICAL: trailing space so model directly predicts the digit
            prompt = f"{few_shot_prefix}{a} + {b} = "
            first_digit = str(answer)[0]  # first predicted digit
            problems.append({
                "a": a, "b": b, "answer": answer,
                "prompt": prompt,
                "ones_digit": answer % 10,
                "tens_digit": (answer // 10) % 10,
                "first_digit": int(first_digit),
                "n_digits": len(str(answer)),
                "carry": 1 if (a % 10 + b % 10) >= 10 else 0,
            })
    return problems


# ═════════════════════════════════════════════════════════════
# STEP 1: COMPUTE FISHER EIGENVECTORS (recompute, not from JSON)
# ═════════════════════════════════════════════════════════════

def compute_fisher_eigenvectors(
    model,
    problems: List[Dict],
    layer: int,
    n_problems: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute top-k Fisher Information eigenvectors at a specific layer.

    Returns:
        eigenvalues: (d_model,) sorted descending
        eigenvectors: (d_model, d_model) columns are eigenvectors, sorted descending
    """
    logger.info(f"  Computing Fisher eigenvectors at layer {layer} ({n_problems} problems)...")
    device = next(model.parameters()).device
    subset = problems[:n_problems]

    d_model = model.cfg.d_model
    fisher_matrix = np.zeros((d_model, d_model), dtype=np.float64)
    n_valid = 0
    hook_name = f"blocks.{layer}.hook_resid_post"

    for i, prob in enumerate(subset):
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True)

        activation_holder = {}

        def capture_hook(act, hook, holder=activation_holder):
            holder['act'] = act
            act.requires_grad_(True)
            act.retain_grad()
            return act

        # Get the first digit token (what model should predict at last position)
        answer_tok = get_first_digit_token_id(model, prob["answer"])

        try:
            with model.hooks(fwd_hooks=[(hook_name, capture_hook)]):
                logits = model(tokens)

            log_probs = F.log_softmax(logits[0, -1], dim=-1)
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if (i + 1) % 50 == 0:
            logger.info(f"    Processed {i + 1}/{len(subset)} ({n_valid} valid)")

    if n_valid < 10:
        logger.error(f"  Only {n_valid} valid — cannot compute Fisher at layer {layer}")
        return None, None

    fisher_matrix /= n_valid

    eigenvalues, eigenvectors = np.linalg.eigh(fisher_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    eff_dim_p = eigenvalues / eigenvalues.sum()
    eff_dim_p = eff_dim_p[eff_dim_p > 1e-10]
    eff_dim = np.exp(-np.sum(eff_dim_p * np.log(eff_dim_p)))

    ratio = eigenvalues[0] / eigenvalues[1] if eigenvalues[1] > 0 else float('inf')
    logger.info(f"  Fisher computed: eff_dim={eff_dim:.2f}, λ₁/λ₂={ratio:.1f}×, n_valid={n_valid}")

    return eigenvalues, eigenvectors


# ═════════════════════════════════════════════════════════════
# STEP 2: COMPUTE SVD/PCA DIRECTIONS (control)
# ═════════════════════════════════════════════════════════════

def compute_pca_directions(
    model,
    problems: List[Dict],
    layer: int,
    n_problems: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute PCA (SVD) directions from activations at a specific layer.

    Returns:
        singular_values: (d_model,) sorted descending
        components: (d_model, d_model) columns are principal components
    """
    logger.info(f"  Computing PCA/SVD directions at layer {layer} ({n_problems} problems)...")
    device = next(model.parameters()).device
    subset = problems[:n_problems]

    hook_name = f"blocks.{layer}.hook_resid_post"
    activations = []

    for i, prob in enumerate(subset):
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
            act = cache[hook_name][0, -1].cpu().float().numpy()
            activations.append(act)
            del cache

        if (i + 1) % 50 == 0:
            logger.info(f"    PCA: Processed {i + 1}/{len(subset)}")

    X = np.stack(activations)  # (n_problems, d_model)
    X_centered = X - X.mean(axis=0)

    # SVD of centered activations
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # Vt rows are principal components; transpose so columns are PCs
    components = Vt.T  # (d_model, n_problems) — columns are PCs

    logger.info(f"  PCA computed: top SVs = {S[:5].round(2)}")
    return S, components


# ═════════════════════════════════════════════════════════════
# STEP 3: FILTER PROBLEMS WHERE MODEL IS CORRECT
# ═════════════════════════════════════════════════════════════

def filter_correct_problems(
    model,
    problems: List[Dict],
    n_test: int = 200,
) -> List[Dict]:
    """Keep only problems where the model predicts the correct first digit.
    
    With trailing-space prompts, the model directly predicts digits:
    - Single-digit answers (0-9): model should predict that digit
    - Multi-digit answers (10+): model should predict the tens digit first
    """
    logger.info(f"  Filtering problems where model predicts correct first digit (testing {n_test})...")
    import random
    subset = random.sample(problems, min(n_test, len(problems)))
    correct = []

    for prob in subset:
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
        with torch.no_grad():
            logits = model(tokens)
        pred_tok = logits[0, -1].argmax(dim=-1).item()
        pred_str = model.tokenizer.decode([pred_tok]).strip()

        expected_first_digit = str(prob["answer"])[0]

        try:
            pred_digit = int(pred_str)
            if pred_str == expected_first_digit:
                prob["_pred_token"] = pred_tok
                prob["_pred_first_digit"] = pred_digit
                correct.append(prob)
        except ValueError:
            continue

    logger.info(f"  Model correct (first digit) on {len(correct)}/{len(subset)} = {len(correct)/len(subset)*100:.1f}%")
    return correct


# ═════════════════════════════════════════════════════════════
# STEP 4: THE PHASE-SHIFT INTERVENTION
# ═════════════════════════════════════════════════════════════

def rotation_matrix_2d(theta: float) -> np.ndarray:
    """2D rotation matrix by angle theta."""
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])


def run_phase_shift_experiment(
    model,
    problems: List[Dict],
    layer: int,
    basis_vectors: np.ndarray,   # (d_model, 2) — the 2D plane to rotate in
    basis_name: str,             # "Fisher" or "PCA"
    n_shifts: int = 9,          # test shifts k=1..9
) -> Dict[str, Any]:
    """
    Core experiment: rotate activations in a 2D subspace and check if
    the predicted first digit changes predictably.

    For each problem, for each shift k=1..9:
      - Apply rotation θ = k × 2π/10 in the 2D basis plane
      - Check if model now predicts a different first digit
      - For single-digit answers: check if (original + k) mod 10 matches

    KEY METRICS:
      - digit_change_rate: How often does rotation change the prediction?
        (Fisher should >> PCA — this is THE test)
      - exact_shift_rate: For single-digit answers, does it shift by k mod 10?
      - numeric_rate: Does the model still predict a number (vs garbage)?
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"PHASE-SHIFT INTERVENTION: {basis_name} plane at Layer {layer}")
    logger.info(f"{'='*60}")
    logger.info(f"  Testing {len(problems)} correct problems × {n_shifts} rotations")

    device = next(model.parameters()).device
    hook_name = f"blocks.{layer}.hook_resid_post"

    # Convert basis to torch tensor on device
    v1 = torch.tensor(basis_vectors[:, 0], dtype=torch.float32, device=device)
    v2 = torch.tensor(basis_vectors[:, 1], dtype=torch.float32, device=device)

    results_by_shift = {k: {"exact_match": 0, "changed": 0, "numeric": 0, "total": 0, "details": []}
                        for k in range(1, n_shifts + 1)}
    total_exact = 0
    total_changed = 0
    total_numeric = 0
    total_tests = 0

    for prob_idx, prob in enumerate(problems):
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
        original_first_digit = prob["first_digit"]
        original_ones = prob["ones_digit"]
        is_single_digit = (prob["n_digits"] == 1)

        # Get clean activation at this layer
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
            clean_act = cache[hook_name][0, -1].clone()  # (d_model,)
            del cache

        # Project activation onto 2D plane (once per problem)
        coeff1 = torch.dot(clean_act.float(), v1)
        coeff2 = torch.dot(clean_act.float(), v2)

        for k in range(1, n_shifts + 1):
            theta = k * 2 * np.pi / 10

            # Rotate in the 2D plane
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            new_coeff1 = coeff1 * cos_t - coeff2 * sin_t
            new_coeff2 = coeff1 * sin_t + coeff2 * cos_t

            # Compute the delta: new_component - old_component
            delta = (new_coeff1 - coeff1) * v1 + (new_coeff2 - coeff2) * v2

            # Create hook that adds this delta
            def rotation_hook(act, hook, d=delta):
                act[:, -1, :] = act[:, -1, :] + d.to(act.dtype).unsqueeze(0)
                return act

            # Run with intervention
            with torch.no_grad():
                with model.hooks(fwd_hooks=[(hook_name, rotation_hook)]):
                    logits = model(tokens)

            pred_tok = logits[0, -1].argmax(dim=-1).item()
            pred_str = model.tokenizer.decode([pred_tok]).strip()

            is_numeric = False
            pred_digit = -1
            try:
                pred_digit = int(pred_str)
                is_numeric = True
            except ValueError:
                pass

            digit_changed = is_numeric and (pred_digit != original_first_digit)
            # For single-digit answers, check exact mod-10 shift
            if is_single_digit:
                expected = (original_ones + k) % 10
            else:
                expected = -99  # no exact expectation for tens digit
            is_exact = is_numeric and is_single_digit and (pred_digit == expected)

            r = results_by_shift[k]
            r["total"] += 1
            if is_numeric:
                r["numeric"] += 1
                total_numeric += 1
            if digit_changed:
                r["changed"] += 1
                total_changed += 1
            if is_exact:
                r["exact_match"] += 1
                total_exact += 1
            r["details"].append({
                "original": original_first_digit,
                "expected": expected if is_single_digit else None,
                "predicted": pred_digit,
                "exact": is_exact,
                "changed": digit_changed,
                "numeric": is_numeric,
                "single_digit": is_single_digit,
            })
            total_tests += 1

        if (prob_idx + 1) % 20 == 0:
            chg_rate = total_changed / total_tests * 100 if total_tests > 0 else 0
            logger.info(f"  Progress: {prob_idx + 1}/{len(problems)}, "
                        f"digit_changed={chg_rate:.1f}%")

    # Summarize
    overall_change_rate = total_changed / total_tests * 100 if total_tests > 0 else 0
    overall_numeric_rate = total_numeric / total_tests * 100 if total_tests > 0 else 0
    # Exact match only counted for single-digit answers
    n_single = sum(1 for k in range(1, n_shifts + 1)
                   for d in results_by_shift[k]["details"] if d["single_digit"])
    overall_exact_rate = total_exact / n_single * 100 if n_single > 0 else 0

    logger.info(f"\n  {'='*50}")
    logger.info(f"  {basis_name} PLANE RESULTS (Layer {layer}):")
    logger.info(f"  {'='*50}")
    logger.info(f"  Digit changed (any): {total_changed}/{total_tests} = {overall_change_rate:.1f}%")
    logger.info(f"  Still numeric: {total_numeric}/{total_tests} = {overall_numeric_rate:.1f}%")
    logger.info(f"  Exact mod-10 shift (single-digit only): {total_exact}/{n_single} = {overall_exact_rate:.1f}%")

    per_shift = {}
    for k in range(1, n_shifts + 1):
        r = results_by_shift[k]
        change_rate = r["changed"] / r["total"] * 100 if r["total"] > 0 else 0
        n_single_k = sum(1 for d in r["details"] if d["single_digit"])
        exact_rate = r["exact_match"] / n_single_k * 100 if n_single_k > 0 else 0
        logger.info(f"  Shift k={k} (θ={k*36}°): changed={change_rate:.1f}%, "
                    f"exact_mod10={exact_rate:.1f}% (of {n_single_k} single-digit)")
        per_shift[k] = {
            "change_rate": change_rate,
            "exact_rate_single_digit": exact_rate,
            "n_changed": r["changed"],
            "n_exact": r["exact_match"],
            "n_total": r["total"],
            "n_single_digit": n_single_k,
        }

    # Shift histogram — what actual digit changes are observed?
    all_details = []
    for k in range(1, n_shifts + 1):
        for d in results_by_shift[k]["details"]:
            all_details.append(d)

    shift_histogram = {}
    for d in all_details:
        if d["numeric"] and d["predicted"] >= 0:
            actual_shift = (d["predicted"] - d["original"]) % 10
            shift_histogram[actual_shift] = shift_histogram.get(actual_shift, 0) + 1

    if shift_histogram:
        logger.info(f"\n  Shift histogram (actual digit changes observed):")
        n_with_pred = sum(shift_histogram.values())
        for s in sorted(shift_histogram.keys()):
            logger.info(f"    Δ={s}: {shift_histogram[s]} ({shift_histogram[s]/n_with_pred*100:.1f}%)")

    return {
        "basis": basis_name,
        "layer": layer,
        "digit_change_rate": overall_change_rate,
        "numeric_rate": overall_numeric_rate,
        "exact_mod10_rate_single_digit": overall_exact_rate,
        "per_shift": per_shift,
        "shift_histogram": shift_histogram,
        "n_problems": len(problems),
        "n_total_tests": total_tests,
        "n_single_digit_tests": n_single,
    }


# ═════════════════════════════════════════════════════════════
# STEP 5: SCALE-SWEEP VARIANT
# ═════════════════════════════════════════════════════════════

def run_causal_intervention_suite(
    model,
    problems: List[Dict],
    layer: int,
    eigenvectors: np.ndarray,   # (d_model, d_model) full eigenvectors
    eigenvalues: np.ndarray,    # (d_model,) eigenvalues
    basis_name: str,            # "Fisher" or "PCA"
    n_dims_list: List[int] = [2, 5, 10],  # number of subspace dims to test
) -> Dict[str, Any]:
    """
    Comprehensive causal intervention suite:
      A. KNOCKOUT: Zero out the top-k subspace component
      B. AMPLIFY: Double the top-k subspace component
      C. ADDITIVE: Add top eigenvector scaled by activation norm
      D. RANDOM CONTROL: Same interventions in random subspace

    If Fisher knockout changes the digit but PCA knockout doesn't,
    that proves the Fisher subspace is the true causal subspace.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"CAUSAL INTERVENTION SUITE: {basis_name} at Layer {layer}")
    logger.info(f"{'='*60}")

    device = next(model.parameters()).device
    hook_name = f"blocks.{layer}.hook_resid_post"

    # Pre-compute basis tensors for each dimensionality
    results = {}

    for n_dims in n_dims_list:
        basis = torch.tensor(eigenvectors[:, :n_dims], dtype=torch.float32, device=device)
        # basis: (d_model, n_dims)

        # Random control basis (same dimensionality) — QR on CPU (MPS unsupported)
        rand_basis_cpu = torch.randn(eigenvectors.shape[0], n_dims, dtype=torch.float32)
        rand_basis_cpu, _ = torch.linalg.qr(rand_basis_cpu)
        rand_basis = rand_basis_cpu.to(device)

        interventions = {
            "knockout": lambda act, b=basis: act - (act @ b) @ b.T,
            "amplify_2x": lambda act, b=basis: act + (act @ b) @ b.T,
            "negate": lambda act, b=basis: act - 2 * (act @ b) @ b.T,
            "rand_knockout": lambda act, b=rand_basis: act - (act @ b) @ b.T,
        }

        dim_results = {}
        for interv_name, interv_fn in interventions.items():
            n_changed = 0
            n_numeric = 0
            n_total = 0
            proj_norms = []
            act_norms = []

            for prob in problems:
                tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
                original_digit = prob["first_digit"]

                # Get clean activation to measure projection magnitude
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens, names_filter=hook_name)
                    clean_act = cache[hook_name][0, -1].float()
                    proj = (clean_act @ basis) @ basis.T
                    proj_norms.append(proj.norm().item())
                    act_norms.append(clean_act.norm().item())
                    del cache

                def hook_fn(act, hook, fn=interv_fn):
                    act_f = act[:, -1, :].float()
                    act[:, -1, :] = fn(act_f).to(act.dtype)
                    return act

                with torch.no_grad():
                    with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
                        logits = model(tokens)

                pred_tok = logits[0, -1].argmax(dim=-1).item()
                pred_str = model.tokenizer.decode([pred_tok]).strip()

                try:
                    pred_digit = int(pred_str)
                    n_numeric += 1
                    if pred_digit != original_digit:
                        n_changed += 1
                except ValueError:
                    n_changed += 1  # non-numeric = definitely changed
                n_total += 1

            change_rate = n_changed / n_total * 100 if n_total > 0 else 0
            numeric_rate = n_numeric / n_total * 100 if n_total > 0 else 0
            mean_proj = np.mean(proj_norms)
            mean_act = np.mean(act_norms)
            proj_frac = mean_proj / mean_act * 100 if mean_act > 0 else 0

            logger.info(f"  [{basis_name} {n_dims}D] {interv_name}: "
                        f"changed={change_rate:.1f}%, numeric={numeric_rate:.1f}%, "
                        f"proj_norm={mean_proj:.1f} ({proj_frac:.1f}% of activation)")

            dim_results[interv_name] = {
                "change_rate": change_rate,
                "numeric_rate": numeric_rate,
                "n_changed": n_changed,
                "n_total": n_total,
                "mean_proj_norm": mean_proj,
                "mean_act_norm": mean_act,
                "proj_fraction": proj_frac,
            }

        results[f"{n_dims}D"] = dim_results

    return {
        "basis": basis_name,
        "layer": layer,
        "n_problems": len(problems),
        "dim_results": results,
    }


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Fisher Causal Phase-Shift Intervention")
    parser.add_argument("--model", type=str, default="phi-3-mini",
                        help="Model key (phi-3-mini, gemma-2b, gpt2-small)")
    parser.add_argument("--layers", type=str, default="24,26",
                        help="Comma-separated layer indices to test")
    parser.add_argument("--operand-range", type=int, default=30,
                        help="Max operand for test problems")
    parser.add_argument("--n-fisher-problems", type=int, default=200,
                        help="Number of problems for Fisher matrix estimation")
    parser.add_argument("--n-test-problems", type=int, default=100,
                        help="Number of correct problems to test interventions on")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--skip-pca", action="store_true",
                        help="Skip PCA control (faster)")
    parser.add_argument("--skip-rotation", action="store_true",
                        help="Skip the phase-shift rotation test (run only knockout suite)")
    parser.add_argument("--direct-answer", action="store_true",
                        help="Use direct-answer mode (for LLaMA 3B: full answer as single token)")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cpu, mps, cuda). Default: auto-detect")
    args = parser.parse_args()

    model_name = resolve_model_name(args.model)
    layers = [int(x.strip()) for x in args.layers.split(",")]

    logger.info(f"{'='*60}")
    logger.info(f"FISHER CAUSAL PHASE-SHIFT INTERVENTION")
    logger.info(f"{'='*60}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Layers: {layers}")
    logger.info(f"Operand range: {args.operand_range}")
    logger.info(f"Fisher problems: {args.n_fisher_problems}")
    logger.info(f"Test problems: {args.n_test_problems}")

    # Load model
    from transformer_lens import HookedTransformer

    device = torch.device(args.device) if args.device else get_device()
    dtype = torch.float32
    if "gemma" in model_name.lower():
        dtype = torch.float32  # Gemma needs float32 on MPS
    elif "phi" in model_name.lower():
        dtype = torch.float32

    logger.info(f"Loading {model_name} on {device} ({dtype})...")
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=dtype)
    model.eval()

    # Generate problems
    if args.direct_answer:
        da_problems, _ = generate_direct_answer_problems(n_per_digit=100, operand_max=args.operand_range)
        logger.info(f"Generated {len(da_problems)} direct-answer problems")
        da_correct = filter_correct_direct_answer(model, da_problems, max_n=len(da_problems))
        # Convert to fisher_phase_shift format (needs first_digit, n_digits, ones_digit, carry)
        problems = []
        for p in da_correct:
            answer = p["answer"]
            problems.append({
                "a": p["a"], "b": p["b"], "answer": answer,
                "prompt": p["prompt"],
                "ones_digit": answer % 10,
                "tens_digit": (answer // 10) % 10,
                "first_digit": int(str(answer)[0]),
                "n_digits": len(str(answer)),
                "carry": 1 if (p["a"] % 10 + p["b"] % 10) >= 10 else 0,
            })
        correct_problems = problems
        logger.info(f"Direct-answer correct problems: {len(correct_problems)}")
    else:
        problems = generate_test_problems(max_operand=args.operand_range)
        logger.info(f"Generated {len(problems)} problems")
        correct_problems = filter_correct_problems(model, problems, n_test=args.n_test_problems * 3)

    if len(correct_problems) < 20:
        logger.error(f"Only {len(correct_problems)} correct — need at least 20. Aborting.")
        return
    test_problems = correct_problems[:args.n_test_problems]
    logger.info(f"Using {len(test_problems)} correct problems for intervention tests")

    all_results = {
        "model": args.model,
        "model_name": model_name,
        "layers": layers,
        "n_test_problems": len(test_problems),
        "operand_range": args.operand_range,
        "experiments": [],
    }

    for layer in layers:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# LAYER {layer}")
        logger.info(f"{'#'*60}")

        # Compute Fisher eigenvectors
        fisher_evals, fisher_evecs = compute_fisher_eigenvectors(
            model, problems, layer, n_problems=args.n_fisher_problems
        )
        if fisher_evals is None:
            logger.error(f"  Fisher computation failed at layer {layer} — skipping")
            continue

        # ── Causal Intervention Suite (Fisher) ──
        fisher_suite = run_causal_intervention_suite(
            model, test_problems[:50], layer,
            fisher_evecs, fisher_evals, "Fisher",
            n_dims_list=[2, 5, 10, 20]
        )
        all_results["experiments"].append(fisher_suite)

        # ── Causal Phase-Shift Rotation (Fisher) ──
        if not args.skip_rotation:
            fisher_shift = run_phase_shift_experiment(
                model, test_problems, layer,
                fisher_evecs[:, :2], "Fisher"
            )
            all_results["experiments"].append(fisher_shift)

        # ── PCA control ──
        if not args.skip_pca:
            pca_svals, pca_components = compute_pca_directions(
                model, problems, layer, n_problems=args.n_fisher_problems
            )
            pca_suite = run_causal_intervention_suite(
                model, test_problems[:50], layer,
                pca_components, pca_svals, "PCA/SVD",
                n_dims_list=[2, 5, 10, 20]
            )
            all_results["experiments"].append(pca_suite)

            # ── Causal Phase-Shift Rotation (PCA) ──
            if not args.skip_rotation:
                pca_shift = run_phase_shift_experiment(
                    model, test_problems, layer,
                    pca_components[:, :2], "PCA/SVD"
                )
                all_results["experiments"].append(pca_shift)

        gc.collect()

    # ── Summary ──
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL SUMMARY")
    logger.info(f"{'='*60}")
    for exp in all_results["experiments"]:
        if "dim_results" in exp:
            logger.info(f"\n  {exp['basis']} L{exp['layer']} Intervention Suite:")
            for dim_key, interventions in exp["dim_results"].items():
                for iname, ires in interventions.items():
                    logger.info(f"    [{dim_key}] {iname}: changed={ires['change_rate']:.1f}%, "
                                f"proj={ires['proj_fraction']:.1f}%")
        elif "digit_change_rate" in exp:
            logger.info(f"  {exp['basis']} L{exp['layer']}: "
                        f"digit_changed={exp['digit_change_rate']:.1f}%, "
                        f"numeric={exp['numeric_rate']:.1f}%")

    # Save results
    output_dir = Path("mathematical_toolkit_results")
    output_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = "_direct" if args.direct_answer else ""
    output_file = output_dir / f"fisher_phase_shift_{args.model}{suffix}_{timestamp}.json"

    # Make JSON-serializable
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    with open(output_file, 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2)
    logger.info(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    import time
    main()
