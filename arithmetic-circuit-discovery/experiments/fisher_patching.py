#!/usr/bin/env python3
"""
Fisher Subspace Activation Patching: The definitive test.

We know from previous experiments:
- Fisher 5D negate changes 52% of predictions (strong causal effect)
- But NO additive steering method can target specific digits (rotation, DIM, probe all fail)

This experiment tests whether the Fisher subspace MEDIATES digit identity via patching:

1. FULL PATCH: Replace entire activation from clean→corrupt. Baseline.
2. FISHER PATCH: Replace ONLY the Fisher-subspace projection. If this changes
   the digit, the Fisher subspace causally encodes digit identity.
3. ORTHO PATCH: Replace ONLY the component orthogonal to Fisher subspace. 
   If this does NOT change the digit, the digit info is in Fisher, not outside.
4. EXCHANGE PATCH: Take clean problem (digit=d1), corrupt problem (digit=d2).
   Replace Fisher projection of corrupt with clean's Fisher projection.
   Does the model now predict d1? This tests directed digit transfer.

This avoids assuming ANY geometry — it uses actual model activations as
the source of the intervention, not synthetic vectors.
"""

import argparse
import logging
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from arithmetic_circuit_scan_updated import (
    generate_direct_answer_problems,
    filter_correct_direct_answer,
)

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


def generate_problems(operand_range=30, prioritize_single_digit=False):
    """Generate addition problems.
    
    If prioritize_single_digit=True, single-digit answer problems come first
    (shuffled), followed by multi-digit (shuffled). This ensures contrastive
    Fisher gets enough single-digit samples.
    """
    problems = []
    for a in range(operand_range):
        for b in range(operand_range):
            answer = a + b
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
    if prioritize_single_digit:
        single = [p for p in problems if p["answer"] < 10]
        multi = [p for p in problems if p["answer"] >= 10]
        np.random.shuffle(single)
        np.random.shuffle(multi)
        problems = single + multi
    else:
        np.random.shuffle(problems)
    return problems


def generate_teacher_forced_problems(n_per_digit=100, operand_max=99):
    """Generate problems for contrastive Fisher via teacher-forcing.

    For ones_digit=0: problems like 12+18=3[0], 21+19=4[0], ...
    Prompt = full expression up to and including tens digit.
    Target token = ones digit.

    This gives unlimited data for every digit class and ensures the
    gradient always flows through the ones-digit prediction, not the
    tens-digit prediction.
    """
    by_digit = defaultdict(list)

    for a in range(operand_max + 1):
        for b in range(operand_max + 1):
            answer = a + b
            ones_digit = answer % 10
            answer_str = str(answer)

            if len(answer_str) == 1:
                # Single-digit answer: prompt ends before answer
                prompt = f"Calculate:\n{a} + {b} = "
                target_str = answer_str
                prefix_str = ""
            else:
                # Multi-digit answer: teacher-force all digits except last
                prefix = answer_str[:-1]
                prompt = f"Calculate:\n{a} + {b} = {prefix}"
                target_str = answer_str[-1]
                prefix_str = prefix

            by_digit[ones_digit].append({
                "prompt": prompt,
                "answer": answer,
                "ones_digit": ones_digit,
                "target_str": target_str,
                "prefix_str": prefix_str,
                "a": a, "b": b,
            })

    # Balance: sample n_per_digit from each class
    problems = []
    for d in range(10):
        pool = by_digit[d]
        np.random.shuffle(pool)
        problems.extend(pool[:n_per_digit])

    np.random.shuffle(problems)
    return problems, by_digit


def compute_fisher_eigenvectors(model, problems, layer, n_problems=200):
    """Compute Fisher Information Matrix eigenvectors."""
    logger.info(f"  Computing Fisher eigenvectors at layer {layer} ({n_problems} problems)...")
    device = next(model.parameters()).device
    hook_name = f"blocks.{layer}.hook_resid_post"
    d_model = model.cfg.d_model

    fisher_matrix = np.zeros((d_model, d_model))
    n_valid = 0

    for i, prob in enumerate(problems[:n_problems]):
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
        answer_str = str(prob["first_digit"])
        answer_toks = model.to_tokens(answer_str, prepend_bos=False)
        answer_tok = answer_toks[0, 0].item()

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
        except Exception:
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

    logger.info(f"  Fisher: eff_dim={eff_dim:.2f}, λ₁/λ₂={eigenvalues[0]/eigenvalues[1]:.1f}×, valid={n_valid}")
    return eigenvalues, eigenvectors, eff_dim


def compute_contrastive_fisher(model, problems, layer, n_problems=200):
    """Compute contrastive Fisher: directions that discriminate between digit classes.

    Instead of one pooled Fisher matrix, we:
    1. Collect individual gradients labeled by ones digit
    2. Compute mean gradient per digit class (mu_k)
    3. Build between-class scatter: S_B = sum_k n_k (mu_k - mu)(mu_k - mu)^T
    4. Top eigenvectors of S_B = digit-discriminative directions

    Also returns the standard Fisher eigenvectors from the same gradients.
    S_B has rank <= 9 (10 classes - 1), so at most 9 contrastive directions.
    """
    logger.info(f"  Computing contrastive Fisher at layer {layer} ({n_problems} problems)...")
    device = next(model.parameters()).device
    hook_name = f"blocks.{layer}.hook_resid_post"
    d_model = model.cfg.d_model

    gradients_by_digit = defaultdict(list)
    all_gradients = []
    n_valid = 0

    for i, prob in enumerate(problems[:n_problems]):
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
        answer_str = str(prob["first_digit"])
        answer_toks = model.to_tokens(answer_str, prepend_bos=False)
        answer_tok = answer_toks[0, 0].item()

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
                all_gradients.append(g)
                n_valid += 1
                # For contrastive: only use single-digit answers where
                # ones_digit == first_digit (no tens/ones mismatch)
                if prob["answer"] < 10:
                    gradients_by_digit[prob["ones_digit"]].append(g)
        except Exception:
            continue
        finally:
            model.zero_grad()

        if (i + 1) % 50 == 0:
            logger.info(f"    Processed {i+1}/{n_problems} ({n_valid} valid)")

    if n_valid < 10:
        raise ValueError(f"Only {n_valid} valid gradients")

    # --- Standard Fisher (for comparison) ---
    fisher_matrix = np.zeros((d_model, d_model))
    for g in all_gradients:
        fisher_matrix += np.outer(g, g)
    fisher_matrix /= n_valid

    std_eigenvalues, std_eigenvectors = np.linalg.eigh(fisher_matrix)
    idx = np.argsort(std_eigenvalues)[::-1]
    std_eigenvalues = std_eigenvalues[idx]
    std_eigenvectors = std_eigenvectors[:, idx]

    total = std_eigenvalues.sum()
    if total > 0:
        p = std_eigenvalues / total
        p = p[p > 1e-10]
        std_eff_dim = np.exp(-np.sum(p * np.log(p)))
    else:
        std_eff_dim = 0

    # --- Contrastive Fisher (between-class scatter) ---
    class_means = {}
    for digit, grads in gradients_by_digit.items():
        class_means[digit] = np.mean(grads, axis=0)

    digits_found = sorted(class_means.keys())
    all_means = np.array([class_means[d] for d in digits_found])
    weights = np.array([len(gradients_by_digit[d]) for d in digits_found])
    overall_mean = np.average(all_means, axis=0, weights=weights)

    S_B = np.zeros((d_model, d_model))
    for digit in digits_found:
        diff = class_means[digit] - overall_mean
        n_k = len(gradients_by_digit[digit])
        S_B += n_k * np.outer(diff, diff)
    S_B /= n_valid

    con_eigenvalues, con_eigenvectors = np.linalg.eigh(S_B)
    idx = np.argsort(con_eigenvalues)[::-1]
    con_eigenvalues = con_eigenvalues[idx]
    con_eigenvectors = con_eigenvectors[:, idx]

    n_nonzero = np.sum(con_eigenvalues > 1e-10 * con_eigenvalues[0])

    n_contrastive_samples = sum(len(v) for v in gradients_by_digit.values())
    logger.info(f"  Standard Fisher: eff_dim={std_eff_dim:.2f}, valid={n_valid}")
    logger.info(f"  Contrastive Fisher: {n_nonzero} non-trivial directions (max 9), {n_contrastive_samples} single-digit samples")
    logger.info(f"    Digits found: {digits_found}, counts: {dict(sorted({d: len(gradients_by_digit[d]) for d in digits_found}.items()))}")
    logger.info(f"    Top contrastive eigenvalues: {con_eigenvalues[:5]}")

    return {
        "standard": (std_eigenvalues, std_eigenvectors, std_eff_dim),
        "contrastive": (con_eigenvalues, con_eigenvectors, int(n_nonzero)),
        "n_valid": n_valid,
        "digits_found": digits_found,
    }


def compute_contrastive_fisher_v3(model, problems_by_digit, layer,
                                   n_per_digit=100):
    """Contrastive Fisher with teacher-forced ones-digit targets.

    Computes:
      F_pooled = (1/N) sum_i g_i g_i^T           standard (pooled) Fisher
      mu_k = (1/n_k) sum_{i in class_k} g_i       mean gradient per class
      mu   = (1/10) sum_k mu_k                     global mean
      S_B  = sum_k n_k (mu_k - mu)(mu_k - mu)^T   between-class scatter

    Top eigenvectors of S_B = directions maximally discriminating digit classes.
    Rank of S_B <= 9 (10 classes - 1), so at most 9 contrastive directions.
    """
    logger.info(f"  Computing contrastive Fisher v3 at layer {layer}...")
    hook_name = f"blocks.{layer}.hook_resid_post"
    d_model = model.cfg.d_model

    class_gradients = defaultdict(list)
    all_gradients = []
    n_valid = 0

    for digit in range(10):
        problems = problems_by_digit[digit][:n_per_digit]
        for prob in problems:
            tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
            # Use [0, -1] not [0, 0]: some tokenizers (Phi-3) prepend a
            # space token before the digit, so last token is the actual digit
            target_tok = model.to_tokens(prob["target_str"],
                                         prepend_bos=False)[0, -1].item()

            holder = {}

            def capture(act, hook, h=holder):
                h['act'] = act
                act.requires_grad_(True)
                act.retain_grad()
                return act

            try:
                with model.hooks(fwd_hooks=[(hook_name, capture)]):
                    logits = model(tokens)

                log_p = F.log_softmax(logits[0, -1].float(), dim=-1)[target_tok]
                log_p.backward(retain_graph=False)

                if 'act' in holder and holder['act'].grad is not None:
                    g = holder['act'].grad[0, -1].detach().cpu().float().numpy()
                    class_gradients[digit].append(g)
                    all_gradients.append(g)
                    n_valid += 1
            except Exception:
                pass
            finally:
                model.zero_grad()

        logger.info(f"    Digit {digit}: {len(class_gradients[digit])} valid gradients")

    logger.info(f"  Total valid: {n_valid}")

    # -- Standard (pooled) Fisher --
    F_pooled = np.zeros((d_model, d_model), dtype=np.float64)
    for g in all_gradients:
        F_pooled += np.outer(g, g)
    F_pooled /= n_valid

    evals_std, evecs_std = np.linalg.eigh(F_pooled)
    idx = np.argsort(evals_std)[::-1]
    evals_std = evals_std[idx]
    evecs_std = evecs_std[:, idx]

    p = evals_std / (evals_std.sum() + 1e-30)
    p = p[p > 1e-10]
    eff_dim = float(np.exp(-np.sum(p * np.log(p))))

    # -- Contrastive (between-class scatter) Fisher --
    class_means = {}
    class_counts = {}
    for d in range(10):
        grads = class_gradients[d]
        if len(grads) == 0:
            continue
        class_means[d] = np.mean(grads, axis=0)
        class_counts[d] = len(grads)

    global_mean = np.mean([class_means[d] for d in class_means], axis=0)

    S_B = np.zeros((d_model, d_model), dtype=np.float64)
    for d, mu_k in class_means.items():
        diff = mu_k - global_mean
        S_B += class_counts[d] * np.outer(diff, diff)
    S_B /= n_valid

    evals_con, evecs_con = np.linalg.eigh(S_B)
    idx = np.argsort(evals_con)[::-1]
    evals_con = evals_con[idx]
    evecs_con = evecs_con[:, idx]

    threshold = evals_con[0] * 1e-4
    n_nontrivial = int(np.sum(evals_con > threshold))
    n_contrastive = min(n_nontrivial, 9)

    logger.info(f"  Standard Fisher: eff_dim={eff_dim:.2f}")
    logger.info(f"  Contrastive: {n_contrastive} non-trivial directions")
    logger.info(f"    Class counts: {dict(sorted(class_counts.items()))}")
    logger.info(f"    Top contrastive eigenvalues: {evals_con[:5]}")

    return {
        "standard": (evals_std, evecs_std, eff_dim),
        "contrastive": (evals_con, evecs_con, n_contrastive),
        "digits_found": sorted(class_means.keys()),
        "class_counts": class_counts,
    }


def generate_balanced_test_problems(min_per_digit=8):
    """Generate test problems with guaranteed coverage of all 10 ones digits.

    Uses operand_range=10 (answers 0-18) for single-digit problems,
    then supplements with targeted pairs for any missing/underrepresented digits.
    """
    base = generate_problems(10)
    by_digit = defaultdict(list)
    for p in base:
        by_digit[p["ones_digit"]].append(p)

    # Check coverage
    for d in range(10):
        if len(by_digit[d]) < min_per_digit:
            # Generate extra problems targeting this ones digit
            for a in range(50):
                for b in range(50):
                    if (a + b) % 10 == d and len(by_digit[d]) < min_per_digit * 2:
                        answer = a + b
                        p = {
                            "prompt": f"Calculate:\n{a} + {b} = ",
                            "answer": answer,
                            "ones_digit": d,
                            "first_digit": int(str(answer)[0]),
                            "n_digits": len(str(answer)),
                            "a": a, "b": b,
                        }
                        if p not in base:
                            base.append(p)
                            by_digit[d].append(p)

    np.random.shuffle(base)
    logger.info(f"  Balanced test: {len(base)} problems, digit counts: {dict(sorted({d: len(v) for d, v in by_digit.items()}.items()))}")
    return base


def get_model_prediction(model, tokens):
    """Get model's predicted token at last position."""
    with torch.no_grad():
        logits = model(tokens)
    pred_tok = logits[0, -1].argmax(dim=-1).item()
    return pred_tok


def filter_correct_problems(model, problems, max_n=100):
    """Filter to problems where model predicts ones digit correctly."""
    correct = []
    for prob in problems[:max_n * 3]:
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
        pred_tok = get_model_prediction(model, tokens)
        pred_str = model.tokenizer.decode([pred_tok]).strip()
        try:
            if int(pred_str) == prob["first_digit"]:
                prob["_tokens"] = tokens
                correct.append(prob)
        except ValueError:
            pass
        if len(correct) >= max_n:
            break
    return correct


def filter_correct_teacher_forced(model, problems, max_n=100):
    """Filter teacher-forced problems where model predicts ones digit correctly."""
    correct = []
    for prob in problems[:max_n * 3]:
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
        pred_tok = get_model_prediction(model, tokens)
        pred_str = model.tokenizer.decode([pred_tok]).strip()
        try:
            if int(pred_str) == prob["ones_digit"]:
                prob["_tokens"] = tokens
                correct.append(prob)
        except ValueError:
            pass
        if len(correct) >= max_n:
            break
    return correct


# ═══════════════════════════════════════════════════════════════
# PATCHING EXPERIMENTS
# ═══════════════════════════════════════════════════════════════

def run_patching_experiment(model, layer, eigenvectors, correct_problems,
                            n_fisher_dims=5, teacher_forced=False, direct_answer=False):
    """
    Run three patching experiments:
    1. Full patch
    2. Fisher-subspace patch
    3. Orthogonal patch
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"ACTIVATION PATCHING EXPERIMENT (Layer {layer}, Fisher {n_fisher_dims}D)")
    logger.info(f"{'='*60}")

    device = next(model.parameters()).device
    hook_name = f"blocks.{layer}.hook_resid_post"

    # Build Fisher projection matrix (n_fisher_dims × d_model)
    fisher_basis = torch.tensor(
        eigenvectors[:, :n_fisher_dims].T,  # (n_fisher_dims, d_model)
        dtype=torch.float32, device=device
    )

    # Group problems by ones digit
    by_digit = defaultdict(list)
    for p in correct_problems:
        if teacher_forced:
            # Teacher-forced: all problems target ones digit
            by_digit[p["ones_digit"]].append(p)
        else:
            # Legacy: only single-digit answers
            if p.get("n_digits", 1) == 1:
                by_digit[p["ones_digit"]].append(p)

    logger.info(f"  Problems by digit: {dict(sorted({d: len(v) for d, v in by_digit.items()}.items()))}")

    # ── Collect clean activations for all problems ──
    logger.info(f"  Caching clean activations...")
    for prob in correct_problems:
        tokens = prob["_tokens"]
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
            prob["_clean_act"] = cache[hook_name][0, -1].clone()
            del cache

    # ── Test 1: FULL PATCH ──
    # For each pair (clean, corrupt) with different digits,
    # replace corrupt's activation with clean's. Does output become clean's digit?
    logger.info(f"\n  TEST 1: FULL ACTIVATION PATCH")
    logger.info(f"  Replace ALL of corrupt's L{layer} activation with clean's")

    full_patch_results = _run_patch_test(
        model, hook_name, by_digit, fisher_basis,
        patch_mode="full", n_fisher_dims=n_fisher_dims, direct_answer=direct_answer
    )

    # ── Test 2: FISHER-SUBSPACE PATCH ──
    logger.info(f"\n  TEST 2: FISHER {n_fisher_dims}D SUBSPACE PATCH")
    logger.info(f"  Replace only the Fisher projection of corrupt with clean's")

    fisher_patch_results = _run_patch_test(
        model, hook_name, by_digit, fisher_basis,
        patch_mode="fisher", n_fisher_dims=n_fisher_dims, direct_answer=direct_answer
    )

    # ── Test 3: ORTHOGONAL PATCH ──
    logger.info(f"\n  TEST 3: ORTHOGONAL PATCH (everything EXCEPT Fisher)")
    logger.info(f"  Replace only the orthogonal component of corrupt with clean's")

    ortho_patch_results = _run_patch_test(
        model, hook_name, by_digit, fisher_basis,
        patch_mode="ortho", n_fisher_dims=n_fisher_dims, direct_answer=direct_answer
    )

    # ── Summary ──
    logger.info(f"\n{'='*60}")
    logger.info(f"PATCHING SUMMARY (Layer {layer}, Fisher {n_fisher_dims}D)")
    logger.info(f"{'='*60}")
    for name, res in [("FULL", full_patch_results), ("FISHER", fisher_patch_results), ("ORTHO", ortho_patch_results)]:
        n = res["total"]
        if n > 0:
            logger.info(f"  {name:8s}: transfer_to_clean={res['transfer']}/{n} ({100*res['transfer']/n:.1f}%), "
                         f"changed={res['changed']}/{n} ({100*res['changed']/n:.1f}%), "
                         f"stayed_corrupt={res['stayed']}/{n} ({100*res['stayed']/n:.1f}%)")

    return {
        "full_patch": full_patch_results,
        "fisher_patch": fisher_patch_results,
        "ortho_patch": ortho_patch_results,
    }


def _run_patch_test(model, hook_name, by_digit, fisher_basis, patch_mode, n_fisher_dims, direct_answer=False):
    """Run a single patching test across digit pairs."""
    device = next(model.parameters()).device
    results = {"total": 0, "transfer": 0, "changed": 0, "stayed": 0, "details": []}

    digits_available = sorted(by_digit.keys())
    n_pairs_per_combo = 3  # Test up to 3 pairs per digit combination

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

                clean_act = clean_prob["_clean_act"]  # (d_model,)
                corrupt_tokens = corrupt_prob["_tokens"]

                # Compute the patch delta
                corrupt_act = corrupt_prob["_clean_act"]

                if patch_mode == "full":
                    # Replace entire activation
                    delta = clean_act - corrupt_act

                elif patch_mode == "fisher":
                    # Replace only Fisher-subspace projection
                    # Project both into Fisher subspace
                    clean_proj = fisher_basis @ clean_act.float()  # (n_fisher,)
                    corrupt_proj = fisher_basis @ corrupt_act.float()  # (n_fisher,)
                    # Delta in Fisher subspace, mapped back to full space
                    delta_fisher = clean_proj - corrupt_proj  # (n_fisher,)
                    delta = (fisher_basis.T @ delta_fisher).to(clean_act.dtype)  # (d_model,)

                elif patch_mode == "ortho":
                    # Replace everything EXCEPT Fisher projection
                    clean_proj = fisher_basis @ clean_act.float()
                    corrupt_proj = fisher_basis @ corrupt_act.float()
                    # Full delta
                    full_delta = clean_act - corrupt_act
                    # Fisher delta
                    fisher_delta = (fisher_basis.T @ (clean_proj - corrupt_proj)).to(clean_act.dtype)
                    # Ortho delta = full - fisher
                    delta = full_delta - fisher_delta

                def hook_fn(act, hook, d=delta):
                    act[:, -1, :] = act[:, -1, :] + d.unsqueeze(0)
                    return act

                with torch.no_grad():
                    with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
                        logits = model(corrupt_tokens)

                pred_tok = logits[0, -1].argmax(dim=-1).item()
                pred_str = model.tokenizer.decode([pred_tok]).strip()

                try:
                    pred_val = int(pred_str)
                    # In direct-answer mode, model predicts full answer (e.g. 21);
                    # extract ones digit for comparison
                    pred_digit = pred_val % 10 if direct_answer else pred_val
                    results["total"] += 1

                    if pred_digit == clean_digit:
                        results["transfer"] += 1  # Successfully transferred clean's digit
                    if pred_digit != corrupt_digit:
                        results["changed"] += 1  # Changed away from corrupt
                    if pred_digit == corrupt_digit:
                        results["stayed"] += 1  # Stayed at corrupt's digit

                    results["details"].append({
                        "clean_digit": clean_digit,
                        "corrupt_digit": corrupt_digit,
                        "predicted": pred_digit,
                        "transferred": pred_digit == clean_digit,
                    })
                except ValueError:
                    results["total"] += 1

    return results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Fisher Subspace Activation Patching")
    parser.add_argument("--model", default="gemma-2b")
    parser.add_argument("--layers", default="20,22", help="Comma-separated layers")
    parser.add_argument("--operand-range", type=int, default=30, help="Range for Fisher computation (legacy)")
    parser.add_argument("--n-per-digit", type=int, default=100,
                        help="Gradients per digit class for teacher-forced Fisher")
    parser.add_argument("--device", default=None, help="Force device (cpu/mps/cuda)")
    parser.add_argument("--standard-only", action="store_true",
                        help="Only run standard Fisher (skip contrastive)")
    parser.add_argument("--direct-answer", action="store_true",
                        help="Use direct-answer mode (for LLaMA 3B: full answer as single token)")
    args = parser.parse_args()

    from transformer_lens import HookedTransformer

    model_name = MODEL_MAP.get(args.model, args.model)
    layers = [int(l) for l in args.layers.split(",")]
    device = args.device if args.device else get_device()

    if device == "mps":
        logger.warning("WARNING: MPS gradients produce incorrect Fisher eigenvectors!")
        logger.warning("Forcing device=cpu for reliable gradients.")
        device = "cpu"

    mode_str = "direct-answer" if args.direct_answer else "teacher-forced"
    logger.info("=" * 60)
    logger.info(f"FISHER SUBSPACE ACTIVATION PATCHING (v3: {mode_str} ones-digit)")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}, Layers: {layers}, Device: {device}")
    logger.info(f"n_per_digit: {args.n_per_digit}, direct_answer: {args.direct_answer}")

    logger.info(f"Loading {model_name} on {device}...")
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=torch.float32)

    if args.direct_answer:
        # Direct-answer mode: full answer predicted as single token (LLaMA 3B)
        fisher_problems_flat, fisher_by_digit_raw = generate_direct_answer_problems(
            n_per_digit=args.n_per_digit, operand_max=99
        )
        # Re-key by ones_digit (generate_direct_answer_problems already keys by ones_digit)
        fisher_by_digit = defaultdict(list)
        for p in fisher_problems_flat:
            fisher_by_digit[p["ones_digit"]].append(p)
        logger.info(f"Generated direct-answer Fisher problems: {len(fisher_problems_flat)} total, "
                    f"{args.n_per_digit} per digit")

        # Filter for correctness
        fisher_correct = filter_correct_direct_answer(model, fisher_problems_flat, max_n=len(fisher_problems_flat))
        fisher_by_digit = defaultdict(list)
        for p in fisher_correct:
            fisher_by_digit[p["ones_digit"]].append(p)
        logger.info(f"Correct Fisher problems: {len(fisher_correct)}")
        for d in range(10):
            logger.info(f"  Digit {d}: {len(fisher_by_digit[d])} correct")

        # Separate test set
        test_problems_flat, _ = generate_direct_answer_problems(
            n_per_digit=20, operand_max=99
        )
        test_correct = filter_correct_direct_answer(model, test_problems_flat, max_n=len(test_problems_flat))
        logger.info(f"Generated direct-answer test problems: {len(test_correct)} correct")
    else:
        # Teacher-forced problems for Fisher computation (100 per digit = 1000 total)
        fisher_problems_flat, fisher_by_digit = generate_teacher_forced_problems(
            n_per_digit=args.n_per_digit, operand_max=99
        )
        logger.info(f"Generated teacher-forced Fisher problems: {len(fisher_problems_flat)} total, "
                    f"{args.n_per_digit} per digit")
        for d in range(10):
            logger.info(f"  Digit {d}: {len(fisher_by_digit[d])} available")

        # Teacher-forced test problems (separate set, 10 per digit = 100 total)
        test_problems_flat, test_by_digit = generate_teacher_forced_problems(
            n_per_digit=15, operand_max=99
        )
        logger.info(f"Generated teacher-forced test problems: {len(test_problems_flat)} total")

    all_results = {"model": model_name, "layers": layers, "device": device,
                   "version": f"v3_{mode_str}"}

    for layer in layers:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# LAYER {layer}")
        logger.info(f"{'#'*60}")

        # Compute both standard and contrastive Fisher via teacher-forcing
        fisher_data = compute_contrastive_fisher_v3(
            model, fisher_by_digit, layer, n_per_digit=args.n_per_digit
        )

        std_eigenvalues, std_eigenvectors, std_eff_dim = fisher_data["standard"]
        con_eigenvalues, con_eigenvectors, n_contrastive = fisher_data["contrastive"]

        # Filter correct test problems
        if args.direct_answer:
            correct = test_correct  # already filtered above
        else:
            correct = filter_correct_teacher_forced(model, test_problems_flat, max_n=150)
        by_digit_counts = defaultdict(int)
        for p in correct:
            by_digit_counts[p["ones_digit"]] += 1
        logger.info(f"Found {len(correct)} correct {mode_str} test problems")
        logger.info(f"  By digit: {dict(sorted(by_digit_counts.items()))}")

        layer_results = {
            "std_eff_dim": std_eff_dim,
            "n_contrastive_dirs": n_contrastive,
            "digits_found": fisher_data["digits_found"],
            "class_counts": fisher_data["class_counts"],
        }

        # --- Standard Fisher patching ---
        logger.info(f"\n{'='*60}")
        logger.info(f"STANDARD FISHER PATCHING (Layer {layer})")
        logger.info(f"{'='*60}")
        for n_dims in [2, 5, 10, 20, 50]:
            if n_dims > std_eigenvectors.shape[1]:
                continue
            result = run_patching_experiment(
                model, layer, std_eigenvectors, correct,
                n_fisher_dims=n_dims, teacher_forced=True,
                direct_answer=args.direct_answer
            )
            layer_results[f"std_fisher_{n_dims}D"] = {
                k: {kk: vv for kk, vv in v.items() if kk != "details"}
                for k, v in result.items()
            }

        # --- Contrastive Fisher patching ---
        if not args.standard_only:
            logger.info(f"\n{'='*60}")
            logger.info(f"CONTRASTIVE FISHER PATCHING (Layer {layer})")
            logger.info(f"{'='*60}")
            max_con = min(n_contrastive, 9)
            for n_dims in [2, 5, 9]:
                if n_dims > max_con:
                    logger.info(f"  Skipping {n_dims}D: only {max_con} contrastive directions")
                    continue
                result = run_patching_experiment(
                    model, layer, con_eigenvectors, correct,
                    n_fisher_dims=n_dims, teacher_forced=True,
                    direct_answer=args.direct_answer
                )
                layer_results[f"contrastive_{n_dims}D"] = {
                    k: {kk: vv for kk, vv in v.items() if kk != "details"}
                    for k, v in result.items()
                }

        all_results[f"layer_{layer}"] = layer_results

    # Save
    out_dir = Path("mathematical_toolkit_results")
    out_dir.mkdir(exist_ok=True)
    suffix = "_direct" if args.direct_answer else ""
    out_path = out_dir / f"fisher_patching_{args.model}_v3{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
