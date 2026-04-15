#!/usr/bin/env python3
"""
Circuit Analysis — Advanced experiments to narrow down the arithmetic circuit.

Builds on the discovery pipeline (arithmetic_circuit_discovery.py) with:

  Tier 1 (High Signal, Fast):
    1. Direct MLP Unembedding   — project core MLP outputs through W_U
    2. Operand B Attention Hunt — find how operand B info reaches the = token
    3. Carry Linear Probe       — does the model learn base-10 carry?

  Tier 2 (Medium Effort):
    4. Activation PCA by Sub-Task — cluster MLP activations by output digit
    5. Ensemble Edge Patching     — prove routing→compute connection

Usage:
    python experiments/circuit_analysis.py --model-key phi-3-mini --experiment 1
    python experiments/circuit_analysis.py --experiment all
    python experiments/circuit_analysis.py --experiment 1 2 3  # Tier 1 only
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from transformer_lens import HookedTransformer
from pathlib import Path
import json
import random
import time
import argparse
import warnings
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.arithmetic_circuit_discovery import (
    generate_arithmetic_prompts,
    get_answer_token_id,
    check_answer_greedy,
    resolve_model_name,
    auto_device_dtype,
    get_device,
    DEFAULT_MODEL,
    OUTPUT_DIR,
)

warnings.filterwarnings("ignore")


# ═════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════

def find_token_positions(model, prompt: str) -> Dict[str, int]:
    """Find semantic token positions in an arithmetic prompt.

    Returns dict with keys: operand_a, plus, operand_b, equals (last token).
    """
    str_tokens = model.to_str_tokens(prompt)
    positions = {"equals": len(str_tokens) - 1}

    for idx, tok in enumerate(str_tokens):
        if tok.strip() == '+':
            positions["plus"] = idx
            positions["operand_a"] = idx - 1
            positions["operand_b"] = idx + 1
            break

    return positions


def generate_carry_dataset(
    n_problems: int = 200,
    max_operand: int = 50,
    seed: int = 77,
) -> List[Dict[str, Any]]:
    """Generate arithmetic problems labeled by whether they require a carry.

    A carry occurs when any column of the addition overflows (digit sum >= 10).
    """
    rng = random.Random(seed)
    problems = []

    few_shot_prefix = "Calculate:\n12 + 7 = 19\n34 + 15 = 49\n"

    for _ in range(n_problems):
        a = rng.randint(1, max_operand)
        b = rng.randint(1, max_operand)
        answer = a + b

        # Determine if carry is needed (ones digit overflow)
        ones_carry = (a % 10) + (b % 10) >= 10
        # For two-digit numbers, also check tens carry
        tens_carry = (a // 10) + (b // 10) + (1 if ones_carry else 0) >= 10

        has_carry = ones_carry or tens_carry

        prompt = f"{few_shot_prefix}{a} + {b} ="

        problems.append({
            "a": a, "b": b, "answer": answer,
            "has_carry": has_carry,
            "ones_carry": ones_carry,
            "tens_carry": tens_carry,
            "clean_prompt": prompt,
        })

    n_carry = sum(1 for p in problems if p["has_carry"])
    print(f"  Generated {n_problems} problems: {n_carry} carry, "
          f"{n_problems - n_carry} no-carry")

    return problems


# ═════════════════════════════════════════════════════════════
# EXPERIMENT 1: DIRECT MLP UNEMBEDDING
# ═════════════════════════════════════════════════════════════

def run_mlp_unembedding(
    model: HookedTransformer,
    problems: List[Dict],
    device: torch.device,
    target_layers: Optional[List[int]] = None,
    top_k: int = 15,
) -> Dict[str, Any]:
    """Project MLP outputs directly through W_U to see what each MLP writes.

    Two analysis modes:
      (a) Direct Logit Attribution: raw mlp_out @ W_U (NO LayerNorm).
          This is the standard approach for decomposing individual component
          contributions. LayerNorm should NOT be applied to individual
          components since it's calibrated for the full residual stream.
      (b) Contribution: logit_lens(resid_post) - logit_lens(resid_post - mlp_out).
          Shows what the MLP changes about the final prediction.
    """
    n_layers = model.cfg.n_layers

    if target_layers is None:
        target_layers = list(range(n_layers // 2, n_layers))

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT 1: DIRECT MLP UNEMBEDDING")
    print(f"{'=' * 70}")
    print(f"  Target layers: {target_layers}")
    print(f"  Problems: {len(problems)}")

    W_U = model.W_U.detach().float().cpu()  # [d_model, vocab]
    has_ln_final = hasattr(model, 'ln_final') and model.ln_final is not None

    # Collect both mlp_out AND resid_post for contribution analysis
    hook_names = []
    for L in target_layers:
        hook_names.append(f"blocks.{L}.hook_mlp_out")
        hook_names.append(f"blocks.{L}.hook_resid_post")

    # Per-layer aggregation — Mode (a): Direct logit attribution
    direct_top_tokens = defaultdict(lambda: defaultdict(list))
    direct_answer_rank = defaultdict(list)
    direct_answer_logit = defaultdict(list)

    # Per-layer aggregation — Mode (b): Contribution (with minus without)
    contrib_top_tokens = defaultdict(lambda: defaultdict(list))
    contrib_answer_boost = defaultdict(list)  # how much does MLP boost the answer?

    n_done = 0
    for prob in problems:
        tokens = model.to_tokens(prob["clean_prompt"])
        answer_tok = get_answer_token_id(model, prob["answer"])
        last_pos = tokens.shape[1] - 1

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)

        for L in target_layers:
            mlp_out = cache[f"blocks.{L}.hook_mlp_out"][0, last_pos, :].float().cpu()
            resid = cache[f"blocks.{L}.hook_resid_post"][0, last_pos, :].float().cpu()

            # ── Mode (a): Direct logit attribution (NO LayerNorm) ──
            direct_logits = mlp_out @ W_U  # [vocab]

            answer_logit_direct = direct_logits[answer_tok].item()
            rank = (direct_logits > answer_logit_direct).sum().item()
            direct_answer_rank[L].append(rank)
            direct_answer_logit[L].append(answer_logit_direct)

            top_vals, top_ids = direct_logits.topk(top_k)
            for val, tid in zip(top_vals.tolist(), top_ids.tolist()):
                tok_str = model.tokenizer.decode([tid]).strip()
                direct_top_tokens[L][tok_str].append(val)

            # ── Mode (b): Contribution (resid WITH - resid WITHOUT this MLP) ──
            resid_without = resid - mlp_out

            if has_ln_final:
                logits_with = model.ln_final(
                    resid.unsqueeze(0).to(device)
                ).squeeze(0).float().cpu() @ W_U
                logits_without = model.ln_final(
                    resid_without.unsqueeze(0).to(device)
                ).squeeze(0).float().cpu() @ W_U
            else:
                logits_with = resid @ W_U
                logits_without = resid_without @ W_U

            contrib_logits = logits_with - logits_without  # [vocab]
            answer_boost = contrib_logits[answer_tok].item()
            contrib_answer_boost[L].append(answer_boost)

            ctop_vals, ctop_ids = contrib_logits.topk(top_k)
            for val, tid in zip(ctop_vals.tolist(), ctop_ids.tolist()):
                tok_str = model.tokenizer.decode([tid]).strip()
                contrib_top_tokens[L][tok_str].append(val)

        del cache
        n_done += 1
        if n_done % 20 == 0:
            print(f"  Processed {n_done}/{len(problems)} problems...")

    # ── Report Mode (a): Direct logit attribution ──
    print(f"\n  --- Mode (a): Direct Logit Attribution (mlp_out @ W_U, no LN) ---")
    print(f"  {'Layer':>6}  {'AnswerRank':>11}  {'AnswerLogit':>12}  "
          f"{'Top tokens (direct)'}")
    print(f"  {'─' * 85}")

    results = {}
    for L in target_layers:
        mean_rank = np.mean(direct_answer_rank[L])
        mean_logit = np.mean(direct_answer_logit[L])

        token_scores = {}
        for tok_str, vals in direct_top_tokens[L].items():
            token_scores[tok_str] = np.mean(vals)
        sorted_tokens = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)

        top_5_str = "  ".join(f"'{t}':{v:.1f}" for t, v in sorted_tokens[:5])

        marker = ""
        if mean_rank < 100:
            marker = " ◀ WRITES ANSWER"
        if mean_rank < 10:
            marker = " ★ STRONG NUMBER WRITER"

        print(f"  L{L:>4d}  {mean_rank:11.1f}  {mean_logit:12.2f}  {top_5_str}{marker}")

        # Count numeric tokens in top 10
        numeric_in_top10 = sum(
            1 for t, _ in sorted_tokens[:10] if t.isdigit()
        )

        results[f"L{L}"] = {
            "layer": L,
            "direct_answer_rank": float(mean_rank),
            "direct_answer_logit": float(mean_logit),
            "direct_top_tokens": {t: float(v) for t, v in sorted_tokens[:20]},
            "numeric_in_top10": numeric_in_top10,
        }

    # ── Report Mode (b): Contribution ──
    print(f"\n  --- Mode (b): MLP Contribution (WITH - WITHOUT this MLP) ---")
    print(f"  {'Layer':>6}  {'AnswerBoost':>12}  {'Top tokens PROMOTED by MLP'}")
    print(f"  {'─' * 85}")

    for L in target_layers:
        mean_boost = np.mean(contrib_answer_boost[L])

        token_scores = {}
        for tok_str, vals in contrib_top_tokens[L].items():
            token_scores[tok_str] = np.mean(vals)
        sorted_tokens = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)

        top_5_str = "  ".join(f"'{t}':{v:.1f}" for t, v in sorted_tokens[:5])

        marker = ""
        if mean_boost > 1.0:
            marker = " ◀ PROMOTES ANSWER"
        if mean_boost > 3.0:
            marker = " ★ STRONG ANSWER PROMOTER"

        print(f"  L{L:>4d}  {mean_boost:12.2f}  {top_5_str}{marker}")

        results[f"L{L}"]["contrib_answer_boost"] = float(mean_boost)
        results[f"L{L}"]["contrib_top_tokens"] = {t: float(v) for t, v in sorted_tokens[:20]}

        # Check if number tokens are promoted
        promoted_numbers = [
            (t, v) for t, v in sorted_tokens[:20] if t.isdigit()
        ]
        if promoted_numbers:
            results[f"L{L}"]["promoted_numbers"] = {
                t: float(v) for t, v in promoted_numbers
            }

    return {"mlp_unembedding": results}


# ═════════════════════════════════════════════════════════════
# EXPERIMENT 2: OPERAND B ATTENTION HUNT
# ═════════════════════════════════════════════════════════════

def run_operand_b_hunt(
    model: HookedTransformer,
    problems: List[Dict],
    device: torch.device,
    early_layers: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Hunt for how operand B information reaches the = token.

    Hypothesis: In early layers (L0-L8), attention heads copy operand A's
    representation onto the + or B token position. By the crossover zone,
    a merged (A, B) representation exists at one position.

    We check:
    - Attention FROM plus/B positions TO operand_a in early layers
    - Attention FROM equals TO operand_b in ALL layers
    - Whether the plus token becomes an "information hub"
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    if early_layers is None:
        early_layers = list(range(min(n_layers, 12)))

    all_layers = list(range(n_layers))

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT 2: OPERAND B ATTENTION HUNT")
    print(f"{'=' * 70}")
    print(f"  Testing hypothesis: A is copied to +/B position in early layers")

    hook_names = [f"blocks.{L}.attn.hook_pattern" for L in all_layers]

    # Track multiple attention patterns
    # (a) FROM plus/B positions TO operand_a (early copy hypothesis)
    plus_to_a = defaultdict(lambda: defaultdict(list))   # L -> H -> [attn weights]
    b_to_a = defaultdict(lambda: defaultdict(list))
    # (b) FROM equals TO operand_b (direct B routing)
    eq_to_b = defaultdict(lambda: defaultdict(list))
    # (c) FROM equals TO plus (plus as info hub)
    eq_to_plus = defaultdict(lambda: defaultdict(list))
    # (d) FROM equals TO operand_a (for comparison)
    eq_to_a = defaultdict(lambda: defaultdict(list))

    n_done = 0
    for prob in problems[:40]:
        tokens = model.to_tokens(prob["clean_prompt"])
        pos = find_token_positions(model, prob["clean_prompt"])

        if "plus" not in pos or "operand_a" not in pos:
            continue

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)

        a_pos = pos["operand_a"]
        b_pos = pos.get("operand_b", pos["plus"] + 1)
        p_pos = pos["plus"]
        e_pos = pos["equals"]

        for L in all_layers:
            hook_name = f"blocks.{L}.attn.hook_pattern"
            if hook_name not in cache:
                continue
            pattern = cache[hook_name][0].float().cpu()  # [n_heads, seq, seq]

            for H in range(n_heads):
                # (a) Early copy: plus attends to A
                if L in early_layers:
                    plus_to_a[L][H].append(pattern[H, p_pos, a_pos].item())
                    b_to_a[L][H].append(pattern[H, b_pos, a_pos].item())

                # (b-d) Equals position attention routing
                eq_to_b[L][H].append(pattern[H, e_pos, b_pos].item())
                eq_to_plus[L][H].append(pattern[H, e_pos, p_pos].item())
                eq_to_a[L][H].append(pattern[H, e_pos, a_pos].item())

        del cache
        n_done += 1

    # ── Report 1: Early copy (plus/B → A) ──
    print(f"\n  --- Early Copy: Does '+' or 'B' attend to operand A? ---")
    print(f"  {'Layer':>6}  {'Head':>5}  {'+ → A':>8}  {'B → A':>8}  {'Note'}")
    print(f"  {'─' * 50}")

    early_copy_results = {}
    early_copy_found = []
    for L in early_layers:
        for H in range(n_heads):
            if H not in plus_to_a[L]:
                continue
            mean_plus_a = np.mean(plus_to_a[L][H])
            mean_b_a = np.mean(b_to_a[L][H])
            if mean_plus_a > 0.05 or mean_b_a > 0.05:
                note = ""
                if mean_plus_a > 0.10:
                    note = "★ COPIES A→+"
                    early_copy_found.append((L, H, mean_plus_a, "plus"))
                if mean_b_a > 0.10:
                    note += " ★ COPIES A→B"
                    early_copy_found.append((L, H, mean_b_a, "b"))
                print(f"  L{L:>4d}  H{H:>4d}  {mean_plus_a:8.4f}  "
                      f"{mean_b_a:8.4f}  {note}")
                early_copy_results[f"L{L}_H{H}"] = {
                    "layer": L, "head": H,
                    "plus_to_a": float(mean_plus_a),
                    "b_to_a": float(mean_b_a),
                }

    if not early_copy_found:
        print("  No strong early-copy heads found (threshold: 0.10)")

    # ── Report 2: Where does = get B information? ──
    print(f"\n  --- Equals-position routing: Where does '=' get info? ---")
    print(f"  {'Layer':>6}  {'Head':>5}  {'= → A':>8}  {'= → B':>8}  "
          f"{'= → +':>8}  {'Primary route'}")
    print(f"  {'─' * 65}")

    eq_routing_results = {}
    for L in all_layers:
        for H in range(n_heads):
            if H not in eq_to_a[L]:
                continue
            mean_a = np.mean(eq_to_a[L][H])
            mean_b = np.mean(eq_to_b[L][H])
            mean_p = np.mean(eq_to_plus[L][H])

            max_attn = max(mean_a, mean_b, mean_p)
            if max_attn < 0.08:
                continue

            route = ""
            if mean_a > 0.10:
                route = "A"
            if mean_b > 0.10:
                route += "+B" if route else "B"
            if mean_p > 0.10:
                route += "+plus" if route else "plus"
            if not route:
                route = "—"

            print(f"  L{L:>4d}  H{H:>4d}  {mean_a:8.4f}  {mean_b:8.4f}  "
                  f"{mean_p:8.4f}  {route}")

            eq_routing_results[f"L{L}_H{H}"] = {
                "layer": L, "head": H,
                "eq_to_a": float(mean_a),
                "eq_to_b": float(mean_b),
                "eq_to_plus": float(mean_p),
            }

    return {
        "operand_b_hunt": {
            "early_copy_heads": early_copy_results,
            "eq_routing": eq_routing_results,
            "hypothesis_confirmed": len(early_copy_found) > 0,
        }
    }


# ═════════════════════════════════════════════════════════════
# EXPERIMENT 3: CARRY LINEAR PROBE
# ═════════════════════════════════════════════════════════════

def run_carry_probe(
    model: HookedTransformer,
    device: torch.device,
    n_problems: int = 300,
    max_operand: int = 50,
    probe_layers: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Train linear probes to detect when 'carry' information is present.

    At each layer, extract the residual stream at the = position.
    Train a logistic regression to predict whether the problem requires carry.
    If the probe achieves high accuracy at layer L, the carry information
    is already encoded in the residual stream by that layer.
    """
    n_layers = model.cfg.n_layers

    if probe_layers is None:
        probe_layers = list(range(n_layers))

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT 3: LINEAR PROBE FOR BASE-10 CARRY")
    print(f"{'=' * 70}")

    # Generate carry-labeled dataset
    problems = generate_carry_dataset(
        n_problems=n_problems, max_operand=max_operand
    )

    hook_names = [f"blocks.{L}.hook_resid_post" for L in probe_layers]

    # Collect activations
    print(f"  Collecting residual stream activations at {len(probe_layers)} layers...")
    activations = {L: [] for L in probe_layers}
    labels = []

    n_done = 0
    for prob in problems:
        tokens = model.to_tokens(prob["clean_prompt"])
        last_pos = tokens.shape[1] - 1

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)

        for L in probe_layers:
            act = cache[f"blocks.{L}.hook_resid_post"][0, last_pos].float().cpu().numpy()
            activations[L].append(act)

        labels.append(1 if prob["has_carry"] else 0)
        del cache
        n_done += 1
        if n_done % 100 == 0:
            print(f"  Collected {n_done}/{len(problems)} activations...")

    labels = np.array(labels)
    n_carry = labels.sum()
    n_no_carry = len(labels) - n_carry
    print(f"  Dataset: {n_carry} carry, {n_no_carry} no-carry")

    if n_carry < 10 or n_no_carry < 10:
        print("  WARNING: Imbalanced dataset — probe results may be unreliable")

    # Train probes per layer
    print(f"\n  {'Layer':>6}  {'CV Accuracy':>12}  {'Std':>8}  {'Verdict'}")
    print(f"  {'─' * 50}")

    results = {}
    for L in probe_layers:
        X = np.array(activations[L])
        # Logistic regression with cross-validation
        clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
        scores = cross_val_score(clf, X, labels, cv=5, scoring='accuracy')

        mean_acc = scores.mean()
        std_acc = scores.std()

        verdict = ""
        if mean_acc > 0.90:
            verdict = "★ CARRY RESOLVED"
        elif mean_acc > 0.75:
            verdict = "◀ CARRY EMERGING"
        elif mean_acc > 0.60:
            verdict = "~ weak signal"

        print(f"  L{L:>4d}  {mean_acc:12.3f}  {std_acc:8.3f}  {verdict}")

        results[f"L{L}"] = {
            "layer": L,
            "cv_accuracy": float(mean_acc),
            "cv_std": float(std_acc),
            "n_samples": len(labels),
        }

    # Also probe for ones_carry specifically
    ones_labels = np.array([1 if p["ones_carry"] else 0 for p in problems])
    if ones_labels.sum() > 10 and (len(ones_labels) - ones_labels.sum()) > 10:
        print(f"\n  --- Ones-digit carry probe ---")
        print(f"  {'Layer':>6}  {'CV Accuracy':>12}  {'Verdict'}")
        print(f"  {'─' * 40}")

        for L in probe_layers:
            X = np.array(activations[L])
            clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
            scores = cross_val_score(clf, X, ones_labels, cv=5, scoring='accuracy')
            mean_acc = scores.mean()

            verdict = "★ ONES CARRY RESOLVED" if mean_acc > 0.90 else ""
            print(f"  L{L:>4d}  {mean_acc:12.3f}  {verdict}")

            results[f"L{L}"]["ones_carry_accuracy"] = float(mean_acc)

    return {"carry_probe": results}


# ═════════════════════════════════════════════════════════════
# EXPERIMENT 4: ACTIVATION PCA BY SUB-TASK
# ═════════════════════════════════════════════════════════════

def run_activation_pca(
    model: HookedTransformer,
    device: torch.device,
    target_layers: Optional[List[int]] = None,
    n_problems: int = 200,
    max_operand: int = 50,
) -> Dict[str, Any]:
    """PCA on MLP activations, colored by output properties.

    Collects MLP output activations at target layers across many problems.
    Runs PCA and checks whether the top components separate:
    - answers by ones digit (0-9)
    - carry vs no-carry
    - answer magnitude

    If top PCs cleanly separate digit classes, the MLP is doing digit-level lookup.
    """
    n_layers = model.cfg.n_layers

    if target_layers is None:
        # Focus on compute-phase MLPs
        target_layers = list(range(n_layers * 2 // 3, n_layers))

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT 4: ACTIVATION PCA BY SUB-TASK")
    print(f"{'=' * 70}")
    print(f"  Target layers: {target_layers}")

    problems = generate_carry_dataset(
        n_problems=n_problems, max_operand=max_operand, seed=88
    )

    hook_names = [f"blocks.{L}.hook_mlp_out" for L in target_layers]

    # Collect
    activations = {L: [] for L in target_layers}
    answers = []
    ones_digits = []
    carries = []

    print(f"  Collecting MLP activations...")
    for prob in problems:
        tokens = model.to_tokens(prob["clean_prompt"])
        last_pos = tokens.shape[1] - 1

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)

        for L in target_layers:
            act = cache[f"blocks.{L}.hook_mlp_out"][0, last_pos].float().cpu().numpy()
            activations[L].append(act)

        answers.append(prob["answer"])
        ones_digits.append(prob["answer"] % 10)
        carries.append(1 if prob["has_carry"] else 0)
        del cache

    answers = np.array(answers)
    ones_digits = np.array(ones_digits)
    carries = np.array(carries)

    print(f"  Collected {len(answers)} activations")

    # PCA per layer
    print(f"\n  {'Layer':>6}  {'Var PC1':>8}  {'Var PC2':>8}  {'Var PC3':>8}  "
          f"{'Digit Sep':>10}  {'Carry Sep':>10}")
    print(f"  {'─' * 65}")

    results = {}
    for L in target_layers:
        X = np.array(activations[L])
        pca = PCA(n_components=min(10, X.shape[1], X.shape[0]))
        X_pca = pca.fit_transform(X)

        var_explained = pca.explained_variance_ratio_

        # Measure separability: within-class vs between-class distance ratio
        # For ones digits
        digit_sep = _cluster_separability(X_pca[:, :3], ones_digits)
        carry_sep = _cluster_separability(X_pca[:, :3], carries)

        marker = ""
        if digit_sep > 1.5:
            marker = " ★ DIGIT CLUSTERS"
        if carry_sep > 1.5:
            marker += " ★ CARRY CLUSTERS"

        print(f"  L{L:>4d}  {var_explained[0]:8.4f}  {var_explained[1]:8.4f}  "
              f"{var_explained[2]:8.4f}  {digit_sep:10.3f}  {carry_sep:10.3f}{marker}")

        results[f"L{L}"] = {
            "layer": L,
            "variance_explained": [float(v) for v in var_explained[:5]],
            "digit_separability": float(digit_sep),
            "carry_separability": float(carry_sep),
        }

    return {"activation_pca": results}


def _cluster_separability(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute between-class / within-class distance ratio.

    Higher values indicate better separation. >1.5 means clearly separable.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    # Between-class: mean pairwise distance between class centroids
    centroids = []
    within_dists = []
    for lbl in unique_labels:
        mask = labels == lbl
        if mask.sum() < 2:
            continue
        class_points = X[mask]
        centroid = class_points.mean(axis=0)
        centroids.append(centroid)
        # Within-class: mean distance to centroid
        dists = np.linalg.norm(class_points - centroid, axis=1)
        within_dists.append(dists.mean())

    if len(centroids) < 2 or not within_dists:
        return 0.0

    centroids = np.array(centroids)
    # Between-class distance
    from itertools import combinations
    between_dists = []
    for i, j in combinations(range(len(centroids)), 2):
        between_dists.append(np.linalg.norm(centroids[i] - centroids[j]))

    mean_between = np.mean(between_dists)
    mean_within = np.mean(within_dists)

    if mean_within < 1e-10:
        return float('inf')

    return mean_between / mean_within


# ═════════════════════════════════════════════════════════════
# EXPERIMENT 5: ENSEMBLE EDGE PATCHING
# ═════════════════════════════════════════════════════════════

def run_ensemble_edge_patching(
    model: HookedTransformer,
    problems: List[Dict],
    device: torch.device,
    router_heads: Optional[List[Tuple[int, int]]] = None,
    compute_mlps: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Bundle top routing heads and patch their combined output into compute MLPs.

    Instead of patching one head at a time (which fails due to redundancy),
    we treat the top-N routing heads as a single logical unit.

    The experiment:
    1. Run clean prompt, cache the summed attention output of all router heads
    2. Run corrupt prompt, but replace the combined router output with clean values
    3. Measure if the model now computes the correct answer

    If recovery is high, we've formally proven the routing→compute connection.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    if router_heads is None:
        # Default: try to load from Stage 2 results, or use heuristic
        # Use heads from the routing zone (first 2/3 of network)
        routing_cutoff = n_layers * 2 // 3
        # Pick all heads in the routing zone — will be refined below
        router_heads = [(L, H) for L in range(routing_cutoff)
                        for H in range(n_heads)]
        print(f"  NOTE: No specific router heads provided. Using all heads in L0-L{routing_cutoff-1}.")
        print(f"  For best results, pass --router-heads from Stage 2/3 results.")

    if compute_mlps is None:
        # Default: last quarter of network
        compute_mlps = list(range(n_layers * 3 // 4, n_layers))

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT 5: ENSEMBLE EDGE PATCHING")
    print(f"{'=' * 70}")
    print(f"  Router heads: {len(router_heads)} heads")
    print(f"  Compute MLPs: {compute_mlps}")

    # Group router heads by layer for efficient hooking
    router_by_layer = defaultdict(list)
    for L, H in router_heads:
        router_by_layer[L].append(H)

    router_layers = sorted(router_by_layer.keys())
    hook_names = [f"blocks.{L}.attn.hook_z" for L in router_layers]

    # Experiment 1: Patch all router outputs (clean→corrupt) at ALL positions
    print(f"\n  --- Patch all router head outputs at final position ---")
    recovery_all = []
    recovery_per_mlp = {L: [] for L in compute_mlps}

    n_done = 0
    for prob in problems:
        clean_tokens = model.to_tokens(prob["clean_prompt"])
        corrupt_tokens = model.to_tokens(prob["corrupt_prompt"])
        answer_tok = get_answer_token_id(model, prob["answer"])

        if clean_tokens.shape[1] != corrupt_tokens.shape[1]:
            continue

        last_pos = clean_tokens.shape[1] - 1

        with torch.no_grad():
            clean_logits, clean_cache = model.run_with_cache(
                clean_tokens, names_filter=hook_names
            )
            corrupt_logits = model(corrupt_tokens)

        clean_logit = clean_logits[0, last_pos, answer_tok].item()
        corrupt_logit = corrupt_logits[0, last_pos, answer_tok].item()
        logit_diff = clean_logit - corrupt_logit

        if abs(logit_diff) < 0.1:
            del clean_cache
            continue

        # Patch all router heads simultaneously
        def make_ensemble_hook(clean_cache_ref, layer, head_list):
            def hook_fn(value, hook):
                clean_act = clean_cache_ref[f"blocks.{layer}.attn.hook_z"]
                for h in head_list:
                    value[0, :, h, :] = clean_act[0, :, h, :]
                return value
            return hook_fn

        fwd_hooks = []
        for L in router_layers:
            heads = router_by_layer[L]
            hook_name = f"blocks.{L}.attn.hook_z"
            fwd_hooks.append((hook_name, make_ensemble_hook(clean_cache, L, heads)))

        with torch.no_grad():
            with model.hooks(fwd_hooks=fwd_hooks):
                patched_logits = model(corrupt_tokens)

        patched_logit = patched_logits[0, last_pos, answer_tok].item()
        recovery = (patched_logit - corrupt_logit) / logit_diff
        recovery_all.append(recovery)

        del clean_cache
        n_done += 1
        if n_done % 10 == 0:
            print(f"  Processed {n_done} problems, "
                  f"mean recovery: {np.mean(recovery_all):.3f}")

    mean_recovery = np.mean(recovery_all) if recovery_all else 0.0
    std_recovery = np.std(recovery_all) if recovery_all else 0.0

    print(f"\n  Ensemble router patching results:")
    print(f"    {len(router_heads)} heads patched simultaneously")
    print(f"    Mean recovery: {mean_recovery:.4f} ± {std_recovery:.4f}")
    if mean_recovery > 0.8:
        print(f"    ★★ STRONG: Router ensemble recovers most of the computation!")
    elif mean_recovery > 0.5:
        print(f"    ◀ MODERATE: Partial recovery — additional routing paths exist")
    elif mean_recovery > 0.2:
        print(f"    ~ WEAK: Some signal but routing is more distributed")
    else:
        print(f"    ✗ MINIMAL: These heads may not be the primary routers")

    # Experiment 2: Ablation test — what happens if we ABLATE routers?
    # (Set router outputs to zero in clean run)
    print(f"\n  --- Ablate all router heads (set to zero in clean run) ---")
    ablated_correct = 0
    baseline_correct = 0
    n_ablated = 0

    for prob in problems[:30]:
        tokens = model.to_tokens(prob["clean_prompt"])

        # Baseline
        if check_answer_greedy(model, tokens, prob["answer"]):
            baseline_correct += 1

        # Ablated
        def make_ablation_hook(layer, head_list):
            def hook_fn(value, hook):
                for h in head_list:
                    value[0, :, h, :] = 0.0
                return value
            return hook_fn

        ablation_hooks = []
        for L in router_layers:
            heads = router_by_layer[L]
            hook_name = f"blocks.{L}.attn.hook_z"
            ablation_hooks.append((hook_name, make_ablation_hook(L, heads)))

        with torch.no_grad():
            with model.hooks(fwd_hooks=ablation_hooks):
                if check_answer_greedy(model, tokens, prob["answer"]):
                    ablated_correct += 1

        n_ablated += 1

    baseline_acc = baseline_correct / max(n_ablated, 1)
    ablated_acc = ablated_correct / max(n_ablated, 1)
    drop = baseline_acc - ablated_acc

    print(f"    Baseline accuracy: {baseline_acc:.1%}")
    print(f"    Ablated accuracy:  {ablated_acc:.1%}")
    print(f"    Accuracy drop:     {drop:+.1%}")

    if drop > 0.3:
        print(f"    ★★ ROUTER ENSEMBLE IS NECESSARY")
    elif drop > 0.1:
        print(f"    ◀ ROUTERS CONTRIBUTE SIGNIFICANTLY")

    return {
        "ensemble_patching": {
            "n_router_heads": len(router_heads),
            "compute_mlps": compute_mlps,
            "mean_recovery": float(mean_recovery),
            "std_recovery": float(std_recovery),
            "n_problems": len(recovery_all),
            "baseline_accuracy": float(baseline_acc),
            "ablated_accuracy": float(ablated_acc),
            "accuracy_drop": float(drop),
        }
    }


# ═════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═════════════════════════════════════════════════════════════

def run_circuit_analysis(
    model_name: str = DEFAULT_MODEL,
    experiments: Optional[List[int]] = None,
    n_problems: int = 40,
    max_operand: int = 50,
    output_dir: str = OUTPUT_DIR,
    device_override: Optional[str] = None,
    dtype_override: Optional[str] = None,
    discovery_results_path: Optional[str] = None,
):
    """Run the circuit analysis experiments."""
    resolved_name = resolve_model_name(model_name)
    device, dtype = auto_device_dtype(resolved_name, device_override, dtype_override)

    if experiments is None:
        experiments = [1, 2, 3, 4, 5]

    print(f"\n{'═' * 70}")
    print(f"CIRCUIT ANALYSIS — Advanced Experiments")
    print(f"{'═' * 70}")
    print(f"  Model: {model_name}")
    if resolved_name != model_name:
        print(f"  Resolved: {resolved_name}")
    print(f"  Device: {device}  dtype: {dtype}")
    print(f"  Experiments: {experiments}")
    print(f"  Operand range: 1–{max_operand}")

    # Load model
    print(f"\nLoading model...")
    t0 = time.time()
    model = HookedTransformer.from_pretrained(resolved_name, device=device, dtype=dtype)
    model.eval()
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    print(f"  Loaded in {time.time() - t0:.1f}s: {n_layers}L, {n_heads}H, "
          f"d_model={model.cfg.d_model}")

    # Load discovery results if available (for router heads, top MLPs)
    discovery = {}
    if discovery_results_path:
        dp = Path(discovery_results_path)
    else:
        dp = Path(output_dir) / "discovery_results.json"
    if dp.exists():
        with open(dp) as f:
            discovery = json.load(f)
        print(f"  Loaded discovery results from {dp}")
    else:
        print(f"  No discovery results found — using defaults")

    # Extract top components from discovery
    top_mlp_layers = [L for L, _ in discovery.get("top_mlps", [])][:5]
    top_head_tuples = [(L, H) for L, H, _ in discovery.get("top_heads", [])][:20]

    # Generate problems
    problems = generate_arithmetic_prompts(
        n_problems=n_problems, max_operand=max_operand, seed=777
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_results = {
        "model": model_name,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "max_operand": max_operand,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }

    # ── Experiment 1: MLP Unembedding ──
    if 1 in experiments:
        t1 = time.time()
        target = top_mlp_layers if top_mlp_layers else None
        r1 = run_mlp_unembedding(model, problems, device, target_layers=target)
        all_results.update(r1)
        print(f"  Experiment 1 done in {time.time() - t1:.1f}s")

    # ── Experiment 2: Operand B Hunt ──
    if 2 in experiments:
        t2 = time.time()
        r2 = run_operand_b_hunt(model, problems, device)
        all_results.update(r2)
        print(f"  Experiment 2 done in {time.time() - t2:.1f}s")

    # ── Experiment 3: Carry Probe ──
    if 3 in experiments:
        t3 = time.time()
        r3 = run_carry_probe(model, device, n_problems=300, max_operand=max_operand)
        all_results.update(r3)
        print(f"  Experiment 3 done in {time.time() - t3:.1f}s")

    # ── Experiment 4: Activation PCA ──
    if 4 in experiments:
        t4 = time.time()
        target = top_mlp_layers if top_mlp_layers else None
        r4 = run_activation_pca(model, device, target_layers=target,
                                 n_problems=200, max_operand=max_operand)
        all_results.update(r4)
        print(f"  Experiment 4 done in {time.time() - t4:.1f}s")

    # ── Experiment 5: Ensemble Edge Patching ──
    if 5 in experiments:
        t5 = time.time()
        router_heads = top_head_tuples if top_head_tuples else None
        compute_mlps = top_mlp_layers[:3] if top_mlp_layers else None
        r5 = run_ensemble_edge_patching(
            model, problems, device,
            router_heads=router_heads,
            compute_mlps=compute_mlps,
        )
        all_results.update(r5)
        print(f"  Experiment 5 done in {time.time() - t5:.1f}s")

    # Save results
    out_path = Path(output_dir) / "circuit_analysis.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")

    print(f"\n{'═' * 70}")
    print(f"CIRCUIT ANALYSIS COMPLETE")
    print(f"{'═' * 70}")

    return all_results


# ═════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Circuit Analysis — Advanced experiments for arithmetic circuit"
    )
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", type=str, default=None,
                             help="Raw TransformerLens / HuggingFace model name")
    model_group.add_argument("--model-key", type=str, default=None,
                             help="Model registry key (e.g. 'pythia-1.4b', 'phi-3-mini')")
    parser.add_argument("--experiment", type=int, nargs='+', default=None,
                        help="Which experiments to run (1-5, default: all)")
    parser.add_argument("--n-problems", type=int, default=40,
                        help="Number of problems")
    parser.add_argument("--operand-range", type=int, default=50,
                        help="Maximum operand value")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Output directory (also looks for discovery_results.json here)")
    parser.add_argument("--discovery-results", type=str, default=None,
                        help="Path to discovery_results.json from Stage 1-3")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cpu, cuda, mps)")
    parser.add_argument("--dtype", type=str, default=None,
                        choices=["float32", "float16", "bfloat16"],
                        help="Force dtype")
    args = parser.parse_args()

    model_name = args.model_key or args.model or DEFAULT_MODEL

    run_circuit_analysis(
        model_name=model_name,
        experiments=args.experiment,
        n_problems=args.n_problems,
        max_operand=args.operand_range,
        output_dir=args.output_dir,
        device_override=args.device,
        dtype_override=args.dtype,
        discovery_results_path=args.discovery_results,
    )
