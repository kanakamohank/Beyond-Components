#!/usr/bin/env python3
"""
Probe Steering v2: Fix the issues from v1.

Key changes:
1. Much more training data (1000+ samples) to prevent overfitting
2. Strong regularization (L2 with proper C tuning)
3. Try multiple layers to find where digit is most linearly readable
4. Use DIFFERENCE-IN-MEANS (DIM) steering vectors instead of probe weights
   - Compute mean activation for each digit class
   - Steering vector = mean(target digit) - mean(source digit)
   - This is the simplest, most robust steering approach
5. Scale sweep to find optimal intervention strength

DIM steering doesn't assume ANY geometry — it just uses the statistical
difference between digit classes as the intervention direction.
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


def generate_problems(operand_range=50):
    """Generate addition problems covering all 10 ones digits well."""
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
    np.random.shuffle(problems)
    return problems


def collect_activations_multi_layer(model, problems, layers, n_problems=500):
    """Collect activations at multiple layers simultaneously."""
    logger.info(f"  Collecting activations at layers {layers} ({n_problems} problems)...")
    hook_names = [f"blocks.{l}.hook_resid_post" for l in layers]

    acts_by_layer = {l: [] for l in layers}
    labels = []

    for i, prob in enumerate(problems[:n_problems]):
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)
            for l, hn in zip(layers, hook_names):
                acts_by_layer[l].append(cache[hn][0, -1].cpu().float().numpy())
            del cache
        labels.append(prob["ones_digit"])

        if (i + 1) % 100 == 0:
            logger.info(f"    Collected {i+1}/{n_problems}")

    labels = np.array(labels)
    for l in layers:
        acts_by_layer[l] = np.array(acts_by_layer[l])

    # Digit distribution
    for d in range(10):
        logger.info(f"    Digit {d}: {(labels == d).sum()} samples")

    return acts_by_layer, labels


def compute_digit_means(activations, labels):
    """Compute mean activation per ones digit."""
    means = {}
    for d in range(10):
        mask = labels == d
        if mask.sum() > 0:
            means[d] = activations[mask].mean(axis=0)
    return means


def evaluate_probe(activations, labels):
    """Train and evaluate a linear probe with proper CV."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    scaler = StandardScaler()
    X = scaler.fit_transform(activations)
    y = labels

    # Use stronger regularization to prevent overfitting
    probe = LogisticRegression(max_iter=2000, C=0.01, solver='lbfgs', random_state=42)
    cv_scores = cross_val_score(probe, X, y, cv=5, scoring='accuracy')

    probe.fit(X, y)
    train_acc = probe.score(X, y)

    return {
        "cv_accuracy": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "train_accuracy": float(train_acc),
    }


# ═══════════════════════════════════════════════════════════════
# DIFFERENCE-IN-MEANS STEERING
# ═══════════════════════════════════════════════════════════════

def dim_steering(model, layer, digit_means, test_problems, scales=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0]):
    """
    Difference-in-Means (DIM) steering: use the vector
    (mean_activation[target_digit] - mean_activation[source_digit])
    as the intervention direction.

    No geometry assumed — just pure statistical difference.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"DIFFERENCE-IN-MEANS STEERING (Layer {layer})")
    logger.info(f"{'='*60}")

    device = next(model.parameters()).device
    hook_name = f"blocks.{layer}.hook_resid_post"

    # Filter to correct problems
    correct = []
    for prob in test_problems[:200]:
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
        with torch.no_grad():
            logits = model(tokens)
        pred_tok = logits[0, -1].argmax(dim=-1).item()
        pred_str = model.tokenizer.decode([pred_tok]).strip()
        try:
            if int(pred_str) == prob["first_digit"]:
                correct.append(prob)
        except ValueError:
            pass
        if len(correct) >= 80:
            break

    # Only use single-digit problems for exact validation
    single = [p for p in correct if p["n_digits"] == 1]
    logger.info(f"  Using {len(single)} correct single-digit problems")
    logger.info(f"  Digit dist: {dict(sorted({d: sum(1 for p in single if p['ones_digit']==d) for d in range(10) if sum(1 for p in single if p['ones_digit']==d) > 0}.items()))}")

    all_results = {}

    # Test specific shift patterns
    for shift_k in [1, 2, 3, 5]:
        logger.info(f"\n  --- Shift k={shift_k}: d → (d+{shift_k})%10 ---")
        best_scale = None
        best_exact = 0
        best_changed = 0

        for scale in scales:
            exact = 0
            changed = 0
            total = 0
            numeric = 0

            for prob in single:
                tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
                original = prob["ones_digit"]
                target = (original + shift_k) % 10

                if original not in digit_means or target not in digit_means:
                    continue

                # DIM steering vector
                steer = digit_means[target] - digit_means[original]
                steer_torch = scale * torch.tensor(steer, dtype=torch.float32, device=device)

                def hook_fn(act, hook, d=steer_torch):
                    act[:, -1, :] = act[:, -1, :] + d.to(act.dtype).unsqueeze(0)
                    return act

                with torch.no_grad():
                    with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
                        logits = model(tokens)

                pred_tok = logits[0, -1].argmax(dim=-1).item()
                pred_str = model.tokenizer.decode([pred_tok]).strip()

                try:
                    pred_digit = int(pred_str)
                    total += 1
                    numeric += 1
                    if pred_digit != original:
                        changed += 1
                    if pred_digit == target:
                        exact += 1
                except ValueError:
                    total += 1

            exact_rate = 100 * exact / total if total > 0 else 0
            changed_rate = 100 * changed / total if total > 0 else 0
            numeric_rate = 100 * numeric / total if total > 0 else 0

            logger.info(f"    scale={scale:5.2f}: exact={exact}/{total} ({exact_rate:.1f}%), "
                         f"changed={changed}/{total} ({changed_rate:.1f}%), "
                         f"numeric={numeric}/{total} ({numeric_rate:.1f}%)")

            if exact > best_exact:
                best_exact = exact
                best_scale = scale
                best_changed = changed

        if best_scale is not None:
            all_results[f"shift_{shift_k}"] = {
                "best_scale": best_scale,
                "best_exact": best_exact,
                "best_changed": best_changed,
                "total": total,
            }

    # Now run the BEST scale for all 9 shifts
    # Find what works best overall
    logger.info(f"\n  --- Comprehensive: all 9 shifts at scale=1.0 and scale=0.5 ---")
    for scale in [0.5, 1.0]:
        logger.info(f"\n  Scale = {scale}:")
        total_exact = 0
        total_changed = 0
        total_tests = 0

        for k in range(1, 10):
            exact = 0
            changed = 0
            total = 0

            for prob in single:
                tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
                original = prob["ones_digit"]
                target = (original + k) % 10

                if original not in digit_means or target not in digit_means:
                    continue

                steer = digit_means[target] - digit_means[original]
                steer_torch = scale * torch.tensor(steer, dtype=torch.float32, device=device)

                def hook_fn(act, hook, d=steer_torch):
                    act[:, -1, :] = act[:, -1, :] + d.to(act.dtype).unsqueeze(0)
                    return act

                with torch.no_grad():
                    with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
                        logits = model(tokens)

                pred_tok = logits[0, -1].argmax(dim=-1).item()
                pred_str = model.tokenizer.decode([pred_tok]).strip()

                try:
                    pred_digit = int(pred_str)
                    total += 1
                    if pred_digit != original:
                        changed += 1
                    if pred_digit == target:
                        exact += 1
                except ValueError:
                    total += 1

            exact_rate = 100 * exact / total if total > 0 else 0
            changed_rate = 100 * changed / total if total > 0 else 0
            logger.info(f"    k={k} (d→d+{k}): exact={exact}/{total} ({exact_rate:.1f}%), "
                         f"changed={changed}/{total} ({changed_rate:.1f}%)")
            total_exact += exact
            total_changed += changed
            total_tests += total

        overall_exact = 100 * total_exact / total_tests if total_tests > 0 else 0
        overall_changed = 100 * total_changed / total_tests if total_tests > 0 else 0
        logger.info(f"    OVERALL: exact={total_exact}/{total_tests} ({overall_exact:.1f}%), "
                     f"changed={total_changed}/{total_tests} ({overall_changed:.1f}%)")

    return all_results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Probe Steering v2")
    parser.add_argument("--model", default="gemma-2b")
    parser.add_argument("--layers", default="18,20,22,24", help="Comma-separated layers")
    parser.add_argument("--operand-range", type=int, default=50)
    parser.add_argument("--n-collect", type=int, default=500, help="Samples for activation collection")
    args = parser.parse_args()

    from transformer_lens import HookedTransformer

    model_name = MODEL_MAP.get(args.model, args.model)
    layers = [int(l) for l in args.layers.split(",")]
    device = get_device()

    logger.info("=" * 60)
    logger.info("PROBE STEERING v2 (Difference-in-Means)")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Layers: {layers}")

    logger.info(f"Loading {model_name} on {device}...")
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=torch.float32)

    problems = generate_problems(args.operand_range)
    logger.info(f"Generated {len(problems)} problems (operand_range={args.operand_range})")

    # Collect activations at all layers
    acts_by_layer, labels = collect_activations_multi_layer(
        model, problems, layers, n_problems=args.n_collect
    )

    # Step 1: Find which layer has best linear readability
    logger.info(f"\n{'='*60}")
    logger.info(f"STEP 1: LINEAR PROBE ACCURACY BY LAYER")
    logger.info(f"{'='*60}")

    best_layer = None
    best_cv = 0
    probe_results = {}

    for layer in layers:
        result = evaluate_probe(acts_by_layer[layer], labels)
        logger.info(f"  Layer {layer}: CV={result['cv_accuracy']:.3f}±{result['cv_std']:.3f}, "
                     f"Train={result['train_accuracy']:.3f}")
        probe_results[layer] = result
        if result["cv_accuracy"] > best_cv:
            best_cv = result["cv_accuracy"]
            best_layer = layer

    logger.info(f"\n  Best layer for linear readability: {best_layer} (CV={best_cv:.3f})")

    # Step 2: DIM steering at each layer
    logger.info(f"\n{'='*60}")
    logger.info(f"STEP 2: DIFFERENCE-IN-MEANS STEERING")
    logger.info(f"{'='*60}")

    # Generate single-digit test problems
    test_problems = []
    for a in range(10):
        for b in range(10):
            answer = a + b
            if answer < 10:
                test_problems.append({
                    "prompt": f"Calculate:\n{a} + {b} = ",
                    "answer": answer,
                    "ones_digit": answer % 10,
                    "first_digit": int(str(answer)[0]),
                    "n_digits": 1,
                    "a": a, "b": b,
                })
    np.random.shuffle(test_problems)

    dim_results = {}
    for layer in layers:
        digit_means = compute_digit_means(acts_by_layer[layer], labels)
        result = dim_steering(model, layer, digit_means, test_problems,
                              scales=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
        dim_results[layer] = result

    # Save results
    out_dir = Path("mathematical_toolkit_results")
    out_dir.mkdir(exist_ok=True)
    save_data = {
        "model": model_name,
        "layers": layers,
        "probe_results": {str(k): v for k, v in probe_results.items()},
        "dim_results": {str(k): v for k, v in dim_results.items()},
    }
    out_path = out_dir / f"probe_steering_v2_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
