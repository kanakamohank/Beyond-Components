#!/usr/bin/env python3
"""
C2: Multi-Layer Frequency-Specific Ablation

Resolves the "individual frequency paradox" from Exp 9:
  - Single-layer k=5 ablation at L19 caused only 0.21% damage
  - But k=5 has 99.87% purity → it's clearly present
  - Hypothesis: info is written redundantly across L14-L25, so ablating
    at just one layer doesn't remove it

This experiment ablates each frequency k=1..5 at ALL layers simultaneously
(L_comp through L_readout). If the frequency is causally necessary,
multi-layer ablation should cause significant damage even though
single-layer ablation doesn't.

Conditions tested:
  1. Baseline (no ablation)
  2. Multi-layer full 9D ablation (should replicate Exp 9 ~12% result)
  3. Multi-layer per-frequency ablation: k=1, k=2, k=3, k=4, k=5
  4. Multi-layer random control (same dimensionality as each freq)
  5. Cumulative frequency ablation: k=5 only, k=5+k=3, k=5+k=3+k=2, etc.

Usage:
    python multilayer_freq_ablation.py --model gemma-2b --device mps
    python multilayer_freq_ablation.py --model phi-3 --device mps
"""

import argparse
import json
import logging
import sys
import time
import gc
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from arithmetic_circuit_scan_updated import (
    generate_teacher_forced_problems,
    filter_correct_teacher_forced,
    generate_single_digit_problems,
    MODEL_MAP,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("mathematical_toolkit_results")
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES (reused from fourier_knockout.py)
# ─────────────────────────────────────────────────────────────────────────────

def check_orthonormal(V: np.ndarray, label: str = ""):
    gram = V.T @ V
    eye = np.eye(V.shape[1])
    err = np.abs(gram - eye).max()
    assert err < 1e-5, f"[SANITY] {label}: not orthonormal, max err = {err:.2e}"


def make_random_orthonormal_basis(d_model: int, n_dims: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    A = rng.randn(d_model, n_dims)
    Q, _ = np.linalg.qr(A)
    assert Q.shape == (d_model, n_dims)
    return Q


def assign_frequencies(U_cols: np.ndarray) -> List[int]:
    """Assign each SVD direction to its dominant Fourier frequency."""
    n_dirs = U_cols.shape[0]
    assignments = []
    for i in range(n_dirs):
        scores = U_cols[i]
        scores_centered = scores - scores.mean()
        dft = np.fft.fft(scores_centered)
        power = np.abs(dft) ** 2
        freq_power = np.zeros(6)
        freq_power[0] = power[0]
        for k in range(1, 5):
            freq_power[k] = power[k] + power[10 - k]
        freq_power[5] = power[5]
        dom_k = int(np.argmax(freq_power[1:])) + 1
        assignments.append(dom_k)
    return assignments


def collect_per_digit_means(model, problems, layer, device) -> np.ndarray:
    """Collect per-digit mean activations at a given layer."""
    hook_name = f"blocks.{layer}.hook_resid_post"
    d_model = model.cfg.d_model
    digit_acts = defaultdict(list)

    for prob in problems:
        digit = int(prob["target_str"])
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
            digit_acts[digit].append(act_vec)

    means = np.zeros((10, d_model))
    for d in range(10):
        if digit_acts[d]:
            means[d] = np.mean(digit_acts[d], axis=0)
    return means


def evaluate_accuracy_multi_layer(
    model,
    problems: List[dict],
    layer_projections: Dict[int, torch.Tensor],
    device: str,
) -> Dict:
    """Evaluate accuracy with ablation at MULTIPLE layers simultaneously."""
    total = 0
    correct = 0
    per_digit_total = defaultdict(int)
    per_digit_correct = defaultdict(int)

    def make_hook(proj):
        def hook_fn(act, hook, p=proj):
            h = act[:, -1, :].float()
            projected = h @ p
            act[:, -1, :] = (h - projected).to(act.dtype)
            return act
        return hook_fn

    fwd_hooks = []
    for layer_idx, proj in layer_projections.items():
        hook_name = f"blocks.{layer_idx}.hook_resid_post"
        fwd_hooks.append((hook_name, make_hook(proj)))

    for prob in problems:
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True).to(device)
        target_digit = int(prob["target_str"])

        with torch.no_grad():
            with model.hooks(fwd_hooks=fwd_hooks):
                logits = model(tokens)

        pred_tok = logits[0, -1].argmax().item()
        pred_str = model.tokenizer.decode([pred_tok]).strip()

        total += 1
        per_digit_total[target_digit] += 1
        try:
            pred_digit = int(pred_str)
            if pred_digit == target_digit:
                correct += 1
                per_digit_correct[target_digit] += 1
        except ValueError:
            pass

    accuracy = correct / total if total > 0 else 0.0
    per_digit_acc = {}
    for d in range(10):
        n = per_digit_total[d]
        c = per_digit_correct[d]
        per_digit_acc[d] = c / n if n > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_digit_accuracy": per_digit_acc,
        "per_digit_total": dict(per_digit_total),
        "per_digit_correct": dict(per_digit_correct),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="C2: Multi-layer frequency-specific ablation")
    parser.add_argument("--model", default="gemma-2b", choices=list(MODEL_MAP.keys()))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--comp-layer", type=int, default=None)
    parser.add_argument("--readout-layer", type=int, default=None)
    parser.add_argument("--n-train-per-digit", type=int, default=100)
    parser.add_argument("--n-test-per-digit", type=int, default=50)
    args = parser.parse_args()

    model_name = MODEL_MAP[args.model]
    device = args.device

    comp_defaults = {"gemma-2b": 19, "phi-3": 26, "llama-3b": 20}
    readout_defaults = {"gemma-2b": 25, "phi-3": 31, "llama-3b": 27}
    comp_layer = args.comp_layer or comp_defaults.get(args.model, 20)
    readout_layer = args.readout_layer or readout_defaults.get(args.model, comp_layer + 6)

    all_layers = list(range(comp_layer, readout_layer + 1))

    logger.info(f"Model: {args.model} ({model_name})")
    logger.info(f"Layer range: L{comp_layer}→L{readout_layer} ({len(all_layers)} layers)")
    logger.info(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=torch.float32 if device == "cpu" else torch.float16,
    )
    model.eval()
    d_model = model.cfg.d_model
    logger.info(f"d_model = {d_model}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: Generate train/test problems
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("  STEP 1: Generate train/test problems")
    logger.info("=" * 60)

    n_train = args.n_train_per_digit
    n_test = args.n_test_per_digit
    n_total_needed = n_train + n_test

    try:
        all_problems, _ = generate_teacher_forced_problems(n_per_digit=n_total_needed)
        all_correct = filter_correct_teacher_forced(model, all_problems, max_n=n_total_needed * 10)
    except Exception as e:
        logger.warning(f"  Teacher-forced failed ({e}), trying single-digit fallback...")
        all_problems = generate_single_digit_problems()
        all_correct = filter_correct_teacher_forced(model, all_problems, max_n=500)

    by_digit = defaultdict(list)
    for p in all_correct:
        by_digit[int(p["target_str"])].append(p)

    min_count = min(len(by_digit[d]) for d in range(10))
    actual_train = min(n_train, min_count - 10)
    actual_test = min(n_test, min_count - actual_train)
    assert actual_train >= 10 and actual_test >= 10

    train_problems = []
    test_problems = []
    for d in range(10):
        digit_probs = by_digit[d][:min_count]
        train_problems.extend(digit_probs[:actual_train])
        test_problems.extend(digit_probs[actual_train:actual_train + actual_test])

    logger.info(f"  Train: {len(train_problems)} ({actual_train}/digit)")
    logger.info(f"  Test:  {len(test_problems)} ({actual_test}/digit)")

    # Sanity: no overlap
    assert len(set(p["prompt"] for p in train_problems) &
               set(p["prompt"] for p in test_problems)) == 0
    logger.info("  [SANITY] Train/test disjoint ✓")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: Compute per-layer Fourier subspaces and frequency assignments
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("  STEP 2: Compute per-layer Fourier subspaces")
    logger.info("=" * 60)

    # For each layer, compute: 9D basis, frequency assignments, per-freq projections
    layer_info = {}  # layer -> {V_9, freq_assignments, freq_to_dirs, per_freq_V}
    for l in all_layers:
        t0 = time.time()
        means_l = collect_per_digit_means(model, train_problems, l, device)
        cent_l = means_l.mean(axis=0, keepdims=True)
        M_l = means_l - cent_l

        U_l, S_l, Vt_l = np.linalg.svd(M_l, full_matrices=False)

        V_9 = Vt_l[:9].T  # (d_model, 9)
        check_orthonormal(V_9, label=f"L{l}")

        freqs = assign_frequencies(U_l[:, :9].T)

        freq_to_dirs = defaultdict(list)
        for dir_idx, freq in enumerate(freqs):
            freq_to_dirs[freq].append(dir_idx)

        # Per-frequency basis vectors
        per_freq_V = {}
        for k in range(1, 6):
            if k in freq_to_dirs:
                dir_indices = freq_to_dirs[k]
                V_k = Vt_l[dir_indices].T  # (d_model, n_dirs)
                per_freq_V[k] = V_k

        layer_info[l] = {
            "V_9": V_9,
            "S": S_l[:9],
            "freq_assignments": freqs,
            "freq_to_dirs": dict(freq_to_dirs),
            "per_freq_V": per_freq_V,
        }
        elapsed = time.time() - t0
        logger.info(f"  L{l}: freqs={freqs}, σ₁={S_l[0]:.1f} ({elapsed:.1f}s)")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: Build multi-layer ablation conditions
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("  STEP 3: Build multi-layer ablation conditions")
    logger.info("=" * 60)

    results = {}

    def run_multi_layer(name, layer_projs):
        logger.info(f"\n  ── {name} ──")
        t0 = time.time()
        res = evaluate_accuracy_multi_layer(model, test_problems, layer_projs, device)
        elapsed = time.time() - t0
        acc = res["accuracy"]
        logger.info(f"  Accuracy: {res['correct']}/{res['total']} = {acc*100:.1f}%  ({elapsed:.1f}s)")
        per_d = res["per_digit_accuracy"]
        logger.info(f"  Per-digit: " +
                    " ".join(f"{d}:{per_d[d]*100:.0f}%" for d in range(10)))
        results[name] = res
        return res

    # 3a. Baseline
    logger.info(f"\n  ── baseline ──")
    t0 = time.time()
    # No hooks
    total = correct = 0
    per_digit_total = defaultdict(int)
    per_digit_correct = defaultdict(int)
    for prob in test_problems:
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True).to(device)
        target_digit = int(prob["target_str"])
        with torch.no_grad():
            logits = model(tokens)
        pred_tok = logits[0, -1].argmax().item()
        pred_str = model.tokenizer.decode([pred_tok]).strip()
        total += 1
        per_digit_total[target_digit] += 1
        try:
            if int(pred_str) == target_digit:
                correct += 1
                per_digit_correct[target_digit] += 1
        except ValueError:
            pass
    baseline_acc = correct / total
    per_d_acc = {d: per_digit_correct[d] / per_digit_total[d] if per_digit_total[d] > 0 else 0
                 for d in range(10)}
    results["baseline"] = {
        "accuracy": baseline_acc, "correct": correct, "total": total,
        "per_digit_accuracy": per_d_acc,
    }
    logger.info(f"  Accuracy: {correct}/{total} = {baseline_acc*100:.1f}%  ({time.time()-t0:.1f}s)")
    logger.info(f"  Per-digit: " +
                " ".join(f"{d}:{per_d_acc[d]*100:.0f}%" for d in range(10)))

    assert baseline_acc > 0.70, f"Baseline too low: {baseline_acc*100:.1f}%"

    # 3b. Multi-layer full 9D ablation (should replicate ~12% from Exp 9)
    full_9d_projs = {}
    for l in all_layers:
        V = layer_info[l]["V_9"]
        P = V @ V.T
        full_9d_projs[l] = torch.tensor(P, dtype=torch.float32, device=device)
    run_multi_layer("multi_full_9d", full_9d_projs)

    # 3c. Multi-layer PER-FREQUENCY ablation — THE KEY TEST
    logger.info("\n" + "=" * 60)
    logger.info("  STEP 4: Multi-layer per-frequency ablation (THE KEY TEST)")
    logger.info("=" * 60)

    # Track dimensionality per frequency per layer
    for k in range(1, 6):
        freq_projs = {}
        dims_per_layer = []
        for l in all_layers:
            V_k = layer_info[l]["per_freq_V"].get(k, None)
            if V_k is not None:
                P_k = V_k @ V_k.T
                freq_projs[l] = torch.tensor(P_k, dtype=torch.float32, device=device)
                dims_per_layer.append(V_k.shape[1])
            else:
                dims_per_layer.append(0)

        dof = f"{min(dims_per_layer)}-{max(dims_per_layer)}D" if dims_per_layer else "0D"
        name = f"multi_freq_k{k} ({dof}, {len(freq_projs)} layers)"
        if freq_projs:
            run_multi_layer(name, freq_projs)
        else:
            logger.warning(f"  k={k}: no layers have this frequency!")

    # 3d. Multi-layer random controls (matching dimensionality of each freq)
    logger.info("\n" + "=" * 60)
    logger.info("  STEP 5: Random controls (matching dimensionality)")
    logger.info("=" * 60)

    for k in range(1, 6):
        rand_projs = {}
        for l in all_layers:
            V_k = layer_info[l]["per_freq_V"].get(k, None)
            if V_k is not None:
                n_dims = V_k.shape[1]
                V_rand = make_random_orthonormal_basis(d_model, n_dims, seed=42 + l + k * 100)
                P_rand = V_rand @ V_rand.T
                rand_projs[l] = torch.tensor(P_rand, dtype=torch.float32, device=device)

        if rand_projs:
            run_multi_layer(f"multi_random_k{k}_dims", rand_projs)

    # Multi-layer full random 9D control
    rand_9d_projs = {}
    for l in all_layers:
        V_rand = make_random_orthonormal_basis(d_model, 9, seed=42 + l)
        rand_9d_projs[l] = torch.tensor(V_rand @ V_rand.T, dtype=torch.float32, device=device)
    run_multi_layer("multi_random_full_9d", rand_9d_projs)

    # 3e. Cumulative frequency ablation (ordered by Gemma importance: k=5, k=3, k=2, k=4, k=1)
    logger.info("\n" + "=" * 60)
    logger.info("  STEP 6: Cumulative frequency ablation")
    logger.info("=" * 60)

    # Order frequencies by singular value dominance (k=5 first for Gemma)
    # Get dominant frequency from comp layer
    comp_freqs = layer_info[comp_layer]["freq_assignments"]
    comp_S = layer_info[comp_layer]["S"]
    # Aggregate: which frequency has the highest total σ²?
    freq_total_sv2 = defaultdict(float)
    for i, f in enumerate(comp_freqs):
        freq_total_sv2[f] += comp_S[i] ** 2
    freq_order = sorted(freq_total_sv2.keys(), key=lambda k: freq_total_sv2[k], reverse=True)
    logger.info(f"  Frequency order by σ² at L{comp_layer}: {freq_order}")
    logger.info(f"  σ² per freq: " +
                ", ".join(f"k={k}: {freq_total_sv2[k]:.0f}" for k in freq_order))

    cumulative_freqs = []
    for k in freq_order:
        cumulative_freqs.append(k)

        # Build projection from stored per-freq basis vectors
        cum_projs = {}
        for l in all_layers:
            basis_vecs = []
            for kk in cumulative_freqs:
                V_k = layer_info[l]["per_freq_V"].get(kk, None)
                if V_k is not None:
                    basis_vecs.append(V_k)
            if basis_vecs:
                V_cum = np.hstack(basis_vecs)  # (d_model, total_dims)
                # Re-orthogonalize (per-freq bases are already orthonormal within
                # each freq, but may have slight overlap between freqs due to
                # imperfect SVD separation)
                Q, _ = np.linalg.qr(V_cum)
                Q = Q[:, :V_cum.shape[1]]  # keep same number of directions
                P_cum = Q @ Q.T
                cum_projs[l] = torch.tensor(P_cum, dtype=torch.float32, device=device)

        freq_str = "+".join(f"k{kk}" for kk in cumulative_freqs)
        total_dims = sum(
            layer_info[comp_layer]["per_freq_V"].get(kk, np.empty((0, 0))).shape[1]
            for kk in cumulative_freqs
            if kk in layer_info[comp_layer]["per_freq_V"]
        )
        run_multi_layer(f"cumulative_{freq_str} ({total_dims}D)", cum_projs)

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("  SUMMARY")
    logger.info("=" * 60)

    logger.info(f"\n  {'Condition':<45} {'Accuracy':>10} {'Delta':>10}")
    logger.info(f"  {'─'*45} {'─'*10} {'─'*10}")
    for name, res in results.items():
        acc = res["accuracy"] * 100
        delta = (res["accuracy"] - baseline_acc) * 100
        marker = ""
        if name == "baseline":
            marker = " ◀"
        elif abs(delta) > 30:
            marker = " ★★★"
        elif abs(delta) > 10:
            marker = " ★★"
        elif abs(delta) > 5:
            marker = " ★"
        logger.info(f"  {name:<45} {acc:>9.1f}% {delta:>+9.1f}%{marker}")

    # Key question: does multi-layer k=5 ablation cause more damage than single-layer?
    k5_result = None
    for name, res in results.items():
        if "freq_k5" in name:
            k5_result = res
            break

    if k5_result:
        k5_acc = k5_result["accuracy"] * 100
        logger.info(f"\n  KEY RESULT: Multi-layer k=5 ablation → {k5_acc:.1f}%")
        logger.info(f"  (Compare: single-layer k=5 at L{comp_layer} was ~99.8% in Exp 9)")
        if k5_acc < baseline_acc * 100 - 5:
            logger.info(f"  ★ RESOLVED: Multi-layer k=5 ablation DOES cause damage!")
            logger.info(f"    Single-layer failed because info is redundant across layers.")
        else:
            logger.info(f"  ⚠ Multi-layer k=5 still doesn't cause much damage.")
            logger.info(f"    The frequency may be truly redundant or encoded differently.")

    # Save
    output = {
        "model": model_name,
        "model_short": args.model,
        "comp_layer": comp_layer,
        "readout_layer": readout_layer,
        "all_layers": all_layers,
        "n_layers": len(all_layers),
        "d_model": d_model,
        "n_train_per_digit": actual_train,
        "n_test_per_digit": actual_test,
        "freq_order_by_sv2": freq_order,
        "per_layer_info": {
            str(l): {
                "freq_assignments": layer_info[l]["freq_assignments"],
                "freq_to_dirs": layer_info[l]["freq_to_dirs"],
                "singular_values": layer_info[l]["S"].tolist(),
            }
            for l in all_layers
        },
        "conditions": {},
    }
    for name, res in results.items():
        output["conditions"][name] = {
            "accuracy": res["accuracy"],
            "correct": res.get("correct", 0),
            "total": res.get("total", 0),
            "per_digit_accuracy": {str(k): v for k, v in res.get("per_digit_accuracy", {}).items()},
        }

    out_path = RESULTS_DIR / f"multilayer_freq_ablation_{args.model}_L{comp_layer}-L{readout_layer}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\n  Results saved to {out_path}")
    logger.info("\n  C2 COMPLETE.")


if __name__ == "__main__":
    main()
