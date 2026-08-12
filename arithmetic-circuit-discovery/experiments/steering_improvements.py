#!/usr/bin/env python3
"""
Investigate steering improvements for Fourier phase rotation.

Tests:
1. Delta scaling: multiply rotation delta by α ∈ {1, 2, 3, 5, 10}
2. W_U-projected delta: project rotation delta onto W_U digit subspace
3. Combined: scale the W_U-projected delta

Runs on logit lens (fast) to compare approaches.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

from experiments.fourier_phase_rotation import (
    compute_digit_fourier_basis, build_frequency_planes,
    compute_rotation_delta, generate_test_problems,
    filter_correct_problems, generate_single_digit_problems,
    filter_correct_single_digit, generate_single_digit_direct_answer_problems,
    MODEL_MAP, COMP_LAYERS,
)
from experiments.arithmetic_circuit_scan_updated import (
    generate_direct_answer_problems, filter_correct_direct_answer,
)


def compute_wu_projector(model, device):
    """Compute the projector onto the W_U digit subspace."""
    digit_token_ids = []
    for d in range(10):
        toks = model.to_tokens(str(d), prepend_bos=False)[0]
        digit_token_ids.append(toks[-1].item())
    digit_token_ids_t = torch.tensor(digit_token_ids, device=device)

    W_U = model.W_U.detach().float()  # (d_model, d_vocab)
    D = W_U[:, digit_token_ids_t]  # (d_model, 10)
    D_centered = D - D.mean(dim=1, keepdim=True)  # center across digits

    U_d, S_d, _ = torch.linalg.svd(D_centered, full_matrices=False)
    # Keep top 9 directions (10 digits - 1 mean = 9 DOF)
    U9 = U_d[:, :9]  # (d_model, 9)
    P_wu = U9 @ U9.T  # (d_model, d_model) — projector onto W_U digit subspace

    return P_wu, digit_token_ids_t, U9, D


def compute_wu_steering_vectors(D, basis_torch, device):
    """
    Compute W_U-informed steering vectors for each shift j.
    
    For shift j: the ideal direction is w_U[target] - w_U[orig], projected onto Fourier subspace.
    We precompute these for all (orig_digit, shift_j) combinations.
    
    Returns:
        wu_steer: dict[j] -> (10, d_model) tensor of per-digit steering vectors
    """
    P_F = basis_torch @ basis_torch.T  # Fourier projector
    wu_steer = {}
    for j in range(1, 10):
        vecs = []
        for d in range(10):
            target = (d + j) % 10
            # Ideal W_U direction: increase target logit, decrease original
            ideal = D[:, target] - D[:, d]  # (d_model,)
            # Project onto Fourier subspace
            ideal_F = P_F @ ideal
            vecs.append(ideal_F)
        wu_steer[j] = torch.stack(vecs)  # (10, d_model)
    return wu_steer


def run_steering_test(model, test_problems, layer, basis_problems,
                      scales, wu_scales, device):
    """Run logit lens with various steering improvements."""

    basis, freq_assignments, svals, digit_scores, freq_purities = \
        compute_digit_fourier_basis(model, basis_problems, layer, n_problems=500)
    freq_planes = build_frequency_planes(basis, freq_assignments, svals)

    hook_name = f"blocks.{layer}.hook_resid_post"

    # Convert to torch
    freq_planes_torch = {}
    for k, plane in freq_planes.items():
        freq_planes_torch[k] = {
            'vecs': torch.tensor(plane['vecs'], dtype=torch.float32, device=device),
            'svals': torch.tensor(plane['svals'], dtype=torch.float32, device=device),
        }
    basis_torch = torch.tensor(basis, dtype=torch.float32, device=device)

    # W_U projector and digit vectors
    P_wu, digit_token_ids_t, U9_wu, D_wu = compute_wu_projector(model, device)
    
    # W_U-informed steering vectors
    wu_steer = compute_wu_steering_vectors(D_wu, basis_torch, device)

    # Cache clean activations
    single_digit = [p for p in test_problems if p.get('n_digits', 1) == 1]
    logger.info(f"  Caching {len(single_digit)} single-digit activations...")

    clean_data = []
    with torch.no_grad():
        for prob in single_digit:
            tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
            act = cache[hook_name][0, -1].clone()
            clean_data.append((act, prob["ones_digit"]))
            del cache

    n_problems = len(clean_data)
    logger.info(f"  Cached {n_problems} activations")

    # Clean baseline
    clean_correct = 0
    with torch.no_grad():
        for act, d_orig in clean_data:
            h = act.unsqueeze(0).unsqueeze(0)
            normed = model.ln_final(h)
            logits = model.unembed(normed)[0, 0]
            pred = logits[digit_token_ids_t].argmax().item()
            if pred == d_orig:
                clean_correct += 1
    logger.info(f"  Clean logit-lens accuracy: {clean_correct}/{n_problems} = "
                f"{clean_correct/n_problems*100:.1f}%")

    # Define all methods to test
    methods = []

    # 1. Scaled coherent rotation
    for alpha in scales:
        methods.append({
            'name': f'coherent_x{alpha}',
            'type': 'scaled',
            'alpha': alpha,
        })

    # 2. W_U-projected rotation (with optional scaling)
    for alpha in wu_scales:
        methods.append({
            'name': f'wu_proj_x{alpha}',
            'type': 'wu_proj',
            'alpha': alpha,
        })

    # 3. W_U-informed steering: use W_U digit difference vectors projected onto Fourier
    for alpha in wu_scales:
        methods.append({
            'name': f'wu_steer_x{alpha}',
            'type': 'wu_steer',
            'alpha': alpha,
        })

    # 3. W_U-aware rotation: rotate in intersection of Fourier and W_U
    # Project Fourier basis onto W_U subspace, re-orthogonalize, rotate there
    F_in_wu = P_wu @ basis_torch  # (d_model, 9) — Fourier dirs projected onto W_U
    # Re-orthogonalize via QR (on CPU since MPS doesn't support linalg_qr)
    Q, R = torch.linalg.qr(F_in_wu.cpu())
    Q = Q.to(device)
    # Only keep directions with significant norm (non-degenerate)
    norms = torch.norm(F_in_wu, dim=0)
    good_dirs = norms > 0.1 * norms.max()
    n_good = good_dirs.sum().item()
    logger.info(f"  W_U-projected Fourier basis: {n_good}/9 directions have significant norm")
    logger.info(f"    Norms: {[f'{n:.3f}' for n in norms.tolist()]}")

    # Run all methods
    logger.info(f"\n{'='*80}")
    logger.info(f"STEERING IMPROVEMENT RESULTS (Layer {layer}, {n_problems} problems)")
    logger.info(f"{'='*80}")
    logger.info(f"  {'Method':<25} | {'exact%':>7} | {'rank':>6} | {'prob':>6} | "
                f"{'Δlogit':>7} | {'changed%':>8} | {'j=1':>5} | {'j=9':>5}")
    logger.info(f"  {'-'*25}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-"
                f"{'-'*7}-+-{'-'*8}-+-{'-'*5}-+-{'-'*5}")

    all_results = {}

    for method in methods:
        total_exact = 0
        total_changed = 0
        total_rank = 0
        total_prob = 0.0
        total_logit_diff = 0.0
        n_total = 0
        per_j_exact = {j: 0 for j in range(1, 10)}
        per_j_count = {j: 0 for j in range(1, 10)}

        with torch.no_grad():
            for j in range(1, 10):
                for act, d_orig in clean_data:
                    target_d = (d_orig + j) % 10
                    n_total += 1
                    per_j_count[j] += 1

                    # Compute base delta
                    delta = compute_rotation_delta(
                        act, freq_planes_torch, j, mode="coherent"
                    )

                    # Apply method
                    if method['type'] == 'scaled':
                        delta = delta * method['alpha']
                    elif method['type'] == 'wu_proj':
                        delta = P_wu @ delta * method['alpha']
                    elif method['type'] == 'wu_steer':
                        # Use W_U-informed direction instead of rotation
                        steer_dir = wu_steer[j][d_orig]  # (d_model,)
                        # Normalize to same magnitude as rotation delta, then scale
                        steer_norm = steer_dir.norm()
                        if steer_norm > 1e-8:
                            delta = steer_dir / steer_norm * delta.norm() * method['alpha']
                        else:
                            delta = delta * method['alpha']

                    rotated = act.float() + delta

                    # Logit lens
                    rot_h = rotated.unsqueeze(0).unsqueeze(0)
                    rot_normed = model.ln_final(rot_h)
                    rot_logits = model.unembed(rot_normed)[0, 0]
                    rot_digit_logits = rot_logits[digit_token_ids_t]
                    pred = rot_digit_logits.argmax().item()

                    # Clean prediction
                    clean_h = act.float().unsqueeze(0).unsqueeze(0)
                    clean_normed = model.ln_final(clean_h)
                    clean_logits = model.unembed(clean_normed)[0, 0]
                    clean_pred = clean_logits[digit_token_ids_t].argmax().item()

                    if pred != clean_pred:
                        total_changed += 1
                    if pred == target_d:
                        total_exact += 1
                        per_j_exact[j] += 1

                    # Rank
                    sorted_idx = rot_digit_logits.argsort(descending=True)
                    rank = (sorted_idx == target_d).nonzero(as_tuple=True)[0].item()
                    total_rank += rank

                    # Prob
                    probs = torch.softmax(rot_digit_logits, dim=0)
                    total_prob += probs[target_d].item()

                    # Logit diff
                    total_logit_diff += (rot_digit_logits[target_d] - rot_digit_logits[d_orig]).item()

        exact_pct = total_exact / n_total * 100
        changed_pct = total_changed / n_total * 100
        mean_rank = total_rank / n_total
        mean_prob = total_prob / n_total
        mean_ldiff = total_logit_diff / n_total
        j1_pct = per_j_exact[1] / per_j_count[1] * 100 if per_j_count[1] > 0 else 0
        j9_pct = per_j_exact[9] / per_j_count[9] * 100 if per_j_count[9] > 0 else 0

        logger.info(f"  {method['name']:<25} | {exact_pct:>6.1f}% | {mean_rank:>6.2f} | "
                    f"{mean_prob:>6.3f} | {mean_ldiff:>+6.2f} | {changed_pct:>7.1f}% | "
                    f"{j1_pct:>4.1f}% | {j9_pct:>4.1f}%")

        all_results[method['name']] = {
            'exact': exact_pct, 'rank': mean_rank, 'prob': mean_prob,
            'logit_diff': mean_ldiff, 'changed': changed_pct,
            'j1': j1_pct, 'j9': j9_pct,
        }

    return all_results


if __name__ == "__main__":
    from transformer_lens import HookedTransformer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma-2b")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--operand-range", type=int, default=30)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--scales", default="1,2,3,5,10",
                        help="Comma-separated scaling factors for coherent rotation")
    parser.add_argument("--wu-scales", default="1,3,5,10,20",
                        help="Comma-separated scaling factors for W_U-projected rotation")
    parser.add_argument("--direct-answer", action="store_true",
                        help="Use direct-answer format (for LLaMA)")
    args = parser.parse_args()

    if args.layer is None:
        args.layer = COMP_LAYERS.get(args.model, 20)

    scales = [float(x) for x in args.scales.split(",")]
    wu_scales = [float(x) for x in args.wu_scales.split(",")]

    model_name = MODEL_MAP[args.model]
    logger.info(f"Loading {model_name} on {args.device}...")
    model = HookedTransformer.from_pretrained(model_name, device=args.device)
    model.eval()

    # Generate problems
    if args.direct_answer:
        da_problems, _ = generate_direct_answer_problems(
            n_per_digit=100, operand_max=args.operand_range)
        da_correct = filter_correct_direct_answer(model, da_problems, max_n=len(da_problems))
        basis_problems = []
        for p in da_correct:
            answer = p["answer"]
            basis_problems.append({
                "a": p["a"], "b": p["b"], "answer": answer,
                "prompt": p["prompt"],
                "ones_digit": answer % 10,
                "tens_digit": (answer // 10) % 10,
                "first_digit": int(str(answer)[0]),
                "n_digits": len(str(answer)),
                "carry": 1 if (p["a"] % 10 + p["b"] % 10) >= 10 else 0,
            })
        sd_da = generate_single_digit_direct_answer_problems()
        test_problems = filter_correct_single_digit(model, sd_da)
    else:
        all_problems = generate_test_problems(max_operand=args.operand_range)
        basis_problems = filter_correct_problems(model, all_problems, n_test=len(all_problems))
        sd_problems = generate_single_digit_problems()
        test_problems = filter_correct_single_digit(model, sd_problems)

    logger.info(f"\nBasis problems: {len(basis_problems)}")
    logger.info(f"Test problems: {len(test_problems)} single-digit")

    results = run_steering_test(
        model, test_problems, args.layer, basis_problems,
        scales=scales, wu_scales=wu_scales, device=args.device,
    )
