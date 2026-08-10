#!/usr/bin/env python3
"""
Diagnose the rotation sign asymmetry in Fourier phase rotation.

Hypothesis: our sin direction is negated relative to the model's encoding,
causing "shift +j" to actually shift by -j.

Tests:
1. Signed correlations of digit scores vs theoretical DFT waves
2. Phase angles of digit projections in each frequency plane
3. Whether negating θ (or equivalently, negating sin direction) fixes the asymmetry
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

from experiments.fourier_phase_rotation import (
    compute_digit_fourier_basis, sanity_check_basis,
    build_frequency_planes, generate_test_problems,
    filter_correct_problems, compute_rotation_delta,
    generate_single_digit_problems, filter_correct_single_digit,
    MODEL_MAP
)

def diagnose_sign(model, problems, layer):
    """Check if the DFT basis sin direction matches the model's encoding."""
    
    basis, freq_assignments, svals, digit_scores, freq_purities = \
        compute_digit_fourier_basis(model, problems, layer, n_problems=500)
    
    digits = np.arange(10)
    n_dirs = len(freq_assignments)
    
    logger.info("\n" + "="*70)
    logger.info("SIGN DIAGNOSTIC: Signed correlations (no abs!)")
    logger.info("="*70)
    
    # For each direction, check SIGNED correlation with theoretical waves
    for i in range(n_dirs):
        k = freq_assignments[i]
        scores = digit_scores[i]
        scores_c = scores - scores.mean()
        
        cos_wave = np.cos(2 * np.pi * k * digits / 10)
        sin_wave = np.sin(2 * np.pi * k * digits / 10)
        
        if np.std(scores_c) > 1e-10:
            cos_corr = np.corrcoef(scores_c, cos_wave)[0, 1]  # SIGNED
            sin_corr = np.corrcoef(scores_c, sin_wave)[0, 1]  # SIGNED
        else:
            cos_corr = sin_corr = 0.0
        
        # Determine which pair this is within its frequency
        freq_dirs = [j for j, f in enumerate(freq_assignments) if f == k]
        pos = freq_dirs.index(i)
        expected = "cos" if pos == 0 else "sin"
        
        sign_ok = "✓" if (expected == "cos" and cos_corr > 0) or \
                         (expected == "sin" and sin_corr > 0) else "✗ NEGATED"
        
        logger.info(f"  Dir {i} (k={k}, {expected}): "
                    f"cos_corr={cos_corr:+.4f}, sin_corr={sin_corr:+.4f}  {sign_ok}")
    
    # Phase angle analysis for each frequency plane
    logger.info("\n" + "="*70)
    logger.info("PHASE ANGLE DIAGNOSTIC")
    logger.info("="*70)
    logger.info("If encoding matches DFT convention: φ(d) ≈ 2πkd/10")
    logger.info("If sin is negated: φ(d) ≈ -2πkd/10 = 2πk(10-d)/10")
    
    freq_dirs_map = {}
    for i, k in enumerate(freq_assignments):
        freq_dirs_map.setdefault(k, []).append(i)
    
    sign_votes = {k: 0 for k in range(1, 6)}  # positive = correct, negative = flipped
    
    for k, dirs in sorted(freq_dirs_map.items()):
        if k == 5:
            # Nyquist: only cos, no phase to check
            logger.info(f"\n  Freq k={k} (Nyquist): cos-only, checking sign of alternation")
            scores = digit_scores[dirs[0]]
            # Theoretical: cos(πd) = (-1)^d = [1, -1, 1, -1, ...]
            alternating = np.array([(-1)**d for d in range(10)], dtype=float)
            corr = np.corrcoef(scores - scores.mean(), alternating)[0, 1]
            logger.info(f"    Correlation with (-1)^d: {corr:+.4f}")
            sign_votes[k] = 1 if corr > 0 else -1
            continue
        
        if len(dirs) < 2:
            continue
            
        cos_scores = digit_scores[dirs[0]]  # should be cos axis
        sin_scores = digit_scores[dirs[1]]  # should be sin axis
        
        logger.info(f"\n  Freq k={k}:")
        logger.info(f"    {'Digit':>5} | {'cos_score':>10} | {'sin_score':>10} | "
                    f"{'φ_actual':>10} | {'φ_theory':>10} | {'φ_negated':>10} | "
                    f"{'err_fwd':>8} | {'err_bwd':>8}")
        logger.info(f"    {'-----':>5}-+-{'----------':>10}-+-{'----------':>10}-+-"
                    f"{'----------':>10}-+-{'----------':>10}-+-{'----------':>10}-+-"
                    f"{'--------':>8}-+-{'--------':>8}")
        
        total_err_fwd = 0
        total_err_bwd = 0
        
        for d in range(10):
            c = cos_scores[d]
            s = sin_scores[d]
            phi_actual = np.arctan2(s, c)
            phi_theory = 2 * np.pi * k * d / 10  # forward convention
            phi_negated = -2 * np.pi * k * d / 10  # backward convention
            
            # Wrap to [-π, π] for comparison
            phi_theory_wrapped = np.arctan2(np.sin(phi_theory), np.cos(phi_theory))
            phi_negated_wrapped = np.arctan2(np.sin(phi_negated), np.cos(phi_negated))
            
            # Angular error
            err_fwd = abs(np.arctan2(np.sin(phi_actual - phi_theory_wrapped), 
                                      np.cos(phi_actual - phi_theory_wrapped)))
            err_bwd = abs(np.arctan2(np.sin(phi_actual - phi_negated_wrapped), 
                                      np.cos(phi_actual - phi_negated_wrapped)))
            
            total_err_fwd += err_fwd
            total_err_bwd += err_bwd
            
            logger.info(f"    {d:>5} | {c:>10.3f} | {s:>10.3f} | "
                        f"{np.degrees(phi_actual):>9.1f}° | "
                        f"{np.degrees(phi_theory_wrapped):>9.1f}° | "
                        f"{np.degrees(phi_negated_wrapped):>9.1f}° | "
                        f"{np.degrees(err_fwd):>7.1f}° | {np.degrees(err_bwd):>7.1f}°")
        
        logger.info(f"    TOTAL angular error: forward={np.degrees(total_err_fwd):.1f}°, "
                    f"backward={np.degrees(total_err_bwd):.1f}°")
        
        if total_err_fwd < total_err_bwd:
            logger.info(f"    → Forward convention MATCHES (sign is correct)")
            sign_votes[k] = 1
        else:
            logger.info(f"    → Backward convention matches better (sin direction is NEGATED)")
            sign_votes[k] = -1
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("SIGN VOTE SUMMARY")
    logger.info("="*70)
    for k, vote in sorted(sign_votes.items()):
        label = "CORRECT" if vote > 0 else "NEGATED"
        logger.info(f"  k={k}: {label}")
    
    n_negated = sum(1 for v in sign_votes.values() if v < 0)
    n_correct = sum(1 for v in sign_votes.values() if v > 0)
    
    if n_negated > n_correct:
        logger.info(f"\n  VERDICT: Sin direction is NEGATED in {n_negated}/{n_negated+n_correct} "
                    f"frequency planes.")
        logger.info(f"  FIX: Negate sin direction in basis computation (flip c_sin → -c_sin)")
        logger.info(f"  This will make j=1 the 'forward +1' shift as intended.")
    elif n_negated == 0:
        logger.info(f"\n  VERDICT: All signs are correct. Asymmetry has a different cause.")
    else:
        logger.info(f"\n  VERDICT: Mixed — {n_negated} negated, {n_correct} correct. "
                    f"Asymmetry is frequency-dependent.")
    
    return sign_votes


def diagnose_asymmetry(model, basis_problems, test_problems, layer):
    """Deep investigation of why backward shifts (j=9) >> forward shifts (j=1).
    
    Args:
        basis_problems: large set for computing basis (can be multi-digit)
        test_problems: single-digit problems for testing (~55)
    """
    
    basis, freq_assignments, svals, digit_scores, freq_purities = \
        compute_digit_fourier_basis(model, basis_problems, layer, n_problems=500)
    freq_planes = build_frequency_planes(basis, freq_assignments, svals)
    
    device = next(model.parameters()).device
    hook_name = f"blocks.{layer}.hook_resid_post"
    
    # Convert to torch
    freq_planes_torch = {}
    for k, plane in freq_planes.items():
        freq_planes_torch[k] = {
            'vecs': torch.tensor(plane['vecs'], dtype=torch.float32, device=device),
            'svals': torch.tensor(plane['svals'], dtype=torch.float32, device=device),
        }
    basis_torch = torch.tensor(basis, dtype=torch.float32, device=device)
    
    # Resolve digit token IDs
    digit_token_ids = []
    for d in range(10):
        toks = model.to_tokens(str(d), prepend_bos=False)[0]
        digit_token_ids.append(toks[-1].item())
    digit_token_ids_t = torch.tensor(digit_token_ids, device=device)
    
    # ─── TEST 1: W_U neighbor structure ───
    logger.info("\n" + "="*70)
    logger.info("TEST 1: W_U NEIGHBOR STRUCTURE")
    logger.info("="*70)
    logger.info("Are backward neighbors (d-1) more similar than forward (d+1) in W_U space?")
    
    W_U = model.W_U.detach().float()  # (d_model, d_vocab)
    digit_vecs = W_U[:, digit_token_ids_t].cpu().numpy()  # (d_model, 10)
    
    # Cosine similarity between each digit and its neighbors
    from numpy.linalg import norm as np_norm
    logger.info(f"  {'Digit':>5} | {'cos(d,d+1)':>12} | {'cos(d,d-1)':>12} | {'winner':>10}")
    logger.info(f"  {'-----':>5}-+-{'------------':>12}-+-{'------------':>12}-+-{'----------':>10}")
    
    fwd_sims, bwd_sims = [], []
    for d in range(10):
        v_d = digit_vecs[:, d]
        v_fwd = digit_vecs[:, (d+1) % 10]
        v_bwd = digit_vecs[:, (d-1) % 10]
        
        cos_fwd = np.dot(v_d, v_fwd) / (np_norm(v_d) * np_norm(v_fwd) + 1e-10)
        cos_bwd = np.dot(v_d, v_bwd) / (np_norm(v_d) * np_norm(v_bwd) + 1e-10)
        fwd_sims.append(cos_fwd)
        bwd_sims.append(cos_bwd)
        
        winner = "bwd" if cos_bwd > cos_fwd else "fwd" if cos_fwd > cos_bwd else "tie"
        logger.info(f"  {d:>5} | {cos_fwd:>+12.4f} | {cos_bwd:>+12.4f} | {winner:>10}")
    
    logger.info(f"  MEAN:  fwd={np.mean(fwd_sims):+.4f}, bwd={np.mean(bwd_sims):+.4f}")
    if np.mean(bwd_sims) > np.mean(fwd_sims):
        logger.info(f"  → W_U backward neighbors are MORE similar → could explain asymmetry")
    else:
        logger.info(f"  → W_U forward neighbors are more similar → asymmetry NOT from W_U")
    
    # Also check in Fourier subspace projection
    logger.info(f"\n  W_U similarity in FOURIER SUBSPACE:")
    digit_vecs_F = basis.T @ digit_vecs  # (9, 10) — digit vecs projected onto Fourier basis
    fwd_sims_F, bwd_sims_F = [], []
    for d in range(10):
        v_d = digit_vecs_F[:, d]
        v_fwd = digit_vecs_F[:, (d+1) % 10]
        v_bwd = digit_vecs_F[:, (d-1) % 10]
        cos_fwd = np.dot(v_d, v_fwd) / (np_norm(v_d) * np_norm(v_fwd) + 1e-10)
        cos_bwd = np.dot(v_d, v_bwd) / (np_norm(v_d) * np_norm(v_bwd) + 1e-10)
        fwd_sims_F.append(cos_fwd)
        bwd_sims_F.append(cos_bwd)
    logger.info(f"  MEAN (Fourier proj): fwd={np.mean(fwd_sims_F):+.4f}, bwd={np.mean(bwd_sims_F):+.4f}")
    
    # ─── TEST 2: Delta magnitude for j=1 vs j=9 ───
    logger.info("\n" + "="*70)
    logger.info("TEST 2: ROTATION DELTA MAGNITUDE")
    logger.info("="*70)
    
    # Use the dedicated single-digit test problems
    single_digit = test_problems
    logger.info(f"  Using {len(single_digit)} single-digit problems")
    
    delta_norms = {j: [] for j in range(1, 10)}
    
    with torch.no_grad():
        for prob in single_digit[:55]:
            tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
            act = cache[hook_name][0, -1]
            del cache
            
            for j in range(1, 10):
                delta = compute_rotation_delta(act, freq_planes_torch, j, mode="coherent")
                delta_norms[j].append(delta.norm().item())
    
    logger.info(f"  {'j':>3} | {'mean ||δ||':>12} | {'target':>8} | {'j vs 10-j':>12}")
    logger.info(f"  {'---':>3}-+-{'------------':>12}-+-{'--------':>8}-+-{'------------':>12}")
    for j in range(1, 10):
        mean_dn = np.mean(delta_norms[j])
        complement = 10 - j
        target = f"d+{j}" if j <= 5 else f"d-{10-j}"
        if j < 5:
            ratio = np.mean(delta_norms[10-j]) / mean_dn if mean_dn > 0 else 0
            comp_str = f"{ratio:.3f}x"
        elif j == 5:
            comp_str = "self"
        else:
            ratio = mean_dn / np.mean(delta_norms[10-j]) if np.mean(delta_norms[10-j]) > 0 else 0
            comp_str = f"{ratio:.3f}x"
        logger.info(f"  {j:>3} | {mean_dn:>12.2f} | {target:>8} | {comp_str:>12}")
    
    # ─── TEST 3: Per-digit success via logit lens ───
    logger.info("\n" + "="*70)
    logger.info("TEST 3: PER-DIGIT SUCCESS RATES (j=1 vs j=9)")
    logger.info("="*70)
    
    # Collect clean activations
    clean_data = []  # (act, ones_digit)
    with torch.no_grad():
        for prob in single_digit[:55]:
            tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
            act = cache[hook_name][0, -1].clone()
            clean_data.append((act, prob["ones_digit"]))
            del cache
    
    digit_counts = [0] * 10
    for _, d in clean_data:
        digit_counts[d] += 1
    logger.info(f"  Answer digit distribution: {digit_counts}")
    
    for j in [1, 9]:
        logger.info(f"\n  Shift j={j} (target = d{'+' if j<=5 else ''}{j if j<=5 else j-10}):")
        per_digit_exact = [0] * 10
        per_digit_total = [0] * 10
        output_confusion = np.zeros((10, 10))  # [original_d, output_d]
        
        with torch.no_grad():
            for act, d_orig in clean_data:
                target_d = (d_orig + j) % 10
                per_digit_total[d_orig] += 1
                
                delta = compute_rotation_delta(act, freq_planes_torch, j, mode="coherent")
                rotated = act.float() + delta
                
                rot_h = rotated.unsqueeze(0).unsqueeze(0)
                rot_normed = model.ln_final(rot_h)
                rot_logits = model.unembed(rot_normed)[0, 0]
                rot_digit_logits = rot_logits[digit_token_ids_t]
                pred = rot_digit_logits.argmax().item()
                
                output_confusion[d_orig, pred] += 1
                if pred == target_d:
                    per_digit_exact[d_orig] += 1
        
        logger.info(f"  {'Orig d':>6} | {'count':>5} | {'target':>6} | {'exact':>5} | {'rate':>6} | {'top output':>15}")
        logger.info(f"  {'------':>6}-+-{'-----':>5}-+-{'------':>6}-+-{'-----':>5}-+-{'------':>6}-+-{'---------------':>15}")
        for d in range(10):
            if per_digit_total[d] == 0:
                continue
            target_d = (d + j) % 10
            rate = per_digit_exact[d] / per_digit_total[d] * 100
            # Top 2 output digits
            row = output_confusion[d]
            top2_idx = np.argsort(row)[::-1][:2]
            top2_str = ", ".join([f"{int(idx)}({int(row[idx])}x)" for idx in top2_idx if row[idx] > 0])
            logger.info(f"  {d:>6} | {per_digit_total[d]:>5} | {target_d:>6} | {per_digit_exact[d]:>5} | {rate:>5.1f}% | {top2_str:>15}")
    
    # ─── TEST 4: Actual shift distribution ───
    logger.info("\n" + "="*70)
    logger.info("TEST 4: ACTUAL SHIFT DISTRIBUTION")
    logger.info("="*70)
    logger.info("What shift does the rotation ACTUALLY produce (averaged over problems)?")
    
    for j in [1, 2, 3, 8, 9]:
        shifts = []
        with torch.no_grad():
            for act, d_orig in clean_data:
                delta = compute_rotation_delta(act, freq_planes_torch, j, mode="coherent")
                rotated = act.float() + delta
                rot_h = rotated.unsqueeze(0).unsqueeze(0)
                rot_normed = model.ln_final(rot_h)
                rot_logits = model.unembed(rot_normed)[0, 0]
                rot_digit_logits = rot_logits[digit_token_ids_t]
                pred = rot_digit_logits.argmax().item()
                actual_shift = (pred - d_orig) % 10
                shifts.append(actual_shift)
        
        shift_counts = [0] * 10
        for s in shifts:
            shift_counts[s] += 1
        target_shift = j
        logger.info(f"  j={j} (target shift={target_shift}): ")
        logger.info(f"    Shift distribution: {shift_counts}")
        logger.info(f"    Mean actual shift: {np.mean(shifts):.2f}")
        mode_shift = np.argmax(shift_counts)
        logger.info(f"    Mode shift: {mode_shift} ({shift_counts[mode_shift]}/{len(shifts)})")
    
    # ─── TEST 5: Activation norm change ───
    logger.info("\n" + "="*70)
    logger.info("TEST 5: ACTIVATION NORM CHANGE")
    logger.info("="*70)
    logger.info("Does the rotation change the activation norm? (LayerNorm is norm-sensitive)")
    
    for j in [1, 5, 9]:
        norms_before, norms_after, norm_ratios = [], [], []
        with torch.no_grad():
            for act, _ in clean_data:
                delta = compute_rotation_delta(act, freq_planes_torch, j, mode="coherent")
                rotated = act.float() + delta
                nb = act.float().norm().item()
                na = rotated.norm().item()
                norms_before.append(nb)
                norms_after.append(na)
                norm_ratios.append(na / nb if nb > 0 else 1)
        logger.info(f"  j={j}: ||act||={np.mean(norms_before):.1f}, "
                    f"||rot||={np.mean(norms_after):.1f}, "
                    f"ratio={np.mean(norm_ratios):.4f}, "
                    f"Δ%={(np.mean(norm_ratios)-1)*100:+.2f}%")


if __name__ == "__main__":
    import argparse
    from transformer_lens import HookedTransformer
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma-2b")
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--operand-range", type=int, default=30)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    
    model_name = MODEL_MAP[args.model]
    logger.info(f"Loading {model_name} on {args.device}...")
    model = HookedTransformer.from_pretrained(model_name, device=args.device)
    
    problems = generate_test_problems(max_operand=args.operand_range)
    # Match main experiment: use ALL problems for basis (not default n_test=200)
    problems = filter_correct_problems(model, problems, n_test=len(problems))
    logger.info(f"Basis problems: {len(problems)} (matching main experiment)")
    
    diagnose_sign(model, problems, args.layer)
    
    # Generate single-digit problems for asymmetry test
    sd_problems = generate_single_digit_problems()
    sd_problems = filter_correct_single_digit(model, sd_problems)
    diagnose_asymmetry(model, problems, sd_problems, args.layer)
