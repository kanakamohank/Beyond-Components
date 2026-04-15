#!/usr/bin/env python3
"""
Experiment 6: DFT of Eigenvector Digit Scores

For each model, take the unembed SVD basis (aligned with contrastive Fisher at >0.98 cosine)
and compute how each basis direction scores the 10 digits. Apply the 10-point DFT to these
scores. If eigenvectors encode Fourier modes, each direction should be dominated by a single
frequency.

Also runs at computation-zone layers using per-digit mean activations (SVD of digit means),
allowing comparison of Fourier structure at computation vs readout layers.

This is the cheapest experiment with the highest interpretability payoff:
  - Readout layer: pure linear algebra on W_U, no inference needed
  - Computation layer: requires one forward pass to collect activations

Usage:
    python eigenvector_dft.py --model gemma-2b --comp-layer 19 --device mps
    python eigenvector_dft.py --model phi-3 --comp-layer 26 --device mps
"""

import argparse
import json
import logging
import sys
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from arithmetic_circuit_scan_updated import (
    MODEL_MAP,
    VALID_PROMPT_FORMATS,
    generate_teacher_forced_problems,
    generate_single_digit_problems,
    generate_direct_answer_problems,
    filter_correct_teacher_forced,
    filter_correct_direct_answer,
    get_context_target_tok,
    get_digit_token_ids,
    compute_unembed_basis,
    log_section,
)

RESULTS_DIR = Path("mathematical_toolkit_results")
RESULTS_DIR.mkdir(exist_ok=True)


def dft_of_scores(scores_matrix: np.ndarray) -> np.ndarray:
    """
    Apply 10-point DFT to each row of scores_matrix.

    Args:
        scores_matrix: (n_dirs, 10) — each row is a digit score pattern

    Returns:
        power_spectrum: (n_dirs, 6) — power at DC + frequencies 1..5
          Column 0: DC (k=0), should be ~0 if centered
          Column 1-4: k=1..4 (each has 2 DOF: |c_k|^2 = Re^2 + Im^2)
          Column 5: k=5 Nyquist (1 DOF: real only since sin(5·2πd/10)=0)
    """
    n_dirs, N = scores_matrix.shape
    assert N == 10, f"Expected 10 digits, got {N}"

    # Full DFT
    dft = np.fft.fft(scores_matrix, axis=1)  # (n_dirs, 10) complex

    # Power at each frequency
    power = np.abs(dft) ** 2  # (n_dirs, 10)

    # Group into frequencies 0..5
    # DFT of real signal: c[k] and c[N-k] are conjugate pairs
    # Power at frequency k = |c[k]|^2 + |c[N-k]|^2 for k=1..4
    # Power at k=0 (DC) = |c[0]|^2
    # Power at k=5 (Nyquist) = |c[5]|^2
    freq_power = np.zeros((n_dirs, 6))
    freq_power[:, 0] = power[:, 0]                          # DC
    for k in range(1, 5):
        freq_power[:, k] = power[:, k] + power[:, 10 - k]   # conjugate pair
    freq_power[:, 5] = power[:, 5]                           # Nyquist

    # Parseval sanity check: sum of grouped power should equal total DFT power
    grouped_total = freq_power.sum(axis=1)
    dft_total = power.sum(axis=1)
    parseval_err = np.abs(grouped_total - dft_total).max()
    assert parseval_err < 1e-6, \
        f"[SANITY] Parseval failed: grouped vs total DFT power mismatch = {parseval_err:.2e}"

    return freq_power


def analyze_layer(label: str, digit_scores: np.ndarray, singular_values: np.ndarray):
    """
    Analyze a set of SVD directions by DFT of their digit score patterns.

    Args:
        label: e.g., "Readout (W_U)" or "Computation L19"
        digit_scores: (n_dirs, 10) — Vt from SVD, each row is how direction scores digits
        singular_values: (n_dirs,) — importance of each direction
    """
    n_dirs = min(9, digit_scores.shape[0])
    scores = digit_scores[:n_dirs]  # top 9 directions
    svals = singular_values[:n_dirs]

    # Center each row (remove DC) for cleaner Fourier analysis
    scores_centered = scores - scores.mean(axis=1, keepdims=True)

    # Sanity: centering should make row sums ~0 (tolerance for float32)
    row_sums = np.abs(scores_centered.sum(axis=1))
    assert row_sums.max() < 1e-5, \
        f"[SANITY] Centering failed: max row sum = {row_sums.max():.2e}"

    # DFT
    freq_power = dft_of_scores(scores_centered)
    # Normalize each row to get fractions
    total_power = freq_power[:, 1:].sum(axis=1, keepdims=True)  # exclude DC
    total_power = np.maximum(total_power, 1e-10)
    freq_frac = freq_power[:, 1:] / total_power  # (n_dirs, 5) for k=1..5

    freq_labels = ["k=1 (ord)", "k=2 (mod5)", "k=3", "k=4", "k=5 (par)"]

    logger.info(f"\n{'='*70}")
    logger.info(f"  {label}")
    logger.info(f"{'='*70}")
    logger.info(f"  {'Dir':>4}  {'σ':>8}  {'k=1':>7}  {'k=2':>7}  {'k=3':>7}  {'k=4':>7}  {'k=5':>7}  {'Dominant':>12}")
    logger.info(f"  {'─'*65}")

    results = []
    for i in range(n_dirs):
        dominant_k = np.argmax(freq_frac[i]) + 1  # +1 because index 0 = k=1
        dominant_pct = freq_frac[i, dominant_k - 1] * 100

        logger.info(
            f"  {i+1:>4}  {svals[i]:>8.2f}  "
            f"{freq_frac[i,0]*100:>6.1f}%  {freq_frac[i,1]*100:>6.1f}%  "
            f"{freq_frac[i,2]*100:>6.1f}%  {freq_frac[i,3]*100:>6.1f}%  "
            f"{freq_frac[i,4]*100:>6.1f}%  "
            f"k={dominant_k} ({dominant_pct:.0f}%)"
        )

        results.append({
            "direction": i + 1,
            "singular_value": float(svals[i]),
            "freq_fractions": {str(k): float(freq_frac[i, k-1]) for k in range(1, 6)},
            "dominant_freq": int(dominant_k),
            "dominant_pct": float(dominant_pct),
            "digit_scores": scores[i].tolist(),
        })

    # Summary: count how many directions are dominated by each frequency
    dom_counts = defaultdict(int)
    for r in results:
        dom_counts[r["dominant_freq"]] += 1

    logger.info(f"\n  Frequency assignment summary:")
    for k in range(1, 6):
        dof = 1 if k == 5 else 2
        expected = dof  # in a perfect Fourier basis, k=1..4 get 2 dirs each, k=5 gets 1
        actual = dom_counts.get(k, 0)
        match = "✓" if actual == expected else f"(expected {expected})"
        logger.info(f"    k={k}: {actual} directions {match}")

    # Check: in a perfect Fourier basis, directions come in cos/sin pairs
    # for k=1..4, and a single direction for k=5. Total = 2+2+2+2+1 = 9.
    perfect_match = all(
        dom_counts.get(k, 0) == (1 if k == 5 else 2)
        for k in range(1, 6)
    )
    if perfect_match:
        logger.info(f"\n  ★ PERFECT FOURIER BASIS: directions pair exactly with frequencies!")
    else:
        logger.info(f"\n  Directions do not form a perfect Fourier basis")

    return results, perfect_match


def collect_per_digit_means(model, problems, layer, device):
    """Collect per-digit mean activations at a given layer."""
    hook_name = f"blocks.{layer}.hook_resid_post"
    d_model = model.cfg.d_model

    digit_acts = defaultdict(list)
    for prob in problems:
        digit = prob["ones_digit"]
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

    # Per-digit means → (10, d_model)
    means = np.zeros((10, d_model))
    for d in range(10):
        if digit_acts[d]:
            means[d] = np.mean(digit_acts[d], axis=0)
            logger.info(f"    Digit {d}: {len(digit_acts[d])} samples")
        else:
            logger.warning(f"    Digit {d}: NO samples!")

    return means


def main():
    parser = argparse.ArgumentParser(description="Experiment 6: DFT of eigenvector digit scores")
    parser.add_argument("--model", default="gemma-2b", choices=list(MODEL_MAP.keys()))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--comp-layer", type=int, default=None,
                        help="Computation-zone layer for activation-based analysis")
    parser.add_argument("--readout-layer", type=int, default=None,
                        help="Readout layer (default: auto from model)")
    parser.add_argument("--prompt-format", default="calculate",
                        choices=VALID_PROMPT_FORMATS,
                        help="Prompt format: calculate, fewshot, minimal, qa, nospace")
    parser.add_argument("--direct-answer", action="store_true",
                        help="Use direct-answer mode (for instruct models that predict full answer as single token)")
    args = parser.parse_args()

    model_name = MODEL_MAP[args.model]
    device = args.device

    # Default readout and computation layers
    readout_defaults = {"gemma-2b": 25, "phi-3": 31, "llama-3b": 27, "llama-3b-it": 27}
    comp_defaults = {"gemma-2b": 19, "phi-3": 26, "llama-3b": 20, "llama-3b-it": 20}

    readout_layer = args.readout_layer or readout_defaults.get(args.model, 24)
    comp_layer = args.comp_layer or comp_defaults.get(args.model, 20)

    prompt_format = args.prompt_format

    logger.info(f"Model: {args.model} ({model_name})")
    logger.info(f"Readout layer: L{readout_layer}, Computation layer: L{comp_layer}")
    logger.info(f"Device: {device}")
    logger.info(f"Prompt format: {prompt_format}")

    # ── Load model ────────────────────────────────────────────────────────
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=torch.float32 if device == "cpu" else torch.float16,
    )
    model.eval()

    all_results = {
        "model": model_name,
        "model_short": args.model,
        "readout_layer": readout_layer,
        "comp_layer": comp_layer,
        "prompt_format": prompt_format,
    }

    # ══════════════════════════════════════════════════════════════════════
    # PART 1: Readout layer — DFT of unembed SVD directions
    # No inference needed, just W_U linear algebra
    # ══════════════════════════════════════════════════════════════════════
    log_section("PART 1: READOUT LAYER (Unembed SVD)")

    digit_ids = get_digit_token_ids(model, prompt_format=prompt_format)

    W_U = model.W_U.detach().float().cpu()          # (d_model, vocab)
    W_digits = W_U[:, digit_ids].numpy()             # (d_model, 10)

    # Center: subtract centroid across 10 digit columns
    centroid = W_digits.mean(axis=1, keepdims=True)  # (d_model, 1)
    W_centered = W_digits - centroid                  # (d_model, 10)

    # Sanity: each row should now have zero mean across 10 digits
    row_means = np.abs(W_centered.mean(axis=1))
    assert row_means.max() < 1e-5, \
        f"[SANITY] W_U centering failed: max row mean = {row_means.max():.2e}"
    logger.info(f"  [SANITY] W_U centering ✓ (max row mean = {row_means.max():.2e})")

    # SVD
    U, S, Vt = np.linalg.svd(W_centered, full_matrices=False)
    # Vt: (10, 10) — rows are digit score patterns of each SVD direction
    # S: (10,) — singular values

    logger.info(f"  Unembed SVD singular values: {S[:9].round(2)}")
    logger.info(f"  Singular value ratios: σ₁/σ₂={S[0]/S[1]:.2f}, σ₁/σ₉={S[0]/S[8]:.2f}")

    readout_results, readout_perfect = analyze_layer(
        f"Readout Layer (Unembed W_U SVD)",
        Vt, S,
    )
    all_results["readout"] = {
        "singular_values": S[:9].tolist(),
        "directions": readout_results,
        "perfect_fourier": readout_perfect,
    }

    # ══════════════════════════════════════════════════════════════════════
    # PART 2: Computation-zone layer — DFT of activation SVD directions
    # Requires inference to collect per-digit mean activations
    # ══════════════════════════════════════════════════════════════════════
    log_section(f"PART 2: COMPUTATION LAYER L{comp_layer} (Activation SVD)")

    # Generate and filter problems
    correct = []
    if args.direct_answer:
        logger.info("  Generating direct-answer problems (instruct model mode)...")
        problems, _ = generate_direct_answer_problems(n_per_digit=100)
        correct = filter_correct_direct_answer(model, problems, max_n=1500)
    else:
        logger.info("  Generating teacher-forced problems...")
        try:
            problems, _ = generate_teacher_forced_problems(n_per_digit=100, prompt_format=prompt_format)
            correct = filter_correct_teacher_forced(model, problems, max_n=1500)
        except Exception as e:
            logger.warning(f"  Teacher-forced generation failed ({e})")

    # Check if we got enough per digit
    by_digit = defaultdict(list)
    for p in correct:
        by_digit[p["ones_digit"]].append(p)
    min_count = min(len(by_digit.get(d, [])) for d in range(10))

    if min_count < 5 and not args.direct_answer:
        logger.warning(
            f"  Multi-digit got only {min_count}/digit "
            f"(counts: {[len(by_digit.get(d, [])) for d in range(10)]}). "
            f"Falling back to single-digit problems..."
        )
        problems = generate_single_digit_problems(prompt_format=prompt_format)
        correct = filter_correct_teacher_forced(model, problems, max_n=500)
        by_digit = defaultdict(list)
        for p in correct:
            by_digit[p["ones_digit"]].append(p)
        min_count = min(len(by_digit.get(d, [])) for d in range(10))

    assert min_count >= 1, \
        f"[SANITY] Insufficient data: min digit count = {min_count} (need ≥1). " \
        f"Per-digit counts: {[len(by_digit.get(d, [])) for d in range(10)]}"
    if min_count < 5:
        logger.warning(f"  ⚠ Low data: {min_count}/digit. Results may be noisy.")
    logger.info(f"  Balanced: {min_count} per digit ({min_count * 10} total)")

    balanced = []
    for d in range(10):
        balanced.extend(by_digit[d][:min_count])

    # Collect per-digit means
    logger.info(f"  Collecting activations at L{comp_layer}...")
    digit_means = collect_per_digit_means(model, balanced, comp_layer, device)

    # Center and SVD
    centroid_comp = digit_means.mean(axis=0, keepdims=True)  # (1, d_model)
    M_centered = digit_means - centroid_comp                  # (10, d_model)

    # Sanity: centering should zero out the column means
    col_means = np.abs(M_centered.mean(axis=0))
    assert col_means.max() < 1e-6, \
        f"[SANITY] Activation centering failed: max col mean = {col_means.max():.2e}"

    U_comp, S_comp, Vt_comp = np.linalg.svd(M_centered, full_matrices=False)
    # M_centered = U_comp @ diag(S_comp) @ Vt_comp
    # U_comp: (10, 10) — COLUMNS are digit loading patterns per direction
    #   U_comp[:, j] = how 10 digits load onto direction j
    # Vt_comp: (10, d_model) — rows are activation-space directions
    # S_comp: (10,) — singular values
    #
    # For DFT analysis, we need digit score patterns = U_comp TRANSPOSED
    # so that row j = U_comp[:, j] = scores of 10 digits for direction j

    logger.info(f"  Activation SVD singular values: {S_comp[:9].round(2)}")

    comp_results, comp_perfect = analyze_layer(
        f"Computation Layer L{comp_layer} (Activation Mean SVD)",
        U_comp.T, S_comp,  # TRANSPOSE: rows of U_comp.T = columns of U_comp = digit patterns
    )
    all_results["computation"] = {
        "layer": comp_layer,
        "singular_values": S_comp[:9].tolist(),
        "directions": comp_results,
        "perfect_fourier": comp_perfect,
        "n_per_digit": min_count,
    }

    # ══════════════════════════════════════════════════════════════════════
    # PART 3: Also check readout-layer activations (not just W_U)
    # ══════════════════════════════════════════════════════════════════════
    log_section(f"PART 3: READOUT LAYER L{readout_layer} (Activation SVD)")

    logger.info(f"  Collecting activations at L{readout_layer}...")
    digit_means_readout = collect_per_digit_means(model, balanced, readout_layer, device)

    centroid_ro = digit_means_readout.mean(axis=0, keepdims=True)
    M_ro = digit_means_readout - centroid_ro

    U_ro, S_ro, Vt_ro = np.linalg.svd(M_ro, full_matrices=False)

    logger.info(f"  Activation SVD singular values: {S_ro[:9].round(2)}")

    ro_act_results, ro_act_perfect = analyze_layer(
        f"Readout Layer L{readout_layer} (Activation Mean SVD)",
        U_ro.T, S_ro,  # TRANSPOSE: same fix as computation layer
    )
    all_results["readout_activations"] = {
        "layer": readout_layer,
        "singular_values": S_ro[:9].tolist(),
        "directions": ro_act_results,
        "perfect_fourier": ro_act_perfect,
    }

    # ══════════════════════════════════════════════════════════════════════
    # COMPARISON SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    log_section("COMPARISON: Computation vs Readout")

    for layer_label, results in [
        ("Readout (W_U)", readout_results),
        (f"Readout L{readout_layer} (activations)", ro_act_results),
        (f"Computation L{comp_layer} (activations)", comp_results),
    ]:
        dom_freqs = [r["dominant_freq"] for r in results]
        dom_pcts = [r["dominant_pct"] for r in results]
        mean_purity = np.mean(dom_pcts)
        logger.info(
            f"  {layer_label:>40}: "
            f"dominant freqs = {dom_freqs}  "
            f"mean purity = {mean_purity:.1f}%"
        )

    # Save results
    fmt_suffix = f"_{prompt_format}" if prompt_format != "calculate" else ""
    if args.direct_answer:
        fmt_suffix += "_direct"
    out_path = RESULTS_DIR / f"eigenvector_dft_{args.model}{fmt_suffix}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
