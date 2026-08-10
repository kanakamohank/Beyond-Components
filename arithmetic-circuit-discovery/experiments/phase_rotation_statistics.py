#!/usr/bin/env python3
"""
Statistical analysis for Fourier phase rotation experiment results.

Computes:
1. Bootstrap 95% CIs for exact_mod10 rates
2. Permutation tests: COHERENT vs TOP2_PCA (frequency-specific vs non-specific)
3. Binomial tests against chance level
"""

import json
import argparse
import logging
import numpy as np
from pathlib import Path
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

RESULT_DIR = Path(__file__).parent.parent / "mathematical_toolkit_results"

# Latest result files (elliptical rotation fix, v4)
RESULT_FILES = {
    "gemma-2b": RESULT_DIR / "fourier_phase_rotation_gemma-2b_20260415_225040.json",
    "phi-3":    RESULT_DIR / "fourier_phase_rotation_phi-3-mini_20260415_232912.json",
    "llama-3b": RESULT_DIR / "fourier_phase_rotation_llama-3b_direct_20260416_044838.json",
}

MODES = ["coherent", "k1_only", "top2_pca"]


def load_results(path):
    """Load JSON results file."""
    with open(path) as f:
        return json.load(f)


def extract_per_problem_outcomes(per_shift_data):
    """
    Extract binary outcome arrays from per-shift data.
    Returns dict: shift_j -> (exact_outcomes, changed_outcomes) each of length n_single.
    Also returns pooled arrays across all shifts.
    """
    exact_pooled = []
    changed_pooled = []
    per_shift = {}

    for j_str, data in per_shift_data.items():
        j = int(j_str)
        n = data["n_single"]
        n_exact = data["n_exact"]
        n_changed = data["n_changed"]

        # Binary outcome arrays: 1 if exact/changed, 0 otherwise
        exact_arr = np.array([1]*n_exact + [0]*(n - n_exact))
        changed_arr = np.array([1]*n_changed + [0]*(n - n_changed))
        per_shift[j] = (exact_arr, changed_arr)
        exact_pooled.extend(exact_arr)
        changed_pooled.extend(changed_arr)

    return per_shift, np.array(exact_pooled), np.array(changed_pooled)


def bootstrap_ci(outcomes, n_boot=10000, ci=0.95, rng=None):
    """
    Bootstrap confidence interval for the mean of binary outcomes.
    Returns (point_estimate, ci_low, ci_high).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(outcomes)
    point = np.mean(outcomes)

    if n == 0:
        return 0.0, 0.0, 0.0

    boot_means = np.array([
        np.mean(rng.choice(outcomes, size=n, replace=True))
        for _ in range(n_boot)
    ])

    alpha = (1 - ci) / 2
    ci_low = np.percentile(boot_means, 100 * alpha)
    ci_high = np.percentile(boot_means, 100 * (1 - alpha))

    return point, ci_low, ci_high


def permutation_test(outcomes_a, outcomes_b, n_perm=10000, rng=None):
    """
    Two-sample permutation test for difference in means.
    H0: mean(A) = mean(B)
    H1: mean(A) > mean(B) (one-sided)
    Returns (observed_diff, p_value).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    obs_diff = np.mean(outcomes_a) - np.mean(outcomes_b)
    pooled = np.concatenate([outcomes_a, outcomes_b])
    n_a = len(outcomes_a)
    n_total = len(pooled)

    # Vectorized: generate all permutation indices at once
    perm_indices = np.array([rng.permutation(n_total) for _ in range(n_perm)])
    perm_a_means = np.array([np.mean(pooled[idx[:n_a]]) for idx in perm_indices])
    perm_b_means = np.array([np.mean(pooled[idx[n_a:]]) for idx in perm_indices])
    perm_diffs = perm_a_means - perm_b_means

    count_ge = np.sum(perm_diffs >= obs_diff)
    p_value = (count_ge + 1) / (n_perm + 1)  # +1 for continuity correction
    return obs_diff, p_value


def binomial_test_vs_chance(n_exact, n_total, chance_rate=1/9):
    """
    One-sided binomial test: is exact_mod10 rate > chance?
    chance_rate = 1/9 ≈ 0.111 (random shift among j=1..9 gives correct digit 1/9 of the time
    if the model outputs a uniform random digit when disrupted).

    Actually, for a random prediction among 10 digits, P(exact match) = 1/10 = 0.1.
    But the shift is designed to produce a SPECIFIC digit, so chance = 1/10 per problem.
    Over 9 shifts × n problems, chance = 1/10 per test.
    """
    # Use 1/10 = 0.10 as chance (random digit prediction)
    result = stats.binomtest(n_exact, n_total, p=0.10, alternative='greater')
    return result.pvalue


def analyze_model(model_name, data, rng):
    """Analyze results for one model."""
    log.info(f"\n{'='*70}")
    log.info(f"  {model_name.upper()}")
    log.info(f"{'='*70}")
    log.info(f"  n_test_problems: {data['n_test_problems']}")
    log.info(f"  operand_range: {data['operand_range']}")

    results = {}

    for layer_key, layer_data in data["experiments"].items():
        log.info(f"\n  --- {layer_key} ---")

        if not layer_data.get("sanity_passed", False):
            log.info(f"    Sanity checks failed — skipping")
            continue

        layer_results = {}

        # Extract outcomes for each mode
        mode_outcomes = {}
        for mode in MODES:
            ps_data = layer_data["phase_shift"][mode]["per_shift"]
            per_shift, exact_pooled, changed_pooled = extract_per_problem_outcomes(ps_data)
            mode_outcomes[mode] = {
                "per_shift": per_shift,
                "exact_pooled": exact_pooled,
                "changed_pooled": changed_pooled,
            }

        # 1. Bootstrap CIs
        log.info(f"\n    [1] Bootstrap 95% CIs (exact_mod10 rate):")
        for mode in MODES:
            exact = mode_outcomes[mode]["exact_pooled"]
            point, ci_lo, ci_hi = bootstrap_ci(exact, n_boot=10000, rng=rng)
            n_exact = int(np.sum(exact))
            n_total = len(exact)
            layer_results[f"{mode}_exact_ci"] = (point, ci_lo, ci_hi)
            log.info(f"      {mode:12s}: {point*100:5.1f}% [{ci_lo*100:5.1f}%, {ci_hi*100:5.1f}%]  ({n_exact}/{n_total})")

        # Also bootstrap CIs for digit_changed
        log.info(f"\n    [1b] Bootstrap 95% CIs (digit_changed rate):")
        for mode in MODES:
            changed = mode_outcomes[mode]["changed_pooled"]
            point, ci_lo, ci_hi = bootstrap_ci(changed, n_boot=10000, rng=rng)
            layer_results[f"{mode}_changed_ci"] = (point, ci_lo, ci_hi)
            log.info(f"      {mode:12s}: {point*100:5.1f}% [{ci_lo*100:5.1f}%, {ci_hi*100:5.1f}%]")

        # 2. Permutation test: COHERENT vs TOP2_PCA
        log.info(f"\n    [2] Permutation test (COHERENT vs TOP2_PCA):")

        # exact_mod10
        coh_exact = mode_outcomes["coherent"]["exact_pooled"]
        pca_exact = mode_outcomes["top2_pca"]["exact_pooled"]
        diff_exact, p_exact = permutation_test(coh_exact, pca_exact, n_perm=10000, rng=rng)
        layer_results["perm_exact_diff"] = diff_exact
        layer_results["perm_exact_p"] = p_exact
        sig_exact = "***" if p_exact < 0.001 else "**" if p_exact < 0.01 else "*" if p_exact < 0.05 else "ns"
        log.info(f"      exact_mod10: Δ={diff_exact*100:+.1f}pp, p={p_exact:.4f} {sig_exact}")

        # digit_changed
        coh_chg = mode_outcomes["coherent"]["changed_pooled"]
        pca_chg = mode_outcomes["top2_pca"]["changed_pooled"]
        diff_chg, p_chg = permutation_test(coh_chg, pca_chg, n_perm=10000, rng=rng)
        layer_results["perm_changed_diff"] = diff_chg
        layer_results["perm_changed_p"] = p_chg
        sig_chg = "***" if p_chg < 0.001 else "**" if p_chg < 0.01 else "*" if p_chg < 0.05 else "ns"
        log.info(f"      digit_changed: Δ={diff_chg*100:+.1f}pp, p={p_chg:.4f} {sig_chg}")

        # 3. Binomial test: is COHERENT exact_mod10 > chance (10%)?
        log.info(f"\n    [3] Binomial test (COHERENT exact_mod10 vs 10% chance):")
        n_exact_coh = int(np.sum(coh_exact))
        n_total_coh = len(coh_exact)
        p_binom = binomial_test_vs_chance(n_exact_coh, n_total_coh, chance_rate=0.10)
        layer_results["binom_p"] = p_binom
        sig_binom = "***" if p_binom < 0.001 else "**" if p_binom < 0.01 else "*" if p_binom < 0.05 else "ns"
        log.info(f"      {n_exact_coh}/{n_total_coh} exact = {n_exact_coh/n_total_coh*100:.1f}% vs 10.0% chance: p={p_binom:.4f} {sig_binom}")

        # 4. Per-shift breakdown for COHERENT (bootstrap CI per shift)
        log.info(f"\n    [4] Per-shift exact_mod10 CIs (COHERENT):")
        for j in sorted(mode_outcomes["coherent"]["per_shift"].keys()):
            exact_j, _ = mode_outcomes["coherent"]["per_shift"][j]
            pt, lo, hi = bootstrap_ci(exact_j, n_boot=5000, rng=rng)
            n_ex = int(np.sum(exact_j))
            n_tot = len(exact_j)
            log.info(f"      j={j}: {pt*100:5.1f}% [{lo*100:5.1f}%, {hi*100:5.1f}%]  ({n_ex}/{n_tot})")

        results[layer_key] = layer_results

    return results


def print_cross_model_summary(all_results):
    """Print cross-model comparison table."""
    log.info(f"\n{'='*70}")
    log.info(f"  CROSS-MODEL SUMMARY")
    log.info(f"{'='*70}")
    log.info(f"\n  {'Model':<12} {'Layer':<6} {'COHERENT exact%':>16} {'95% CI':>20} {'vs PCA p':>10} {'vs chance p':>12}")
    log.info(f"  {'-'*12} {'-'*6} {'-'*16} {'-'*20} {'-'*10} {'-'*12}")

    for model_name, model_results in all_results.items():
        for layer_key, lr in model_results.items():
            pt, lo, hi = lr["coherent_exact_ci"]
            p_perm = lr["perm_exact_p"]
            p_binom = lr["binom_p"]
            sig_perm = "***" if p_perm < 0.001 else "**" if p_perm < 0.01 else "*" if p_perm < 0.05 else "ns"
            sig_binom = "***" if p_binom < 0.001 else "**" if p_binom < 0.01 else "*" if p_binom < 0.05 else "ns"
            log.info(f"  {model_name:<12} {layer_key:<6} {pt*100:>14.1f}%  [{lo*100:5.1f}%, {hi*100:5.1f}%]  {p_perm:>8.4f}{sig_perm:>3s} {p_binom:>10.4f}{sig_binom:>3s}")


def main():
    parser = argparse.ArgumentParser(description="Statistical analysis of phase rotation results")
    parser.add_argument("--n-boot", type=int, default=10000, help="Number of bootstrap samples")
    parser.add_argument("--n-perm", type=int, default=10000, help="Number of permutations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    log.info("=" * 70)
    log.info("  FOURIER PHASE ROTATION — STATISTICAL ANALYSIS")
    log.info("=" * 70)
    log.info(f"  Bootstrap samples: {args.n_boot}")
    log.info(f"  Permutation tests: {args.n_perm}")
    log.info(f"  Random seed: {args.seed}")

    all_results = {}

    for model_name, path in RESULT_FILES.items():
        if not path.exists():
            log.warning(f"  {model_name}: result file not found at {path}")
            continue

        data = load_results(path)
        results = analyze_model(model_name, data, rng)
        all_results[model_name] = results

    if all_results:
        print_cross_model_summary(all_results)

    # Save results
    out_path = RESULT_DIR / "fourier_phase_rotation_statistics.json"
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tuple):
            return list(obj)
        return obj

    serializable = {}
    for model, model_res in all_results.items():
        serializable[model] = {}
        for layer, layer_res in model_res.items():
            serializable[model][layer] = {
                k: convert(v) for k, v in layer_res.items()
            }

    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=convert)
    log.info(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
