import torch
import numpy as np
import gc
from transformer_lens import HookedTransformer

# -----------------------------------------------------------------------
# CONSTANTS & PROMPTS
# -----------------------------------------------------------------------
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
COMPASS_LAYER = 24
COMPASS_HEAD = 28
COMPASS_DIMS = (3, 7)  # Validated compass dims from svd_stats_ov_helix_circuit.md

# Semantic Categories for Validation (expanded to 5 prompts/category for reliability)
PROMPTS = {
    "Operations": [
        "The software company launched a new program",
        "The scientists conducted several successful experiments",
        "The military unit commenced their tactical operations",
        "The hospital began a new outreach program",
        "The factory streamlined its manufacturing operations",
    ],
    "Teams": [
        "The championship was won by the rival teams",
        "The presentation was delivered by the management team",
        "The community organized into small neighborhood groups",
        "The project was completed by cross-functional teams",
        "The volunteers formed several cleanup groups",
    ],
    "Tools": [
        "The carpenter organized his workbench and tools",
        "The mechanic reached for the heavy duty tools",
        "The digital parser broke the text into individual tokens",
        "The surgeon carefully sterilized the surgical tools",
        "The programmer debugged using diagnostic tools",
    ],
    "Geography": [
        "The explorers mapped out the uncharted territory",
        "The wildlife reserve covers several large areas",
        "The climate varies significantly across these regions",
        "The borders divided the ancient territory",
        "The drought affected the surrounding areas",
    ]
}

# -----------------------------------------------------------------------
# HELPER: Circular spread (handles wraparound)
# -----------------------------------------------------------------------
def circular_spread(angles_deg):
    """Compute circular standard deviation of angles in degrees.
    Uses the mean resultant length method from circular statistics."""
    rads = np.deg2rad(angles_deg)
    C = np.mean(np.cos(rads))
    S = np.mean(np.sin(rads))
    R = np.sqrt(C**2 + S**2)  # mean resultant length, 0=uniform, 1=identical
    # Circular std: sqrt(-2 * ln(R)), in degrees
    if R > 1e-9:
        circ_std = np.degrees(np.sqrt(-2 * np.log(R)))
    else:
        circ_std = 180.0  # maximum dispersion
    return circ_std

def circular_mean(angles_deg):
    """Compute circular mean of angles in degrees."""
    rads = np.deg2rad(angles_deg)
    mean_rad = np.arctan2(np.mean(np.sin(rads)), np.mean(np.cos(rads)))
    return float(np.degrees(mean_rad) % 360)

# -----------------------------------------------------------------------
# EXPERIMENT 1: LayerNorm Angle Preservation Test
# -----------------------------------------------------------------------
def test_layernorm_preservation(model):
    print(f"\n{'=' * 75}")
    print(f"EXPERIMENT 1: LayerNorm Angle Preservation Test")
    print(f"Claim: Angles survive LayerNorm; magnitudes don't.")
    print(f"Target: L{COMPASS_LAYER} H{COMPASS_HEAD}, SVD Dims {COMPASS_DIMS}")
    print(f"{'=' * 75}")

    # Extract SVD Reading Subspace for the known compass
    W_V = model.W_V[COMPASS_LAYER, COMPASS_HEAD].detach().float().cpu()
    W_O = model.W_O[COMPASS_LAYER, COMPASS_HEAD].detach().float().cpu()
    W_OV = W_V @ W_O
    U, _, _ = torch.linalg.svd(W_OV, full_matrices=False)

    u1 = U[:, COMPASS_DIMS[0]]
    u2 = U[:, COMPASS_DIMS[1]]

    hook_pre = f"blocks.{COMPASS_LAYER}.hook_resid_pre"
    hook_post_ln = f"blocks.{COMPASS_LAYER}.ln1.hook_normalized"

    print(f"\n  {'Category':<12} {'Token':<12} | {'Pre-LN°':<9} {'Post-LN°':<10} {'Δ Angle':<8} | {'Mag Pre':<9} {'Mag Post':<9} {'Mag Ratio'}")
    print(f"  {'-' * 95}")

    all_deltas = []
    all_mag_ratios = []

    for category, prompt_list in PROMPTS.items():
        for prompt in prompt_list:
            tokens = model.to_tokens(prompt)
            last_token_str = model.tokenizer.decode([tokens[0, -1].item()]).strip()

            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=[hook_pre, hook_post_ln])

            # 1. Pre-LayerNorm Projections
            resid_pre = cache[hook_pre][0, -1, :].cpu().float()
            c1_pre, c2_pre = (resid_pre @ u1).item(), (resid_pre @ u2).item()
            angle_pre = np.degrees(np.arctan2(c2_pre, c1_pre)) % 360
            mag_pre = np.sqrt(c1_pre**2 + c2_pre**2)

            # 2. Post-LayerNorm Projections
            resid_post = cache[hook_post_ln][0, -1, :].cpu().float()
            c1_post, c2_post = (resid_post @ u1).item(), (resid_post @ u2).item()
            angle_post = np.degrees(np.arctan2(c2_post, c1_post)) % 360
            mag_post = np.sqrt(c1_post**2 + c2_post**2)

            # 3. Deltas
            angle_diff = abs(angle_pre - angle_post)
            angle_diff = min(angle_diff, 360 - angle_diff)  # Handle wraparound
            mag_ratio = mag_post / (mag_pre + 1e-9)

            all_deltas.append(angle_diff)
            all_mag_ratios.append(mag_ratio)

            print(f"  {category:<12} {last_token_str:<12} | {angle_pre:>6.1f}°  {angle_post:>6.1f}°    {angle_diff:>5.2f}°   | {mag_pre:>7.3f}  {mag_post:>7.3f}  {mag_ratio:>7.4f}x")

            del cache

    # Summary statistics
    print(f"\n  {'=' * 75}")
    print(f"  EXPERIMENT 1 SUMMARY")
    print(f"  {'=' * 75}")
    print(f"  Angle deltas:    mean={np.mean(all_deltas):.2f}°, max={np.max(all_deltas):.2f}°, std={np.std(all_deltas):.2f}°")
    print(f"  Magnitude ratios: mean={np.mean(all_mag_ratios):.4f}x, std={np.std(all_mag_ratios):.4f}")
    print()
    # Check normalization type
    norm_type = model.cfg.normalization_type
    print(f"  Normalization type: {norm_type}")
    if 'RMS' in str(norm_type):
        print(f"  NOTE: {norm_type} is pure scalar scaling → angles are EXACTLY preserved (mathematical identity)")
        print(f"        Standard LayerNorm (with mean subtraction) would show approximate, not exact, preservation.")
        print(f"        The architecture CHOICE of RMSNorm guarantees angle invariance.")
    print()
    if np.mean(all_deltas) < 5.0:
        print(f"  --> CONFIRMED: Normalization preserves angles (mean Δ={np.mean(all_deltas):.2f}°)")
        print(f"      Magnitudes change by {np.mean(all_mag_ratios):.4f}x on average")
        print(f"      Angles are the ONLY information surviving normalization in this subspace")
    elif np.mean(all_deltas) < 20.0:
        print(f"  --> PARTIAL: LayerNorm introduces moderate angle shift (mean Δ={np.mean(all_deltas):.2f}°)")
        print(f"      The preservation is imperfect; other mechanisms may contribute")
    else:
        print(f"  --> REJECTED: LayerNorm significantly alters angles (mean Δ={np.mean(all_deltas):.2f}°)")
        print(f"      The angle-preservation theory needs revision")

# -----------------------------------------------------------------------
# HELPER: Fisher Circular Discriminability
# -----------------------------------------------------------------------
def fisher_circular_discriminability(layer_cat_angles, layer, categories):
    """Compute Fisher-like discriminability on the circle.
    Returns: between-class variance / within-class variance.
    High values = categories are separable; low = they overlap."""
    # Collect within-class circular variances
    within_vars = []
    cat_means = []
    for cat in categories:
        angles = layer_cat_angles[layer][cat]
        cm = circular_mean(angles)
        cat_means.append(cm)
        cs = circular_spread(angles)
        within_vars.append(cs ** 2)  # variance = std^2

    mean_within_var = np.mean(within_vars)
    between_var = circular_spread(cat_means) ** 2

    fisher = between_var / max(mean_within_var, 1.0)  # avoid div-by-zero
    return fisher, circular_spread(cat_means), np.mean([np.sqrt(v) for v in within_vars]), cat_means

# -----------------------------------------------------------------------
# EXPERIMENT 2 (CORRECTED): Full Depth Profile of Semantic Separation
# Fixes from review:
#   1. Uses Fisher discriminability (between/within variance ratio)
#   2. Includes random-shuffle baseline to expose early-layer artifacts
#   3. Expanded to 5 prompts per category for reliability
# -----------------------------------------------------------------------
def test_full_depth_semantic_profile(model):
    print(f"\n{'=' * 75}")
    print(f"EXPERIMENT 2 (CORRECTED): Full Depth Profile")
    print(f"Fixed projection: L{COMPASS_LAYER} H{COMPASS_HEAD} SVD dims {COMPASS_DIMS}")
    print(f"Metric: Fisher discriminability (between-class / within-class variance)")
    print(f"Baseline: random category shuffle (same tokens, scrambled labels)")
    print(f"{'=' * 75}")

    # Extract FIXED reading directions from L24 H28
    W_V = model.W_V[COMPASS_LAYER, COMPASS_HEAD].detach().float().cpu()
    W_O = model.W_O[COMPASS_LAYER, COMPASS_HEAD].detach().float().cpu()
    W_OV = W_V @ W_O
    U, _, _ = torch.linalg.svd(W_OV, full_matrices=False)
    u1 = U[:, COMPASS_DIMS[0]]
    u2 = U[:, COMPASS_DIMS[1]]

    # Pre-tokenize all prompts
    categories = list(PROMPTS.keys())
    all_tokens = {}
    n_per_cat = None
    for category, prompt_list in PROMPTS.items():
        all_tokens[category] = [model.to_tokens(p) for p in prompt_list]
        n_per_cat = len(prompt_list)

    all_hook_names = [f"blocks.{l}.hook_resid_pre" for l in range(model.cfg.n_layers)]
    n_prompts = len(categories) * n_per_cat
    print(f"\n  Running {n_prompts} prompts ({n_per_cat}/category), all {model.cfg.n_layers} layers...")

    # Collect angles: layer -> category -> [angles]
    layer_cat_angles = {l: {cat: [] for cat in categories} for l in range(model.cfg.n_layers)}

    for category in categories:
        for tokens in all_tokens[category]:
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=all_hook_names)
            for layer in range(model.cfg.n_layers):
                resid = cache[f"blocks.{layer}.hook_resid_pre"][0, -1, :].cpu().float()
                c1 = (resid @ u1).item()
                c2 = (resid @ u2).item()
                angle_deg = np.degrees(np.arctan2(c2, c1)) % 360
                layer_cat_angles[layer][category].append(angle_deg)
            del cache
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Compute random baseline: shuffle all angles into fake categories
    # (same angles, random grouping → measures chance-level Fisher)
    n_shuffles = 50
    rng = np.random.default_rng(42)

    print(f"\n  {'Layer':<6} | {'Fisher':<8} | {'Baseline':<10} | {'Ratio':<7} | {'Btwn-Std':<10} | {'Within-Std':<11} | {'Verdict'}")
    print(f"  {'-' * 85}")

    results = []
    for layer in range(model.cfg.n_layers):
        fisher, between_std, within_std, cat_means = fisher_circular_discriminability(
            layer_cat_angles, layer, categories)

        # Random baseline: pool all angles, shuffle into 4 groups
        all_angles = []
        for cat in categories:
            all_angles.extend(layer_cat_angles[layer][cat])
        all_angles = np.array(all_angles)

        baseline_fishers = []
        for _ in range(n_shuffles):
            shuffled = rng.permutation(all_angles)
            fake_layer = {layer: {}}
            for i, cat in enumerate(categories):
                fake_layer[layer][cat] = shuffled[i*n_per_cat:(i+1)*n_per_cat].tolist()
            bf, _, _, _ = fisher_circular_discriminability(fake_layer, layer, categories)
            baseline_fishers.append(bf)
        baseline_mean = np.mean(baseline_fishers)

        ratio = fisher / max(baseline_mean, 0.01)
        results.append((layer, fisher, baseline_mean, ratio, between_std, within_std))

        # Verdict based on ratio over baseline
        if ratio > 3.0:
            verdict = "🔥 STRONG (>>baseline)"
        elif ratio > 1.5:
            verdict = "✦  ABOVE baseline"
        elif ratio > 1.0:
            verdict = "·  ~baseline"
        else:
            verdict = "   BELOW baseline"

        compass = " ◀" if layer == COMPASS_LAYER else ""
        print(f"  L{layer:<4} | {fisher:>6.2f} | {baseline_mean:>8.2f}   | {ratio:>5.2f}x | {between_std:>7.1f}°   | {within_std:>7.1f}°     | {verdict}{compass}")

    # Summary
    print(f"\n  {'=' * 75}")
    print(f"  EXPERIMENT 2 SUMMARY")
    print(f"  {'=' * 75}")

    peak_layer, peak_fisher = max(results, key=lambda x: x[3])[:2]  # by ratio over baseline
    peak_ratio = max(results, key=lambda x: x[3])[3]
    print(f"  Peak discriminability (vs baseline): L{peak_layer} (ratio={peak_ratio:.2f}x)")
    print(f"  Compass layer: L{COMPASS_LAYER}")

    compass_result = results[COMPASS_LAYER]
    print(f"  Compass Fisher={compass_result[1]:.2f}, baseline={compass_result[2]:.2f}, ratio={compass_result[3]:.2f}x")

    # Depth profile bar chart (by ratio over baseline)
    print(f"\n  Depth profile (Fisher ratio over random baseline):")
    max_ratio = max(r[3] for r in results)
    for layer, fisher, baseline, ratio, _, _ in results:
        bar_len = int(40 * ratio / max(max_ratio, 0.01))
        bar = "█" * bar_len
        compass = " ◀" if layer == COMPASS_LAYER else ""
        print(f"  L{layer:>2} | {bar:<40} {ratio:>5.2f}x{compass}")


# -----------------------------------------------------------------------
# EXECUTION
# -----------------------------------------------------------------------
if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[0] Loading {MODEL_NAME} on {device}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=torch.bfloat16)
    model.eval()

    test_layernorm_preservation(model)
    test_full_depth_semantic_profile(model)

    print("\n  Investigation completed successfully.")
