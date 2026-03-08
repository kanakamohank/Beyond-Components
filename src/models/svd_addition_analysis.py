import torch
import numpy as np
import itertools
from transformer_lens import HookedTransformer

# ============================================================
# STEP 0: PREREQUISITES
# ============================================================

model = HookedTransformer.from_pretrained("gpt-j-6b")
# Or train your own 2-layer addition model for cleaner signals

# ============================================================
# PHASE 1: LOCATE CLOCK HEADS (with actual code)
# ============================================================

def find_clock_heads(model, n_test_pairs=50, tie_threshold=0.05):
    """
    Returns list of (layer, head) tuples with high TIE on addition.
    Uses minimal operand perturbation: change b by 1, observe answer shift.
    """
    import random

    # Use multiple prompt pairs to get stable TIE estimates
    test_pairs = [
        (f"What is {a} + {b}?", f"What is {a} + {b+1}?",
         str(a+b), str(a+b+1))
        for a, b in [(random.randint(10,50), random.randint(10,40))
                     for _ in range(n_test_pairs)]
    ]

    tie_accumulator = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)

    for clean_p, corr_p, clean_ans, corr_ans in test_pairs:
        try:
            clean_ans_idx = model.to_single_token(" " + clean_ans)
        except:
            continue

        clean_tokens = model.to_tokens(clean_p)
        corr_tokens = model.to_tokens(corr_p)

        _, clean_cache = model.run_with_cache(clean_tokens)
        base_logits = model(corr_tokens)
        base_prob = torch.softmax(base_logits[0,-1,:], dim=-1)[clean_ans_idx].item()

        for layer in range(model.cfg.n_layers):
            for head in range(model.cfg.n_heads):
                clean_out = clean_cache[
                    "blocks.{}.attn.hook_result".format(layer)
                ][0, :, head, :]

                def make_hook(c_out, h_idx):
                    def hook(value, hook):
                        value[0, :, h_idx, :] = c_out
                        return value
                    return hook

                patched_logits = model.run_with_hooks(
                    corr_tokens,
                    fwd_hooks=[(
                        f"blocks.{layer}.attn.hook_result",
                        make_hook(clean_out, head)
                    )]
                )
                patched_prob = torch.softmax(
                    patched_logits[0,-1,:], dim=-1
                )[clean_ans_idx].item()

                tie_accumulator[layer, head] += (patched_prob - base_prob)

    tie_accumulator /= n_test_pairs

    clock_heads = [
        (l, h)
        for l in range(model.cfg.n_layers)
        for h in range(model.cfg.n_heads)
        if tie_accumulator[l, h] > tie_threshold
    ]

    print(f"Found {len(clock_heads)} clock heads")
    return clock_heads, tie_accumulator


# ============================================================
# PHASE 2: EXTRACT OV CIRCUIT (professor's corrected version)
# ============================================================

def get_OV_circuit(model, layer, head):
    W_V = model.W_V[layer, head]  # [d_model, d_head]
    W_O = model.W_O[layer, head]  # [d_head, d_model]
    return W_V @ W_O              # [d_model, d_model] CORRECT

def get_number_token_index(model, prompt, number):
    str_tokens = model.to_str_tokens(prompt)
    n_str = str(number)
    matches = [i for i, t in enumerate(str_tokens) if t.strip() == n_str]
    if len(matches) != 1:
        return None, False
    return matches[0], True


# ============================================================
# PHASE 3: COLLECT RESIDUAL STREAM ACTIVATIONS
# ============================================================

def collect_arithmetic_activations(model, layer,
                                   prompt_template="What is {n} + 5?",
                                   n_range=range(100)):
    """
    Returns (acts_tensor, valid_ns) with proper token index tracking.
    """
    acts = []
    valid_ns = []

    for n in n_range:
        prompt = prompt_template.format(n=n)
        tokens = model.to_tokens(prompt)

        token_idx, valid = get_number_token_index(model, prompt, n)
        if not valid:
            continue

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        resid_pre = cache[f"blocks.{layer}.hook_resid_pre"]

        # Verify shape before indexing
        expected_d_model = model.cfg.d_model
        assert resid_pre.shape[-1] == expected_d_model

        acts.append(resid_pre[0, token_idx, :].cpu())
        valid_ns.append(n)

    if len(acts) < 20:
        raise ValueError(f"Only {len(acts)} valid tokens found — "
                         f"check tokenizer compatibility")

    return torch.stack(acts), valid_ns


# ============================================================
# PHASE 4: GEOMETRIC HELIX DETECTION (with fixes)
# ============================================================

def safe_angle_linearity(angles_np, valid_ns):
    """Unwrap and correlate with safety check for small periods."""
    mean_delta = np.abs(np.diff(angles_np)).mean()
    if mean_delta > np.pi * 0.8:
        return 0.0  # Unwrap unreliable for T < ~4

    unwrapped = np.unwrap(angles_np)
    return np.corrcoef(valid_ns, unwrapped)[0, 1]

def find_helix_planes(model, layer, head, acts_tensor, valid_ns,
                      top_k=10, cv_threshold=0.2, lin_threshold=0.9):
    """
    Tests all pairs of top-k singular directions for helix structure.
    Returns ranked list of (k1, k2, radius_cv, angle_linearity).
    """
    W_OV = get_OV_circuit(model, layer, head)
    U, S, Vt = torch.linalg.svd(W_OV)

    results = []

    for k1, k2 in itertools.combinations(range(top_k), 2):
        v1, v2 = Vt[k1].cpu(), Vt[k2].cpu()

        coords = torch.stack([
            acts_tensor @ v1,
            acts_tensor @ v2
        ], dim=1)  # [N_valid, 2]

        radii = coords.norm(dim=1)
        angles = torch.atan2(coords[:, 1], coords[:, 0]).numpy()

        radius_cv = (radii.std() / radii.mean()).item()
        angle_lin = safe_angle_linearity(angles, valid_ns)

        results.append((k1, k2, radius_cv, angle_lin))

    # Sort by combined score (low CV, high linearity)
    results.sort(key=lambda x: x[2] - abs(x[3]))

    # Report all candidates above threshold (not just first)
    found = [(k1, k2, cv, lin) for k1, k2, cv, lin in results
             if cv < cv_threshold and abs(lin) > lin_threshold]

    if found:
        print(f"\nLayer {layer}, Head {head}: "
              f"{len(found)} helix planes found")
        for k1, k2, cv, lin in found:
            print(f"  Directions ({k1},{k2}): CV={cv:.3f}, Lin={lin:.3f}")

        # Report distribution baseline for context
        all_cvs = [r[2] for r in results]
        print(f"  CV baseline: mean={np.mean(all_cvs):.3f}, "
              f"min={np.min(all_cvs):.3f} (found threshold: {cv_threshold})")

    return found, U, S, Vt


# ============================================================
# PHASE 5: VALIDATE COMPUTATION (not just representation)
# Checks that U (output) side also shows helix for the SUM
# ============================================================

def validate_addition_computation(model, layer, head,
                                  k1, k2, U, Vt,
                                  fixed_a=13, b_range=range(5, 80)):
    """
    Critical test: does the OUTPUT side encode the SUM helically?
    This distinguishes 'n is stored helically' from 'addition is computed'.
    """
    u1, u2 = U[:, k1].cpu(), U[:, k2].cpu()

    sum_acts = []
    valid_sums = []

    for b in b_range:
        target_sum = fixed_a + b
        prompt = f"What is {fixed_a} + {b}?"
        tokens = model.to_tokens(prompt)

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        # Get residual stream AFTER this layer at ANSWER position
        resid_post = cache[f"blocks.{layer}.hook_resid_post"]
        ans_pos = resid_post.shape[1] - 1  # last token = answer position

        sum_acts.append(resid_post[0, ans_pos, :].cpu())
        valid_sums.append(target_sum)

    sum_tensor = torch.stack(sum_acts)

    # Project onto U directions (writing/output side)
    coords = torch.stack([
        sum_tensor @ u1,
        sum_tensor @ u2
    ], dim=1)

    radii = coords.norm(dim=1)
    angles = torch.atan2(coords[:, 1], coords[:, 0]).numpy()

    radius_cv = (radii.std() / radii.mean()).item()
    angle_lin = safe_angle_linearity(angles, valid_sums)

    print(f"\nOutput/Sum side validation:")
    print(f"  Radius CV: {radius_cv:.3f}  (want < 0.2)")
    print(f"  Angle linearity w.r.t. sum: {angle_lin:.3f}  (want > 0.9)")

    if radius_cv < 0.2 and abs(angle_lin) > 0.9:
        print("  ✓ Output side is also helical w.r.t. sum")
        print("  → Evidence that this head performs addition via rotation")
    else:
        print("  ✗ Output side is not helix-structured w.r.t. sum")
        print("  → Head reads helically but may not compute addition here")

    return radius_cv, angle_lin


# ============================================================
# PHASE 6: CAUSAL INTERVENTION (the decisive test)
# ============================================================

def causal_phase_shift_test(model, layer, head, Vt, k1, k2,
                            test_cases=None):
    """
    Rotates activation in helix plane by angle corresponding to +delta.
    If model answer shifts by delta: causal proof of rotation mechanism.
    Stratified by carry/no-carry.
    """
    if test_cases is None:
        # Stratify: no-carry and carry cases
        test_cases = {
            'no_carry': [(13, 24, 5), (21, 31, 3)],  # a, b, delta
            'carry':    [(13, 28, 5), (17, 27, 6)],  # ones digit sum >= 10
        }

    v1, v2 = Vt[k1].cpu(), Vt[k2].cpu()

    for case_type, cases in test_cases.items():
        print(f"\n{case_type.upper()} cases:")
        successes = 0

        for a, b, delta in cases:
            prompt = f"What is {a} + {b}?"
            correct_answer = a + b
            expected_answer = a + b + delta  # after rotation by delta

            # Angle to rotate in the ones-digit plane (period T=10)
            theta = 2 * np.pi * delta / 10

            tokens = model.to_tokens(prompt)
            _, cache = model.run_with_cache(tokens)

            token_idx, valid = get_number_token_index(model, prompt, a)
            if not valid:
                continue

            resid_pre = cache[f"blocks.{layer}.hook_resid_pre"].cpu()
            h = resid_pre[0, token_idx, :]

            # Rotate in (v1, v2) plane by theta
            c1 = (h @ v1).item()
            c2 = (h @ v2).item()

            h_rotated = (h
                         - c1 * v1 - c2 * v2  # remove original
                         + (c1*np.cos(theta) - c2*np.sin(theta)) * v1
                         + (c1*np.sin(theta) + c2*np.cos(theta)) * v2)

            def rotation_hook(value, hook,
                              idx=token_idx, h_rot=h_rotated):
                value[0, idx, :] = h_rot.to(value.device)
                return value

            patched_logits = model.run_with_hooks(
                tokens.to(model.cfg.device),
                fwd_hooks=[(
                    f"blocks.{layer}.hook_resid_pre",
                    rotation_hook
                )]
            )

            predicted = patched_logits[0, -1, :].argmax().item()
            predicted_str = model.to_str_tokens([predicted])[0]

            success = str(expected_answer % 10) in predicted_str
            successes += int(success)

            print(f"  {a}+{b}: rotate by +{delta} → "
                  f"predicted={predicted_str}, "
                  f"expected ones digit={expected_answer%10} "
                  f"{'✓' if success else '✗'}")

        print(f"  Success rate: {successes}/{len(cases)}")
        if case_type == 'carry':
            print("  (carry failures expected if helix only "
                  "encodes ones-digit rotation)")


# ============================================================
# MAIN EXECUTION
# ============================================================

def run_full_pipeline(model):

    print("="*60)
    print("PHASE 1: Finding Clock Heads")
    print("="*60)
    clock_heads, tie_matrix = find_clock_heads(model)

    for layer, head in clock_heads[:5]:  # Top 5 by TIE
        print(f"\n{'='*60}")
        print(f"Analyzing Layer {layer}, Head {head}")
        print(f"{'='*60}")

        # Phase 2+3: Collect activations
        acts_tensor, valid_ns = collect_arithmetic_activations(
            model, layer,
            prompt_template="What is {n} + 5?"
        )

        # Phase 4: Geometric helix detection
        found_planes, U, S, Vt = find_helix_planes(
            model, layer, head, acts_tensor, valid_ns
        )

        if not found_planes:
            print("  No helix found, skipping")
            continue

        # Take best plane and validate
        best_k1, best_k2 = found_planes[0][0], found_planes[0][1]

        # Phase 5: Validate computation (not just representation)
        validate_addition_computation(
            model, layer, head, best_k1, best_k2, U, Vt
        )

        # Phase 6: Causal intervention
        causal_phase_shift_test(
            model, layer, head, Vt, best_k1, best_k2
        )