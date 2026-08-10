#!/usr/bin/env python3
"""
End-to-End Execution of the Geometric SVD Framework for Arithmetic Circuit Discovery.
Validates the trigonometric computation mechanisms in pre-trained models.
"""

import argparse
import torch
import numpy as np
from transformer_lens import HookedTransformer

# Assuming you saved the previous rewrite as src/models/arithmetic_pipeline.py
from src.models.arithmetic_pipeline import GeometricArithmeticPipeline

def generate_test_data(num_samples=20):
    """Generates clean, no-carry addition pairs for testing."""
    import random
    np.random.seed(42)
    random.seed(42)

    pairs = []
    while len(pairs) < num_samples:
        a = random.randint(10, 40)
        b = random.randint(10, 40)
        # Ensure no carry for baseline testing
        if (a % 10) + (b % 10) < 10:
            pairs.append((a, b))

    numbers = list(set([a for a, _ in pairs] + [b for _, b in pairs]))
    return sorted(numbers), pairs

def main():
    parser = argparse.ArgumentParser(description="Run Geometric SVD Framework")
    parser.add_argument("--model", type=str, default="gpt2-small", choices=["gpt2-small", "gpt2-medium"])
    args = parser.parse_args()

    # Roadmap targets (75% depth principle)
    target_layer = 9 if args.model == "gpt2-small" else 18
    target_head = 9 if args.model == "gpt2-small" else 15

    print("="*60)
    print(f"🚀 Initializing Geometric Arithmetic Pipeline")
    print(f"Model: {args.model}")
    print(f"Targeting 75% Depth: Layer {target_layer}, Head {target_head}")
    print("="*60)

    # 1. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading model to {device}...")
    model = HookedTransformer.from_pretrained(args.model, device=device)
    pipeline = GeometricArithmeticPipeline(model)

    # 2. Generate Data
    numbers, operand_pairs = generate_test_data(num_samples=30)
    print(f"Generated {len(numbers)} unique operands and {len(operand_pairs)} no-carry addition pairs.")

    # ==========================================
    # PHASE 3: Input Plane Geometric Testing
    # ==========================================
    print("\n" + "="*40)
    print("PHASE 3: Testing Input Plane (Reading Directions)")
    print("="*40)

    plane_result = pipeline.test_input_plane(
        layer=target_layer,
        head=target_head,
        numbers=numbers,
        prompt_template="{n} + 5 ="
    )

    if not plane_result:
        print("❌ FAILED: No 2D SVD plane met the strict geometric criteria (CV < 0.2, Linearity > 0.9).")
        return

    k1, k2 = plane_result['k1'], plane_result['k2']
    v1, v2 = plane_result['v1'], plane_result['v2']

    print(f"✅ SUCCESS: Helix plane found at SVD components {k1} and {k2}!")
    print(f"   Radius CV (constant radius): {plane_result['cv']:.4f} (< 0.2 threshold)")
    print(f"   Angle Linearity (rotation):  {plane_result['lin']:.4f} (> 0.9 threshold)")


    # ==========================================
    # PHASE 4: Output Plane Computation Testing
    # ==========================================
    print("\n" + "="*40)
    print("PHASE 4: Testing Output Plane (Writing Computation)")
    print("="*40)

    out_result = pipeline.test_output_plane(
        layer=target_layer,
        head=target_head,
        operand_pairs=operand_pairs,
        u_k1=k1,
        u_k2=k2
    )

    if out_result['is_output_helix']:
        print("✅ SUCCESS: Output vectors form a helix representing the SUM (a+b)!")
    else:
        print("⚠️ WARNING: Output vectors do not perfectly match the target sum geometry.")
    print(f"   Output Radius CV: {out_result['cv']:.4f}")
    print(f"   Output Linearity: {out_result['linearity']:.4f}")

    # ==========================================
    # PHASE 5: MLP Interaction Pipeline Analysis
    # ==========================================
    print("\n" + "="*40)
    print("PHASE 5: Measuring MLP Subspace Alignment")
    print("="*40)

    mlp_result = pipeline.measure_mlp_alignment(
        layer=target_layer,
        head=target_head,
        u_k1=k1,
        u_k2=k2
    )

    if mlp_result['is_coupled']:
        print("✅ SUCCESS: Strong coupling detected between Attention Output and MLP Input.")
    else:
        print("⚠️ WARNING: Weak coupling detected.")
    print(f"   Maximum Cosine Similarity: {mlp_result['max_alignment']:.4f}")

    # ==========================================
    # PHASE 6: Causal Verification
    # ==========================================
    print("\n" + "="*40)
    print("PHASE 6: Causal Phase-Shift Intervention")
    print("="*40)

    # We estimate period T ≈ 74.2 for GPT-2 Small, or 55.4 for GPT-2 Medium based on the roadmap anomaly
    period = 74.2 if args.model == "gpt2-small" else 55.4
    print(f"Using empirically derived BPE-warped period T = {period}")

    success_count = 0
    test_cases = [(12, 14), (21, 15), (30, 11)]
    shift_amount = 2  # We will force the model to add 2 to the result purely via math

    for a, b in test_cases:
        causal_res = pipeline.causal_phase_shift(
            layer=target_layer,
            head=target_head,
            v1=v1, v2=v2,
            a=a, b=b,
            shift_delta=shift_amount,
            period=period
        )

        expected_str = str(causal_res['expected_shifted_sum'])
        model_out = causal_res['model_output'].strip()

        status = "✅" if expected_str in model_out else "❌"
        if expected_str in model_out: success_count += 1

        print(f"   Prompt: {a} + {b} = (True Sum: {causal_res['original_sum']})")
        print(f"   Intervention: Rotate vector by +{shift_amount} steps")
        print(f"   Expected Output: {expected_str} | Model Output: '{model_out}' {status}\n")

    print("="*60)
    print(f"🎯 EXPERIMENT COMPLETE: {success_count}/{len(test_cases)} Causal Interventions Successful")
    print("="*60)

if __name__ == "__main__":
    main()