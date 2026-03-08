#!/usr/bin/env python3
"""
Demonstration script for helix visualization with MaskedTransformerCircuit.

This script shows how to reproduce the Neel and Tegmark/Kattamaneni helix approach
using the Beyond Components SVD-based circuit discovery framework.

Usage:
    python examples/helix_demo.py --config configs/gt_config.yaml --output_dir helix_results/
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import argparse
import yaml
from pathlib import Path

from src.models.masked_transformer_circuit import MaskedTransformerCircuit
from src.utils.helix_visualization import (
    add_helix_analysis_to_circuit,
    find_helix_directions,
    visualize_2d_helix,
    create_3d_helix_visualization,
    visualize_phase_shift_analysis,
    generate_helix_comparison_report
)
from transformer_lens import HookedTransformer


def collect_arithmetic_activations(model, layer, prompt_template="What is {n} + 5?",
                                 n_range=range(10, 50)):
    """
    Collect activations for arithmetic prompts following the helix paper approach.

    Args:
        model: HookedTransformer model
        layer: Layer to extract activations from
        prompt_template: Template for arithmetic prompts
        n_range: Range of numbers to test

    Returns:
        Tuple of (activations_tensor, valid_numbers)
    """
    activations = []
    valid_numbers = []

    for n in n_range:
        prompt = prompt_template.format(n=n)
        tokens = model.to_tokens(prompt)

        # Find the position of the number token
        str_tokens = model.to_str_tokens(tokens[0])
        n_str = str(n)

        # Look for exact match or number within token
        token_idx = None
        for i, token in enumerate(str_tokens):
            if n_str in token.strip():
                token_idx = i
                break

        if token_idx is None:
            print(f"Warning: Could not find token for number {n} in prompt '{prompt}'")
            continue

        # Get activations at the number token position
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        # Extract residual stream activations before the specified layer
        resid_pre = cache[f'blocks.{layer}.hook_resid_pre']
        activation = resid_pre[0, token_idx, :].cpu()

        activations.append(activation)
        valid_numbers.append(n)

    if len(activations) == 0:
        raise ValueError("No valid activations collected. Check tokenizer and prompt format.")

    return torch.stack(activations), valid_numbers


def demonstrate_helix_detection(model, circuit, output_dir):
    """Demonstrate helix detection on arithmetic tasks."""
    print("="*60)
    print("PHASE 1: Helix Detection in Arithmetic Reasoning")
    print("="*60)

    # Test different arithmetic patterns
    arithmetic_tasks = [
        ("addition", "What is {n} + 3?", range(10, 40)),
        ("subtraction", "What is {n} - 2?", range(12, 42)),
        ("modular", "{n} mod 10 =", range(10, 50))
    ]

    helix_results = {}

    for task_name, prompt_template, n_range in arithmetic_tasks:
        print(f"\nTesting {task_name} with prompt: '{prompt_template}'")

        helix_results[task_name] = {}

        # Test a few key layers (middle layers often show arithmetic reasoning)
        test_layers = [model.cfg.n_layers // 3, model.cfg.n_layers // 2, 2 * model.cfg.n_layers // 3]
        test_layers = [l for l in test_layers if l < model.cfg.n_layers]

        for layer in test_layers:
            print(f"\n  Layer {layer}:")

            try:
                # Collect activations
                activations, values = collect_arithmetic_activations(
                    model, layer, prompt_template, n_range
                )
                print(f"    Collected {len(values)} valid activations")

                layer_results = {}

                # Test a few heads
                test_heads = [0, model.cfg.n_heads // 2, model.cfg.n_heads - 1]
                test_heads = [h for h in test_heads if h < model.cfg.n_heads]

                for head in test_heads:
                    try:
                        # Find helix directions
                        helix_dirs = find_helix_directions(
                            circuit, layer, head, activations, values, top_k=8
                        )

                        if helix_dirs and helix_dirs[0]['is_helix']:
                            best = helix_dirs[0]
                            print(f"    Head {head}: ✓ HELIX FOUND!")
                            print(f"      Radius CV: {best['radius_cv']:.3f}")
                            print(f"      Angle Linearity: {best['angle_linearity']:.3f}")
                            print(f"      Period: {best['estimated_period']:.1f}")

                            # Create visualizations
                            task_dir = output_dir / task_name / f"L{layer}H{head}"
                            task_dir.mkdir(parents=True, exist_ok=True)

                            # 2D visualization
                            visualize_2d_helix(
                                best, values,
                                title=f"{task_name.title()} - Layer {layer}, Head {head}",
                                output_path=str(task_dir / "helix_2d.png")
                            )

                            # 3D interactive visualization
                            fig_3d = create_3d_helix_visualization(
                                best, values,
                                title=f"{task_name.title()} - Layer {layer}, Head {head}",
                                output_path=str(task_dir / "helix_3d.html")
                            )

                            layer_results[head] = best

                        else:
                            print(f"    Head {head}: No helix structure found")

                    except Exception as e:
                        print(f"    Head {head}: Error - {e}")

                helix_results[task_name][layer] = layer_results

            except Exception as e:
                print(f"  Layer {layer}: Error collecting activations - {e}")

    return helix_results


def demonstrate_phase_shift_analysis(model, circuit, helix_results, output_dir):
    """Demonstrate phase shift analysis for arithmetic operations."""
    print("\n" + "="*60)
    print("PHASE 2: Phase Shift Analysis")
    print("="*60)

    # Find best helix for demonstration
    best_helix_info = None

    for task, layers in helix_results.items():
        for layer, heads in layers.items():
            for head, result in heads.items():
                if result['is_helix'] and result['angle_linearity'] > 0.85:
                    best_helix_info = (task, layer, head, result)
                    break
            if best_helix_info:
                break
        if best_helix_info:
            break

    if not best_helix_info:
        print("No suitable helix found for phase shift analysis")
        return

    task, layer, head, helix_result = best_helix_info
    print(f"Using best helix from {task}: Layer {layer}, Head {head}")

    # Perform phase shift analysis
    base_values = list(range(10, 20))  # Test numbers 10-19

    try:
        shift_results = visualize_phase_shift_analysis(
            circuit, layer, head,
            base_values=base_values,
            shift_amount=1,
            helix_directions=helix_result['direction_indices'],
            output_dir=str(output_dir / "phase_analysis")
        )

        print(f"Phase shift analysis completed for +1 operation")
        print(f"Results saved to {output_dir / 'phase_analysis'}")

    except Exception as e:
        print(f"Phase shift analysis failed: {e}")


def demonstrate_comparison_report(circuit, output_dir):
    """Generate comprehensive comparison between approaches."""
    print("\n" + "="*60)
    print("PHASE 3: Generating Comparison Report")
    print("="*60)

    try:
        report = generate_helix_comparison_report(
            circuit,
            arithmetic_task="modular_addition",
            output_dir=str(output_dir / "comparison")
        )

        print("Comparison report generated successfully!")
        print("Key findings:")
        print(f"- Model: {report['model_info']['n_layers']} layers, {report['model_info']['n_heads']} heads")
        print(f"- Analysis task: {report['task']}")
        print(f"- Report saved to: {output_dir / 'comparison'}")

    except Exception as e:
        print(f"Comparison report generation failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Helix Visualization Demo")
    parser.add_argument("--config", default="configs/gt_config.yaml",
                       help="Config file path")
    parser.add_argument("--output_dir", default="helix_demo_output",
                       help="Output directory for visualizations")
    parser.add_argument("--model_name", default="gpt2-small",
                       help="Model name for transformer_lens")

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🔬 Helix Visualization Demo")
    print("Reproducing Neel & Tegmark/Kattamaneni helix approach")
    print("="*60)

    # Load model
    print(f"Loading model: {args.model_name}")
    model = HookedTransformer.from_pretrained(args.model_name)

    # Create masked circuit
    print("Initializing MaskedTransformerCircuit...")
    circuit = MaskedTransformerCircuit(
        model=model,
        mask_init_value=0.99,
        cache_svd=True,
        force_recompute_svd=False
    )

    # Add helix analysis capabilities
    print("Adding helix analysis methods...")
    add_helix_analysis_to_circuit(circuit)

    print(f"Output directory: {output_dir}")
    print("="*60)

    # Run demonstrations
    try:
        # Phase 1: Helix Detection
        helix_results = demonstrate_helix_detection(model, circuit, output_dir)

        # Phase 2: Phase Shift Analysis
        demonstrate_phase_shift_analysis(model, circuit, helix_results, output_dir)

        # Phase 3: Comparison Report
        demonstrate_comparison_report(circuit, output_dir)

        print("\n" + "="*60)
        print("🎉 Demo completed successfully!")
        print(f"Check {output_dir} for all generated visualizations")
        print("="*60)

        # Print summary of findings
        total_helices = sum(
            len(heads) for task_layers in helix_results.values()
            for heads in task_layers.values()
        )

        print(f"\nSUMMARY:")
        print(f"- Total helix structures found: {total_helices}")
        print(f"- Tasks tested: {list(helix_results.keys())}")
        print(f"- Visualizations saved to: {output_dir}")

        # List key files generated
        key_files = [
            "*/helix_2d.png - 2D helix projections",
            "*/helix_3d.html - Interactive 3D helix visualizations",
            "phase_analysis/ - Phase shift analysis for arithmetic operations",
            "comparison/ - Comparison report between standard SVD and helix approaches"
        ]

        print(f"\nKey output files:")
        for file_desc in key_files:
            print(f"  📁 {file_desc}")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()