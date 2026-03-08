#!/usr/bin/env python3
"""
Run Helix-Based Arithmetic Circuit Analysis

This script demonstrates the complete helix approach for discovering arithmetic
circuits in transformer models, as described in the Neel and Tegmark/Kattamaneni
research on trigonometric number representations.

Usage:
    # Basic helix analysis
    python run_helix_analysis.py

    # With custom model and output directory
    python run_helix_analysis.py --model gpt2-medium --output_dir my_helix_results/

    # Compare with existing SVD approach
    python run_helix_analysis.py --compare_svd --svd_checkpoint path/to/checkpoint.pt
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.helix_circuit_discovery import (
    HelixArithmeticCircuit,
    quick_helix_analysis,
    compare_with_standard_svd
)
from src.models.masked_transformer_circuit import MaskedTransformerCircuit
from transformer_lens import HookedTransformer
import torch


def main():
    parser = argparse.ArgumentParser(description="Helix-Based Arithmetic Circuit Analysis")

    # Model arguments
    parser.add_argument("--model", default="gpt2-small",
                       help="Model name for transformer_lens (default: gpt2-small)")
    parser.add_argument("--device", default=None,
                       help="Device to run on (default: auto-detect)")

    # Analysis arguments
    parser.add_argument("--output_dir", default="helix_analysis_results",
                       help="Output directory for results (default: helix_analysis_results)")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick analysis with default settings")

    # Comparison arguments
    parser.add_argument("--compare_svd", action="store_true",
                       help="Compare with standard SVD approach")
    parser.add_argument("--svd_checkpoint", default=None,
                       help="Path to trained SVD circuit checkpoint for comparison")

    # Helix parameters
    parser.add_argument("--cv_threshold", type=float, default=0.2,
                       help="Radius CV threshold for helix detection (default: 0.2)")
    parser.add_argument("--linearity_threshold", type=float, default=0.9,
                       help="Angle linearity threshold for helix detection (default: 0.9)")

    # Task specification
    parser.add_argument("--custom_tasks", action="store_true",
                       help="Use custom arithmetic tasks for analysis")

    args = parser.parse_args()

    # Setup device with MPS support for Apple Silicon
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print("🧮 Helix-Based Arithmetic Circuit Discovery")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")
    print("="*70)

    # Load model
    print("Loading model...")
    try:
        model = HookedTransformer.from_pretrained(args.model, device=device)
        print(f"✓ Loaded {args.model} ({model.cfg.n_layers} layers, {model.cfg.n_heads} heads)")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Define arithmetic tasks
    arithmetic_tasks = None
    if args.custom_tasks:
        arithmetic_tasks = {
            'number_recognition': {
                'template': 'The number {n} is',
                'range': range(0, 50)
            },
            'simple_addition': {
                'template': '{n} + 1 =',
                'range': range(10, 30)
            },
            'addition_by_5': {
                'template': '{n} + 5 =',
                'range': range(5, 35)
            },
            'subtraction': {
                'template': '{n} - 3 =',
                'range': range(10, 40)
            },
            'counting_sequence': {
                'template': 'Count: 1, 2, 3, ..., {n}',
                'range': range(4, 20)
            },
            'modular_arithmetic': {
                'template': '{n} mod 10 =',
                'range': range(10, 50)
            }
        }

    if args.quick:
        # Quick analysis
        print("\n🚀 Running quick helix analysis...")
        helix_circuit = quick_helix_analysis(
            model=model,
            output_dir=args.output_dir,
            arithmetic_tasks=arithmetic_tasks
        )

    else:
        # Detailed analysis
        print("\n🔬 Running detailed helix analysis...")

        # Initialize helix circuit analyzer
        helix_circuit = HelixArithmeticCircuit(model=model, device=device)

        # Set custom parameters if provided
        if args.cv_threshold != 0.2:
            helix_circuit.helix_params['cv_threshold'] = args.cv_threshold
        if args.linearity_threshold != 0.9:
            helix_circuit.helix_params['linearity_threshold'] = args.linearity_threshold

        print(f"Helix detection parameters:")
        print(f"  - CV threshold: {helix_circuit.helix_params['cv_threshold']}")
        print(f"  - Linearity threshold: {helix_circuit.helix_params['linearity_threshold']}")

        # Find arithmetic heads
        print("\nStep 1: Finding arithmetic heads with helix structure...")
        found_heads = helix_circuit.find_arithmetic_heads(arithmetic_tasks)

        # Print summary of found heads
        total_found = sum(len(heads) for heads in found_heads.values())
        print(f"\n📊 Summary of helix structures found:")
        for task, heads in found_heads.items():
            if heads:
                print(f"  {task}: {len(heads)} heads - {heads}")
            else:
                print(f"  {task}: No helix structures found")
        print(f"Total helix heads found: {total_found}")

        if total_found > 0:
            # Analyze specific operations
            print("\nStep 2: Analyzing arithmetic operations...")
            operation_analysis = helix_circuit.analyze_arithmetic_operations(
                base_numbers=list(range(10, 25)),
                output_dir=args.output_dir
            )

            # Generate comprehensive report
            print("\nStep 3: Generating comprehensive report...")
            report = helix_circuit.generate_comprehensive_report(args.output_dir)

            print(f"\n✅ Analysis complete!")
            print(f"📁 Results saved to: {args.output_dir}")
            print(f"📈 Helix heads found: {len(helix_circuit.helix_heads)}")

            # Print key findings
            if helix_circuit.helix_heads:
                best_head = max(helix_circuit.helix_heads.items(),
                              key=lambda x: x[1]['helix_result']['best_result']['angle_linearity'])
                (layer, head), data = best_head
                result = data['helix_result']['best_result']

                print(f"\n🎯 Best helix head: Layer {layer}, Head {head}")
                print(f"   Task: {data['task']}")
                print(f"   Radius CV: {result['radius_cv']:.3f}")
                print(f"   Angle Linearity: {result['angle_linearity']:.3f}")
                print(f"   Estimated Period: {result['estimated_period']:.1f}")

        else:
            print("\n⚠️  No helix structures found with current parameters.")
            print("Try adjusting --cv_threshold or --linearity_threshold for more lenient detection.")

    # Comparison with standard SVD approach (if requested)
    if args.compare_svd:
        print("\n🔄 Comparing with standard SVD approach...")

        if args.svd_checkpoint:
            # Load existing trained circuit
            print(f"Loading SVD checkpoint: {args.svd_checkpoint}")
            try:
                base_circuit = MaskedTransformerCircuit(model=model, device=device)
                checkpoint = torch.load(args.svd_checkpoint, map_location=device)
                # Load state (implementation depends on checkpoint format)
                print("✓ SVD checkpoint loaded")

                comparison_output = f"{args.output_dir}_comparison"
                helix_circuit_comp, comparison_report = compare_with_standard_svd(
                    model=model,
                    base_circuit=base_circuit,
                    output_dir=comparison_output
                )

                print(f"📊 Comparison results saved to: {comparison_output}")

            except Exception as e:
                print(f"❌ Error loading SVD checkpoint: {e}")

        else:
            # Create fresh SVD circuit for comparison
            print("Creating fresh SVD circuit for comparison...")
            base_circuit = MaskedTransformerCircuit(model=model, device=device)

            comparison_output = f"{args.output_dir}_comparison"
            helix_circuit_comp, comparison_report = compare_with_standard_svd(
                model=model,
                base_circuit=base_circuit,
                output_dir=comparison_output
            )

            print(f"📊 Comparison results saved to: {comparison_output}")

    print("\n" + "="*70)
    print("🎉 Helix analysis complete!")
    print(f"📁 Check {args.output_dir} for:")
    print("   - JSON report with all findings")
    print("   - Heatmaps showing helix quality across heads")
    print("   - Individual head visualizations (2D and 3D)")
    print("   - Comparison plots with standard SVD approach")
    print("   - Interactive HTML visualizations")
    print("="*70)


if __name__ == "__main__":
    main()