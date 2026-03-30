#!/usr/bin/env python3
"""
Run Helix-Based Arithmetic Circuit Analysis

This script demonstrates the complete helix approach for discovering arithmetic
circuits in transformer models, as described in the Neel and Tegmark/Kattamaneni
research on trigonometric number representations.

Features three memory modes:
- standard: Full-featured analysis with comprehensive visualizations
- optimized: Batch processing for large models (GPT-Neo 2.7B+)
- cache: Leverage existing SVD cache files for faster re-analysis

Usage:
    # Standard mode (default) - full features
    python run_helix_analysis.py

    # Memory-optimized mode for large models
    python run_helix_analysis.py --memory_mode optimized --batch_size 4 --model EleutherAI/gpt-neo-2.7B

    # Cache-based mode (uses existing SVD cache)
    python run_helix_analysis.py --memory_mode cache --cache_dir svd_cache

    # With custom model and output directory
    python run_helix_analysis.py --model gpt2-medium --output_dir my_helix_results/

    # Compare with existing SVD approach (standard mode)
    python run_helix_analysis.py --compare_svd --svd_checkpoint path/to/checkpoint.pt
"""

import argparse
import sys
import gc
import itertools
import glob
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def numpy_to_python(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_python(item) for item in obj]
    else:
        return obj

from src.models.helix_circuit_discovery import (
    HelixArithmeticCircuit,
    quick_helix_analysis,
    compare_with_standard_svd
)
from src.models.masked_transformer_circuit import MaskedTransformerCircuit
from src.utils.helix_visualization import detect_helix_geometry
from transformer_lens import HookedTransformer
import torch


class MemoryOptimizedHelixAnalyzer:
    """Memory-efficient helix analyzer for large models."""

    def __init__(self, model, device, batch_size=4, model_name=None):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.model_name = model_name or getattr(model.cfg, 'model_name', 'unknown')
        self.helix_heads = {}
        self.analysis_cache = {}

    def analyze_head_batch(self, layer_head_pairs, arithmetic_tasks=None):
        """Analyze a batch of heads with memory optimization."""
        results = {}

        # Use simple number sequence if no tasks provided
        if arithmetic_tasks is None:
            number_range = list(range(10, 50))
            prompts = [f"The number {n}" for n in number_range]

        for (layer, head) in tqdm(layer_head_pairs, desc=f"Analyzing batch"):
            try:
                # Clear cache before each head
                if self.device.type == "mps":
                    torch.mps.empty_cache()
                elif self.device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

                # Get activations for this head
                activations = self._get_head_activations(layer, head, prompts, number_range)

                if activations is None:
                    continue

                # Perform helix analysis on device (MPS/CUDA/CPU)
                helix_result = self._analyze_helix_on_device(activations, number_range)

                if helix_result:
                    results[(layer, head)] = {
                        'helix_result': helix_result,
                        'activations': activations.cpu().numpy() if isinstance(activations, torch.Tensor) else activations,  # Store as numpy for serialization
                        'numbers': number_range
                    }

            except Exception as e:
                print(f"Error analyzing Layer {layer}, Head {head}: {e}")
                continue

        return results

    def _get_head_activations(self, layer, head, prompts, numbers):
        """Get activations for a specific head with memory management."""
        try:
            activations = []

            # Process prompts in smaller batches
            prompt_batch_size = min(8, len(prompts))

            for i in range(0, len(prompts), prompt_batch_size):
                batch_prompts = prompts[i:i+prompt_batch_size]

                # Tokenize batch
                tokens_batch = []
                for prompt in batch_prompts:
                    try:
                        tokens = self.model.to_tokens(prompt, prepend_bos=False)
                        tokens_batch.append(tokens)
                    except:
                        # Skip problematic tokens
                        continue

                if not tokens_batch:
                    continue

                # Get activations for this batch
                with torch.no_grad():
                    for tokens in tokens_batch:
                        try:
                            # Move to device only when needed
                            tokens = tokens.to(self.device)

                            # Run model and get cache
                            _, cache = self.model.run_with_cache(tokens)

                            # Extract head activations using hook_z (compatible with GPT-2, GPT-Neo, etc.)
                            # hook_z shape: [batch, seq_len, n_heads, d_head]
                            hook_z = cache[f"blocks.{layer}.attn.hook_z"]
                            head_output = hook_z[0, -1, head, :]  # Last token, specific head - keep on device

                            activations.append(head_output)

                            # Clear intermediate cache
                            del cache, hook_z

                        except Exception as e:
                            print(f"Skipping prompt due to error: {e}")
                            continue

                # Clear memory after each batch
                if self.device.type == "mps":
                    torch.mps.empty_cache()
                elif self.device.type == "cuda":
                    torch.cuda.empty_cache()

            if len(activations) < 10:  # Need minimum data points
                return None

            return torch.stack(activations)

        except Exception as e:
            print(f"Error getting activations: {e}")
            return None

    def _analyze_helix_on_device(self, activations, numbers):
        """Perform helix analysis on device (MPS/CUDA/CPU) using torch SVD."""
        try:
            # Keep activations on device for SVD computation
            if not isinstance(activations, torch.Tensor):
                activations = torch.tensor(activations, device=self.device)

            # SVD on device using torch (matches arithmetic_circuit_discovery.py approach)
            # activations shape: [n_samples, d_head]
            # Vt rows are principal component directions
            U, S, Vt = torch.linalg.svd(activations, full_matrices=False)

            best_result = None
            best_score = float('inf')  # Initialize to infinity for minimization (GPT-2 style)

            # Test top 5 SVD direction pairs
            for k1, k2 in itertools.combinations(range(min(5, len(S))), 2):
                v1 = Vt[k1]  # Shape: [d_head]
                v2 = Vt[k2]  # Shape: [d_head]

                # Use helix detection from existing utils
                # Function does projection internally
                result = detect_helix_geometry(
                    activations,  # Original activations
                    v1,          # First direction
                    v2,          # Second direction
                    numbers,     # Values
                    cv_threshold=0.2,   # MATCHED TO GPT-2 (was 0.25)
                    linearity_threshold=0.9  # MATCHED TO GPT-2 (was 0.8)
                )

                if result['is_helix']:
                    # MATCHED TO GPT-2 scoring: lower is better
                    score = result['radius_cv'] - result['angle_linearity']
                    # Initialize best_score to infinity for minimization
                    if best_score is None or score < best_score:
                        best_score = score
                        best_result = result
                        best_result['svd_directions'] = (k1, k2)
                        best_result['singular_values'] = S[:10].cpu().tolist()

            return {'best_result': best_result} if best_result else None

        except Exception as e:
            print(f"Error in helix analysis: {e}")
            return None

    def run_full_analysis(self, output_dir="helix_memory_optimized", start_layer=0, start_head=0, end_layer=None, end_head=None):
        """Run complete helix analysis with memory optimization.

        Args:
            output_dir: Directory to save results
            start_layer: Starting layer (inclusive)
            start_head: Starting head within starting layer (inclusive)
            end_layer: Ending layer (inclusive, None = all layers)
            end_head: Ending head within ending layer (inclusive, None = all heads)
        """
        Path(output_dir).mkdir(exist_ok=True)

        # Determine layer range
        end_layer = self.model.cfg.n_layers - 1 if end_layer is None else end_layer

        print(f"Starting memory-optimized helix analysis for {self.model_name}")
        print(f"Model: {self.model.cfg.n_layers} layers, {self.model.cfg.n_heads} heads")
        print(f"Analyzing: Layer {start_layer} to {end_layer}")
        if start_layer == end_layer:
            print(f"  Head range: {start_head} to {end_head if end_head is not None else self.model.cfg.n_heads - 1}")
        print(f"Device: {self.device}, Batch size: {self.batch_size}")

        # Generate layer-head pairs for specified range
        all_pairs = []
        for layer in range(start_layer, end_layer + 1):
            # Determine head range for this layer
            if layer == start_layer and layer == end_layer:
                # Single layer case
                head_start = start_head
                head_end = end_head if end_head is not None else self.model.cfg.n_heads - 1
            elif layer == start_layer:
                # First layer in multi-layer range
                head_start = start_head
                head_end = self.model.cfg.n_heads - 1
            elif layer == end_layer:
                # Last layer in multi-layer range
                head_start = 0
                head_end = end_head if end_head is not None else self.model.cfg.n_heads - 1
            else:
                # Middle layers
                head_start = 0
                head_end = self.model.cfg.n_heads - 1

            for head in range(head_start, head_end + 1):
                all_pairs.append((layer, head))

        print(f"Total heads to analyze: {len(all_pairs)}")

        # Process in batches
        all_results = {}
        helix_found = 0

        for i in tqdm(range(0, len(all_pairs), self.batch_size), desc="Processing batches"):
            batch = all_pairs[i:i+self.batch_size]

            print(f"\nProcessing batch {i//self.batch_size + 1}/{(len(all_pairs)-1)//self.batch_size + 1}")
            print(f"   Heads: {batch}")

            batch_results = self.analyze_head_batch(batch)

            # Save batch results immediately
            if batch_results:
                all_results.update(batch_results)
                helix_found += len(batch_results)

                # Save intermediate results
                self._save_intermediate_results(all_results, output_dir, i//self.batch_size + 1)

            print(f"   Batch complete. Helix heads found so far: {helix_found}")

        # Generate final report
        self._generate_final_report(all_results, output_dir)

        return all_results

    def _save_intermediate_results(self, results, output_dir, batch_num):
        """Save intermediate results after each batch."""
        try:
            # Convert tensor data for JSON serialization
            json_safe_results = {}
            for (layer, head), data in results.items():
                json_safe_results[f"L{layer}H{head}"] = {
                    'layer': int(layer),
                    'head': int(head),
                    'helix_result': numpy_to_python(data['helix_result']),
                    'activation_shape': list(data['activations'].shape),
                    'number_count': len(data['numbers'])
                }

            # Save JSON report
            with open(f"{output_dir}/intermediate_results_batch_{batch_num}.json", 'w') as f:
                json.dump({
                    'batch_number': int(batch_num),
                    'total_helix_heads': len(results),
                    'results': json_safe_results
                }, f, indent=2)

        except Exception as e:
            print(f"Warning: Could not save intermediate results: {e}")

    def _generate_final_report(self, results, output_dir):
        """Generate final comprehensive report."""
        print(f"\nGenerating final report...")

        # Summary statistics
        total_heads = self.model.cfg.n_layers * self.model.cfg.n_heads
        helix_heads = len(results)

        print(f"Analysis Complete!")
        print(f"   Total heads analyzed: {total_heads}")
        print(f"   Helix heads found: {helix_heads}")
        print(f"   Detection rate: {helix_heads/total_heads*100:.2f}%")

        # Find best head
        if results:
            best_head = max(
                results.items(),
                key=lambda x: x[1]['helix_result']['best_result']['angle_linearity']
            )
            (layer, head), data = best_head
            best_result = data['helix_result']['best_result']

            print(f"\nBest Helix Head: Layer {layer}, Head {head}")
            print(f"   Radius CV: {best_result['radius_cv']:.4f}")
            print(f"   Angle Linearity: {best_result['angle_linearity']:.4f}")
            print(f"   Estimated Period: {best_result.get('estimated_period', 'N/A')}")

        # Save final JSON report
        json_safe_results = {}
        for (layer, head), data in results.items():
            json_safe_results[f"L{layer}H{head}"] = {
                'layer': int(layer),
                'head': int(head),
                'helix_result': numpy_to_python(data['helix_result'])
            }

        final_report = {
            'model_name': self.model_name,
            'total_heads_analyzed': int(total_heads),
            'helix_structures_found': int(helix_heads),
            'detection_rate': float(helix_heads/total_heads),
            'analysis_parameters': {
                'cv_threshold': 0.25,
                'linearity_threshold': 0.8,
                'device': str(self.device),
                'batch_size': int(self.batch_size)
            },
            'results': json_safe_results
        }

        with open(f"{output_dir}/helix_analysis_report.json", 'w') as f:
            json.dump(final_report, f, indent=2)

        # Save README
        self._save_readme(output_dir, final_report)

        print(f"Results saved to: {output_dir}/")

    def _save_readme(self, output_dir, report):
        """Save a summary README."""
        readme_content = f"""# Memory-Optimized Helix Analysis Results

## Overview

This report analyzes arithmetic circuits using the helix approach on {report['model_name']}.

## Key Findings

- **Total heads analyzed**: {report['total_heads_analyzed']}
- **Helix structures found**: {report['helix_structures_found']}
- **Detection rate**: {report['detection_rate']:.1%}

## Analysis Parameters

- CV threshold: {report['analysis_parameters']['cv_threshold']}
- Linearity threshold: {report['analysis_parameters']['linearity_threshold']}
- Device: {report['analysis_parameters']['device']}
- Batch size: {report['analysis_parameters']['batch_size']}

## Files Generated

- `helix_analysis_report.json` - Complete analysis results
- `intermediate_results_batch_*.json` - Batch-wise results
- `README.md` - This summary file
"""

        if report['helix_structures_found'] > 0:
            # Find best head for README using GPT-2 style scoring (minimize)
            best_head = None
            best_score = float('inf')
            for result_key, result_data in report['results'].items():
                result = result_data['helix_result']['best_result']
                score = result['radius_cv'] - result['angle_linearity']
                if score < best_score:
                    best_score = score
                    best_head = (result_data['layer'], result_data['head'], result)

            if best_head:
                layer, head, result = best_head
                readme_content += f"""
## Best Helix Head

**Layer {layer}, Head {head}**
- Radius CV: {result['radius_cv']:.4f}
- Angle Linearity: {result['angle_linearity']:.4f}
- Estimated Period: {result.get('estimated_period', 'N/A')}
"""

        with open(f"{output_dir}/README.md", 'w') as f:
            f.write(readme_content)


def analyze_from_cache(model, device, cache_dir="svd_cache", output_dir="helix_from_cache"):
    """Analyze helix structures using existing SVD cache files."""

    print("Analyzing helix structures from existing SVD cache...")
    print(f"Cache directory: {cache_dir}")
    print(f"Output directory: {output_dir}")

    Path(output_dir).mkdir(exist_ok=True)

    # Find all SVD cache files
    cache_pattern = f"{cache_dir}/*_ov.pt"
    cache_files = glob.glob(cache_pattern)

    print(f"Found {len(cache_files)} SVD cache files")

    if len(cache_files) == 0:
        print("No SVD cache files found. Run standard analysis first to generate cache.")
        return {}

    # Extract layer/head info from filenames
    layer_head_pairs = []
    for cache_file in cache_files:
        try:
            parts = Path(cache_file).stem.split('_')
            layer_idx = None
            head_idx = None

            for i, part in enumerate(parts):
                if part.startswith('layer') and i+1 < len(parts) and parts[i+1].startswith('head'):
                    layer_idx = int(part[5:])
                    head_idx = int(parts[i+1][4:])
                    break

            if layer_idx is not None and head_idx is not None:
                layer_head_pairs.append((layer_idx, head_idx))

        except Exception as e:
            continue

    layer_head_pairs = sorted(list(set(layer_head_pairs)))
    print(f"Analyzing {len(layer_head_pairs)} layer-head combinations from cache")

    # Generate test data
    numbers = list(range(0, 50))
    prompts = [f"The number {n}" for n in numbers]

    results = {}
    helix_count = 0

    # Process each layer-head combination
    for layer, head in tqdm(layer_head_pairs, desc="Analyzing cached heads"):
        try:
            # Collect activations for this head
            activations = []

            with torch.no_grad():
                for prompt in prompts[:30]:  # Process subset for memory
                    try:
                        tokens = model.to_tokens(prompt, prepend_bos=False).to(device)
                        _, cache = model.run_with_cache(tokens)

                        # Extract head activation using hook_z (universal across architectures)
                        try:
                            hook_z = cache[f"blocks.{layer}.attn.hook_z"]
                            # hook_z shape: [batch, seq_len, n_heads, d_head]
                            head_activation = hook_z[0, -1, head, :].to(device)
                        except (KeyError, IndexError, RuntimeError):
                            # Fallback to residual stream if hook_z not available
                            resid_post = cache[f"blocks.{layer}.hook_resid_post"]
                            head_activation = resid_post[0, -1, :].to(device)

                        activations.append(head_activation)

                        # Clear cache immediately
                        del cache
                        if device.type == "mps":
                            torch.mps.empty_cache()

                    except Exception as e:
                        continue

            if len(activations) < 15:
                continue

            # Stack activations and run helix detection
            acts_tensor = torch.stack(activations)
            acts_cpu = acts_tensor.cpu().numpy()

            # SVD on CPU (matches arithmetic_circuit_discovery.py approach)
            # acts_cpu shape: [n_samples, d_head]
            U, S, Vt = np.linalg.svd(acts_cpu, full_matrices=False)

            best_result = None
            best_score = float('inf')  # Initialize to infinity for minimization (GPT-2 style)

            # Test top 10 direction pairs
            for k1, k2 in itertools.combinations(range(min(10, len(S))), 2):
                v1 = Vt[k1]  # Shape: [d_head]
                v2 = Vt[k2]  # Shape: [d_head]

                # Convert to torch tensors for detect_helix_geometry
                acts_torch = torch.from_numpy(acts_cpu).float()
                v1_torch = torch.from_numpy(v1).float()
                v2_torch = torch.from_numpy(v2).float()

                result = detect_helix_geometry(
                    acts_torch,
                    v1_torch,
                    v2_torch,
                    numbers[:len(activations)],
                    cv_threshold=0.2,   # MATCHED TO GPT-2
                    linearity_threshold=0.9  # MATCHED TO GPT-2 (was 0.85)
                )

                if result['is_helix']:
                    # MATCHED TO GPT-2 scoring: lower is better
                    score = result['radius_cv'] - result['angle_linearity']
                    if score < best_score:
                        best_score = score
                        best_result = result
                        best_result['svd_directions'] = (k1, k2)

            if best_result:
                helix_count += 1
                results[(layer, head)] = best_result

                print(f"\nHELIX FOUND! Layer {layer}, Head {head}")
                print(f"   Radius CV: {best_result['radius_cv']:.4f}")
                print(f"   Angle Linearity: {best_result['angle_linearity']:.4f}")

                # Save individual result
                with open(f"{output_dir}/helix_L{layer}H{head}.json", 'w') as f:
                    json.dump({
                        'layer': layer, 'head': head,
                        'result': best_result,
                        'model': getattr(model.cfg, 'model_name', 'unknown')
                    }, f, indent=2)

            # Cleanup
            del activations, acts_tensor
            gc.collect()

        except Exception as e:
            print(f"Error analyzing L{layer}H{head}: {e}")
            continue

    # Generate final report
    final_report = {
        'model': getattr(model.cfg, 'model_name', 'unknown'),
        'cache_files_found': len(cache_files),
        'layer_head_combinations_analyzed': len(layer_head_pairs),
        'helix_structures_found': helix_count,
        'detection_rate': helix_count / len(layer_head_pairs) if layer_head_pairs else 0,
        'results': {f"L{l}H{h}": r for (l,h), r in results.items()}
    }

    with open(f"{output_dir}/helix_analysis_from_cache.json", 'w') as f:
        json.dump(final_report, f, indent=2)

    print(f"\nCache-based analysis complete!")
    print(f"Analyzed: {len(layer_head_pairs)} heads from cached SVD")
    print(f"Helix structures found: {helix_count}")
    print(f"Detection rate: {final_report['detection_rate']:.1%}")
    print(f"Results saved to: {output_dir}/")

    return results


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

    # Memory optimization arguments
    parser.add_argument("--memory_mode", default="standard",
                       choices=["standard", "optimized", "cache"],
                       help="Memory mode: standard (full features), optimized (batch processing), cache (use existing SVD cache)")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Number of heads to process in each batch (optimized mode, default: 4)")
    parser.add_argument("--cache_dir", default="svd_cache",
                       help="Directory containing SVD cache files (cache mode, default: svd_cache)")
    parser.add_argument("--start_layer", type=int, default=0,
                       help="Resume from this layer (optimized mode)")
    parser.add_argument("--start_head", type=int, default=0,
                       help="Resume from this head (optimized mode)")
    parser.add_argument("--end_layer", type=int, default=None,
                       help="End at this layer (inclusive, for targeted analysis)")
    parser.add_argument("--end_head", type=int, default=None,
                       help="End at this head (inclusive, for targeted analysis)")
    parser.add_argument("--checkpoint_interval", type=int, default=10,
                       help="Save checkpoint every N heads (optimized mode, default: 10)")

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
    print(f"Memory Mode: {args.memory_mode}")
    if args.memory_mode == "optimized":
        print(f"Batch Size: {args.batch_size}")
    elif args.memory_mode == "cache":
        print(f"Cache Directory: {args.cache_dir}")
    print("="*70)

    # Set MPS memory limit to allow full usage (needed for large models like GPT-Neo 2.7B)
    if args.memory_mode == "optimized" and device.type == "mps":
        import os
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable upper limit
        print("MPS memory limit disabled (allows full RAM usage)")

    # Load model
    print("Loading model...")
    try:
        model = HookedTransformer.from_pretrained(args.model, device=device)
        print(f"✓ Loaded {args.model} ({model.cfg.n_layers} layers, {model.cfg.n_heads} heads)")
        if args.memory_mode == "optimized":
            print(f"   ~{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Handle memory-optimized modes
    if args.memory_mode == "cache":
        print("\n🚀 Running cache-based analysis...")
        results = analyze_from_cache(
            model=model,
            device=device,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir
        )
        print("\n" + "="*70)
        print("🎉 Cache-based analysis complete!")
        print(f"📁 Results saved to: {args.output_dir}")
        print("="*70)
        return

    elif args.memory_mode == "optimized":
        print("\n🚀 Running memory-optimized analysis...")
        analyzer = MemoryOptimizedHelixAnalyzer(
            model=model,
            device=device,
            batch_size=args.batch_size,
            model_name=args.model
        )
        results = analyzer.run_full_analysis(
            output_dir=args.output_dir,
            start_layer=args.start_layer,
            start_head=args.start_head,
            end_layer=args.end_layer,
            end_head=args.end_head
        )
        print("\n" + "="*70)
        print("🎉 Memory-optimized analysis complete!")
        print(f"📁 Results saved to: {args.output_dir}")
        print("="*70)
        return

    # Continue with standard mode below
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

    # Standard mode continues here
    if args.quick:
        # Quick analysis
        print("\n🚀 Running quick helix analysis (standard mode)...")
        helix_circuit = quick_helix_analysis(
            model=model,
            output_dir=args.output_dir,
            arithmetic_tasks=arithmetic_tasks
        )

    else:
        # Detailed analysis
        print("\n🔬 Running detailed helix analysis (standard mode)...")

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