#!/usr/bin/env python3
"""
Memory-Optimized Helix Analysis for Large Models

This script implements a memory-efficient version of the helix analysis
that can handle large models like GPT-Neo 2.7B by processing heads in batches
and using CPU fallback for SVD computations when necessary.
"""

import argparse
import sys
import gc
from pathlib import Path
import torch
import json
import numpy as np
from tqdm import tqdm

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.helix_visualization import detect_helix_geometry
from transformer_lens import HookedTransformer


class MemoryOptimizedHelixAnalyzer:
    """Memory-efficient helix analyzer for large models."""

    def __init__(self, model, device, batch_size=4):
        self.model = model
        self.device = device
        self.batch_size = batch_size
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

                # Perform helix analysis with CPU fallback for SVD
                helix_result = self._analyze_helix_cpu_fallback(activations, number_range)

                if helix_result:
                    results[(layer, head)] = {
                        'helix_result': helix_result,
                        'activations': activations.cpu().numpy() if isinstance(activations, torch.Tensor) else activations,
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

                            # Extract head activations (residual stream after attention)
                            resid_post = cache[f"blocks.{layer}.attn.hook_result"]
                            head_output = resid_post[0, -1, head, :].cpu()  # Last token, specific head

                            activations.append(head_output)

                            # Clear intermediate cache
                            del cache, resid_post

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

    def _analyze_helix_cpu_fallback(self, activations, numbers):
        """Perform helix analysis with CPU fallback for SVD."""
        try:
            # Move to CPU for SVD computation
            if isinstance(activations, torch.Tensor):
                acts_cpu = activations.cpu().numpy()
            else:
                acts_cpu = activations

            # Simple SVD on CPU
            U, S, Vt = np.linalg.svd(acts_cpu.T, full_matrices=False)

            best_result = None
            best_score = 0

            # Test top 5 SVD direction pairs
            for k1, k2 in itertools.combinations(range(min(5, len(S))), 2):
                v1 = Vt[k1]
                v2 = Vt[k2]

                # Project activations onto these directions
                coords = np.column_stack([acts_cpu @ v1, acts_cpu @ v2])

                # Use helix detection from existing utils
                result = detect_helix_geometry(
                    coords,
                    numbers,
                    cv_threshold=0.25,
                    linearity_threshold=0.8
                )

                if result['is_helix']:
                    score = result['angle_linearity'] * (1 - result['radius_cv'])
                    if score > best_score:
                        best_score = score
                        best_result = result
                        best_result['svd_directions'] = (k1, k2)
                        best_result['singular_values'] = S[:10].tolist()

            return {'best_result': best_result} if best_result else None

        except Exception as e:
            print(f"Error in helix analysis: {e}")
            return None

    def run_full_analysis(self, output_dir="helix_neo27b_optimized"):
        """Run complete helix analysis with memory optimization."""
        Path(output_dir).mkdir(exist_ok=True)

        print(f"🔍 Starting memory-optimized helix analysis for {self.model.cfg.name_or_path}")
        print(f"📊 Model: {self.model.cfg.n_layers} layers, {self.model.cfg.n_heads} heads")
        print(f"💾 Device: {self.device}, Batch size: {self.batch_size}")

        # Generate all layer-head pairs
        all_pairs = [
            (layer, head)
            for layer in range(self.model.cfg.n_layers)
            for head in range(self.model.cfg.n_heads)
        ]

        print(f"🎯 Total heads to analyze: {len(all_pairs)}")

        # Process in batches
        all_results = {}
        helix_found = 0

        for i in tqdm(range(0, len(all_pairs), self.batch_size), desc="Processing batches"):
            batch = all_pairs[i:i+self.batch_size]

            print(f"\n📦 Processing batch {i//self.batch_size + 1}/{(len(all_pairs)-1)//self.batch_size + 1}")
            print(f"   Heads: {batch}")

            batch_results = self.analyze_head_batch(batch)

            # Save batch results immediately
            if batch_results:
                all_results.update(batch_results)
                helix_found += len(batch_results)

                # Save intermediate results
                self._save_intermediate_results(all_results, output_dir, i//self.batch_size + 1)

            print(f"   ✓ Batch complete. Helix heads found so far: {helix_found}")

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
                    'layer': layer,
                    'head': head,
                    'helix_result': data['helix_result'],
                    'activation_shape': data['activations'].shape,
                    'number_count': len(data['numbers'])
                }

            # Save JSON report
            with open(f"{output_dir}/intermediate_results_batch_{batch_num}.json", 'w') as f:
                json.dump({
                    'batch_number': batch_num,
                    'total_helix_heads': len(results),
                    'results': json_safe_results
                }, f, indent=2)

        except Exception as e:
            print(f"Warning: Could not save intermediate results: {e}")

    def _generate_final_report(self, results, output_dir):
        """Generate final comprehensive report."""
        print(f"\n📈 Generating final report...")

        # Summary statistics
        total_heads = self.model.cfg.n_layers * self.model.cfg.n_heads
        helix_heads = len(results)

        print(f"🎯 Analysis Complete!")
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

            print(f"\n🏆 Best Helix Head: Layer {layer}, Head {head}")
            print(f"   Radius CV: {best_result['radius_cv']:.4f}")
            print(f"   Angle Linearity: {best_result['angle_linearity']:.4f}")
            print(f"   Estimated Period: {best_result.get('estimated_period', 'N/A')}")

        # Save final JSON report
        json_safe_results = {}
        for (layer, head), data in results.items():
            json_safe_results[f"L{layer}H{head}"] = {
                'layer': layer,
                'head': head,
                'helix_result': data['helix_result']
            }

        final_report = {
            'model_name': self.model.cfg.name_or_path,
            'total_heads_analyzed': total_heads,
            'helix_structures_found': helix_heads,
            'detection_rate': helix_heads/total_heads,
            'analysis_parameters': {
                'cv_threshold': 0.25,
                'linearity_threshold': 0.8,
                'device': str(self.device),
                'batch_size': self.batch_size
            },
            'results': json_safe_results
        }

        with open(f"{output_dir}/helix_analysis_report.json", 'w') as f:
            json.dump(final_report, f, indent=2)

        # Save README
        self._save_readme(output_dir, final_report)

        print(f"📁 Results saved to: {output_dir}/")

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
"""

        if report['helix_structures_found'] > 0:
            # Find best head for README
            best_head = None
            best_score = 0
            for result_key, result_data in report['results'].items():
                result = result_data['helix_result']['best_result']
                score = result['angle_linearity']
                if score > best_score:
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


def main():
    parser = argparse.ArgumentParser(description="Memory-Optimized Helix Analysis")
    parser.add_argument("--model", default="EleutherAI/gpt-neo-2.7B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default="helix_neo27b_optimized")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Number of heads to process in each batch")

    args = parser.parse_args()

    # Setup device
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print("🧮 Memory-Optimized Helix Analysis")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print("="*50)

    # Load model with memory optimization
    print("Loading model...")
    try:
        # Set memory fraction for MPS
        if device.type == "mps":
            import os
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.7'

        model = HookedTransformer.from_pretrained(args.model, device=device)
        print(f"✓ Loaded {args.model}")
        print(f"  {model.cfg.n_layers} layers, {model.cfg.n_heads} heads")
        print(f"  ~{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Initialize analyzer
    analyzer = MemoryOptimizedHelixAnalyzer(model, device, args.batch_size)

    # Run analysis
    results = analyzer.run_full_analysis(args.output_dir)

    print(f"\n🎉 Analysis complete! Results in {args.output_dir}/")


if __name__ == "__main__":
    main()