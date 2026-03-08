#!/usr/bin/env python3
"""
Ultra-Lightweight Helix Analysis for GPT-Neo 2.7B

This version avoids the MaskedTransformerCircuit entirely and implements
direct helix detection to minimize memory usage.
"""

import argparse
import sys
import gc
from pathlib import Path
import torch
import json
import numpy as np
from tqdm import tqdm
import glob
import itertools

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from transformer_lens import HookedTransformer


def simple_helix_detection(activations, numbers, cv_threshold=0.2, linearity_threshold=0.85):
    """Lightweight helix detection without heavy dependencies."""
    try:
        # Convert to numpy for SVD
        if isinstance(activations, torch.Tensor):
            acts = activations.cpu().numpy()
        else:
            acts = activations

        # Simple SVD on CPU (much lighter than the full circuit analysis)
        U, S, Vt = np.linalg.svd(acts.T, full_matrices=False)

        best_result = None
        best_score = 0

        # Test top 10 direction pairs (matching successful analyses)
        max_dirs = min(10, len(S)-1)

        for k1 in range(max_dirs):
            for k2 in range(k1+1, max_dirs+1):
                try:
                    v1, v2 = Vt[k1], Vt[k2]

                    # Project to 2D
                    coords = np.column_stack([acts @ v1, acts @ v2])

                    # Calculate radius consistency (CV)
                    radii = np.linalg.norm(coords, axis=1)
                    radius_mean = np.mean(radii)

                    if radius_mean < 1e-6:
                        continue

                    radius_cv = np.std(radii) / radius_mean

                    # Calculate angle linearity
                    angles = np.arctan2(coords[:, 1], coords[:, 0])

                    # Improved linearity check (correlation with numbers)
                    if len(angles) == len(numbers) and len(angles) > 5:
                        try:
                            # Unwrap angles more carefully
                            angles_unwrapped = np.unwrap(angles)

                            # Multiple linearity checks
                            if len(set(numbers)) > 3:  # Ensure variety in numbers
                                corr_matrix = np.corrcoef(numbers, angles_unwrapped)
                                angle_linearity = abs(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0

                                # Additional check: R-squared from linear fit
                                if angle_linearity > 0.5:
                                    try:
                                        z = np.polyfit(numbers, angles_unwrapped, 1)
                                        p = np.poly1d(z)
                                        ss_res = np.sum((angles_unwrapped - p(numbers)) ** 2)
                                        ss_tot = np.sum((angles_unwrapped - np.mean(angles_unwrapped)) ** 2)
                                        r_squared = 1 - (ss_res / (ss_tot + 1e-8))
                                        angle_linearity = max(angle_linearity, abs(r_squared))
                                    except:
                                        pass
                            else:
                                angle_linearity = 0
                        except:
                            angle_linearity = 0
                    else:
                        angle_linearity = 0

                    # Check if it's a helix
                    if radius_cv < cv_threshold and angle_linearity > linearity_threshold:
                        score = angle_linearity * (1 - radius_cv)
                        if score > best_score:
                            best_score = score
                            best_result = {
                                'is_helix': True,
                                'radius_cv': radius_cv,
                                'angle_linearity': angle_linearity,
                                'svd_directions': (k1, k2),
                                'singular_values': S[:5].tolist(),
                                'estimated_period': 2 * np.pi / (angles_unwrapped[-1] - angles_unwrapped[0]) * len(numbers) if angle_linearity > 0.5 else None
                            }

                except Exception as e:
                    continue

        if best_result is None:
            return {'is_helix': False, 'radius_cv': 1.0, 'angle_linearity': 0.0}

        return best_result

    except Exception as e:
        print(f"Error in helix detection: {e}")
        return {'is_helix': False, 'radius_cv': 1.0, 'angle_linearity': 0.0}


def analyze_from_existing_cache(model, device, cache_dir="svd_cache", output_dir="helix_from_cache"):
    """Analyze helix structures using existing SVD cache files."""

    print("🔍 Analyzing helix structures from existing SVD cache...")
    print(f"📁 Cache directory: {cache_dir}")
    print(f"💾 Output directory: {output_dir}")

    Path(output_dir).mkdir(exist_ok=True)

    # Find all SVD cache files for this model
    model_name_clean = "gpt-neo-2.7B"  # Hard-coded since we know we're analyzing GPT-Neo 2.7B
    cache_pattern = f"{cache_dir}/{model_name_clean}_*_ov.pt"
    cache_files = glob.glob(cache_pattern)

    print(f"📊 Found {len(cache_files)} SVD cache files")

    if len(cache_files) == 0:
        print("❌ No SVD cache files found. Need to run original analysis first.")
        return {}

    # Extract layer/head info from filenames
    layer_head_pairs = []
    for cache_file in cache_files:
        try:
            # Parse filename like: gpt-neo-2.7B_32_20_2560_128_d1d57596_layer15_head3_ov.pt
            parts = Path(cache_file).stem.split('_')
            layer_idx = None
            head_idx = None

            for i, part in enumerate(parts):
                if part.startswith('layer') and i+1 < len(parts) and parts[i+1].startswith('head'):
                    layer_idx = int(part[5:])  # Remove 'layer' prefix
                    head_idx = int(parts[i+1][4:])  # Remove 'head' prefix
                    break

            if layer_idx is not None and head_idx is not None:
                layer_head_pairs.append((layer_idx, head_idx))

        except Exception as e:
            continue

    layer_head_pairs = sorted(list(set(layer_head_pairs)))
    print(f"🎯 Analyzing {len(layer_head_pairs)} layer-head combinations from cache")

    # Generate test data (matching successful GPT-2 analysis)
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
                for prompt in prompts[:30]:  # Process in batches for memory
                    try:
                        tokens = model.to_tokens(prompt, prepend_bos=False).to(device)
                        _, cache = model.run_with_cache(tokens)

                        # Extract head activation using improved method
                        try:
                            attn_result = cache[f"blocks.{layer}.attn.hook_result"]
                            if attn_result.ndim >= 4 and attn_result.size(2) > head:
                                head_activation = attn_result[0, -1, head, :].to(device)
                            else:
                                resid_post = cache[f"blocks.{layer}.hook_resid_post"]
                                head_activation = resid_post[0, -1, :].to(device)
                        except (KeyError, IndexError, RuntimeError):
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

            # Stack activations and run improved helix detection
            acts_tensor = torch.stack(activations)
            result = simple_helix_detection(acts_tensor, numbers[:len(activations)])

            if result.get('is_helix', False):
                helix_count += 1
                results[(layer, head)] = result

                print(f"\n✅ HELIX FOUND! Layer {layer}, Head {head}")
                print(f"   Radius CV: {result['radius_cv']:.4f}")
                print(f"   Angle Linearity: {result['angle_linearity']:.4f}")
                print(f"   Score: {result.get('score', 0):.4f}")

                # Save individual result immediately
                with open(f"{output_dir}/helix_L{layer}H{head}.json", 'w') as f:
                    json.dump({
                        'layer': layer, 'head': head,
                        'result': result,
                        'model': model.cfg.name_or_path
                    }, f, indent=2)

            # Cleanup memory aggressively
            del activations, acts_tensor
            gc.collect()

        except Exception as e:
            print(f"Error analyzing L{layer}H{head}: {e}")
            continue

    # Generate final report
    final_report = {
        'model': model.cfg.name_or_path,
        'cache_files_found': len(cache_files),
        'layer_head_combinations_analyzed': len(layer_head_pairs),
        'helix_structures_found': helix_count,
        'detection_rate': helix_count / len(layer_head_pairs) if layer_head_pairs else 0,
        'analysis_parameters': {
            'cv_threshold': 0.2,
            'linearity_threshold': 0.85,
            'number_range': f"0-{len(numbers)-1}",
            'svd_directions_tested': 10,
            'method': 'cache_based_improved_detection'
        },
        'results': {f"L{l}H{h}": r for (l,h), r in results.items()}
    }

    with open(f"{output_dir}/helix_analysis_from_cache.json", 'w') as f:
        json.dump(final_report, f, indent=2)

    print(f"\n🎉 Cache-based analysis complete!")
    print(f"📊 Cache files found: {len(cache_files)}")
    print(f"🎯 Analyzed: {len(layer_head_pairs)} heads from cached SVD")
    print(f"✨ Helix structures found: {helix_count}")
    print(f"📈 Detection rate: {final_report['detection_rate']:.1%}")
    print(f"📁 Results saved to: {output_dir}/")

    return results


def analyze_single_head_minimal(model, layer, head, device):
    """Analyze a single head with minimal memory footprint."""
    try:
        # Generate test data (expanded range for better helix detection)
        numbers = list(range(0, 50))  # Expanded range matching successful analyses
        prompts = [f"The number {n}" for n in numbers]

        activations = []

        # Process one prompt at a time to minimize memory
        for prompt in prompts:
            try:
                # Tokenize
                tokens = model.to_tokens(prompt, prepend_bos=False)

                with torch.no_grad():
                    # Get just the specific activation we need
                    _, cache = model.run_with_cache(tokens)

                    # Extract head output - use residual stream after attention
                    try:
                        # Try attention result hook first
                        attn_result = cache[f"blocks.{layer}.attn.hook_result"]
                        if attn_result.ndim >= 4 and attn_result.size(2) > head:
                            head_activation = attn_result[0, -1, head, :].to(device)
                        else:
                            # Fallback to residual stream
                            resid_post = cache[f"blocks.{layer}.hook_resid_post"]
                            head_activation = resid_post[0, -1, :].to(device)
                    except (KeyError, IndexError, RuntimeError):
                        # Final fallback: use residual stream
                        resid_post = cache[f"blocks.{layer}.hook_resid_post"]
                        head_activation = resid_post[0, -1, :].to(device)

                    activations.append(head_activation)

                    # Immediately clear cache
                    del cache, resid_post

                # Clear MPS cache after each prompt
                if device.type == "mps":
                    torch.mps.empty_cache()

            except Exception as e:
                print(f"Skipping prompt '{prompt}': {e}")
                continue

        if len(activations) < 10:
            return None

        # Stack and analyze
        acts_tensor = torch.stack(activations)
        result = simple_helix_detection(acts_tensor, numbers)

        # Clear activations
        del activations, acts_tensor
        gc.collect()

        return result if result['is_helix'] else None

    except Exception as e:
        print(f"Error analyzing L{layer}H{head}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Ultra-Lite Helix Analysis")
    parser.add_argument("--model", default="EleutherAI/gpt-neo-2.7B")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output_dir", default="helix_neo27b_lite")
    parser.add_argument("--start_layer", type=int, default=0, help="Resume from this layer")
    parser.add_argument("--start_head", type=int, default=0, help="Resume from this head")
    parser.add_argument("--use_cache", action="store_true",
                       help="Use existing SVD cache files for analysis (faster, leverages previous work)")
    parser.add_argument("--cache_dir", default="svd_cache", help="Directory containing SVD cache files")

    args = parser.parse_args()

    device = torch.device(args.device)
    Path(args.output_dir).mkdir(exist_ok=True)

    print("🚀 Ultra-Lite Helix Analysis")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Starting from: Layer {args.start_layer}, Head {args.start_head}")
    print("="*50)

    # Load model
    print("Loading model...")
    try:
        model = HookedTransformer.from_pretrained(args.model, device=device)
        print(f"✓ Model loaded: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return

    # Analyze heads one by one
    results = {}
    total_heads = model.cfg.n_layers * model.cfg.n_heads
    processed = 0
    helix_found = 0

    # Choose analysis method
    if args.use_cache:
        print(f"\n🚀 Using cache-based analysis from {args.cache_dir}")
        results = analyze_from_existing_cache(model, device, args.cache_dir, args.output_dir)
        processed = len(results)
        helix_found = len(results)
        total_heads = processed  # Only count analyzed heads
    else:
        print(f"\n🔍 Analyzing {total_heads} heads...")

        for layer in range(args.start_layer, model.cfg.n_layers):
            start_head = args.start_head if layer == args.start_layer else 0

            for head in range(start_head, model.cfg.n_heads):
                processed += 1
                print(f"\n[{processed:3d}/{total_heads}] Layer {layer:2d}, Head {head:2d}... ", end="", flush=True)

                try:
                    result = analyze_single_head_minimal(model, layer, head, device)

                    if result:
                        helix_found += 1
                        results[(layer, head)] = result
                        print(f"✓ HELIX! CV={result['radius_cv']:.3f}, Lin={result['angle_linearity']:.3f}")

                        # Save intermediate result
                        with open(f"{args.output_dir}/found_L{layer}H{head}.json", 'w') as f:
                            json.dump({
                                'layer': layer, 'head': head,
                                'result': result,
                                'progress': f"{processed}/{total_heads}"
                            }, f, indent=2)
                    else:
                        print("✗")

                except Exception as e:
                    print(f"❌ {e}")
                    continue

                # Save checkpoint every 10 heads
                if processed % 10 == 0:
                    checkpoint = {
                        'last_layer': layer,
                        'last_head': head,
                        'total_processed': processed,
                        'helix_found': helix_found,
                        'results_count': len(results)
                    }

                    with open(f"{args.output_dir}/checkpoint.json", 'w') as f:
                        json.dump(checkpoint, f, indent=2)

                    print(f"   📊 Checkpoint: {helix_found}/{processed} helixes found ({helix_found/processed*100:.1f}%)")

    # Final report
    print(f"\n🎉 Analysis Complete!")
    print(f"   Processed: {processed}/{total_heads} heads")
    print(f"   Helix structures found: {helix_found}")
    print(f"   Detection rate: {helix_found/processed*100:.1f}%")

    if results:
        # Find best helix
        best_head = max(results.items(), key=lambda x: x[1]['angle_linearity'])
        (layer, head), result = best_head

        print(f"\n🏆 Best Helix: Layer {layer}, Head {head}")
        print(f"   Radius CV: {result['radius_cv']:.4f}")
        print(f"   Angle Linearity: {result['angle_linearity']:.4f}")
        print(f"   Period: {result.get('estimated_period', 'N/A')}")

    # Save final report
    final_report = {
        'model': args.model,
        'total_heads': total_heads,
        'processed': processed,
        'helix_found': helix_found,
        'detection_rate': helix_found/processed if processed > 0 else 0,
        'results': {f"L{l}H{h}": r for (l,h), r in results.items()}
    }

    with open(f"{args.output_dir}/final_report.json", 'w') as f:
        json.dump(final_report, f, indent=2)

    print(f"\n📁 Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()