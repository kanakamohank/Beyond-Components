#!/usr/bin/env python3
"""
Complete GPT-Neo 2.7B Helix Analysis - Layers 20-25
Focus on the critical discovery zone where helix structures are expected.
"""

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
                                'score': score,
                                'estimated_period': 2 * np.pi / (angles_unwrapped[-1] - angles_unwrapped[0]) * len(numbers) if angle_linearity > 0.5 else None
                            }

                except Exception as e:
                    continue

        if best_result is None:
            return {'is_helix': False, 'radius_cv': 1.0, 'angle_linearity': 0.0, 'score': 0.0}

        return best_result

    except Exception as e:
        print(f"Error in helix detection: {e}")
        return {'is_helix': False, 'radius_cv': 1.0, 'angle_linearity': 0.0, 'score': 0.0}


def analyze_remaining_heads():
    """Analyze the remaining heads in layers 20-25 where helix structures are expected."""

    print("🎯 GPT-Neo 2.7B Critical Zone Analysis - Layers 20-25")
    print("="*60)

    device = torch.device("mps")
    output_dir = Path("helix_neo27b_critical_zone")
    output_dir.mkdir(exist_ok=True)

    # Load model
    print("Loading GPT-Neo 2.7B...")
    model = HookedTransformer.from_pretrained("EleutherAI/gpt-neo-2.7B", device=device)
    print(f"✓ Model loaded: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads")

    # Focus on critical layers (20-25) where helix structures are expected
    target_layers = range(20, 26)  # Layers 20-25

    # Generate test data (matching successful GPT-2 analysis)
    numbers = list(range(0, 50))
    prompts = [f"The number {n}" for n in numbers]

    results = {}
    helix_count = 0
    total_heads_analyzed = 0

    print(f"\n🔍 Analyzing critical layers {list(target_layers)} (expected helix zone)")
    print(f"📊 Total heads to analyze: {len(target_layers) * model.cfg.n_heads}")

    for layer in target_layers:
        print(f"\n📍 Layer {layer} ({layer/model.cfg.n_layers*100:.1f}% depth)")

        for head in range(model.cfg.n_heads):
            total_heads_analyzed += 1
            print(f"  Head {head:2d}... ", end="", flush=True)

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
                    print("❌ insufficient data")
                    continue

                # Stack activations and run helix detection
                acts_tensor = torch.stack(activations)
                result = simple_helix_detection(acts_tensor, numbers[:len(activations)])

                if result.get('is_helix', False):
                    helix_count += 1
                    results[(layer, head)] = result

                    print(f"✅ HELIX FOUND!")
                    print(f"     CV: {result['radius_cv']:.4f}, Linearity: {result['angle_linearity']:.4f}, Score: {result['score']:.4f}")

                    # Save individual result immediately
                    result_data = {
                        'layer': layer,
                        'head': head,
                        'model': 'EleutherAI/gpt-neo-2.7B',
                        'result': result,
                        'depth_percentage': layer / model.cfg.n_layers * 100
                    }

                    with open(output_dir / f"helix_L{layer}H{head}.json", 'w') as f:
                        json.dump(result_data, f, indent=2)

                    print(f"     💾 Saved to helix_L{layer}H{head}.json")

                else:
                    print("❌")

                # Cleanup memory aggressively
                del activations, acts_tensor
                gc.collect()

            except Exception as e:
                print(f"❌ Error: {e}")
                continue

    # Generate final summary report
    final_report = {
        'model': 'EleutherAI/gpt-neo-2.7B',
        'analysis_type': 'critical_zone_layers_20_25',
        'layers_analyzed': list(target_layers),
        'total_heads_analyzed': total_heads_analyzed,
        'helix_structures_found': helix_count,
        'detection_rate': helix_count / total_heads_analyzed if total_heads_analyzed > 0 else 0,
        'helix_threshold_parameters': {
            'cv_threshold': 0.2,
            'linearity_threshold': 0.85,
            'method': 'improved_svd_detection'
        },
        'discovered_helixes': {f"L{l}H{h}": r for (l,h), r in results.items()}
    }

    with open(output_dir / "critical_zone_analysis_report.json", 'w') as f:
        json.dump(final_report, f, indent=2)

    print(f"\n🎉 Critical Zone Analysis Complete!")
    print(f"📊 Layers analyzed: {list(target_layers)}")
    print(f"🎯 Total heads analyzed: {total_heads_analyzed}")
    print(f"✨ Helix structures found: {helix_count}")
    if helix_count > 0:
        print(f"📈 Detection rate: {final_report['detection_rate']:.1%}")
        print(f"\n🏆 Discovered Helixes:")
        for (layer, head), result in results.items():
            print(f"   Layer {layer}, Head {head}: CV={result['radius_cv']:.4f}, Linearity={result['angle_linearity']:.4f}")
    else:
        print("❌ No helix structures detected in critical zone")
        print("🤔 This suggests GPT-Neo 2.7B may not use helical arithmetic encoding")

    print(f"📁 Results saved to: {output_dir}/")

    return results


if __name__ == "__main__":
    analyze_remaining_heads()