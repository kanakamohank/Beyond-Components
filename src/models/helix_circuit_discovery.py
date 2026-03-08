"""
Helix-based Arithmetic Circuit Discovery

This module implements the Neel and Tegmark/Kattamaneni helix approach for discovering
arithmetic circuits in transformer models. It provides a complete framework for:

1. Detecting helical number representations in attention heads
2. Analyzing trigonometric patterns in arithmetic computations
3. Visualizing the "Clock Algorithm" for mathematical operations
4. Comparing helix-based discoveries with traditional SVD approaches

Key Features:
- Helical geometry detection using SVD directions
- 3D interactive helix visualizations
- Phase shift analysis for arithmetic operations
- Integration with existing MaskedTransformerCircuit framework
- Comprehensive comparison tools

Based on research:
- "Language Models Use Trigonometry to Do Addition"
- "The Clock and the Pizza: Two Stories in Mechanistic Explanation"
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import itertools
from typing import Dict, List, Tuple, Optional, Any, Union
import os
import json
from pathlib import Path
from transformer_lens import HookedTransformer
from tqdm import tqdm
import logging

# Import existing framework components
from .masked_transformer_circuit import MaskedTransformerCircuit

logger = logging.getLogger(__name__)


class HelixArithmeticCircuit:
    """
    Helix-based arithmetic circuit discovery and analysis.

    This class implements the complete helix approach for finding and analyzing
    arithmetic circuits in transformer models, building on the SVD framework
    from MaskedTransformerCircuit.
    """

    def __init__(self,
                 model: HookedTransformer,
                 base_circuit: Optional[MaskedTransformerCircuit] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize helix-based circuit discovery.

        Args:
            model: HookedTransformer model to analyze
            base_circuit: Optional existing MaskedTransformerCircuit for comparison
            device: Device to run computations on
        """
        self.model = model
        self.device = device if device is not None else next(model.parameters()).device

        # Create or use existing base circuit
        if base_circuit is not None:
            self.base_circuit = base_circuit
        else:
            self.base_circuit = MaskedTransformerCircuit(
                model=model,
                device=self.device,
                cache_svd=True,
                mask_init_value=0.99
            )

        # Configuration
        self.cfg = model.cfg
        self.n_layers = model.cfg.n_layers
        self.n_heads = model.cfg.n_heads
        self.d_model = model.cfg.d_model

        # Helix discovery results
        self.helix_heads = {}  # {(layer, head): helix_info}
        self.arithmetic_patterns = {}  # Store discovered patterns

        # Analysis parameters
        self.helix_params = {
            'cv_threshold': 0.2,        # Max coefficient of variation for radius
            'linearity_threshold': 0.9, # Min correlation for angle linearity
            'top_k_directions': 10,     # Number of SVD directions to test
            'min_period': 2.0,          # Minimum reasonable period
            'max_period': 100.0         # Maximum reasonable period
        }

    def collect_number_activations(self,
                                 layer: int,
                                 prompt_template: str = "The number {n}",
                                 n_range: range = range(0, 100),
                                 position: str = "last") -> Tuple[torch.Tensor, List[int]]:
        """
        Collect activations for number tokens to analyze geometric structure.

        Args:
            layer: Layer to extract activations from
            prompt_template: Template for prompts containing numbers
            n_range: Range of numbers to test
            position: Position to extract ('last', 'number', or specific index)

        Returns:
            Tuple of (activations tensor [N, d_model], valid numbers list)
        """
        activations = []
        valid_numbers = []

        for n in tqdm(n_range, desc=f"Collecting activations for layer {layer}"):
            prompt = prompt_template.format(n=n)
            tokens = self.model.to_tokens(prompt)

            # Determine position to extract activation
            if position == "last":
                pos_idx = -1
            elif position == "number":
                # Try to find the number token
                str_tokens = self.model.to_str_tokens(tokens[0])
                pos_idx = None
                n_str = str(n)

                for i, token in enumerate(str_tokens):
                    if n_str in token.strip():
                        pos_idx = i
                        break

                if pos_idx is None:
                    logger.warning(f"Could not find number {n} in tokens for prompt '{prompt}'")
                    continue
            else:
                pos_idx = int(position)

            # Get activations
            with torch.no_grad():
                _, cache = self.model.run_with_cache(tokens)

            # Extract residual stream before the layer
            hook_name = f'blocks.{layer}.hook_resid_pre'
            if hook_name in cache:
                resid = cache[hook_name]
                activation = resid[0, pos_idx, :].cpu()

                activations.append(activation)
                valid_numbers.append(n)
            else:
                logger.warning(f"Hook {hook_name} not found in cache")

        if len(activations) == 0:
            raise ValueError("No valid activations collected")

        return torch.stack(activations), valid_numbers

    def detect_helix_in_head(self,
                           layer: int,
                           head: int,
                           activations: torch.Tensor,
                           numbers: List[int]) -> Dict[str, Any]:
        """
        Detect helical structure in a specific attention head.

        Args:
            layer: Layer index
            head: Head index
            activations: Activations tensor [N, d_model]
            numbers: Corresponding number values

        Returns:
            Dictionary with helix detection results
        """
        # Get SVD components for this head
        head_key = f'differential_head_{layer}_{head}'

        # Use OV circuit directions (where computation happens)
        ov_cache_key = f"{head_key}_ov"
        if ov_cache_key not in self.base_circuit.svd_cache:
            logger.warning(f"SVD cache not found for {ov_cache_key}")
            return {'is_helix': False, 'error': 'SVD cache not found'}

        U_ov, S_ov, Vh_ov, _ = self.base_circuit.svd_cache[ov_cache_key]

        # Test pairs of SVD directions for helix structure
        results = []
        top_k = min(self.helix_params['top_k_directions'], Vh_ov.shape[0])

        for k1, k2 in itertools.combinations(range(top_k), 2):
            direction_1 = Vh_ov[k1].cpu()
            direction_2 = Vh_ov[k2].cpu()

            # Project activations onto 2D plane
            coords = torch.stack([
                activations @ direction_1,
                activations @ direction_2
            ], dim=1)

            # Compute polar coordinates
            radii = coords.norm(dim=1)
            angles = torch.atan2(coords[:, 1], coords[:, 0]).numpy()

            # Helix quality metrics
            radius_cv = (radii.std() / radii.mean()).item() if radii.mean() > 0 else float('inf')

            # Angle linearity with numbers
            mean_delta = np.abs(np.diff(angles)).mean()
            if mean_delta > np.pi * 0.8:
                angle_linearity = 0.0  # Unwrap unreliable
            else:
                unwrapped_angles = np.unwrap(angles)
                if len(numbers) > 1:
                    angle_linearity = abs(np.corrcoef(numbers, unwrapped_angles)[0, 1])
                else:
                    angle_linearity = 0.0

            # Estimate period
            if len(numbers) > 2:
                angle_diffs = np.diff(np.unwrap(angles))
                number_diffs = np.diff(numbers)
                valid_mask = number_diffs != 0

                if valid_mask.sum() > 0:
                    periods = 2 * np.pi / (angle_diffs[valid_mask] / number_diffs[valid_mask])
                    finite_periods = periods[np.isfinite(periods)]

                    if len(finite_periods) > 0:
                        estimated_period = np.median(finite_periods)
                    else:
                        estimated_period = float('inf')
                else:
                    estimated_period = float('inf')
            else:
                estimated_period = float('inf')

            # Quality score (lower is better)
            quality_score = radius_cv - angle_linearity

            result = {
                'direction_indices': (k1, k2),
                'singular_values': (S_ov[k1].item(), S_ov[k2].item()),
                'radius_cv': radius_cv,
                'angle_linearity': angle_linearity,
                'estimated_period': estimated_period,
                'quality_score': quality_score,
                'coords': coords.numpy(),
                'radii': radii.numpy(),
                'angles': angles,
                'direction_1': direction_1,
                'direction_2': direction_2
            }

            results.append(result)

        # Sort by quality (best first)
        results.sort(key=lambda x: x['quality_score'])

        # Check if best result meets helix criteria
        if results:
            best = results[0]
            is_helix = (
                best['radius_cv'] < self.helix_params['cv_threshold'] and
                best['angle_linearity'] > self.helix_params['linearity_threshold'] and
                self.helix_params['min_period'] <= best['estimated_period'] <= self.helix_params['max_period']
            )

            return {
                'is_helix': is_helix,
                'best_result': best,
                'all_results': results,
                'layer': layer,
                'head': head
            }
        else:
            return {'is_helix': False, 'error': 'No direction pairs found'}

    def find_arithmetic_heads(self,
                            arithmetic_tasks: Optional[Dict[str, Dict]] = None,
                            layers_to_test: Optional[List[int]] = None) -> Dict[str, List[Tuple[int, int]]]:
        """
        Find heads that exhibit helical structure for arithmetic operations.

        Args:
            arithmetic_tasks: Dict of {task_name: {template, range}} for testing
            layers_to_test: List of layer indices to test (default: middle layers)

        Returns:
            Dictionary mapping task names to lists of (layer, head) tuples
        """
        if arithmetic_tasks is None:
            arithmetic_tasks = {
                'numbers': {
                    'template': 'The number is {n}.',
                    'range': range(10, 50)
                },
                'addition': {
                    'template': '{n} + 5 =',
                    'range': range(10, 40)
                },
                'counting': {
                    'template': 'Count: {n}',
                    'range': range(1, 21)
                }
            }

        if layers_to_test is None:
            # Test middle layers where arithmetic reasoning typically occurs
            layers_to_test = [
                self.n_layers // 4,
                self.n_layers // 2,
                3 * self.n_layers // 4
            ]
            layers_to_test = [l for l in layers_to_test if l < self.n_layers]

        found_heads = {}

        for task_name, task_config in arithmetic_tasks.items():
            print(f"\nTesting task: {task_name}")
            found_heads[task_name] = []

            for layer in layers_to_test:
                print(f"  Layer {layer}:")

                try:
                    # Collect activations for this task
                    activations, numbers = self.collect_number_activations(
                        layer=layer,
                        prompt_template=task_config['template'],
                        n_range=task_config['range']
                    )

                    print(f"    Collected {len(numbers)} activations")

                    # Test heads in this layer
                    for head in range(self.n_heads):
                        helix_result = self.detect_helix_in_head(layer, head, activations, numbers)

                        if helix_result['is_helix']:
                            best = helix_result['best_result']
                            print(f"    Head {head}: ✓ HELIX FOUND!")
                            print(f"      CV: {best['radius_cv']:.3f}, "
                                  f"Linearity: {best['angle_linearity']:.3f}, "
                                  f"Period: {best['estimated_period']:.1f}")

                            found_heads[task_name].append((layer, head))

                            # Store detailed results
                            self.helix_heads[(layer, head)] = {
                                'task': task_name,
                                'helix_result': helix_result,
                                'activations': activations,
                                'numbers': numbers
                            }

                except Exception as e:
                    print(f"    Error in layer {layer}: {e}")

        return found_heads

    def visualize_helix_head(self,
                           layer: int,
                           head: int,
                           output_dir: Optional[str] = None,
                           show_3d: bool = True) -> Dict[str, Any]:
        """
        Create comprehensive visualizations for a helix head.

        Args:
            layer: Layer index
            head: Head index
            output_dir: Directory to save visualizations
            show_3d: Whether to create 3D interactive plots

        Returns:
            Dictionary with visualization results and figure objects
        """
        if (layer, head) not in self.helix_heads:
            raise ValueError(f"No helix data found for Layer {layer}, Head {head}")

        helix_data = self.helix_heads[(layer, head)]
        helix_result = helix_data['helix_result']['best_result']
        numbers = helix_data['numbers']

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        visualizations = {}

        # 1. 2D Helix Projection
        fig_2d, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        coords = helix_result['coords']
        radii = helix_result['radii']
        angles = helix_result['angles']

        # Scatter plot in SVD space
        scatter = ax1.scatter(coords[:, 0], coords[:, 1], c=numbers,
                            cmap='viridis', s=60, alpha=0.7, edgecolors='black')
        ax1.set_xlabel('SVD Direction 1', fontweight='bold')
        ax1.set_ylabel('SVD Direction 2', fontweight='bold')
        ax1.set_title(f'L{layer}H{head} Helix Projection\n'
                     f'CV: {helix_result["radius_cv"]:.3f}', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        plt.colorbar(scatter, ax=ax1, label='Number Value')

        # Angle linearity plot
        unwrapped_angles = np.unwrap(angles)
        ax2.scatter(numbers, unwrapped_angles, c=numbers, cmap='viridis', s=60, alpha=0.7)
        ax2.set_xlabel('Number Value', fontweight='bold')
        ax2.set_ylabel('Unwrapped Angle (rad)', fontweight='bold')
        ax2.set_title(f'Angle Linearity: {helix_result["angle_linearity"]:.3f}\n'
                     f'Period: {helix_result["estimated_period"]:.1f}', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Add trend line
        if len(numbers) > 1:
            z = np.polyfit(numbers, unwrapped_angles, 1)
            p = np.poly1d(z)
            ax2.plot(numbers, p(numbers), 'r--', linewidth=2, alpha=0.8)

        plt.tight_layout()

        if output_dir:
            fig_2d.savefig(output_dir / f'helix_2d_L{layer}H{head}.png', dpi=300, bbox_inches='tight')

        visualizations['fig_2d'] = fig_2d

        # 2. 3D Interactive Helix (if requested)
        if show_3d:
            fig_3d = go.Figure()

            # Main helix points
            fig_3d.add_trace(go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=numbers,
                mode='markers+lines',
                marker=dict(
                    size=8,
                    color=numbers,
                    colorscale='Viridis',
                    colorbar=dict(title="Number"),
                    line=dict(width=1, color='black')
                ),
                line=dict(color='rgba(255,107,53,0.6)', width=4),
                name='Number Helix',
                text=[f'n={n}' for n in numbers],
                hovertemplate='Number: %{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
            ))

            # Projection on base
            fig_3d.add_trace(go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=[min(numbers)] * len(numbers),
                mode='markers',
                marker=dict(size=4, color='gray', opacity=0.5),
                name='Projection',
                showlegend=False
            ))

            # Add ideal helix curve
            period = helix_result['estimated_period']
            if np.isfinite(period) and period > 0:
                t_ideal = np.linspace(min(numbers), max(numbers), 100)
                mean_radius = np.mean(radii)

                # Estimate phase offset
                phase_offset = unwrapped_angles[0] - 2*np.pi*numbers[0]/period if len(numbers) > 0 else 0

                x_ideal = mean_radius * np.cos(2*np.pi*t_ideal/period + phase_offset)
                y_ideal = mean_radius * np.sin(2*np.pi*t_ideal/period + phase_offset)

                fig_3d.add_trace(go.Scatter3d(
                    x=x_ideal, y=y_ideal, z=t_ideal,
                    mode='lines',
                    line=dict(color='red', width=3, dash='dash'),
                    name=f'Ideal Helix (T={period:.1f})',
                    opacity=0.7
                ))

            fig_3d.update_layout(
                title=f'Layer {layer}, Head {head} - 3D Helix Structure<br>'
                      f'<sub>Task: {helix_data["task"]}</sub>',
                scene=dict(
                    xaxis_title='SVD Direction 1',
                    yaxis_title='SVD Direction 2',
                    zaxis_title='Number Value',
                    bgcolor='white'
                ),
                width=800, height=700
            )

            if output_dir:
                fig_3d.write_html(output_dir / f'helix_3d_L{layer}H{head}.html')

            visualizations['fig_3d'] = fig_3d

        # 3. Comparison with standard SVD approach
        fig_comparison = self._create_comparison_plot(layer, head)

        if output_dir:
            fig_comparison.savefig(output_dir / f'comparison_L{layer}H{head}.png',
                                 dpi=300, bbox_inches='tight')

        visualizations['fig_comparison'] = fig_comparison

        return visualizations

    def _create_comparison_plot(self, layer: int, head: int) -> plt.Figure:
        """Create comparison between helix and standard SVD approaches."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Get helix data
        helix_data = self.helix_heads[(layer, head)]
        helix_result = helix_data['helix_result']['best_result']

        # Get standard SVD data
        head_key = f'differential_head_{layer}_{head}'
        if hasattr(self.base_circuit, 'ov_masks') and head_key in self.base_circuit.ov_masks:
            standard_mask = torch.sigmoid(self.base_circuit.ov_masks[head_key]).detach().cpu().numpy()
        else:
            standard_mask = np.ones(self.base_circuit.d_head + 1) * 0.5  # Default

        # Top row: Standard SVD approach
        # Mask values
        axes[0, 0].bar(range(len(standard_mask)), standard_mask,
                      color='steelblue', alpha=0.7)
        axes[0, 0].set_title('Standard SVD Masks', fontweight='bold')
        axes[0, 0].set_ylabel('Mask Value')
        axes[0, 0].grid(True, alpha=0.3)

        # Singular values
        ov_cache_key = f"{head_key}_ov"
        if ov_cache_key in self.base_circuit.svd_cache:
            _, S_ov, _, _ = self.base_circuit.svd_cache[ov_cache_key]
            S_values = S_ov.detach().cpu().numpy()[:len(standard_mask)]

            axes[0, 1].bar(range(len(S_values)), S_values,
                          color='orange', alpha=0.7)
            axes[0, 1].set_title('Singular Values', fontweight='bold')
            axes[0, 1].set_ylabel('Singular Value')
            axes[0, 1].grid(True, alpha=0.3)

            # Masked singular values
            masked_S = standard_mask * S_values
            axes[0, 2].bar(range(len(masked_S)), masked_S,
                          color='green', alpha=0.7)
            axes[0, 2].set_title('Masked Singular Values', fontweight='bold')
            axes[0, 2].set_ylabel('Masked Value')
            axes[0, 2].grid(True, alpha=0.3)

        # Bottom row: Helix approach
        # Helix quality metrics
        metrics = ['Radius CV', 'Angle Linearity', 'Period/10']
        values = [
            helix_result['radius_cv'],
            helix_result['angle_linearity'],
            helix_result['estimated_period'] / 10  # Scale for visualization
        ]
        colors = ['red' if v > 0.2 else 'green' for i, v in enumerate(values)]
        colors[1] = 'green' if values[1] > 0.9 else 'red'  # Linearity (higher is better)

        axes[1, 0].bar(metrics, values, color=colors, alpha=0.7)
        axes[1, 0].set_title('Helix Quality Metrics', fontweight='bold')
        axes[1, 0].set_ylabel('Metric Value')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Helix projection
        coords = helix_result['coords']
        numbers = helix_data['numbers']
        scatter = axes[1, 1].scatter(coords[:, 0], coords[:, 1], c=numbers,
                                   cmap='viridis', s=40, alpha=0.7)
        axes[1, 1].set_title('Helix Projection', fontweight='bold')
        axes[1, 1].set_xlabel('Direction 1')
        axes[1, 1].set_ylabel('Direction 2')
        axes[1, 1].set_aspect('equal')

        # Phase progression
        angles = np.unwrap(helix_result['angles'])
        axes[1, 2].plot(numbers, angles, 'o-', color='purple', alpha=0.7)
        axes[1, 2].set_title('Phase Progression', fontweight='bold')
        axes[1, 2].set_xlabel('Number Value')
        axes[1, 2].set_ylabel('Phase (rad)')
        axes[1, 2].grid(True, alpha=0.3)

        fig.suptitle(f'SVD vs Helix Comparison - Layer {layer}, Head {head}',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        return fig

    def analyze_arithmetic_operations(self,
                                    base_numbers: List[int] = None,
                                    operations: Dict[str, int] = None,
                                    output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze how arithmetic operations correspond to phase shifts in helix space.

        Args:
            base_numbers: Numbers to test operations on
            operations: Dict of {operation_name: shift_amount}
            output_dir: Directory to save analysis results

        Returns:
            Dictionary with operation analysis results
        """
        if base_numbers is None:
            base_numbers = list(range(10, 20))

        if operations is None:
            operations = {
                'add_1': 1,
                'add_2': 2,
                'subtract_1': -1,
                'multiply_by_2_mod_10': None  # Special case
            }

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        results = {'operations': {}}

        # Find best helix head for analysis
        if not self.helix_heads:
            print("No helix heads found. Run find_arithmetic_heads() first.")
            return results

        # Use the best helix head
        best_head = max(self.helix_heads.items(),
                       key=lambda x: x[1]['helix_result']['best_result']['angle_linearity'])
        (layer, head), helix_data = best_head

        print(f"Using Layer {layer}, Head {head} for operation analysis")

        # Get helix parameters
        helix_result = helix_data['helix_result']['best_result']
        period = helix_result['estimated_period']

        # Analyze each operation
        for op_name, shift in operations.items():
            print(f"\nAnalyzing operation: {op_name}")

            if shift is None:
                # Special handling for multiplication
                continue

            # Theoretical phase shift
            theoretical_phase_shift = 2 * np.pi * shift / period if np.isfinite(period) else None

            results['operations'][op_name] = {
                'shift_amount': shift,
                'theoretical_phase_shift': theoretical_phase_shift,
                'period': period
            }

        # Create visualization
        if output_dir:
            self._visualize_operation_analysis(results, base_numbers, output_dir)

        return results

    def _visualize_operation_analysis(self, results, base_numbers, output_dir):
        """Create visualizations for operation analysis."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Phase Shift vs Operation',
                'Predicted vs Actual Results',
                'Operation Visualization',
                'Helix Rotation Demo'
            ],
            specs=[[{'type': 'bar'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter3d'}]]
        )

        # Extract data for plotting
        operations = list(results['operations'].keys())
        shifts = [results['operations'][op]['shift_amount'] for op in operations]
        phase_shifts = [results['operations'][op]['theoretical_phase_shift'] for op in operations]

        # Phase shift comparison
        fig.add_trace(
            go.Bar(x=operations, y=phase_shifts, name='Theoretical Phase Shift'),
            row=1, col=1
        )

        fig.update_layout(
            title='Arithmetic Operation Analysis via Helix Phase Shifts',
            height=800
        )

        fig.write_html(output_dir / 'operation_analysis.html')

    def generate_comprehensive_report(self, output_dir: str) -> Dict[str, Any]:
        """
        Generate comprehensive report comparing helix and standard approaches.

        Args:
            output_dir: Directory to save report and visualizations

        Returns:
            Dictionary with complete analysis results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Compile all results
        report = {
            'model_info': {
                'model_name': getattr(self.cfg, 'model_name', 'unknown'),
                'n_layers': self.n_layers,
                'n_heads': self.n_heads,
                'd_model': self.d_model
            },
            'helix_summary': {
                'total_heads_tested': self.n_layers * self.n_heads,
                'helix_heads_found': len(self.helix_heads),
                'helix_heads_by_layer': {}
            },
            'comparison_metrics': {},
            'key_findings': []
        }

        # Organize helix heads by layer
        for (layer, head), data in self.helix_heads.items():
            if layer not in report['helix_summary']['helix_heads_by_layer']:
                report['helix_summary']['helix_heads_by_layer'][layer] = []

            report['helix_summary']['helix_heads_by_layer'][layer].append({
                'head': head,
                'task': data['task'],
                'quality_metrics': {
                    'radius_cv': data['helix_result']['best_result']['radius_cv'],
                    'angle_linearity': data['helix_result']['best_result']['angle_linearity'],
                    'period': data['helix_result']['best_result']['estimated_period']
                }
            })

        # Generate visualizations for all helix heads
        print("Generating visualizations for all helix heads...")
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)

        for (layer, head) in tqdm(self.helix_heads.keys(), desc="Creating visualizations"):
            head_viz_dir = viz_dir / f'L{layer}H{head}'
            self.visualize_helix_head(layer, head, str(head_viz_dir))

        # Create summary visualizations
        self._create_summary_visualizations(report, output_dir)

        # Save report as JSON
        with open(output_dir / 'helix_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Create markdown summary
        self._create_markdown_summary(report, output_dir)

        print(f"\n✓ Comprehensive report saved to {output_dir}")
        print(f"✓ Found {len(self.helix_heads)} helix heads across {len(set(l for l, h in self.helix_heads.keys()))} layers")

        return report

    def _create_summary_visualizations(self, report, output_dir):
        """Create summary visualizations for the report."""
        # Helix quality heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Create matrices for visualization
        cv_matrix = np.full((self.n_layers, self.n_heads), np.nan)
        linearity_matrix = np.full((self.n_layers, self.n_heads), np.nan)

        for (layer, head), data in self.helix_heads.items():
            result = data['helix_result']['best_result']
            cv_matrix[layer, head] = result['radius_cv']
            linearity_matrix[layer, head] = result['angle_linearity']

        # Radius CV heatmap
        im1 = ax1.imshow(cv_matrix, cmap='RdYlBu_r', aspect='auto')
        ax1.set_title('Radius CV (lower = better)', fontweight='bold')
        ax1.set_xlabel('Head')
        ax1.set_ylabel('Layer')
        plt.colorbar(im1, ax=ax1)

        # Angle linearity heatmap
        im2 = ax2.imshow(linearity_matrix, cmap='RdYlGn', aspect='auto')
        ax2.set_title('Angle Linearity (higher = better)', fontweight='bold')
        ax2.set_xlabel('Head')
        ax2.set_ylabel('Layer')
        plt.colorbar(im2, ax=ax2)

        plt.suptitle('Helix Quality Across All Heads', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'helix_quality_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_markdown_summary(self, report, output_dir):
        """Create markdown summary report."""
        with open(output_dir / 'README.md', 'w') as f:
            f.write("# Helix-Based Arithmetic Circuit Analysis\n\n")
            f.write("## Overview\n\n")
            f.write(f"This report analyzes arithmetic circuits using the helix approach on ")
            f.write(f"{report['model_info']['model_name']} ")
            f.write(f"({report['model_info']['n_layers']} layers, {report['model_info']['n_heads']} heads).\n\n")

            f.write("## Key Findings\n\n")
            f.write(f"- **Total heads analyzed**: {report['helix_summary']['total_heads_tested']}\n")
            f.write(f"- **Helix structures found**: {report['helix_summary']['helix_heads_found']}\n")
            f.write(f"- **Layers with helix heads**: {len(report['helix_summary']['helix_heads_by_layer'])}\n\n")

            f.write("## Helix Heads by Layer\n\n")
            for layer, heads in report['helix_summary']['helix_heads_by_layer'].items():
                f.write(f"### Layer {layer}\n\n")
                for head_info in heads:
                    f.write(f"- **Head {head_info['head']}** ({head_info['task']})\n")
                    f.write(f"  - Radius CV: {head_info['quality_metrics']['radius_cv']:.3f}\n")
                    f.write(f"  - Angle Linearity: {head_info['quality_metrics']['angle_linearity']:.3f}\n")
                    f.write(f"  - Period: {head_info['quality_metrics']['period']:.1f}\n")
                f.write("\n")

            f.write("## Files Generated\n\n")
            f.write("- `helix_analysis_report.json` - Complete analysis results\n")
            f.write("- `helix_quality_heatmap.png` - Quality metrics across all heads\n")
            f.write("- `visualizations/` - Individual head visualizations\n")
            f.write("  - `L{layer}H{head}/` - Per-head analysis\n")
            f.write("    - `helix_2d_L{layer}H{head}.png` - 2D helix projection\n")
            f.write("    - `helix_3d_L{layer}H{head}.html` - Interactive 3D visualization\n")
            f.write("    - `comparison_L{layer}H{head}.png` - SVD vs Helix comparison\n")


# Convenience functions for quick analysis
def quick_helix_analysis(model: HookedTransformer,
                        output_dir: str = "helix_analysis",
                        arithmetic_tasks: Optional[Dict] = None) -> HelixArithmeticCircuit:
    """
    Perform quick helix analysis on a model.

    Args:
        model: HookedTransformer to analyze
        output_dir: Directory to save results
        arithmetic_tasks: Custom arithmetic tasks (optional)

    Returns:
        HelixArithmeticCircuit instance with results
    """
    print("🔬 Starting Helix-Based Arithmetic Circuit Analysis")
    print("="*60)

    # Initialize helix circuit analyzer
    helix_circuit = HelixArithmeticCircuit(model)

    # Find arithmetic heads
    print("Phase 1: Finding arithmetic heads...")
    found_heads = helix_circuit.find_arithmetic_heads(arithmetic_tasks)

    print(f"Found helix structures in {len(helix_circuit.helix_heads)} heads")

    # Generate comprehensive report
    print("Phase 2: Generating comprehensive report...")
    report = helix_circuit.generate_comprehensive_report(output_dir)

    print(f"✓ Analysis complete! Results saved to {output_dir}")

    return helix_circuit


def compare_with_standard_svd(model: HookedTransformer,
                             base_circuit: MaskedTransformerCircuit,
                             output_dir: str = "helix_vs_svd_comparison"):
    """
    Compare helix approach with standard SVD circuit discovery.

    Args:
        model: HookedTransformer model
        base_circuit: Trained MaskedTransformerCircuit for comparison
        output_dir: Directory to save comparison results
    """
    print("🔍 Helix vs Standard SVD Comparison")
    print("="*60)

    # Run helix analysis
    helix_circuit = HelixArithmeticCircuit(model, base_circuit)

    # Find helix heads
    helix_heads = helix_circuit.find_arithmetic_heads()

    # Generate comparison report
    comparison_report = helix_circuit.generate_comprehensive_report(output_dir)

    print(f"✓ Comparison complete! Results saved to {output_dir}")

    return helix_circuit, comparison_report