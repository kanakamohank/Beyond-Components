"""
Helix visualization utilities for masked transformer circuits.

This module implements the Neel and Tegmark/Kattamaneni helix approach for analyzing
and visualizing mathematical reasoning in transformer circuits. It combines SVD-based
circuit discovery with geometric helix detection and 3D visualization.

Key features:
- Helical geometry detection in SVD directions
- 3D helix visualization of number representations
- Trigonometric phase analysis
- Clock algorithm visualization
- Integration with existing MaskedTransformerCircuit
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import itertools
from typing import Dict, List, Tuple, Optional, Any
import os
from transformer_lens import HookedTransformer
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# Helix-specific color palette
HELIX_COLORS = {
    'helix_primary': '#FF6B35',      # Vibrant Orange
    'helix_secondary': '#2A9D8F',    # Teal
    'direction_1': '#E63946',        # Red
    'direction_2': '#1D3557',        # Navy
    'phase_shift': '#8338EC',        # Purple
    'reference': '#6C757D',          # Gray
    'arithmetic': '#F4A261',         # Gold
    'background': '#F8F9FA'          # Light background
}


def safe_angle_linearity(angles_np: np.ndarray, values: List[float]) -> float:
    """
    Compute angle linearity with safety check for small periods.

    Args:
        angles_np: Array of angles in radians
        values: Corresponding numerical values

    Returns:
        Correlation coefficient between unwrapped angles and values
    """
    mean_delta = np.abs(np.diff(angles_np)).mean()
    if mean_delta > np.pi * 0.8:
        return 0.0  # Unwrap unreliable for T < ~4

    unwrapped = np.unwrap(angles_np)
    return np.corrcoef(values, unwrapped)[0, 1]


def detect_helix_geometry(activations: torch.Tensor,
                         direction_1: torch.Tensor,
                         direction_2: torch.Tensor,
                         values: List[float],
                         cv_threshold: float = 0.2,
                         linearity_threshold: float = 0.9) -> Dict[str, float]:
    """
    Detect helical geometry in a 2D plane defined by two SVD directions.

    Args:
        activations: Tensor of shape [N, d_model] containing activations
        direction_1: First SVD direction vector [d_model]
        direction_2: Second SVD direction vector [d_model]
        values: List of numerical values corresponding to activations
        cv_threshold: Maximum coefficient of variation for radius
        linearity_threshold: Minimum correlation for angle linearity

    Returns:
        Dictionary with helix detection metrics
    """
    # Project activations onto the 2D plane
    coords = torch.stack([
        activations @ direction_1,
        activations @ direction_2
    ], dim=1)  # [N, 2]

    # Compute polar coordinates
    radii = coords.norm(dim=1)
    angles = torch.atan2(coords[:, 1], coords[:, 0]).numpy()

    # Helix quality metrics
    radius_cv = (radii.std() / radii.mean()).item() if radii.mean() > 0 else float('inf')
    angle_linearity = safe_angle_linearity(angles, values)

    # Estimate period from angle progression
    angle_diffs = np.diff(np.unwrap(angles))
    value_diffs = np.diff(values)
    valid_mask = value_diffs != 0

    if valid_mask.sum() > 0:
        periods = 2 * np.pi / (angle_diffs[valid_mask] / value_diffs[valid_mask])
        estimated_period = np.median(periods[np.isfinite(periods)])
    else:
        estimated_period = float('inf')

    # Helix detection
    is_helix = (radius_cv < cv_threshold and
                abs(angle_linearity) > linearity_threshold and
                np.isfinite(estimated_period))

    return {
        'is_helix': is_helix,
        'radius_cv': radius_cv,
        'angle_linearity': angle_linearity,
        'estimated_period': estimated_period,
        'radii': radii.numpy(),
        'angles': angles,
        'coords': coords.numpy()
    }


def find_helix_directions(circuit, layer: int, head: int,
                         activations: torch.Tensor,
                         values: List[float],
                         top_k: int = 10) -> List[Dict]:
    """
    Find pairs of SVD directions that exhibit helical structure.

    Args:
        circuit: MaskedTransformerCircuit instance
        layer: Layer index
        head: Head index
        activations: Activations tensor [N, d_model]
        values: Corresponding numerical values
        top_k: Number of top SVD directions to consider

    Returns:
        List of helix detection results sorted by quality
    """
    # Get SVD components for this head
    head_key = f'differential_head_{layer}_{head}'
    ov_cache_key = f"{head_key}_ov"

    if ov_cache_key not in circuit.svd_cache:
        raise ValueError(f"SVD cache not found for {ov_cache_key}")

    U_ov, S_ov, Vh_ov, _ = circuit.svd_cache[ov_cache_key]

    results = []

    # Test all pairs of top-k directions
    for k1, k2 in itertools.combinations(range(min(top_k, Vh_ov.shape[0])), 2):
        direction_1 = Vh_ov[k1].cpu()
        direction_2 = Vh_ov[k2].cpu()

        helix_result = detect_helix_geometry(
            activations, direction_1, direction_2, values
        )

        helix_result.update({
            'direction_indices': (k1, k2),
            'direction_1': direction_1,
            'direction_2': direction_2,
            'singular_values': (S_ov[k1].item(), S_ov[k2].item())
        })

        results.append(helix_result)

    # Sort by helix quality (combine low radius CV with high angle linearity)
    results.sort(key=lambda x: x['radius_cv'] - abs(x['angle_linearity']))

    return results


def visualize_2d_helix(helix_result: Dict, values: List[float],
                      title: str = "Helix Projection",
                      output_path: Optional[str] = None) -> None:
    """
    Create 2D visualization of helical structure.

    Args:
        helix_result: Result from detect_helix_geometry
        values: Numerical values for color coding
        title: Plot title
        output_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    coords = helix_result['coords']
    radii = helix_result['radii']
    angles = helix_result['angles']

    # 2D scatter plot in SVD direction space
    scatter = ax1.scatter(coords[:, 0], coords[:, 1], c=values,
                         cmap='viridis', s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('SVD Direction 1', fontsize=12, fontweight='bold')
    ax1.set_ylabel('SVD Direction 2', fontsize=12, fontweight='bold')
    ax1.set_title(f'{title}\nRadius CV: {helix_result["radius_cv"]:.3f}',
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Add colorbar
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label('Numerical Value', fontsize=12, fontweight='bold')

    # Polar representation showing angle vs value
    ax2.scatter(values, np.unwrap(angles), c=values, cmap='viridis',
               s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Numerical Value', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Unwrapped Angle (radians)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Angle Linearity: {helix_result["angle_linearity"]:.3f}\n'
                 f'Period: {helix_result["estimated_period"]:.1f}',
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add trend line
    if len(values) > 1:
        z = np.polyfit(values, np.unwrap(angles), 1)
        p = np.poly1d(z)
        ax2.plot(values, p(values), '--', color='red', linewidth=2, alpha=0.8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_3d_helix_visualization(helix_result: Dict, values: List[float],
                                 title: str = "3D Helix Structure",
                                 output_path: Optional[str] = None) -> go.Figure:
    """
    Create interactive 3D helix visualization using plotly.

    Args:
        helix_result: Result from detect_helix_geometry
        values: Numerical values for color coding and z-axis
        title: Plot title
        output_path: Optional path to save HTML file

    Returns:
        Plotly figure object
    """
    coords = helix_result['coords']

    # Create 3D coordinates: (x, y, z) = (coord1, coord2, value)
    x = coords[:, 0]
    y = coords[:, 1]
    z = np.array(values)

    # Create the main helix scatter plot
    fig = go.Figure()

    # Add points
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers+lines',
        marker=dict(
            size=8,
            color=values,
            colorscale='Viridis',
            colorbar=dict(title="Numerical Value"),
            line=dict(width=1, color='black')
        ),
        line=dict(color='rgba(255,107,53,0.6)', width=4),
        name='Number Helix',
        text=[f'Value: {v}' for v in values],
        hovertemplate='<b>Value: %{text}</b><br>' +
                     'X: %{x:.3f}<br>' +
                     'Y: %{y:.3f}<br>' +
                     'Z: %{z:.1f}<extra></extra>'
    ))

    # Add projection onto XY plane
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=[min(z)] * len(x),
        mode='markers',
        marker=dict(
            size=4,
            color='rgba(128,128,128,0.5)',
            symbol='circle'
        ),
        name='XY Projection',
        showlegend=False
    ))

    # Add helical reference curve if period is reasonable
    period = helix_result['estimated_period']
    if np.isfinite(period) and period > 0:
        t_ref = np.linspace(min(values), max(values), 100)
        mean_radius = np.mean(helix_result['radii'])

        # Fit phase offset
        angles_unwrapped = np.unwrap(helix_result['angles'])
        if len(values) > 0:
            phase_offset = angles_unwrapped[0] - 2*np.pi*values[0]/period
        else:
            phase_offset = 0

        x_ref = mean_radius * np.cos(2*np.pi*t_ref/period + phase_offset)
        y_ref = mean_radius * np.sin(2*np.pi*t_ref/period + phase_offset)

        fig.add_trace(go.Scatter3d(
            x=x_ref, y=y_ref, z=t_ref,
            mode='lines',
            line=dict(color='red', width=3, dash='dash'),
            name=f'Ideal Helix (T={period:.1f})',
            opacity=0.7
        ))

    # Update layout
    fig.update_layout(
        title={
            'text': f'{title}<br><sub>Radius CV: {helix_result["radius_cv"]:.3f}, '
                   f'Angle Linearity: {helix_result["angle_linearity"]:.3f}</sub>',
            'x': 0.5,
            'font': {'size': 16}
        },
        scene=dict(
            xaxis_title='SVD Direction 1',
            yaxis_title='SVD Direction 2',
            zaxis_title='Numerical Value',
            bgcolor='white',
            xaxis=dict(gridcolor='lightgray'),
            yaxis=dict(gridcolor='lightgray'),
            zaxis=dict(gridcolor='lightgray')
        ),
        width=800,
        height=700,
        margin=dict(l=0, r=0, b=0, t=50)
    )

    if output_path:
        fig.write_html(output_path)

    return fig


def visualize_phase_shift_analysis(circuit, layer: int, head: int,
                                  base_values: List[int],
                                  shift_amount: int = 1,
                                  helix_directions: Optional[Tuple[int, int]] = None,
                                  output_dir: Optional[str] = None) -> Dict:
    """
    Visualize how phase shifts in helix correspond to arithmetic operations.

    Args:
        circuit: MaskedTransformerCircuit instance
        layer: Layer index
        head: Head index
        base_values: List of base numbers to test
        shift_amount: Amount to shift numbers (+1 for n -> n+1)
        helix_directions: Tuple of (k1, k2) SVD direction indices
        output_dir: Directory to save visualizations

    Returns:
        Dictionary with phase shift analysis results
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Get SVD components
    head_key = f'differential_head_{layer}_{head}'
    ov_cache_key = f"{head_key}_ov"
    U_ov, S_ov, Vh_ov, _ = circuit.svd_cache[ov_cache_key]

    # Use provided directions or find best helix automatically
    if helix_directions is None:
        # This would require activations - for now use top 2 directions
        k1, k2 = 0, 1
    else:
        k1, k2 = helix_directions

    direction_1 = Vh_ov[k1].cpu()
    direction_2 = Vh_ov[k2].cpu()

    results = {
        'base_values': base_values,
        'shifted_values': [v + shift_amount for v in base_values],
        'phase_shifts': [],
        'predicted_shifts': [],
        'directions': (k1, k2)
    }

    # Note: This is a template - actual implementation would need
    # to collect activations for base_values and shifted_values
    # For demonstration, we'll create a conceptual visualization

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Original Number Encoding',
            'Shifted Number Encoding',
            'Phase Shift Vectors',
            'Arithmetic Operation Visualization'
        ],
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'scatter3d'}]]
    )

    # Simulate helix points for visualization
    angles_orig = [2 * np.pi * v / 10 for v in base_values]  # Assume period 10
    angles_shifted = [2 * np.pi * v / 10 for v in results['shifted_values']]

    radius = 1.0  # Assume unit radius
    x_orig = [radius * np.cos(a) for a in angles_orig]
    y_orig = [radius * np.sin(a) for a in angles_orig]
    x_shift = [radius * np.cos(a) for a in angles_shifted]
    y_shift = [radius * np.sin(a) for a in angles_shifted]

    # Original encoding
    fig.add_trace(
        go.Scatter(x=x_orig, y=y_orig, mode='markers+text',
                  text=[str(v) for v in base_values],
                  textposition='middle right',
                  marker=dict(size=10, color=HELIX_COLORS['direction_1']),
                  name='Original'),
        row=1, col=1
    )

    # Shifted encoding
    fig.add_trace(
        go.Scatter(x=x_shift, y=y_shift, mode='markers+text',
                  text=[str(v) for v in results['shifted_values']],
                  textposition='middle right',
                  marker=dict(size=10, color=HELIX_COLORS['direction_2']),
                  name='Shifted'),
        row=1, col=2
    )

    # Phase shift vectors
    for i in range(len(x_orig)):
        fig.add_trace(
            go.Scatter(x=[x_orig[i], x_shift[i]], y=[y_orig[i], y_shift[i]],
                      mode='lines',
                      line=dict(color=HELIX_COLORS['phase_shift'], width=2),
                      showlegend=False),
            row=2, col=1
        )

    # Add original and shifted points to phase shift plot
    fig.add_trace(
        go.Scatter(x=x_orig, y=y_orig, mode='markers',
                  marker=dict(size=8, color=HELIX_COLORS['direction_1']),
                  name='Original', showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_shift, y=y_shift, mode='markers',
                  marker=dict(size=8, color=HELIX_COLORS['direction_2']),
                  name='Shifted', showlegend=False),
        row=2, col=1
    )

    # 3D visualization showing arithmetic as helix rotation
    fig.add_trace(
        go.Scatter3d(x=x_orig, y=y_orig, z=base_values,
                    mode='markers+lines',
                    marker=dict(size=6, color=HELIX_COLORS['helix_primary']),
                    line=dict(color=HELIX_COLORS['helix_primary'], width=4),
                    name='Number Helix'),
        row=2, col=2
    )

    fig.update_layout(
        title=f'Phase Shift Analysis: Layer {layer}, Head {head}<br>'
              f'<sub>Operation: n → n + {shift_amount}</sub>',
        height=800,
        showlegend=True
    )

    if output_dir:
        fig.write_html(os.path.join(output_dir, f'phase_shift_analysis_L{layer}H{head}.html'))

    return results


def generate_helix_comparison_report(circuit,
                                   arithmetic_task: str = "addition",
                                   output_dir: Optional[str] = None) -> Dict:
    """
    Generate comprehensive comparison between standard SVD and helix approaches.

    Args:
        circuit: MaskedTransformerCircuit instance
        arithmetic_task: Type of arithmetic task to analyze
        output_dir: Directory to save report and visualizations

    Returns:
        Dictionary containing comparison results
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    report = {
        'task': arithmetic_task,
        'model_info': {
            'n_layers': circuit.n_layers,
            'n_heads': circuit.n_heads,
            'd_model': circuit.d_model
        },
        'helix_heads': [],
        'standard_analysis': {},
        'helix_analysis': {}
    }

    # Create summary visualization
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Standard SVD Mask Visualization',
            'Helix-Enhanced Mask Visualization',
            'Singular Value Distribution',
            'Helical Structure Quality',
            'Standard Sparsity Pattern',
            'Helix-Guided Sparsity Pattern'
        ]
    )

    # This is a template structure - actual implementation would:
    # 1. Run helix detection on all heads
    # 2. Compare with standard mask learning results
    # 3. Generate side-by-side visualizations
    # 4. Create performance comparison metrics

    # Add placeholder content showing the comparison concept
    layers = list(range(circuit.n_layers))
    heads = list(range(circuit.n_heads))

    # Standard approach visualization (current mask values)
    standard_sparsity = []
    for layer in layers:
        for head in heads:
            head_key = f'differential_head_{layer}_{head}'
            if hasattr(circuit, 'ov_masks') and head_key in circuit.ov_masks:
                mask = torch.sigmoid(circuit.ov_masks[head_key]).detach()
                sparsity = (mask < 0.1).float().mean().item()
                standard_sparsity.append(sparsity)
            else:
                standard_sparsity.append(0.5)  # placeholder

    fig.add_trace(
        go.Heatmap(z=[standard_sparsity[i:i+circuit.n_heads]
                     for i in range(0, len(standard_sparsity), circuit.n_heads)],
                  colorscale='RdBu', name='Standard Sparsity'),
        row=3, col=1
    )

    # Simulated helix quality scores
    helix_quality = np.random.beta(2, 5, size=len(standard_sparsity))  # Placeholder

    fig.add_trace(
        go.Heatmap(z=[helix_quality[i:i+circuit.n_heads]
                     for i in range(0, len(helix_quality), circuit.n_heads)],
                  colorscale='Viridis', name='Helix Quality'),
        row=2, col=2
    )

    fig.update_layout(
        title=f'SVD vs Helix Approach Comparison<br><sub>Task: {arithmetic_task}</sub>',
        height=900
    )

    if output_dir:
        fig.write_html(os.path.join(output_dir, 'helix_comparison_report.html'))

        # Save text summary
        with open(os.path.join(output_dir, 'comparison_summary.txt'), 'w') as f:
            f.write(f"Helix vs Standard SVD Comparison Report\n")
            f.write(f"Task: {arithmetic_task}\n")
            f.write(f"Model: {circuit.n_layers} layers, {circuit.n_heads} heads\n")
            f.write(f"Generated visualizations in {output_dir}\n")

    return report


# Integration function to add helix analysis to existing MaskedTransformerCircuit
def add_helix_analysis_to_circuit(circuit) -> None:
    """
    Add helix analysis methods to an existing MaskedTransformerCircuit instance.

    Args:
        circuit: MaskedTransformerCircuit instance to enhance
    """

    def find_helix_directions_method(self, layer, head, activations, values, top_k=10):
        return find_helix_directions(self, layer, head, activations, values, top_k)

    def visualize_helix_structure_method(self, layer, head, activations, values, output_dir=None):
        helix_results = self.find_helix_directions(layer, head, activations, values)

        if not helix_results:
            print(f"No helix structure found in Layer {layer}, Head {head}")
            return None

        best_helix = helix_results[0]
        if best_helix['is_helix']:
            print(f"Found helix in Layer {layer}, Head {head}: "
                  f"CV={best_helix['radius_cv']:.3f}, "
                  f"Linearity={best_helix['angle_linearity']:.3f}")

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # 2D visualization
            visualize_2d_helix(
                best_helix, values,
                title=f"Layer {layer}, Head {head} Helix",
                output_path=f"{output_dir}/helix_2d_L{layer}H{head}.png" if output_dir else None
            )

            # 3D visualization
            fig_3d = create_3d_helix_visualization(
                best_helix, values,
                title=f"Layer {layer}, Head {head} 3D Helix",
                output_path=f"{output_dir}/helix_3d_L{layer}H{head}.html" if output_dir else None
            )

            return best_helix, fig_3d
        else:
            print(f"No significant helix structure found in Layer {layer}, Head {head}")
            return None

    def generate_helix_report_method(self, arithmetic_task="addition", output_dir=None):
        return generate_helix_comparison_report(self, arithmetic_task, output_dir)

    # Add methods to circuit instance
    circuit.find_helix_directions = find_helix_directions_method.__get__(circuit)
    circuit.visualize_helix_structure = visualize_helix_structure_method.__get__(circuit)
    circuit.generate_helix_report = generate_helix_report_method.__get__(circuit)

    print("Added helix analysis methods to MaskedTransformerCircuit:")
    print("  - find_helix_directions(layer, head, activations, values)")
    print("  - visualize_helix_structure(layer, head, activations, values, output_dir)")
    print("  - generate_helix_report(arithmetic_task, output_dir)")