"""
Wrapper script to generate visualizations for strict threshold helixes.

This script uses the existing visualization functions from src/utils/helix_visualization.py
to create 2D and 3D plots for all 11 helixes found with strict thresholds (CV<0.2, Linearity>0.9).
"""

import json
import os
import numpy as np
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Import existing visualization functions
from src.utils.helix_visualization import (
    visualize_2d_helix,
    create_3d_helix_visualization,
    HELIX_COLORS
)


def load_helix_data(layer_dir):
    """Load helix analysis results from JSON file."""
    report_path = os.path.join(layer_dir, 'helix_analysis_report.json')

    if not os.path.exists(report_path):
        return None

    with open(report_path, 'r') as f:
        data = json.load(f)

    return data


def create_summary_dashboard(all_helixes, output_path):
    """Create an overview dashboard showing all helixes."""

    n_helixes = len(all_helixes)
    n_cols = 3
    n_rows = (n_helixes + n_cols - 1) // n_cols

    # Create subplot titles with metrics
    subplot_titles = []
    for helix_key, info in all_helixes.items():
        title = f"{helix_key}<br>CV:{info['cv']:.3f} L:{abs(info['lin']):.3f}"
        subplot_titles.append(title)

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        specs=[[{'type': 'scatter'}] * n_cols for _ in range(n_rows)],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )

    for idx, (helix_key, helix_info) in enumerate(all_helixes.items()):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        coords = np.array(helix_info['coords'])
        values = helix_info['values']

        # Add scatter trace
        fig.add_trace(
            go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode='markers+lines',
                marker=dict(
                    size=6,
                    color=values,
                    colorscale='Viridis',
                    showscale=(idx == 0),
                    colorbar=dict(
                        title="Value",
                        x=1.15,
                        len=0.3
                    ) if idx == 0 else None,
                    line=dict(width=0.5, color='black')
                ),
                line=dict(color=HELIX_COLORS['helix_primary'], width=2),
                name=helix_key,
                showlegend=False,
                hovertemplate='%{marker.color}<extra></extra>'
            ),
            row=row, col=col
        )

        # Set equal aspect ratio for each subplot
        fig.update_xaxes(
            scaleanchor=f"y{idx+1}",
            scaleratio=1,
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            row=row, col=col
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            row=row, col=col
        )

    fig.update_layout(
        title={
            'text': '<b>All 11 Strict Threshold Helixes</b><br>'
                   '<sub>CV<0.2, Linearity>0.9 | GPT-Neo 2.7B | Layers 4, 8, 11, 13</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2C3E50'}
        },
        height=300 * n_rows,
        width=1400,
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='#F8F9FA',
        font=dict(family="Arial", size=11)
    )

    fig.write_html(output_path)
    print(f"\n✓ Created summary dashboard: {output_path}")


def main():
    """Main visualization generation function."""

    print("\n" + "=" * 75)
    print("  HELIX VISUALIZATION GENERATOR - Strict Thresholds (CV<0.2, L>0.9)")
    print("=" * 75)

    # Define layer directories
    layer_dirs = {
        4: 'helix_layer4_strict',
        8: 'helix_layer8_strict',
        11: 'helix_layer11_strict',
        13: 'helix_layer13_strict'
    }

    # Create output directory
    output_dir = 'helix_visualizations_strict'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}/")

    # Number values (0-39 based on the analysis)
    values = list(range(40))

    # Track all helixes for summary
    all_helixes = {}
    helix_count = 0

    # Process each layer
    for layer_num, layer_dir in layer_dirs.items():
        data = load_helix_data(layer_dir)

        if data is None:
            print(f"\n⚠ Warning: No data found for layer {layer_num}")
            continue

        print(f"\n{'─' * 75}")
        print(f"Layer {layer_num}: {data['helix_structures_found']} helix(es) found")
        print(f"{'─' * 75}")

        # Process each helix in this layer
        for helix_key, helix_data in data['results'].items():
            helix_count += 1
            result = helix_data['helix_result']['best_result']

            print(f"\n{helix_count}. {helix_key}")
            print(f"   CV: {result['radius_cv']:.4f} | "
                  f"Linearity: {result['angle_linearity']:7.4f} | "
                  f"Period: {result['estimated_period']:8.2f}")

            # Create output filenames
            base_name = f"{helix_key.replace('L', 'layer').replace('H', '_head')}"
            plot_2d_path = os.path.join(output_dir, f"{base_name}_2d.png")
            plot_3d_path = os.path.join(output_dir, f"{base_name}_3d.html")

            # Convert coords to numpy array (stored as list in JSON)
            result_copy = result.copy()
            result_copy['coords'] = np.array(result['coords'])
            result_copy['radii'] = np.array(result['radii'])
            result_copy['angles'] = np.array(result['angles'])

            # Generate 2D visualization using existing function
            try:
                visualize_2d_helix(
                    result_copy,
                    values,
                    title=f"{helix_key} Helix Structure",
                    output_path=plot_2d_path
                )
                print(f"   ✓ 2D plot saved: {base_name}_2d.png")
            except Exception as e:
                print(f"   ✗ Error generating 2D plot: {e}")

            # Generate 3D visualization using existing function
            try:
                create_3d_helix_visualization(
                    result_copy,
                    values,
                    title=f"{helix_key} - 3D Helix Structure",
                    output_path=plot_3d_path
                )
                print(f"   ✓ 3D plot saved: {base_name}_3d.html")
            except Exception as e:
                print(f"   ✗ Error generating 3D plot: {e}")

            # Store for summary dashboard
            all_helixes[helix_key] = {
                'coords': result['coords'],
                'values': values,
                'cv': result['radius_cv'],
                'lin': result['angle_linearity'],
                'period': result['estimated_period']
            }

    # Create summary dashboard
    print("\n" + "=" * 75)
    print("Creating summary dashboard...")
    print("=" * 75)

    summary_path = os.path.join(output_dir, 'all_helixes_summary.html')
    create_summary_dashboard(all_helixes, summary_path)

    # Create README file
    readme_path = os.path.join(output_dir, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write("=" * 75 + "\n")
        f.write("  Helix Visualizations - Strict Thresholds (CV<0.2, Linearity>0.9)\n")
        f.write("=" * 75 + "\n\n")
        f.write(f"Total helixes visualized: {helix_count}\n")
        f.write(f"Model: GPT-Neo 2.7B\n")
        f.write(f"Layers analyzed: 4, 8, 11, 13\n\n")

        f.write("Files Generated:\n")
        f.write("-" * 75 + "\n")
        f.write("- all_helixes_summary.html : Interactive dashboard showing all helixes\n")
        f.write("- *_2d.png                 : 2D matplotlib plots (SVD space + angle)\n")
        f.write("- *_3d.html                : Interactive 3D plotly visualizations\n\n")

        f.write("Helix Quality Metrics:\n")
        f.write("-" * 75 + "\n")
        f.write(f"{'Helix':<10} | {'CV':<8} | {'Linearity':<10} | {'Period':<10}\n")
        f.write("-" * 75 + "\n")

        for helix_key, info in sorted(all_helixes.items()):
            f.write(f"{helix_key:<10} | {info['cv']:<8.4f} | "
                   f"{info['lin']:>10.4f} | {info['period']:>10.2f}\n")

        f.write("\n" + "=" * 75 + "\n")
        f.write("Best Helix: L8H5 (CV=0.0625, Linearity=0.9897, Period=143.58)\n")
        f.write("=" * 75 + "\n")

    print(f"✓ Created README: {readme_path}")

    # Final summary
    print("\n" + "=" * 75)
    print("  COMPLETE!")
    print("=" * 75)
    print(f"✓ Generated {helix_count * 2 + 1} visualization files:")
    print(f"  • {helix_count} 2D plots (.png)")
    print(f"  • {helix_count} 3D interactive plots (.html)")
    print(f"  • 1 summary dashboard (.html)")
    print(f"\nView all helixes: open {summary_path}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
