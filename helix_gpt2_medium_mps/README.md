# Helix-Based Arithmetic Circuit Analysis

## Overview

This report analyzes arithmetic circuits using the helix approach on gpt2-medium (24 layers, 16 heads).

## Key Findings

- **Total heads analyzed**: 384
- **Helix structures found**: 1
- **Layers with helix heads**: 1

## Helix Heads by Layer

### Layer 18

- **Head 15** (counting)
  - Radius CV: 0.049
  - Angle Linearity: 0.934
  - Period: 55.4

## Files Generated

- `helix_analysis_report.json` - Complete analysis results
- `helix_quality_heatmap.png` - Quality metrics across all heads
- `visualizations/` - Individual head visualizations
  - `L{layer}H{head}/` - Per-head analysis
    - `helix_2d_L{layer}H{head}.png` - 2D helix projection
    - `helix_3d_L{layer}H{head}.html` - Interactive 3D visualization
    - `comparison_L{layer}H{head}.png` - SVD vs Helix comparison
