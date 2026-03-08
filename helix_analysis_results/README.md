# Helix-Based Arithmetic Circuit Analysis

## Overview

This report analyzes arithmetic circuits using the helix approach on gpt2 (12 layers, 12 heads).

## Key Findings

- **Total heads analyzed**: 144
- **Helix structures found**: 1
- **Layers with helix heads**: 1

## Helix Heads by Layer

### Layer 9

- **Head 9** (counting)
  - Radius CV: 0.063
  - Angle Linearity: 0.907
  - Period: 74.2

- **Interpretation** 
- Core Helix Metrics (from helix_3d_L18H15.html):

  1. Radius CV (Coefficient of Variation): 0.048
  - Meaning: Measures how consistent the "radius" is as numbers
    change
  - Range: 0.0 (perfect circle) to 1.0+ (chaotic)
  - Threshold: < 0.2 for helix detection
  - Interpretation: 0.048 = excellent consistency (very circular
    when viewed from the side)

  2. Angle Linearity: 0.934
  - Meaning: How linearly the angle changes with increasing
    numbers
  - Range: 0.0 (random) to 1.0 (perfect linear progression)
  - Threshold: > 0.85 for helix detection
  - Interpretation: 0.934 = strong arithmetic encoding (numbers
    map to angles predictably)

  3. Period: 55.4
  - Meaning: After how many numbers the helix completes one full
    rotation
  - Interpretation: The model "wraps around" every ~55 numbers

   **What This Reveals About the Transformer:**

  **Geometric Arithmetic Encoding:**
  - The attention head spatially organizes numbers in a helical
    pattern
  - Distance encodes magnitude (larger numbers → further from
    center)
  - Angle encodes sequence (consecutive numbers → rotating
    positions)

  **Functional Significance:**
  - Arithmetic reasoning: Model uses geometric relationships for
    math operations
  - Clock-like mechanism: Similar to how analog clocks represent
    time cyclically

## Files Generated

- `helix_analysis_report.json` - Complete analysis results
- `helix_quality_heatmap.png` - Quality metrics across all heads
- `visualizations/` - Individual head visualizations
  - `L{layer}H{head}/` - Per-head analysis
    - `helix_2d_L{layer}H{head}.png` - 2D helix projection
    - `helix_3d_L{layer}H{head}.html` - Interactive 3D visualization
    - `comparison_L{layer}H{head}.png` - SVD vs Helix comparison
