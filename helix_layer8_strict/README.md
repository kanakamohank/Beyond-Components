# Memory-Optimized Helix Analysis Results

## Overview

This report analyzes arithmetic circuits using the helix approach on EleutherAI/gpt-neo-2.7B.

## Key Findings

- **Total heads analyzed**: 640
- **Helix structures found**: 2
- **Detection rate**: 0.3%

## Analysis Parameters

- CV threshold: 0.25
- Linearity threshold: 0.8
- Device: mps
- Batch size: 4

## Files Generated

- `helix_analysis_report.json` - Complete analysis results
- `intermediate_results_batch_*.json` - Batch-wise results
- `README.md` - This summary file

## Best Helix Head

**Layer 8, Head 5**
- Radius CV: 0.0625
- Angle Linearity: 0.9897
- Estimated Period: 143.58362207314994
