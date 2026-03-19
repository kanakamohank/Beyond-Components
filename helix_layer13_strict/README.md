# Memory-Optimized Helix Analysis Results

## Overview

This report analyzes arithmetic circuits using the helix approach on EleutherAI/gpt-neo-2.7B.

## Key Findings

- **Total heads analyzed**: 640
- **Helix structures found**: 1
- **Detection rate**: 0.2%

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

**Layer 13, Head 18**
- Radius CV: 0.0889
- Angle Linearity: -0.9622
- Estimated Period: -194.8711087775782
