# Memory-Optimized Helix Analysis Results

## Overview

This report analyzes arithmetic circuits using the helix approach on EleutherAI/gpt-neo-2.7B.

## Key Findings

- **Total heads analyzed**: 640
- **Helix structures found**: 4
- **Detection rate**: 0.6%

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

**Layer 11, Head 4**
- Radius CV: 0.1163
- Angle Linearity: 0.9597
- Estimated Period: 84.9566048473902
