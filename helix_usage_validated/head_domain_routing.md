# Head x domain routing analysis (StereoSet scans)

Specialization index per head = max over domains of the normalized pass-count vector.
  1.00 -> fully one-domain;  0.25 -> perfectly spread across 4 domains.

| Model | Unique heads (>=1 pass) | top-4 ensemble mean spec | Verdict |
|---|---|---|---|
| gpt2 | 6 | 0.80 | specialized |
| gemma | 1 | 1.00 | specialized |
| phi3 | 12 | 0.53 | mixed |
| llama | 12 | 0.47 | mixed |

## Interpretation
- **specialized (>= 0.55)**: heads pass mostly on one domain. A **routed ensemble** (pick heads per incoming bias_type) should reduce overshoot.
- **generic (<= 0.35)**: heads pass across many domains. The ensemble is fine; overshoot is an alpha problem and per-domain alpha (with dev/test split) is the fix.
- **mixed**: both effects matter; per-domain routing *and* per-domain alpha likely help.

Per-model top-12 tables are in the stdout of this script; heatmaps are saved as head_domain_heatmap_{tag}.png.