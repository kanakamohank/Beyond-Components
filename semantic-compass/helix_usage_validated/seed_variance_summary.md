# Seed variance: CrowS-Pairs SS under routed + per-domain α

Head selection and probe tie-breaks are randomized by `--seed`; everything else is deterministic. Four seeds per model: 0 (paper default), 1, 2, 3. `mean ± std` is over these four runs. Column `seed0` is the paper-default deterministic value for reference.

## phi3

| condition | domain | seed0 | mean ± std | min | max |
|---|---|---|---|---|---|
| baseline | gender | 57.63 | 57.63 ± 0.00 | 57.63 | 57.63 |
| baseline | race | 52.52 | 52.52 ± 0.00 | 52.52 | 52.52 |
| baseline | profession | 57.56 | 57.56 ± 0.00 | 57.56 | 57.56 |
| baseline | religion | 59.05 | 59.05 ± 0.00 | 59.05 | 59.05 |
| baseline | overall | 55.26 | 55.26 ± 0.00 | 55.26 | 55.26 |
| routed_calib | gender | 45.80 | 50.67 ± 3.24 | 45.80 | 52.29 |
| routed_calib | race | 34.30 | 34.30 ± 0.00 | 34.30 | 34.30 |
| routed_calib | profession | 58.72 | 58.72 ± 0.00 | 58.72 | 58.72 |
| routed_calib | religion | 61.90 | 61.90 ± 0.00 | 61.90 | 61.90 |
| routed_calib | overall | 43.89 | 45.09 ± 0.81 | 43.89 | 45.50 |

## gemma

| condition | domain | seed0 | mean ± std | min | max |
|---|---|---|---|---|---|
| baseline | gender | 51.15 | 51.15 ± 0.00 | 51.15 | 51.15 |
| baseline | race | 56.01 | 56.01 ± 0.00 | 56.01 | 56.01 |
| baseline | profession | 63.95 | 63.95 ± 0.00 | 63.95 | 63.95 |
| baseline | religion | 63.81 | 63.81 ± 0.00 | 63.81 | 63.81 |
| baseline | overall | 56.87 | 56.87 ± 0.00 | 56.87 | 56.87 |
| routed_calib | gender | 32.44 | 32.44 ± 0.00 | 32.44 | 32.44 |
| routed_calib | race | 31.98 | 31.98 ± 0.00 | 31.98 | 31.98 |
| routed_calib | profession | 65.12 | 62.35 ± 1.86 | 61.05 | 65.12 |
| routed_calib | religion | 60.00 | 60.00 ± 0.00 | 60.00 | 60.00 |
| routed_calib | overall | 40.28 | 39.83 ± 0.30 | 39.62 | 40.28 |
