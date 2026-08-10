# Debias-method head-to-head: INLP vs SentenceDebias vs routed+calib

CrowS-Pairs Stereotype Score (SS) per (model, domain). Ideal SS = 50 (no stereotypical preference). `*_delta` is the SS change relative to baseline (negative = debiasing in the intended direction when baseline > 50). INLP and SentenceDebias are single-domain interventions trained on `gender` or `race` and evaluated at layer L[model]; the same `gender` hook is also applied to `profession` rows (following Liang 2020). `routed+calib` is our per-domain SVD-plane ensemble at calibrated α.

## gpt2  (layer L10 for INLP / SentenceDebias)

| domain | n | baseline | INLP | SentDebias | routed+calib | routed+matched (M*) | Δ INLP | Δ Sent | Δ calib | Δ matched |
|---|---|---|---|---|---|---|---|---|---|---|
| gender | 262 |  48.47 |  61.07 |  43.89 |  33.59 |  48.47 (M=0.00) | +12.60 | -4.58 | -14.89 | +0.00 |
| race | 516 |  50.00 |  40.70 |  49.61 |  53.29 |  50.00 (M=0.00) | -9.30 | -0.39 | +3.29 | +0.00 |
| profession | 172 |  61.63 |  63.37 |  52.33 |  57.56 | -- | +1.74 | -9.30 | -4.07 | -- |
| religion | 105 |  60.00 | -- | -- |  57.14 | -- | -- | -- | -2.86 | -- |
| overall | 1055 |  52.51 |  40.70 |  49.61 |  49.48 | -- | -11.81 | -2.90 | -3.03 | -- |

## phi3  (layer L24 for INLP / SentenceDebias)

| domain | n | baseline | INLP | SentDebias | routed+calib | routed+matched (M*) | Δ INLP | Δ Sent | Δ calib | Δ matched |
|---|---|---|---|---|---|---|---|---|---|---|
| gender | 262 |  57.63 |  63.74 |  53.05 |  45.80 |  49.62 (M=0.25) | +6.11 | -4.58 | -11.83 | -8.02 |
| race | 516 |  52.52 |  50.39 |  50.97 |  34.30 |  52.52 (M=0.00) | -2.13 | -1.55 | -18.22 | +0.00 |
| profession | 172 |  57.56 |  60.47 |  58.14 |  58.72 | -- | +2.91 | +0.58 | +1.16 | -- |
| religion | 105 |  59.05 | -- | -- |  61.90 | -- | -- | -- | +2.86 | -- |
| overall | 1055 |  55.26 |  50.39 |  50.97 |  43.89 | -- | -4.87 | -4.29 | -11.37 | -- |

## gemma  (layer L21 for INLP / SentenceDebias)

| domain | n | baseline | INLP | SentDebias | routed+calib | routed+matched (M*) | Δ INLP | Δ Sent | Δ calib | Δ matched |
|---|---|---|---|---|---|---|---|---|---|---|
| gender | 262 |  51.15 |  58.78 |  54.20 |  32.44 |  51.15 (M=0.00) | +7.63 | +3.05 | -18.70 | +0.00 |
| race | 516 |  56.01 |  56.40 |  53.88 |  31.98 |  46.71 (M=0.25) | +0.39 | -2.13 | -24.03 | -9.30 |
| profession | 172 |  63.95 |  64.53 |  62.79 |  65.12 | -- | +0.58 | -1.16 | +1.16 | -- |
| religion | 105 |  63.81 | -- | -- |  60.00 | -- | -- | -- | -3.81 | -- |
| overall | 1055 |  56.87 |  56.40 |  53.88 |  40.28 | -- | -0.48 | -3.00 | -16.59 | -- |

## llama  (layer L22 for INLP / SentenceDebias)

| domain | n | baseline | INLP | SentDebias | routed+calib | routed+matched (M*) | Δ INLP | Δ Sent | Δ calib | Δ matched |
|---|---|---|---|---|---|---|---|---|---|---|
| gender | 262 |  54.20 |  57.25 |  51.53 |  55.34 |  54.20 (M=0.00) | +3.05 | -2.67 | +1.15 | +0.00 |
| race | 516 |  50.97 |  46.90 |  45.16 |  45.35 |  49.81 (M=0.25) | -4.07 | -5.81 | -5.62 | -1.16 |
| profession | 172 |  66.86 |  67.44 |  66.86 |  69.77 | -- | +0.58 | +0.00 | +2.91 | -- |
| religion | 105 |  74.29 | -- | -- |  72.38 | -- | -- | -- | -1.90 | -- |
| overall | 1055 |  56.68 |  46.90 |  45.16 |  54.50 | -- | -9.78 | -11.53 | -2.18 | -- |

## Per-cell |Δ50| (distance from neutral; lower is better)

Only gender and race cells shown — these are the domains where INLP / SentenceDebias are trained. `matched` = best-multiplier compass run from target_ss50_sweep.py; `--` where the sweep was not run.

| model | domain | baseline | INLP | SentDebias | routed+calib | routed+matched |
|---|---|---|---|---|---|---|
| gpt2 | gender |   1.53 |  11.07 |   6.11 |  16.41 |   1.53 |
| gpt2 | race |   0.00 |   9.30 |   0.39 |   3.29 |   0.00 |
| phi3 | gender |   7.63 |  13.74 |   3.05 |   4.20 |   0.38 |
| phi3 | race |   2.52 |   0.39 |   0.97 |  15.70 |   2.52 |
| gemma | gender |   1.15 |   8.78 |   4.20 |  17.56 |   1.15 |
| gemma | race |   6.01 |   6.40 |   3.88 |  18.02 |   3.29 |
| llama | gender |   4.20 |   7.25 |   1.53 |   5.34 |   4.20 |
| llama | race |   0.97 |   3.10 |   4.84 |   4.65 |   0.19 |
