# Flip-ratio summary per (model, domain)

`mean_debias_shift` is in nats and measures `logprob(anti-stereo) - logprob(stereo)` shift under the routed + per-domain-alpha intervention relative to baseline. Positive values mean the intervention on average pushes probability mass off the stereotype on that row.

`flip_anti` / `flip_stereo` count rows whose preferred sentence *changed sign* under the intervention. `chi2 / chi2_p` tests the null that flips are 50/50 (df=1, asymptotic).

## gpt2

| domain | n | mean Δ (nats) | flip→anti | flip→stereo | flip→anti rate | flip→stereo rate | net flip | flip ratio | χ²(1) | p |
|---|---|---|---|---|---|---|---|---|---|---|
| gender | 262 | +0.856 | 54 | 15 | 0.206 | 0.057 | +39 | 3.60 | 22.04 | 2.67e-06 |
| race | 516 | -0.083 | 8 | 25 | 0.016 | 0.048 | -17 | 0.32 | 8.76 | 0.00308 |
| profession | 172 | +0.330 | 9 | 2 | 0.052 | 0.012 | +7 | 4.50 | 4.45 | 0.0348 |
| religion | 105 | +0.236 | 5 | 2 | 0.048 | 0.019 | +3 | 2.50 | 1.29 | 0.257 |

## phi3

| domain | n | mean Δ (nats) | flip→anti | flip→stereo | flip→anti rate | flip→stereo rate | net flip | flip ratio | χ²(1) | p |
|---|---|---|---|---|---|---|---|---|---|---|
| gender | 262 | +1.163 | 51 | 20 | 0.195 | 0.076 | +31 | 2.55 | 13.54 | 0.000234 |
| race | 516 | +1.983 | 103 | 9 | 0.200 | 0.017 | +94 | 11.44 | 78.89 | 6.56e-19 |
| profession | 172 | -0.173 | 5 | 7 | 0.029 | 0.041 | -2 | 0.71 | 0.33 | 0.564 |
| religion | 105 | +0.337 | 6 | 9 | 0.057 | 0.086 | -3 | 0.67 | 0.60 | 0.439 |

## gemma

| domain | n | mean Δ (nats) | flip→anti | flip→stereo | flip→anti rate | flip→stereo rate | net flip | flip ratio | χ²(1) | p |
|---|---|---|---|---|---|---|---|---|---|---|
| gender | 262 | +3.048 | 65 | 16 | 0.248 | 0.061 | +49 | 4.06 | 29.64 | 5.2e-08 |
| race | 516 | +2.461 | 131 | 7 | 0.254 | 0.014 | +124 | 18.71 | 111.42 | 4.79e-26 |
| profession | 172 | +0.329 | 8 | 10 | 0.047 | 0.058 | -2 | 0.80 | 0.22 | 0.637 |
| religion | 105 | +0.680 | 11 | 7 | 0.105 | 0.067 | +4 | 1.57 | 0.89 | 0.346 |

## llama

| domain | n | mean Δ (nats) | flip→anti | flip→stereo | flip→anti rate | flip→stereo rate | net flip | flip ratio | χ²(1) | p |
|---|---|---|---|---|---|---|---|---|---|---|
| gender | 262 | -0.038 | 4 | 7 | 0.015 | 0.027 | -3 | 0.57 | 0.82 | 0.366 |
| race | 516 | +0.376 | 32 | 3 | 0.062 | 0.006 | +29 | 10.67 | 24.03 | 9.49e-07 |
| profession | 172 | -0.985 | 11 | 16 | 0.064 | 0.093 | -5 | 0.69 | 0.93 | 0.336 |
| religion | 105 | +0.089 | 3 | 1 | 0.029 | 0.010 | +2 | 3.00 | 1.00 | 0.317 |

## Interpretation guide

- `net_flip > 0` and `flip_ratio > 1` at p < 0.05 = intervention is directionally debiasing on that (model, domain) cell.
- A single domain with `flip_ratio < 1` while other domains are positive suggests the intervention is catching a non-stereotype lexical confound in that domain (e.g. the Phi-3 profession top-regressions were driven by name swaps, not poor/rich).
- `mean_debias_shift` aggregates *all* row-level shifts including non-flips, so it can be positive even when flip counts are balanced; disagreement between mean shift and flip ratio is itself informative.