# Pre-trained Large Language Models Use Fourier Features to Compute Addition

**arxiv ID:** 2406.03445
**URL:** https://arxiv.org/html/2406.03445
**Authors:** Tianyi Zhou, Deqing Fu, Vatsal Sharan, Robin Jia (University of Southern California)
**Published:** June 2024 (NeurIPS 2024)
**Models Studied:** GPT-2-XL (1.5B, primary), GPT-2-small, GPT-J (6B), Phi-2 (2.7B), GPT-3.5, GPT-4, PaLM-2

---

## TL;DR
Pre-trained LLMs add integers via a **two-mechanism Fourier-domain split**: **MLPs use low-frequency Fourier features to approximate the magnitude** of `a+b`, and **attention layers use high-frequency Fourier features to perform modular classification** (mod 2, mod 5, mod 10). This division of labor is causally validated by frequency-band ablations and originates in pre-trained **token embeddings** — freezing pre-trained `W^E` alone makes a randomly-initialized GPT-2-small reach **100%** addition accuracy.

## Problem
Do LLMs apply mathematical principles when solving math problems, or just reproduce memorized patterns? Prior interpretability work focused on **toy** transformers trained from scratch (Nanda et al. on modular addition) or **narrow** tasks (Hanna et al. on greater-than in GPT-2-small). The arithmetic mechanism in **pre-trained LLMs** doing **integer addition** had not been characterized.

## Method

### Step 1 — Behavioral logit lens
Apply `W^U h^{(ℓ)}` at each layer ℓ. **Finding:** at GPT-2-XL layers 20–30 (out of 48), top-1 prediction is rarely the exact answer but is almost always **within ±10** of it. Sharpens to exact at later layers. Hypothesis: model first **approximates magnitude**, then **refines**.

### Step 2 — Per-module logit decomposition
Because residual stream updates linearly:
```
h^{(ℓ)} = h^{(ℓ-1)} + Attn^{(ℓ)} + MLP^{(ℓ)}
```
the per-module contributions to logits are well-defined:
```
L_Attn^{(ℓ)} := W^U · Attn^{(ℓ)}
L_MLP^{(ℓ)}  := W^U · MLP^{(ℓ)}
```

**Two computation patterns observed:**
- **Approximation:** smooth Gaussian-shaped logit profile peaking near `a+b`
- **Classification:** comb-shaped profile peaking at every number congruent to `(a+b) mod c`
  - MLP layer 33 → mod-2 comb (peaks at every even number)
  - MLP layer 45 → mod-10 comb

### Step 3 — Fourier transform of per-module logits
With `p = 521` (single-token integer range +1), build a Fourier basis matrix `F` indexed by frequency `k`, period `T_k = (p−1)/k`:
```
ω_k = 2πk/(p−1)
T_k = (p−1)/k
```

DFT of per-module logits:
```
L̂_Attn^{(ℓ)} = F · L_Attn^{(ℓ)}
L̂_MLP^{(ℓ)}  = F · L_MLP^{(ℓ)}
```

**Key finding:** spectra are **approximately sparse**. Outlier components cluster at periods **2, 2.5, 5, 10**. MLPs additionally show low-frequency outliers (the approximation); attention is dominated by the high-frequency outliers (the classification).

### Step 4 — Mechanistic interpretation
- **High-frequency wave** (e.g., period 2): peaks at every even number — phase shift aligns peaks with `(a+b) mod 2`. Provides **fine-grained** unit-digit discrimination.
- **Low-frequency wave** (e.g., period 520): broad bump near `a+b`. Cannot place a sharp peak at the exact answer but assigns more mass to nearby numbers. Provides **magnitude approximation**.
- The two **multiply** in logit space: low-freq picks the neighborhood (around 108, not 178), high-freq picks the exact integer within that neighborhood (108 vs. 109 vs. 110).

### Step 5 — Causal ablation via Fourier filtering
Define a filter operator that finds the closest activation `h̃` whose unembedded projection has zero Fourier energy in a chosen band Γ:
```
F(h; Γ) = argmin_{h̃} ||h̃ − h||_2  s.t.  (F · W^U h̃)_k = 0  ∀ k ∈ Γ
```
Closed-form linear projection. Threshold τ = 50 separates "low" (k < 50) from "high" (k ≥ 50) frequency.

- **Low-pass** = zero out high frequencies → kills classification
- **High-pass** = zero out low frequencies → kills approximation

Apply filter to `Attn`, `MLP`, or both, at every layer, at inference time. No retraining.

## Datasets
Synthetic addition: `a + b` with `a, b ≤ 260` (so `a+b ≤ 520` stays within GPT-2-XL's single-token integer range). Five natural-language templates (e.g., `"Put together 15 and 93. Answer:"`). 80/10/10 train/val/test split. Fine-tune GPT-2-XL → **99.74%** test accuracy.

## Key Results

### Headline ablation table (GPT-2-XL, fine-tuned)

| Module | Removed | Val Loss | Accuracy |
|---|---|---|---|
| None (baseline) | — | 0.0073 | **99.74%** |
| ATTN & MLP | Low-Frequency | 4.0842 | 5.94% |
| ATTN | Low-Frequency | 0.0352 | **99.12%** |
| MLP | Low-Frequency | 2.1399 | **35.89%** |
| ATTN & MLP | High-Frequency | 1.8598 | 27.08% |
| ATTN | High-Frequency | 0.5943 | **78.36%** |
| MLP | High-Frequency | 0.1213 | **98.10%** |

**The two diagnostic numbers:**
- **MLP / Low-freq** ablation: 99.74 → 35.89% (catastrophic) → MLP **does** approximation
- **ATTN / High-freq** ablation: 99.74 → 78.36% (substantial) → ATTN **does** classification
- **MLP / High-freq** ablation: 99.74 → 98.10% (negligible) → MLP doesn't do classification
- **ATTN / Low-freq** ablation: 99.74 → 99.12% (negligible) → ATTN doesn't do approximation

**Error pattern signatures:**
- Ablating low-freq from MLP → **off-by-10/50/100** errors (magnitude wrong, mod-10 right)
- Ablating high-freq from ATTN → **off-by-<6** errors (magnitude right, mod-10 wrong)

These match exactly what theory predicts.

### Pre-training origin experiments

| Setup | Accuracy |
|---|---|
| GPT-2-XL fine-tuned | 99.74% |
| GPT-2-XL **trained from scratch** | **94.44%** (no Fourier outliers in embeddings or logits) |
| GPT-2-small **from scratch** | 53.95% |
| GPT-2-small with **frozen pre-trained `W^E`** (other layers random) | **100%** (5 seeds, faster convergence) |

**Conclusion:** the Fourier-feature inductive bias for addition lives in the **token embedding matrix** `W^E`, not in the architecture or in fine-tuning dynamics.

### Embedding analysis
For `W^E ∈ ℝ^{p × D}` (p = 521): apply column-wise DFT, take L2 norm of each row → p-dim vector measuring magnitude per Fourier component. Pre-trained GPT-2-XL embeddings have **outlier components at periods 2, 2.5, 5, 10**. t-SNE + k-means clusters by magnitude and by multiples of 10. Same pattern in pre-trained Phi-2, GPT-J.

### Generalization to other models
- **Phi-2 (2.7B), 4-shot:** 73% of absolute errors are multiples of 10
- **GPT-J (6B), 4-shot:** 93% of errors are multiples of 10
- Both show the same period-2/2.5/5/10 outlier pattern in MLP and attention spectra over their last 15 layers

**Closed-source models (0-shot, behavioral evidence only):**
- GPT-3.5: 100% of absolute errors are multiples of 10
- GPT-4: 100%
- PaLM-2: 87%

## Limitations
1. **Single-token range only:** numbers ≤ 260 due to GPT-2-XL tokenization.
2. **Closed-source claims** (GPT-3.5/4, PaLM-2) are behavioral only — no internals access.
3. **Multi-digit / carry / multiplication** addressed only in appendices.
4. **No formal sparsity quantification** — "approximately sparse" is asserted via visual inspection of spectra.
5. **Ablation methodology**: Fourier filter minimizes L₂ in **activation space**, not logit space — could leave the W^U-kernel intact while modifying the residual stream.

## Key Equations Recap

**Per-module logit lens:**
```
L_Attn^{(ℓ)} = W^U · Attn^{(ℓ)}
L_MLP^{(ℓ)}  = W^U · MLP^{(ℓ)}
```

**Fourier basis on ℤ/(p−1)ℤ** (p = 521):
```
ω_k = 2πk/(p−1),    T_k = (p−1)/k,    f_k = k/(p−1)
û = F · u
```

**Fourier-filter ablation:**
```
F(h; Γ) = argmin_{h̃} ||h̃ − h||_2  s.t.  (F · W^U h̃)_k = 0  ∀ k ∈ Γ
```

## Critical Quotes

*"MLP layers primarily approximate the magnitude of the answer using low-frequency features, while attention layers primarily perform modular addition... using high-frequency features."*

*"Pre-trained LLMs do not memorize but compute via interplay of approximation and classification in the frequency domain."*

*"Pre-training instills these features primarily in the token embeddings, which act as inductive bias."*

## Tags
`fourier-features` `mechanistic-interpretability` `arithmetic` `mlp-approximation` `attention-classification` `pre-training-inductive-bias` `frequency-domain` `logit-lens` `causal-ablation` `gpt-2-xl`

---

## Honest Critical Assessment

### Weakest assumption
The Fourier-filter ablation `F(h; Γ)` minimizes L₂ in **activation space** subject to a constraint on **logit-space** frequency content. Since `W^U` projects from D ≈ 1600 dims down to vocab, its kernel is huge. The filter can satisfy the logit-space constraint while making large changes in directions **invisible to W^U** but consequential for downstream layers. The paper does not test sensitivity to this choice.

### Likely cherry-picked
Outlier periods 2, 2.5, 5, 10 are not surprising for arithmetic mod 10 — but no formal null distribution or permutation test is reported. A clean counterfactual ("do other periods like 3, 7 appear as outliers in non-arithmetic prompts?") is not run.

### Likely failure modes
- **Multi-digit numbers** beyond single-token range (≤ 260) — entire mechanism rests on ℤ/(p−1)ℤ structure
- **Modular arithmetic with wraparound** — paper notes integer addition has no wraparound, which is *why* mod-10 high-freq doesn't disturb the mod-100 boundary
- **Multiplication** — answer's mod-10 is *not* a simple function of operands' mod-10, so the mechanism doesn't trivially port

### Overhyped relative to evidence
- The frozen-embedding result is at GPT-2-**small** scale (53.95% → 100%), not GPT-2-XL (94.44% → 99.74%). Transferring "the inductive bias is entirely in W^E" to bigger models is a smaller delta.
- Closed-source model claims read confirmatorily but are pure behavioral evidence.
- Per-module attribution is single-layer (`Attn^{(ℓ)}` for one ℓ at a time) — cross-layer interactions (LayerNorm rescaling, attention reading from earlier MLP outputs) are ignored.

---

## Why This Matters for the Beyond-Components Repository

This is the **canonical reference paper** for the Fourier-feature mechanism the repo's `paper/main.tex` builds on. Direct relevance to:

- **`paper/main.tex`** — the repo's "Fourier Basis of Digit Arithmetic" paper extends this with:
  - **Subspace-level** evidence (9-D Fourier basis = irreducible reps of ℤ/10ℤ) vs. Zhou et al.'s **frequency-component-level** evidence
  - **Trig identity verification** (CP tensor decomposition) as a *mechanism* claim, beyond the *what* claim here
  - **Random-subspace specificity controls** (0% effect) — Zhou et al. lacks this
  - **Progressive rotation into W_U** — orthogonal to Zhou et al.'s observation, complementary
  - **Three modern models** (Gemma 2B, Phi-3 Mini, LLaMA 3.2-3B) vs. GPT-2-XL primarily

- **`ARITHMETIC_CIRCUIT_PLAN.md`** — Zhou et al.'s approximation-vs-classification framing maps onto the repo's compute-layer (low-freq dominant?) vs. readout-layer (high-freq for digit ID?) split. The repo's `fourier_decomposition.py` (Step 3) and `fourier_knockout.py` (Step 4) are direct descendants of Zhou et al.'s methodology, with subspace-level granularity.

See `comparison_modular_digit_vs_repo.md` for the full deep comparison.
