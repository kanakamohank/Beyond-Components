# SVD Stats — OV Helix Circuit

Summary of best geometric (helix) representations found via SVD scanning of OV matrices across model variants.

---

## GPT-2 Small

### Run 1 (90 single-token numbers, 0–100)

| Property | Value |
|---|---|
| **Best Layer/Head** | Layer 10, Head 2 |
| **SVD Dims** | 1, 2 |
| **Circular Variance (CV)** | 0.334 |
| **Linearity** | 0.981 |
| **Matrix Type** | OV (Computation) |

**Variance Analysis:**

| Metric | Value |
|---|---|
| Top 1 Dim explains | 0.6% |
| Top 2 Dims explain | 3.0% |
| Top 3 Dims explain | 7.2% |
| Dims to reach 90% var | 640 |
| Dims to reach 95% var | 702 |

### Run 2 (99 single-token numbers, 0–100)

| Property | Value |
|---|---|
| **Best Layer/Head** | Layer 10, Head 2 |
| **SVD Dims** | 2, 7 |
| **Circular Variance (CV)** | 0.336 |
| **Linearity** | 0.978 |
| **Matrix Type** | OV (Computation) |

**Variance Analysis:**

| Metric | Value |
|---|---|
| Top 1 Dim explains | 0.8% |
| Top 2 Dims explain | 3.1% |
| Top 3 Dims explain | 7.2% |
| Dims to reach 90% var | 637 |
| Dims to reach 95% var | 699 |

**Visualization:** `geometry_gpt2-small_L10H2_OV.png`

---

## GPT-2 Medium

### Run (99 single-token numbers, 0–100)

Progression of candidates found during scan:

#### Candidate 1

| Property | Value |
|---|---|
| **Layer/Head** | Layer 19, Head 12 |
| **SVD Dims** | 1, 5 |
| **Circular Variance (CV)** | 0.337 |
| **Linearity** | 0.876 |
| **Matrix Type** | OV (Computation) |

#### Candidate 2

| Property | Value |
|---|---|
| **Layer/Head** | Layer 19, Head 12 |
| **SVD Dims** | 2, 4 |
| **Circular Variance (CV)** | 0.343 |
| **Linearity** | 0.977 |
| **Matrix Type** | OV (Computation) |

#### Candidate 3 (Best)

| Property | Value |
|---|---|
| **Best Layer/Head** | Layer 19, Head 12 |
| **SVD Dims** | 3, 4 |
| **Circular Variance (CV)** | 0.191 |
| **Linearity** | 0.983 |
| **Matrix Type** | OV (Computation) |

**Variance Analysis (Layer 19, Head 12 — shared across all candidates):**

| Metric | Value |
|---|---|
| Top 1 Dim explains | 0.3% |
| Top 2 Dims explain | 1.0% |
| Top 3 Dims explain | 4.1% |
| Dims to reach 90% var | 864 |
| Dims to reach 95% var | 943 |

**Visualization:** `geometry_gpt2-medium_L19H12_OV.png`

---

## Gemma-2 2B (`google/gemma-2-2b`)

### Run (Online Scanner — contextualized embeddings via shallow pass to Layer 1)

- **Scanner:** `online_svd_scanner.py` (on-the-fly W_OV = W_V @ W_O, no cache)
- **Embeddings:** Contextualized (residual stream after Layer 0 fuses multi-digit tokens)
- **Matrices scanned:** 26 layers × 8 heads = 208 OV matrices

| Property | Value |
|---|---|
| **Best Layer/Head** | Layer 9, Head 1 |
| **SVD Dims** | 1, 2 |
| **Circular Variance (CV)** | 0.293 |
| **Linearity** | 0.999 |
| **Matrix Type** | OV (Computation) |

**Singular Value Hypothesis:**

| Metric | Value |
|---|---|
| Sigma 1 | 0.9268 |
| Sigma 2 | 0.8232 |
| Ratio (ideal = 1.0) | 1.1259 |

**Fourier Frequency Mapping (Top 10 SVD Dims):**

| Dim | Dominant Period | Signal Strength |
|---|---|---|
| 0 | 3.3 | 10.2% |
| **1** | **3.3** | **13.9%** |
| **2** | **9.9** | **22.5%** |
| 3 | 9.9 | 11.8% |
| 4 | 3.3 | 12.1% |
| 5 | 2.0 | 14.2% |
| 6 | 9.9 | 18.5% |
| 7 | 3.3 | 9.6% |
| 8 | 4.9 | 13.9% |
| 9 | 4.9 | 13.3% |

> **Note:** Unlike GPT-2 models (which lock to T≈99), Gemma-2 2B shows a rich multi-frequency spectrum. The winning Dim 2 peaks at T=9.9 (base-10 structure), while Dim 1 peaks at T=3.3. Additional periods at 4.9 and 2.0 suggest the model encodes multiple arithmetic bases simultaneously.

**Visualizations:**
- `geometry_google_gemma-2-2b_L9H1_OV.png`
- `fft_spectrum_google_gemma-2-2b_L9H1.png`
- `google_gemma-2-2b_L9H1_OV_sine_cosine_unpacking_L_H_dims_1_2.png`

### Run 2 (Simulated 4-Bit Quantization)

- **Method:** Simulated quantization on W_OV — `round(W_OV / scale) * scale` with scale = max(|W_OV|) / 7
- **SVD performed on:** `W_OV_quantized` (precision crushed to 4-bit equivalent)

Progression of candidates during scan:

#### Candidate 1

| Property | Value |
|---|---|
| **Layer/Head** | Layer 9, Head 7 |
| **SVD Dims** | 1, 3 |
| **Circular Variance (CV)** | 0.290 |
| **Linearity** | 0.994 |
| **Matrix Type** | OV (Computation) |

**Singular Value Hypothesis:**

| Metric | Value |
|---|---|
| Sigma 1 | 0.9583 |
| Sigma 3 | 0.8800 |
| Ratio (ideal = 1.0) | 1.0889 |

#### Candidate 2 (Best)

| Property | Value |
|---|---|
| **Best Layer/Head** | Layer 11, Head 3 |
| **SVD Dims** | 0, 7 |
| **Circular Variance (CV)** | 0.249 |
| **Linearity** | 0.994 |
| **Matrix Type** | OV (Computation) |

**Singular Value Hypothesis:**

| Metric | Value |
|---|---|
| Sigma 0 | 0.9597 |
| Sigma 7 | 0.5783 |
| Ratio (ideal = 1.0) | 1.6596 |

**Fourier Frequency Mapping (Top 10 SVD Dims):**

| Dim | Dominant Period | Signal Strength |
|---|---|---|
| **0** | **9.9** | **10.9%** |
| 1 | 9.9 | 15.1% |
| 2 | 2.5 | 10.9% |
| 3 | 9.9 | 12.2% |
| 4 | 2.5 | 14.6% |
| 5 | 9.9 | 12.1% |
| 6 | 99.0 | 9.8% |
| **7** | **9.9** | **18.8%** |
| 8 | 9.9 | 13.5% |
| 9 | 3.3 | 11.1% |

> **Note:** Under 4-bit quantization, the best head **shifts** from L9H1 → L11H3, and the winning dims spread from (1,2) → (0,7). Despite the σ ratio degrading to 1.66, the helix actually gets **tighter** (CV 0.293 → 0.249). T=9.9 dominates even more strongly (7 of 10 dims), suggesting base-10 structure is robust to quantization noise. The T=3.3 signal weakens while T=2.5 emerges.

**Visualizations:**
- `geometry_google_gemma-2-2b_L11H3_OV_4bit_quant.png`
- `fft_spectrum_google_gemma-2-2b_L11H3_4bit_quant.png`
- `google_gemma-2-2b_L11H3_OV_sine_cosine_unpacking_L_H_dims_0_7_4bit_quant.png`

---

## Phi-3 Mini (`microsoft/Phi-3-mini-4k-instruct`)

### Run (Online Scanner — contextualized embeddings via shallow pass to Layer 1)

- **Scanner:** `online_svd_scanner.py` (on-the-fly W_OV = W_V @ W_O, no cache)
- **Embeddings:** Contextualized (residual stream after Layer 0 fuses multi-digit tokens)
- **Matrices scanned:** 32 layers × 32 heads = 1024 OV matrices

Progression of candidates during scan:

#### Candidate 1

| Property | Value |
|---|---|
| **Layer/Head** | Layer 0, Head 8 |
| **SVD Dims** | 2, 4 |
| **Circular Variance (CV)** | 0.286 |
| **Linearity** | 0.993 |
| **Matrix Type** | OV (Computation) |

**Singular Value Hypothesis:**

| Metric | Value |
|---|---|
| Sigma 2 | 0.0377 |
| Sigma 4 | 0.0305 |
| Ratio (ideal = 1.0) | 1.2367 |

#### Candidate 2

| Property | Value |
|---|---|
| **Layer/Head** | Layer 0, Head 8 |
| **SVD Dims** | 4, 8 |
| **Circular Variance (CV)** | 0.225 |
| **Linearity** | 0.992 |
| **Matrix Type** | OV (Computation) |

**Singular Value Hypothesis:**

| Metric | Value |
|---|---|
| Sigma 4 | 0.0305 |
| Sigma 8 | 0.0250 |
| Ratio (ideal = 1.0) | 1.2223 |

#### Candidate 3

| Property | Value |
|---|---|
| **Layer/Head** | Layer 1, Head 29 |
| **SVD Dims** | 0, 5 |
| **Circular Variance (CV)** | 0.209 |
| **Linearity** | 0.999 |
| **Matrix Type** | OV (Computation) |

**Singular Value Hypothesis:**

| Metric | Value |
|---|---|
| Sigma 0 | 0.1042 |
| Sigma 5 | 0.0974 |
| Ratio (ideal = 1.0) | 1.0701 |

#### Candidate 4

| Property | Value |
|---|---|
| **Layer/Head** | Layer 4, Head 7 |
| **SVD Dims** | 0, 5 |
| **Circular Variance (CV)** | 0.180 |
| **Linearity** | 0.984 |
| **Matrix Type** | OV (Computation) |

**Singular Value Hypothesis:**

| Metric | Value |
|---|---|
| Sigma 0 | 0.8280 |
| Sigma 5 | 0.6930 |
| Ratio (ideal = 1.0) | 1.1949 |

#### Candidate 5

| Property | Value |
|---|---|
| **Layer/Head** | Layer 12, Head 10 |
| **SVD Dims** | 3, 7 |
| **Circular Variance (CV)** | 0.184 |
| **Linearity** | 0.997 |
| **Matrix Type** | OV (Computation) |

**Singular Value Hypothesis:**

| Metric | Value |
|---|---|
| Sigma 3 | 2.2163 |
| Sigma 7 | 1.9859 |
| Ratio (ideal = 1.0) | 1.1161 |

#### Candidate 6

| Property | Value |
|---|---|
| **Layer/Head** | Layer 16, Head 21 |
| **SVD Dims** | 6, 9 |
| **Circular Variance (CV)** | 0.162 |
| **Linearity** | 0.992 |
| **Matrix Type** | OV (Computation) |

**Singular Value Hypothesis:**

| Metric | Value |
|---|---|
| Sigma 6 | 1.9244 |
| Sigma 9 | 1.8584 |
| Ratio (ideal = 1.0) | 1.0355 |

#### Candidate 7 (Best)

| Property | Value |
|---|---|
| **Best Layer/Head** | Layer 24, Head 28 |
| **SVD Dims** | 3, 7 |
| **Circular Variance (CV)** | 0.141 |
| **Linearity** | 0.999 |
| **Matrix Type** | OV (Computation) |

**Singular Value Hypothesis:**

| Metric | Value |
|---|---|
| Sigma 3 | 3.9990 |
| Sigma 7 | 3.7503 |
| Ratio (ideal = 1.0) | 1.0663 |

**Fourier Frequency Mapping (Top 10 SVD Dims):**

| Dim | Dominant Period | Signal Strength |
|---|---|---|
| 0 | 2.5 | 10.7% |
| 1 | 4.9 | 16.0% |
| 2 | 9.9 | 17.9% |
| **3** | **9.9** | **18.4%** |
| 4 | 3.3 | 9.4% |
| 5 | 4.9 | 9.0% |
| 6 | 3.3 | 10.5% |
| **7** | **9.9** | **35.6%** |
| 8 | 9.9 | 12.2% |
| 9 | 2.0 | 14.5% |

> **Note:** Phi-3 Mini shows the richest candidate progression — 7 successively better heads across layers 0–24, demonstrating helical structure at multiple depths. The winning Dim 7 has the strongest single-dim Fourier signal seen so far (35.6% at T=9.9). Like Gemma-2 2B, the spectrum is multi-frequency (T=9.9, 4.9, 3.3, 2.5, 2.0), but T=9.9 dominates 4 of 10 dims. Singular values grow with layer depth (0.03 at L0 → 4.0 at L24), and the σ ratios stay moderate (1.07–1.24), with the best candidate at 1.07.

**Visualizations:**
- `geometry_microsoft_Phi-3-mini-4k-instruct_L24H28_OV.png`
- `fft_spectrum_microsoft_Phi-3-mini-4k-instruct_L24H28.png`
- `microsoft_Phi-3-mini-4k-instruct_L24H28_OV_sine_cosine_unpacking_L_H_dims_3_7.png`

---

## GPT-J 6B (`EleutherAI/gpt-j-6B`)

### Run (Online Scanner — contextualized embeddings via shallow pass to Layer 1)

- **Scanner:** `online_svd_scanner.py` (on-the-fly W_OV = W_V @ W_O, no cache)
- **Embeddings:** Contextualized (residual stream after Layer 0 fuses multi-digit tokens)
- **Matrices scanned:** 28 layers × 16 heads = 448 OV matrices
- **Selection criteria:** CV < 0.4, Linearity > 0.9

Only one candidate passed the threshold:

| Property | Value |
|---|---|
| **Best Layer/Head** | Layer 3, Head 13 |
| **SVD Dims** | 2, 3 |
| **Circular Variance (CV)** | 0.273 |
| **Linearity** | 0.928 |
| **Matrix Type** | OV (Computation) |

**Singular Value Hypothesis:**

| Metric | Value |
|---|---|
| Sigma 2 | 1.1891 |
| Sigma 3 | 1.1729 |
| Ratio (ideal = 1.0) | 1.0138 |

**Fourier Frequency Mapping (Top 10 SVD Dims):**

| Dim | Dominant Period | Signal Strength |
|---|---|---|
| 0 | 49.5 | 8.0% |
| 1 | 99.0 | 15.2% |
| **2** | **99.0** | **27.7%** |
| **3** | **99.0** | **23.5%** |
| 4 | 99.0 | 16.2% |
| 5 | 19.8 | 7.2% |
| 6 | 33.0 | 6.6% |
| 7 | 99.0 | 9.3% |
| 8 | 99.0 | 10.1% |
| 9 | 99.0 | 15.2% |

> **Note:** GPT-J 6B behaves like a GPT-2 family model — T=99 dominates (7 of 10 dims), with sub-harmonics at T=49.5, 33.0, and 19.8. The winning dims (2, 3) have the strongest signals (27.7% and 23.5%). Despite being a 6B-parameter model, only one candidate passed the relaxed threshold (CV < 0.4, Lin > 0.9), and the linearity (0.928) is the lowest across all models tested. However, the σ ratio (1.014) is the **best seen so far** — near-perfect singular value balance.

**Visualizations:**
- `geometry_EleutherAI_gpt-j-6B_L3H13_OV.png`
- `fft_spectrum_EleutherAI_gpt-j-6B_L3H13.png`
- `EleutherAI_gpt-j-6B_L3H13_OV_sine_cosine_unpacking_L_H_dims_2_3.png`

---

## Gemma 7B (`google/gemma-7b`)

### Run (Online Scanner — contextualized embeddings via shallow pass to Layer 1)

- **Scanner:** `online_svd_scanner.py` (on-the-fly W_OV = W_V @ W_O, no cache)
- **Embeddings:** Contextualized (residual stream after Layer 0 fuses multi-digit tokens)
- **Matrices scanned:** 28 layers × 16 heads = 448 OV matrices

Only one candidate passed the threshold:

| Property | Value |
|---|---|
| **Best Layer/Head** | Layer 14, Head 2 |
| **SVD Dims** | 2, 5 |
| **Circular Variance (CV)** | 0.220 |
| **Linearity** | 0.997 |
| **Matrix Type** | OV (Computation) |

**Singular Value Hypothesis:**

| Metric | Value |
|---|---|
| Sigma 2 | 0.1441 |
| Sigma 5 | 0.1387 |
| Ratio (ideal = 1.0) | 1.0388 |

**Fourier Frequency Mapping (Top 10 SVD Dims):**

| Dim | Dominant Period | Signal Strength |
|---|---|---|
| 0 | 9.9 | 19.7% |
| 1 | 9.9 | 14.4% |
| **2** | **9.9** | **16.4%** |
| 3 | 4.9 | 12.4% |
| 4 | 9.9 | 20.7% |
| **5** | **9.9** | **13.5%** |
| 6 | 3.3 | 11.3% |
| 7 | 9.9 | 13.7% |
| 8 | 4.9 | 7.2% |
| 9 | 4.9 | 13.9% |

> **Note:** Gemma 7B strongly confirms the Gemma-family pattern — T=9.9 dominates (6 of 10 dims), with sub-harmonics at T=4.9 and T=3.3. Compared to its smaller sibling Gemma-2 2B, Gemma 7B achieves better CV (0.220 vs 0.293) and a tighter σ ratio (1.039 vs 1.126), while maintaining near-perfect linearity (0.997). The best head sits at mid-depth (L14) rather than early layers.

**Visualizations:**
- `geometry_google_gemma-7b_L14H2_OV.png`
- `fft_spectrum_google_gemma-7b_L14H2.png`
- `google_gemma-7b_L14H2_OV_sine_cosine_unpacking_L_H_dims_2_5.png`

---

## 8. Llama 3.2 3B (`meta-llama/Llama-3.2-3B`) — Online Scan (bf16)

**Scanner:** `scan_model_on_the_fly` (Online — contextualized embeddings, bf16)
**Matrices Scanned:** 672 (28 layers × 24 heads)

**Best Candidate:**

| Metric | Value |
|---|---|
| **Layer / Head** | L1 H17 |
| **SVD Dims** | 2, 6 |
| **Circular Variance (CV)** | 0.312 |
| **Linearity** | 0.934 |
| **Matrix Type** | OV (Computation) |

**Singular Value Hypothesis:**

| Metric | Value |
|---|---|
| Sigma 2 | 0.0919 |
| Sigma 6 | 0.0732 |
| Ratio (ideal = 1.0) | 1.2555 |

**Fourier Frequency Mapping (Top 10 SVD Dims):**

| Dim | Dominant Period | Signal Strength |
|---|---|---|
| 0 | 99.0 | 9.9% |
| 1 | 99.0 | 9.3% |
| **2** | **99.0** | **7.9%** |
| 3 | 99.0 | 14.3% |
| 4 | 2.0 | 10.6% |
| 5 | 2.0 | 11.9% |
| **6** | **2.0** | **23.6%** |
| 7 | 2.0 | 11.1% |
| 8 | 2.0 | 9.7% |
| 9 | 99.0 | 5.8% |

> **Note:** Llama 3.2 3B shows a clear split in Fourier behavior: Dims 0–3 and 9 lock to T=99.0 (single-rotation, GPT-family pattern), while Dims 4–8 lock to T=2.0 (a strong even/odd parity signal). The best SVD pair (2, 6) bridges these two regimes — Dim 2 at T=99.0 and Dim 6 at T=2.0. The σ ratio (1.256) is the highest among all models tested, indicating significant anisotropy in the encoding plane. The best head sits very early (L1), similar to GPT-J 6B (L3), suggesting that Llama places its arithmetic encoding in early layers. CV (0.312) and linearity (0.934) are moderate — comparable to GPT-2 Small (0.336 / 0.978) and GPT-J 6B (0.273 / 0.928).

**Visualizations:**
- `geometry_meta-llama_Llama-3.2-3B_L1H17_OV.png`
- `fft_spectrum_meta-llama_Llama-3.2-3B_L1H17.png`
- `meta-llama_Llama-3.2-3B_L1H17_OV_sine_cosine_unpacking_L_H_dims_2_6.png`

---

## Cross-Model Comparison

| Model | Scanner | Best Head | SVD Dims | CV | Linearity | σ Ratio | Dominant Period |
|---|---|---|---|---|---|---|---|
| **GPT-2 Small** | Offline | L10 H2 | 2, 7 | 0.336 | 0.978 | — | 99.0 |
| **GPT-2 Medium** | Offline | L19 H12 | 3, 4 | 0.191 | 0.983 | — | 99.0 |
| **GPT-J 6B** | Online (bf16) | L3 H13 | 2, 3 | 0.273 | 0.928 | 1.0138 | 99.0 |
| **Gemma 7B** | Online (bf16) | L14 H2 | 2, 5 | 0.220 | 0.997 | 1.0388 | 9.9 |
| **Gemma-2 2B** | Online (bf16) | L9 H1 | 1, 2 | 0.293 | 0.999 | 1.1259 | 3.3 / 9.9 |
| **Gemma-2 2B (4-bit)** | Online (sim. quant) | L11 H3 | 0, 7 | 0.249 | 0.994 | 1.6596 | 9.9 |
| **Phi-3 Mini** | Online (bf16) | L24 H28 | 3, 7 | 0.141 | 0.999 | 1.0663 | 9.9 |
| **Llama 3.2 3B** | Online (bf16) | L1 H17 | 2, 6 | 0.312 | 0.934 | 1.2555 | 99.0 / 2.0 |

---

## Key Observations

- **All models converge on a single dedicated head** — GPT-2 Small (L10H2), GPT-2 Medium (L19H12), GPT-J 6B (L3H13), Gemma 7B (L14H2), Gemma-2 2B (L9H1), Phi-3 Mini (L24H28), Llama 3.2 3B (L1H17).
- **Phi-3 Mini achieves the best overall helix** — lowest CV (0.141), perfect linearity (0.999), and the strongest single-dim Fourier signal (35.6%).
- **Gemma 7B improves on Gemma-2 2B** — better CV (0.220 vs 0.293), tighter σ ratio (1.039 vs 1.126), and cleaner Fourier spectrum. Scaling within the same architecture family does help.
- **GPT-J 6B has the best σ ratio (1.014) but weakest linearity (0.928)** — near-perfect singular value balance doesn't guarantee a clean helix.
- **GPT-J 6B clusters with GPT-2 in Fourier behavior** — all three GPT-family models lock to T≈99, while Gemma and Phi-3 use T≈9.9 (base-10). This is an architectural/training-data divide.
- **Helix quality does not scale with model size alone** — GPT-J 6B (6B params) has the weakest linearity; Phi-3 Mini (3.8B) has the best. But within the same family (Gemma), the larger model does produce a cleaner helix.
- **Low circular variance ranking:** Phi-3 Mini (0.141) > GPT-2 Medium (0.191) > Gemma 7B (0.220) > GPT-J 6B (0.273) > Gemma-2 2B (0.293) > Llama 3.2 3B (0.312) > GPT-2 Small (0.336).
- **4-bit quantization preserves the helix** — Gemma-2 2B's CV improves (0.293 → 0.249) and linearity stays high (0.994) under simulated quantization.
- **Fourier spectra split into two families:** GPT-family (T≈99, single rotation) vs modern architectures (T≈9.9, multi-frequency base-10 encoding). **Llama 3.2 3B bridges both** — T=99.0 in low dims and T=2.0 (even/odd parity) in higher dims.
- **Matrices scanned:** 2086 cached OV/QK per GPT-2 model; 672 for Llama 3.2 3B; 448 for GPT-J 6B and Gemma 7B; 208 for Gemma-2 2B; 1024 for Phi-3 Mini.
- **Llama 3.2 3B has the highest σ ratio (1.256)** — the most anisotropic encoding plane, suggesting the helix is more elliptical than circular. Combined with moderate CV (0.312) and the lowest linearity among larger models (0.934), Llama's arithmetic encoding is functional but geometrically impure.
- **Early-layer arithmetic heads:** Both Llama 3.2 3B (L1) and GPT-J 6B (L3) place their best head in early layers, while Gemma (L9/L14) and Phi-3 (L24) use mid-to-late layers. This may reflect differences in how these architectures distribute computation.

---

## Experiment: Corrected Pipeline — Live Contextualized Helix Verification

### Objective

The earlier SVD scans (above) found helical structure by projecting **static token embeddings** through the right singular vectors (Vh) of each head's OV matrix. The `run_corrected_pipeline` tests whether that helix survives when probed with **live, contextualized activations** from actual arithmetic prompts.

The pipeline uses 100 prompts of the form `"What is {n} + 5?"` (n = 0–99) and captures the residual stream at `hook_resid_pre` of the target layer. It then searches for a clean **T=10 modular clock** — a 2D plane where the angle encodes the ones digit (n mod 10) and completes one full rotation every 10 numbers.

**Key sub-functions:**

1. **`collect_activations`** — Runs each prompt through the model. Captures the residual stream vector at two positions:
   - **"operand"** — the token immediately before the `+` operator (the fused representation of `n`)
   - **"final"** — the last token in the sequence (`?`)

2. **`analyze_helix`** — Applies PCA to the collected activation matrix, then tests all pairs of PCA components for circular geometry. Reports four metrics:
   - **Radius CV** — coefficient of variation of radii from center (target < 0.20 for a clean circle)
   - **Phase Match (ones)** — `mean(cos(θ_actual − θ_ideal))` where `θ_ideal = (n%10 / 10) × 2π`. Measures whether the angle explicitly encodes the ones digit mod 10 (target > 0.90)
   - **Lin (raw n)** — Pearson correlation of unwrapped angles with raw `n` (measures monotonic angular progression regardless of period)
   - **Period T** — `2π / slope` from linear fit of unwrapped angles vs `n` (target ≈ 10.0 for a mod-10 clock)

3. **`svd_reading_directions`** — Projects the operand activations into the top-10 **left singular vectors (U columns)** of the OV matrix (the head's *reading* subspace), then runs PCA + `analyze_helix` on that reduced 10-D space.

4. **`sweep_layers`** — Repeats the PCA circle search across layers 8–25 at both token positions, finding the layer with the best phase-match vs CV trade-off.

**Verdict criteria:**
- ✅ **CLEAN T=10 HELIX:** CV < 0.20 AND Phase > 0.90 AND 9 < T < 11
- ⚠️ **PARTIAL HELIX:** CV < 0.35 AND Phase > 0.75
- 🔶 **WRONG PERIOD:** Lin(raw n) > 0.70 but T ≠ 10
- ❌ **NO CLEAN HELIX:** none of the above

---

### Results: Gemma 7B (`google/gemma-7b`) — Layer 14, Head 2

#### Direct PCA — Operand Position, Layer 14

| Metric | Value | Target |
|---|---|---|
| Best PC pair | (10, 11) | — |
| Radius CV | 0.468 | < 0.20 |
| Phase Match (ones) | 0.009 | > 0.90 |
| Lin (raw n) | 0.976 | — |
| Period T | 9.88 | ~10.0 |
| **Verdict** | ❌ NO CLEAN HELIX | |

#### Direct PCA — Final Token Position, Layer 14

| Metric | Value | Target |
|---|---|---|
| Best PC pair | (9, 10) | — |
| Radius CV | 0.587 | < 0.20 |
| Phase Match (ones) | 0.098 | > 0.90 |
| Lin (raw n) | 0.919 | — |
| Period T | 17.21 | ~10.0 |
| **Verdict** | 🔶 WRONG PERIOD (T=17.2) | |

#### SVD Reading-Direction PCA — Layer 14, Head 2

| Metric | Value | Target |
|---|---|---|
| Best PC pair | (5, 9) | — |
| Radius CV | 0.550 | < 0.20 |
| Phase Match (ones) | −0.083 | > 0.90 |
| Lin (raw n) | 0.961 | — |
| Period T | 18.29 | ~10.0 |
| **Verdict** | 🔶 WRONG PERIOD (T=18.3) | |

#### Layer Sweep (Layers 8–25)

| Layer | Position | CV | Phase(ones) | Period | Verdict |
|---|---|---|---|---|---|
| 8 | operand | 0.461 | 0.288 | 89.71 | ❌ None |
| 8 | final | 0.467 | 0.057 | 57.82 | ❌ None |
| 9 | operand | 0.398 | 0.362 | 1150.57 | ❌ None |
| 9 | final | 0.454 | 0.070 | 59.46 | ❌ None |
| 10 | operand | 0.509 | 0.288 | 1321.54 | ❌ None |
| 10 | final | 0.514 | 0.180 | 202.76 | ❌ None |
| 11 | operand | 0.457 | 0.287 | 25.66 | ❌ None |
| 11 | final | 0.538 | 0.192 | 73.01 | ❌ None |
| 12 | operand | 0.478 | 0.329 | 47.58 | ❌ None |
| 12 | final | 0.466 | 0.121 | 91.26 | ❌ None |
| 13 | operand | 0.501 | 0.479 | 1318.27 | ❌ None |
| 13 | final | 0.388 | 0.013 | 269.33 | ❌ None |
| 14 | operand | 0.514 | 0.283 | 22.12 | ❌ None |
| 14 | final | 0.393 | 0.022 | 17012.88 | ❌ None |
| 15 | operand | 0.443 | 0.297 | 27.18 | ❌ None |
| 15 | final | 0.390 | 0.031 | 90.56 | ❌ None |
| 16 | operand | 0.502 | 0.338 | 36.74 | ❌ None |
| 16 | final | 0.358 | 0.048 | 90.00 | ❌ None |
| 17 | operand | 0.511 | 0.404 | 15.68 | ❌ None |
| 17 | final | 0.367 | −0.047 | 381.02 | ❌ None |
| 18 | operand | 0.477 | 0.380 | 63.09 | ❌ None |
| 18 | final | 0.374 | 0.029 | 94.57 | ❌ None |
| 19 | operand | 0.493 | 0.524 | 41.73 | ❌ None |
| 19 | final | 0.558 | 0.210 | 50.94 | ❌ None |
| 20 | operand | 0.423 | 0.467 | 84.04 | ❌ None |
| 20 | final | 0.501 | 0.010 | 154.44 | ❌ None |
| 21 | operand | 0.424 | 0.526 | 16.10 | ❌ None |
| 21 | final | 0.511 | 0.079 | 160.04 | ❌ None |
| 22 | operand | 0.445 | 0.496 | 16.05 | ❌ None |
| 22 | final | 0.521 | 0.087 | 138.84 | ❌ None |
| 23 | operand | 0.546 | 0.260 | 725.09 | ❌ None |
| 23 | final | 0.507 | −0.004 | 45.66 | ❌ None |
| 24 | operand | 0.514 | 0.104 | 45.21 | ❌ None |
| 24 | final | 0.479 | 0.005 | 126.21 | ❌ None |
| 25 | operand | 0.694 | 0.369 | 48.54 | ❌ None |
| 25 | final | 0.480 | 0.079 | 27.88 | ❌ None |

**Best layer:** Layer 21, operand position (CV=0.424, Phase=0.526, T=16.1)

#### Gemma 7B Summary

| Analysis | CV | Phase | Period T |
|---|---|---|---|
| Direct PCA operand L14 | 0.468 | 0.009 | 9.9 |
| Direct PCA final L14 | 0.587 | 0.098 | 17.2 |
| SVD-Reading L14H2 | 0.550 | −0.083 | 18.3 |
| Best layer sweep | 0.424 | 0.526 | 16.1 |

**Overall Verdict: ❌ No clean T=10 helix found in any layer or position for Gemma 7B.**

---

### Results: Phi-3 Mini (`microsoft/Phi-3-mini-4k-instruct`) — Layer 24, Head 28

#### Direct PCA — Operand Position, Layer 24

| Metric | Value | Target |
|---|---|---|
| Best PC pair | (3, 4) | — |
| Radius CV | 0.448 | < 0.20 |
| Phase Match (ones) | 0.652 | > 0.90 |
| Lin (raw n) | 0.991 | — |
| Period T | 12.69 | ~10.0 |
| **Verdict** | 🔶 WRONG PERIOD (T=12.7) | |

#### Direct PCA — Final Token Position, Layer 24

| Metric | Value | Target |
|---|---|---|
| Best PC pair | (6, 11) | — |
| Radius CV | 0.568 | < 0.20 |
| Phase Match (ones) | −0.049 | > 0.90 |
| Lin (raw n) | 0.937 | — |
| Period T | 25.09 | ~10.0 |
| **Verdict** | 🔶 WRONG PERIOD (T=25.1) | |

#### SVD Reading-Direction PCA — Layer 24, Head 28

| Metric | Value | Target |
|---|---|---|
| Best PC pair | (1, 8) | — |
| Radius CV | 0.609 | < 0.20 |
| Phase Match (ones) | −0.002 | > 0.90 |
| Lin (raw n) | 0.952 | — |
| Period T | 13.07 | ~10.0 |
| **Verdict** | 🔶 WRONG PERIOD (T=13.1) | |

#### Layer Sweep (Layers 8–25)

| Layer | Position | CV | Phase(ones) | Period | Verdict |
|---|---|---|---|---|---|
| 8 | operand | 0.389 | 0.114 | 84.44 | ❌ None |
| 8 | final | 0.484 | 0.030 | 100.71 | ❌ None |
| 9 | operand | 0.459 | 0.291 | 99.44 | ❌ None |
| 9 | final | 0.582 | 0.153 | 90.55 | ❌ None |
| 10 | operand | 0.402 | 0.193 | 27.37 | ❌ None |
| 10 | final | 0.569 | 0.102 | 90.03 | ❌ None |
| 11 | operand | 0.382 | 0.258 | 107.91 | ❌ None |
| 11 | final | 0.534 | 0.091 | 383.72 | ❌ None |
| 12 | operand | 0.618 | 0.268 | 17.74 | ❌ None |
| 12 | final | 0.581 | 0.256 | 59.50 | ❌ None |
| 13 | operand | 0.430 | 0.063 | 25.95 | ❌ None |
| 13 | final | 0.553 | 0.081 | 142.36 | ❌ None |
| 14 | operand | 0.506 | 0.378 | 106.41 | ❌ None |
| 14 | final | 0.470 | 0.117 | 119.08 | ❌ None |
| 15 | operand | 0.450 | 0.428 | 24.96 | ❌ None |
| 15 | final | 0.550 | 0.167 | 75.15 | ❌ None |
| 16 | operand | 0.466 | 0.106 | 19.80 | ❌ None |
| 16 | final | 0.496 | 0.119 | 195.96 | ❌ None |
| 17 | operand | 0.413 | 0.207 | 25.35 | ❌ None |
| 17 | final | 0.537 | 0.171 | 35.34 | ❌ None |
| 18 | operand | 0.454 | 0.355 | 15.61 | ❌ None |
| 18 | final | 0.543 | 0.075 | 128.02 | ❌ None |
| 19 | operand | 0.427 | 0.036 | 96.36 | ❌ None |
| 19 | final | 0.457 | 0.115 | 85.26 | ❌ None |
| 20 | operand | 0.463 | 0.035 | 144.38 | ❌ None |
| 20 | final | 0.446 | 0.061 | 94.90 | ❌ None |
| 21 | operand | 0.408 | −0.005 | 220.60 | ❌ None |
| 21 | final | 0.568 | 0.113 | 263.58 | ❌ None |
| 22 | operand | 0.461 | 0.258 | 80.36 | ❌ None |
| 22 | final | 0.546 | 0.156 | 207.44 | ❌ None |
| 23 | operand | 0.433 | 0.403 | 55.31 | ❌ None |
| 23 | final | 0.517 | 0.130 | 145.71 | ❌ None |
| 24 | operand | 0.448 | 0.652 | 12.69 | 🔶 T=12.7≠10 |
| 24 | final | 0.520 | 0.041 | 152.42 | ❌ None |
| 25 | operand | 0.381 | 0.322 | 39.76 | ❌ None |
| 25 | final | 0.538 | 0.066 | 829.05 | ❌ None |

**Best layer:** Layer 24, operand position (CV=0.448, Phase=0.652, T=12.7)

#### Phi-3 Mini Summary

| Analysis | CV | Phase | Period T |
|---|---|---|---|
| Direct PCA operand L24 | 0.448 | 0.652 | 12.7 |
| Direct PCA final L24 | 0.568 | −0.049 | 25.1 |
| SVD-Reading L24H28 | 0.609 | −0.002 | 13.1 |
| Best layer sweep | 0.448 | 0.652 | 12.7 |

**Overall Verdict: 🔶 Partial angular structure at Layer 24, but T≈12.7 ≠ 10 and CV too high. No clean T=10 helix.**

---

### Corrected Pipeline — Key Findings

- **Neither model produces a clean T=10 modular helix with live activations.** All CVs are > 0.35 (target < 0.20) and all phase-match scores are < 0.66 (target > 0.90). This is a **negative result** that constrains the helix hypothesis.
- **Static SVD ≠ Live activations.** The earlier scans found strong helices (CV=0.141–0.293, linearity=0.928–0.999) using static embeddings projected through Vh. Those structures exist in the *weight geometry* of W_OV, but do **not** cleanly manifest as a T=10 clock in the live residual stream during arithmetic.
- **Lin(raw n) remains high (0.92–0.99)** even when phase-match fails — meaning the angles *do* increase monotonically with n, but the period is wrong (T≈13–18 instead of 10). The model uses a continuous angular encoding, not a discrete mod-10 clock.
- **Phi-3 Mini is closer than Gemma 7B.** Phi-3 achieves Phase=0.652 at L24 (the strongest signal in the sweep), while Gemma 7B's best is Phase=0.526 at L21. This aligns with Phi-3's superior static helix quality (CV=0.141).
- **Operand position consistently outperforms final position** — the representation of the number itself carries more angular information than the aggregated final-token representation.
- **The helix hypothesis survives as a weight-space phenomenon** — the geometric structure in W_OV is real and consistent across models, but during live inference the residual stream spreads across many more dimensions than the 2D plane can capture.

---

## Experiment: Phase-Corrected Pipeline — Fixing the Phase Offset Bug

### The Bug

The previous `run_corrected_pipeline` computed phase match as:

```
Phase = mean(cos(θ_actual − θ_ideal))     where θ_ideal = (n%10 / 10) × 2π
```

This **assumes the helix starts at angle 0 for digit 0**. But PCA axes are arbitrary — the entire circle can be rotated by any constant offset φ. A perfect mod-10 clock rotated by, say, 120° would score *poorly* under the original metric because every angle is shifted by a constant that the formula doesn't account for.

### The Fix (`run_phase_corrected_pipeline`)

The new `phase_match_optimized` function searches over 720 evenly-spaced phase offsets φ ∈ [0, 2π) and finds the φ that **maximizes** alignment:

```
Phase_corr = max_φ  mean(cos(θ_actual − θ_ideal − φ))
```

This makes the metric **rotationally invariant** — it measures whether the *relative spacing* of angles matches a mod-10 pattern, regardless of where "digit 0" happens to land on the PCA circle.

### Methodology Comparison: `run_corrected_pipeline` vs `run_phase_corrected_pipeline`

| Aspect | Previous (`run_corrected_pipeline`) | New (`run_phase_corrected_pipeline`) |
|---|---|---|
| **Phase metric** | Fixed origin: `mean(cos(θ − θ_ideal))` | Phase-invariant: `max_φ mean(cos(θ − θ_ideal − φ))` over 720 offsets |
| **Reports** | Single phase score | Both Phase_ORIGINAL (buggy, for reference) and Phase_CORRECTED |
| **SVD reading directions** | Yes — projects into top-10 U columns of OV matrix | No — only raw PCA on residual stream |
| **PCA components** | 10 PCs, tests pairs from top 8 | 15 PCs, tests pairs from top 12 (wider search) |
| **Scoring function** | `phase − cv` | `phase_corr − cv − 0.3 × |T−10|/10` (penalizes wrong period) |
| **New verdict** | — | 🔷 MONOTONE HELIX (Vector Translation) when Lin(raw n) > 0.90 but phase_corr < 0.85 |
| **Verdict thresholds** | CV < 0.20, Phase > 0.90, 9 < T < 11 | CV < 0.20, Phase_corr > 0.85, 9 < T < 11 |

### Impact of the Fix

The phase correction reveals that the original pipeline **systematically underreported** phase alignment. In many cases, a strong rotational pattern existed but was hidden by an arbitrary PCA phase offset. For example:

- **Gemma 7B L21 operand:** Phase 0.526 → **0.885** (offset = ~unknown, but the original metric was seeing the clock "rotated away")
- **Phi-3 Mini L24 operand:** Phase 0.652 → **0.765** (modest gain — the original offset happened to be closer to zero)
- **Llama 3.2 3B L1 operand:** Phase 0.156 → **0.192** (minimal gain — genuinely weak modular structure)

---

### Results: Gemma 7B (`google/gemma-7b`) — Phase-Corrected, Layer 14, Head 2

#### Direct PCA — Operand Position, Layer 14

| Metric | Value | Target |
|---|---|---|
| Best PC pair | (3, 4) | — |
| Radius CV | 0.4890 | < 0.20 |
| Lin (raw n) | 0.9937 | — |
| Phase ORIGINAL | −0.2977 | (buggy — reference) |
| **Phase CORRECTED** | **0.6274** | > 0.85 |
| Phase offset | 118.5° | — |
| Period T | 11.26 | ~10.0 |
| **Verdict** | 🔷 MONOTONE HELIX (Vector Translation) | |

#### Direct PCA — Final Token Position, Layer 14

| Metric | Value | Target |
|---|---|---|
| Best PC pair | (9, 10) | — |
| Radius CV | 0.5867 | < 0.20 |
| Lin (raw n) | 0.9187 | — |
| Phase ORIGINAL | 0.0986 | (buggy — reference) |
| **Phase CORRECTED** | **0.0998** | > 0.85 |
| Phase offset | 351.5° | — |
| Period T | 17.21 | ~10.0 |
| **Verdict** | 🔷 MONOTONE HELIX (Vector Translation) | |

#### Layer Sweep (Layers 8–25, Corrected Phase Metric)

| Layer | Position | CV | Phase_corr | Phase_orig | Period | Verdict |
|---|---|---|---|---|---|---|
| 8 | operand | 0.488 | 0.632 | 0.308 | 12.51 | 🔶 T=12.5 |
| 8 | final | 0.597 | 0.370 | −0.064 | 91.07 | ❌ Linear Translation |
| 9 | operand | 0.461 | 0.769 | 0.296 | 19.52 | 🔶 T=19.5 |
| 9 | final | 0.653 | 0.491 | −0.271 | 24.14 | ❌ Linear Translation |
| 10 | operand | 0.440 | 0.612 | −0.533 | 14.71 | 🔶 T=14.7 |
| 10 | final | 0.513 | 0.298 | 0.030 | 31.45 | ❌ Linear Translation |
| 11 | operand | 0.363 | 0.655 | −0.294 | 11.12 | 🔶 T=11.1 |
| 11 | final | 0.627 | 0.351 | 0.156 | 121.16 | ❌ Linear Translation |
| 12 | operand | 0.418 | 0.677 | −0.338 | 10.85 | 🔶 T=10.9 |
| 12 | final | 0.527 | 0.274 | −0.107 | 58.64 | ❌ Linear Translation |
| 13 | operand | 0.514 | 0.710 | 0.093 | 64.02 | 🔶 T=64.0 |
| 13 | final | 0.569 | 0.319 | −0.178 | 34.36 | ❌ Linear Translation |
| 14 | operand | 0.446 | 0.612 | −0.423 | 34.86 | 🔶 T=34.9 |
| 14 | final | 0.412 | 0.099 | −0.095 | 112.47 | ❌ Linear Translation |
| 15 | operand | 0.499 | 0.708 | −0.303 | 10.07 | 🔶 T=10.1 |
| 15 | final | 0.438 | 0.089 | −0.085 | 164.97 | ❌ Linear Translation |
| 16 | operand | 0.548 | 0.637 | 0.009 | 298.01 | 🔶 T=298.0 |
| 16 | final | 0.358 | 0.049 | 0.048 | 90.00 | ❌ Linear Translation |
| 17 | operand | 0.539 | 0.650 | 0.013 | 28.06 | 🔶 T=28.1 |
| 17 | final | 0.367 | 0.047 | −0.047 | 381.02 | ❌ Linear Translation |
| 18 | operand | 0.493 | 0.851 | 0.086 | 13.64 | 🔶 T=13.6 |
| 18 | final | 0.522 | 0.206 | 0.074 | 96.24 | ❌ Linear Translation |
| 19 | operand | 0.465 | 0.872 | 0.088 | 11.64 | 🔶 T=11.6 |
| 19 | final | 0.558 | 0.223 | 0.209 | 50.95 | ❌ Linear Translation |
| 20 | operand | 0.450 | 0.761 | 0.148 | 16.03 | 🔶 T=16.0 |
| 20 | final | 0.554 | 0.151 | −0.137 | 178.88 | ❌ Linear Translation |
| **21** | **operand** | **0.392** | **0.885** | **0.215** | **10.00** | **🔶 T=10.0** |
| 21 | final | 0.543 | 0.272 | −0.071 | 196.96 | ❌ Linear Translation |
| 22 | operand | 0.398 | 0.627 | 0.174 | 21.20 | 🔶 T=21.2 |
| 22 | final | 0.576 | 0.177 | −0.136 | 39.53 | ❌ Linear Translation |
| 23 | operand | 0.425 | 0.851 | −0.294 | 11.84 | 🔶 T=11.8 |
| 23 | final | 0.545 | 0.222 | −0.213 | 29.26 | ❌ Linear Translation |
| 24 | operand | 0.581 | 0.839 | −0.268 | 12.74 | 🔶 T=12.7 |
| 24 | final | 0.505 | 0.083 | −0.016 | 38.73 | ❌ Linear Translation |
| 25 | operand | 0.530 | 0.603 | −0.280 | 72.39 | 🔶 T=72.4 |
| 25 | final | 0.508 | 0.196 | 0.016 | 52.71 | ❌ Linear Translation |

**Best layer:** L21 operand — Phase_corr=0.885, CV=0.392, T=10.00

#### Gemma 7B Phase-Corrected Summary

| Analysis | CV | Phase_orig | Phase_corr | Phase Offset | Period T | Verdict |
|---|---|---|---|---|---|---|
| Direct PCA operand L14 | 0.489 | −0.298 | 0.627 | 118.5° | 11.3 | 🔷 Monotone |
| Direct PCA final L14 | 0.587 | 0.099 | 0.100 | 351.5° | 17.2 | 🔷 Monotone |
| Best layer sweep (L21 op) | 0.392 | 0.215 | **0.885** | — | **10.0** | 🔶 T=10.0 |

**Overall Verdict: 🔶 Near-miss. L21 operand achieves Phase_corr=0.885 (above 0.85 threshold) and T=10.00, but CV=0.392 is too high (target < 0.20). The mod-10 clock pattern exists but the circle is too noisy to confirm a clean clock face. The original pipeline reported Phase=0.526 at this same layer — the bug was hiding 68% of the true alignment signal.**

---

### Results: Phi-3 Mini (`microsoft/Phi-3-mini-4k-instruct`) — Phase-Corrected, Layer 24, Head 28

#### Direct PCA — Operand Position, Layer 24

| Metric | Value | Target |
|---|---|---|
| Best PC pair | (4, 5) | — |
| Radius CV | 0.4474 | < 0.20 |
| Lin (raw n) | 0.9792 | — |
| Phase ORIGINAL | 0.1429 | (buggy — reference) |
| **Phase CORRECTED** | **0.7648** | > 0.85 |
| Phase offset | 281.0° | — |
| Period T | 14.36 | ~10.0 |
| **Verdict** | 🔷 MONOTONE HELIX (Vector Translation) | |

#### Direct PCA — Final Token Position, Layer 24

| Metric | Value | Target |
|---|---|---|
| Best PC pair | (6, 11) | — |
| Radius CV | 0.5679 | < 0.20 |
| Lin (raw n) | 0.9372 | — |
| Phase ORIGINAL | −0.0484 | (buggy — reference) |
| **Phase CORRECTED** | **0.1239** | > 0.85 |
| Phase offset | 247.0° | — |
| Period T | 25.09 | ~10.0 |
| **Verdict** | 🔷 MONOTONE HELIX (Vector Translation) | |

#### Layer Sweep (Layers 8–25, Corrected Phase Metric)

| Layer | Position | CV | Phase_corr | Phase_orig | Period | Verdict |
|---|---|---|---|---|---|---|
| 8 | operand | 0.477 | 0.467 | −0.222 | 81.17 | ❌ Linear Translation |
| 8 | final | 0.461 | 0.028 | −0.028 | 211.42 | ❌ Linear Translation |
| 9 | operand | 0.392 | 0.589 | −0.107 | 17.21 | ❌ Linear Translation |
| 9 | final | 0.583 | 0.196 | 0.082 | 68.86 | ❌ Linear Translation |
| 10 | operand | 0.419 | 0.591 | −0.569 | 11.97 | ❌ Linear Translation |
| 10 | final | 0.569 | 0.113 | 0.102 | 90.03 | ❌ Linear Translation |
| 11 | operand | 0.487 | 0.691 | −0.300 | 23.46 | 🔶 T=23.5 |
| 11 | final | 0.581 | 0.172 | 0.001 | 32.96 | ❌ Linear Translation |
| 12 | operand | 0.589 | 0.527 | −0.198 | 39.73 | ❌ Linear Translation |
| 12 | final | 0.581 | 0.270 | 0.256 | 59.50 | ❌ Linear Translation |
| 13 | operand | 0.545 | 0.808 | −0.140 | 11.75 | 🔶 T=11.7 |
| 13 | final | 0.499 | 0.126 | 0.010 | 187.84 | ❌ Linear Translation |
| 14 | operand | 0.519 | 0.724 | −0.058 | 16.76 | 🔶 T=16.8 |
| 14 | final | 0.462 | 0.143 | 0.078 | 913.71 | ❌ Linear Translation |
| 15 | operand | 0.449 | 0.557 | 0.428 | 24.96 | ❌ Linear Translation |
| 15 | final | 0.550 | 0.236 | 0.167 | 75.15 | ❌ Linear Translation |
| 16 | operand | 0.551 | 0.707 | −0.111 | 18.61 | 🔶 T=18.6 |
| 16 | final | 0.496 | 0.135 | 0.119 | 195.96 | ❌ Linear Translation |
| 17 | operand | 0.397 | 0.340 | 0.103 | 19.17 | ❌ Linear Translation |
| 17 | final | 0.537 | 0.193 | 0.171 | 35.34 | ❌ Linear Translation |
| 18 | operand | 0.415 | 0.608 | 0.043 | 10.99 | 🔶 T=11.0 |
| 18 | final | 0.518 | 0.152 | −0.049 | 95.35 | ❌ Linear Translation |
| 19 | operand | 0.393 | 0.527 | −0.520 | 16.07 | ❌ Linear Translation |
| 19 | final | 0.457 | 0.178 | 0.115 | 85.26 | ❌ Linear Translation |
| 20 | operand | 0.396 | 0.542 | −0.537 | 15.88 | ❌ Linear Translation |
| 20 | final | 0.446 | 0.100 | 0.061 | 94.90 | ❌ Linear Translation |
| 21 | operand | 0.478 | 0.575 | −0.133 | 217.71 | ❌ Linear Translation |
| 21 | final | 0.463 | 0.133 | −0.133 | 57.66 | ❌ Linear Translation |
| 22 | operand | 0.447 | 0.689 | −0.222 | 11.73 | 🔶 T=11.7 |
| 22 | final | 0.419 | 0.152 | −0.008 | 90.66 | ❌ Linear Translation |
| 23 | operand | 0.442 | 0.798 | 0.183 | 12.17 | 🔶 T=12.2 |
| 23 | final | 0.411 | 0.116 | −0.104 | 91.09 | ❌ Linear Translation |
| **24** | **operand** | **0.447** | **0.765** | **0.142** | **14.36** | **🔶 T=14.4** |
| 24 | final | 0.408 | 0.156 | −0.130 | 115.06 | ❌ Linear Translation |
| **25** | **operand** | **0.432** | **0.822** | **0.221** | **11.74** | **🔶 T=11.7** |
| 25 | final | 0.395 | 0.141 | −0.106 | 42.80 | ❌ Linear Translation |

**Best layer:** L25 operand — Phase_corr=0.822, CV=0.432, T=11.74

#### Phi-3 Mini Phase-Corrected Summary

| Analysis | CV | Phase_orig | Phase_corr | Phase Offset | Period T | Verdict |
|---|---|---|---|---|---|---|
| Direct PCA operand L24 | 0.447 | 0.143 | 0.765 | 281.0° | 14.4 | 🔷 Monotone |
| Direct PCA final L24 | 0.568 | −0.048 | 0.124 | 247.0° | 25.1 | 🔷 Monotone |
| Best layer sweep (L25 op) | 0.432 | 0.221 | **0.822** | — | **11.7** | 🔶 T=11.7 |

**Overall Verdict: 🔶 The phase correction reveals Phi-3 was closer than the original metric showed (L25 operand: Phase_corr=0.822, just below the 0.85 threshold), but T≈11.7 ≠ 10 and CV=0.432 remains too high. Note: the best layer shifted from L24 to L25 under the corrected metric. The original pipeline reported Phase=0.652 at L24 — the bug was hiding ~17% of the alignment signal.**

---

### Results: Llama 3.2 3B (`meta-llama/Llama-3.2-3B`) — Phase-Corrected, Layer 1, Head 17

#### Direct PCA — Operand Position, Layer 1

| Metric | Value | Target |
|---|---|---|
| Best PC pair | (4, 9) | — |
| Radius CV | 0.5603 | < 0.20 |
| Lin (raw n) | 0.9868 | — |
| Phase ORIGINAL | 0.1566 | (buggy — reference) |
| **Phase CORRECTED** | **0.1922** | > 0.85 |
| Phase offset | 324.5° | — |
| Period T | 10.68 | ~10.0 |
| **Verdict** | 🔷 MONOTONE HELIX (Vector Translation) | |

#### Direct PCA — Final Token Position, Layer 1

| Metric | Value | Target |
|---|---|---|
| Best PC pair | (8, 10) | — |
| Radius CV | 0.6306 | < 0.20 |
| Lin (raw n) | 0.8590 | — |
| Phase ORIGINAL | −0.0087 | (buggy — reference) |
| **Phase CORRECTED** | **0.1275** | > 0.85 |
| Phase offset | 94.0° | — |
| Period T | 26.71 | ~10.0 |
| **Verdict** | ❌ NO CLEAN HELIX | |

#### Layer Sweep (Layers 8–25, Corrected Phase Metric)

| Layer | Position | CV | Phase_corr | Phase_orig | Period | Verdict |
|---|---|---|---|---|---|---|
| 8 | operand | 0.377 | 0.179 | 0.177 | 180.26 | ❌ Linear Translation |
| 8 | final | 0.652 | 0.400 | 0.354 | 1396.56 | ❌ Linear Translation |
| 9 | operand | 0.411 | 0.159 | 0.128 | 201.68 | ❌ Linear Translation |
| 9 | final | 0.489 | 0.146 | −0.036 | 97.78 | ❌ Linear Translation |
| 10 | operand | 0.397 | 0.141 | −0.048 | 169.48 | ❌ Linear Translation |
| 10 | final | 0.685 | 0.373 | 0.373 | 48.48 | ❌ Linear Translation |
| 11 | operand | 0.504 | 0.331 | −0.331 | 105.64 | ❌ Linear Translation |
| 11 | final | 0.662 | 0.344 | 0.343 | 36.53 | ❌ Linear Translation |
| 12 | operand | 0.441 | 0.567 | 0.374 | 26.60 | ❌ Linear Translation |
| 12 | final | 0.535 | 0.170 | −0.015 | 224.19 | ❌ Linear Translation |
| 13 | operand | 0.352 | 0.522 | 0.105 | 150.01 | ❌ Linear Translation |
| 13 | final | 0.523 | 0.186 | −0.045 | 220.80 | ❌ Linear Translation |
| 14 | operand | 0.498 | 0.453 | −0.082 | 160.75 | ❌ Linear Translation |
| 14 | final | 0.550 | 0.153 | −0.118 | 115.80 | ❌ Linear Translation |
| 15 | operand | 0.448 | 0.438 | −0.188 | 43.90 | ❌ Linear Translation |
| 15 | final | 0.616 | 0.198 | −0.017 | 18.98 | ❌ Linear Translation |
| 16 | operand | 0.452 | 0.524 | −0.195 | 14.00 | ❌ Linear Translation |
| 16 | final | 0.637 | 0.215 | 0.168 | 85.56 | ❌ Linear Translation |
| 17 | operand | 0.482 | 0.476 | −0.237 | 24.24 | ❌ Linear Translation |
| 17 | final | 0.568 | 0.160 | 0.016 | 212.16 | ❌ Linear Translation |
| 18 | operand | 0.557 | 0.518 | 0.239 | 51.88 | ❌ Linear Translation |
| 18 | final | 0.619 | 0.275 | −0.215 | 35.03 | ❌ Linear Translation |
| 19 | operand | 0.527 | 0.421 | 0.125 | 61.87 | ❌ Linear Translation |
| 19 | final | 0.584 | 0.253 | −0.215 | 28.80 | ❌ Linear Translation |
| 20 | operand | 0.463 | 0.325 | −0.190 | 11642.15 | ❌ Linear Translation |
| 20 | final | 0.656 | 0.253 | −0.151 | 27.92 | ❌ Linear Translation |
| 21 | operand | 0.405 | 0.244 | −0.085 | 71.85 | ❌ Linear Translation |
| 21 | final | 0.623 | 0.206 | 0.042 | 25.44 | ❌ Linear Translation |
| 22 | operand | 0.380 | 0.208 | −0.062 | 1122.08 | ❌ Linear Translation |
| 22 | final | 0.624 | 0.210 | 0.046 | 25.44 | ❌ Linear Translation |
| 23 | operand | 0.581 | 0.348 | −0.342 | 197.43 | ❌ Linear Translation |
| 23 | final | 0.534 | 0.166 | −0.160 | 447.24 | ❌ Linear Translation |
| 24 | operand | 0.584 | 0.303 | −0.075 | 100.69 | ❌ Linear Translation |
| 24 | final | 0.558 | 0.141 | −0.085 | 266.70 | ❌ Linear Translation |
| 25 | operand | 0.515 | 0.321 | −0.094 | 114.82 | ❌ Linear Translation |
| 25 | final | 0.557 | 0.154 | −0.105 | 57.18 | ❌ Linear Translation |

**Best layer:** L13 operand — Phase_corr=0.522, CV=0.352, T=150.01

#### Llama 3.2 3B Phase-Corrected Summary

| Analysis | CV | Phase_orig | Phase_corr | Phase Offset | Period T | Verdict |
|---|---|---|---|---|---|---|
| Direct PCA operand L1 | 0.560 | 0.157 | 0.192 | 324.5° | 10.7 | 🔷 Monotone |
| Direct PCA final L1 | 0.631 | −0.009 | 0.128 | 94.0° | 26.7 | ❌ No helix |
| Best layer sweep (L13 op) | 0.352 | 0.105 | **0.522** | — | **150.0** | ❌ Linear Translation |

**Overall Verdict: ❌ No modular helix found at any layer. All phase-corrected scores remain below 0.60. Even the best layer (L13) has a wildly wrong period (T=150). Llama 3.2 3B appears to use a pure Vector Translation mechanism for arithmetic — it encodes numbers as monotonically-ordered vectors (Lin=0.987) but without any modular/circular structure.**

---

### Phase-Corrected Pipeline — Cross-Model Comparison

| Model | Best Layer/Pos | Phase_orig | Phase_corr | CV | Period T | Verdict |
|---|---|---|---|---|---|---|
| **Gemma 7B** | L21 operand | 0.215 | **0.885** | 0.392 | **10.00** | 🔶 Near-miss (CV too high) |
| **Phi-3 Mini** | L25 operand | 0.221 | **0.822** | 0.432 | 11.74 | 🔶 Close (T≠10, CV too high) |
| **Llama 3.2 3B** | L13 operand | 0.105 | 0.522 | 0.352 | 150.01 | ❌ No modular structure |

### Phase-Corrected Pipeline — Key Findings

- **The phase bug was real and significant.** Original phase scores were suppressed by 17–68% due to arbitrary PCA axis rotation. After correction, Gemma 7B's best score jumps from 0.526 to **0.885** — crossing the 0.85 threshold for phase match.
- **Gemma 7B L21 operand is the closest to a confirmed T=10 clock:** Phase_corr=0.885, T=10.00 (exact!), but CV=0.392 (target < 0.20). The mod-10 angular pattern is present but embedded in a noisy, non-circular distribution.
- **Phi-3 Mini's best layer shifted from L24 to L25** under the corrected metric, with Phase_corr=0.822 (just below threshold) and T=11.7.
- **Llama 3.2 3B shows no modular structure at all** — max Phase_corr=0.522, periods are wildly wrong (T=150). It uses a pure **Vector Translation** mechanism: numbers are encoded as monotonically-ordered vectors (Lin(raw n) up to 0.987) with no circular/modular component.
- **All final-token positions remain weak** — the aggregated representation at the last token consistently destroys any angular structure. Modular information, where it exists, lives at the operand position.
- **Two arithmetic mechanisms confirmed:**
  - **🔷 Vector Translation** (all models, dominant): Numbers encoded as linearly-ordered vectors. High Lin(raw n), low Phase_corr.
  - **🔶 Partial Modular Clock** (Gemma 7B, Phi-3 only): Weak mod-10 angular pattern at specific layers, but too noisy (CV > 0.35) to qualify as a clean clock face.
- **No model achieves ✅ CLEAN T=10 CLOCK FACE** — the threshold requires CV < 0.20, Phase_corr > 0.85, and 9 < T < 11 simultaneously. Gemma 7B comes closest (2 of 3 criteria met).

---

## Experiment: Final Pipeline — Fourier Isolation + Causal Phase-Shift

### Objective

The phase-corrected pipeline showed that high CV (> 0.35) is the remaining barrier to confirming a clean clock face. The **Final Pipeline** (`run_final_pipeline`) tests two hypotheses:

1. **Superposition Hypothesis:** The high CV is caused by other signals (e.g., tens-digit encoding, positional info) being superimposed on the same activation dimensions. If we **Fourier-isolate** only the T-periodic component, the circle should become clean.
2. **Causal Hypothesis:** If the helix truly controls the ones digit of the arithmetic output, **rotating** the activation in the SVD reading plane by δ/T should shift the model's predicted ones digit by δ.

### Line-by-Line Analysis of `run_final_pipeline`

The pipeline has three steps:

#### Step 1: `fourier_isolate_TN` — Fourier Isolation

Strips the residual stream down to *only* the T-periodic component:

1. Constructs a 3-column regression basis matrix: `[cos(2πn/T), sin(2πn/T), 1]` for each n in valid_ns
2. Solves the least-squares regression `A = lstsq(basis, activations)` — finds the best linear combination of cos, sin, and constant that approximates each activation dimension independently
3. Reconstructs **only** the periodic part: `acts_T = basis[:, :2] @ A[:2, :]` — drops the constant (DC offset) term, keeping only the cos/sin projection
4. Computes **R²** — what fraction of total activation variance is captured by the T-periodic component alone

**Purpose:** If a mod-T clock exists but is buried under superposition noise from other circuits, this extracts it. The isolated matrix is rank-2 by construction (it lives in a 2D cos/sin subspace).

#### Step 2: `check_after_isolation` — Circle Test on Isolated Component

Runs the same PCA + phase-corrected circle analysis on the Fourier-isolated activations:

1. Mean-centers the isolated activations and runs PCA (capped at 5 components; the matrix is rank-2, so only 2 will have non-trivial variance)
2. Tests all PCA pairs for circular geometry: CV, phase-corrected match (360 offsets), and detected period
3. **Verdict logic:**
   - CV < 0.20 AND Phase_corr > 0.85 → ✅ "CLEAN CLOCK FACE after isolation. Superposition resolved."
   - CV improved significantly → ⚠️ "Superposition partially explains it."
   - CV unchanged → ❌ "High CV is intrinsic, not a superposition artifact."

#### Step 3: `causal_phase_shift_test` — Causal Intervention

Tests whether the SVD reading plane **causally controls** the ones digit of the model's arithmetic output:

1. **Extracts the reading plane:** Computes `W_OV = W_V @ W_O` for the target head, takes SVD, and extracts the top-2 left singular vectors (U columns) as v1, v2 — the head's "reading directions"
2. **Generates test cases:** 20 no-carry and 20 carry addition problems (e.g., `45+12`, `5+15`), each with a rotation delta δ ∈ {1, 2}
3. **For each test case:**
   - Runs the prompt `"Math:\n10 + 10 = 20\n21 + 13 = 34\n{a} + {b} ="` through the model (few-shot format to constrain output)
   - Captures the residual stream `h_orig` at the operand position (token before `+`)
   - **Rotates** `h_orig` by angle `θ = 2π·δ/T` in the (v1, v2) plane:
     ```
     h_rot = h_orig − c1·v1 − c2·v2 + (cos(θ)·c1 − sin(θ)·c2)·v1 + (sin(θ)·c1 + cos(θ)·c2)·v2
     ```
     where c1 = h_orig·v1 and c2 = h_orig·v2
   - Hooks the rotated activation back into the model via `model.hooks()` context manager and generates 4 tokens
   - Extracts the first number from the output and checks if its ones digit equals `(a + b + δ) % 10`
4. **Reports success rate** separately for no-carry and carry cases

**Purpose:** This is the **gold standard** test. If rotating by δ/T in the SVD plane shifts the ones digit by δ, it proves the helix is not just a geometric curiosity — it's a **causal mechanism** the model uses for computation. 0% success means the plane is geometrically structured but not causally linked to the output.

---

### Results: Gemma 7B (`google/gemma-7b`) — Final Pipeline, L21 H2, T=10.0

#### Step 1: Fourier Isolation

| Metric | Value |
|---|---|
| Target period | T = 10.0 |
| R² (variance explained) | **9.2%** |

> The T=10 component explains only 9.2% of the total activation variance at L21. The remaining ~91% is other signals (positional encoding, semantic content, tens-digit info, etc.).

#### Step 2: Circle Test After Isolation

| Metric | Value | Target |
|---|---|---|
| CV after isolation | 0.4432 | < 0.20 |
| Phase_corr after isolation | 0.6472 | > 0.85 |
| Period T (detected) | 1333.21 | ~10.0 |
| **Verdict** | ❌ CV unchanged. High CV is intrinsic. | |

> **Fourier isolation did NOT help Gemma 7B.** CV stayed at 0.443 (was 0.392 before isolation) and phase actually dropped from 0.885 to 0.647. The detected period exploded to T=1333 — the isolated component does not form a clean circle. The high CV is **not** caused by superposition; the T=10 signal itself is geometrically impure.

#### Step 3: Causal Phase-Shift

| Case Type | Success Rate | Examples |
|---|---|---|
| **No-carry** | **0/20 (0.0%)** | 45+12 rot+1: pred=7, exp=8 ✗; 22+20 rot+1: pred=2, exp=3 ✗ |
| **Carry** | **0/20 (0.0%)** | 5+15 rot+2: pred=0, exp=2 ✗; 18+26 rot+1: pred=4, exp=5 ✗ |

> **Complete causal failure.** Rotating the activation in the SVD reading plane has zero effect on the ones digit. The model outputs the **original correct answer** every time (e.g., 45+12=57 regardless of rotation). The SVD plane at L21 H2 is not causally responsible for computing the ones digit.

#### Gemma 7B Final Summary

| Metric | Value | Interpretation |
|---|---|---|
| T=10 variance explained | 9.2% | Weak — T=10 is a minor component |
| CV after isolation | 0.443 | ❌ High CV is intrinsic |
| Phase after isolation | 0.647 | Below 0.85 threshold |
| Causal no-carry | 0.0% | ❌ Not causally linked |
| Causal carry | 0.0% | ❌ Not causally linked |

**Overall Verdict: ❌ The T=10 signal at L21 is a geometric artifact, not a causal mechanism. Fourier isolation cannot rescue the circle (CV stays high), and rotating in the SVD plane has zero causal effect on the model's output.**

---

### Results: Phi-3 Mini (`microsoft/Phi-3-mini-4k-instruct`) — Final Pipeline, L25 H28, T=11.74

#### Step 1: Fourier Isolation

| Metric | Value |
|---|---|
| Target period | T = 11.74 |
| R² (variance explained) | **1.5%** |

> The T=11.74 component explains only 1.5% of total activation variance at L25 — even less than Gemma 7B's 9.2%. This is an extremely faint signal.

#### Step 2: Circle Test After Isolation

| Metric | Value | Target |
|---|---|---|
| CV after isolation | **0.1022** | < 0.20 ✅ |
| Phase_corr after isolation | **0.9403** | > 0.85 ✅ |
| Period T (detected) | 11.74 | ~11.74 ✅ |
| **Verdict** | ✅ **CLEAN CLOCK FACE after isolation. Superposition resolved.** | |

> **Breakthrough result for Phi-3 Mini.** After Fourier isolation, the circle becomes near-perfect: CV drops from 0.432 to **0.102** and Phase jumps to **0.940**. This confirms that Phi-3 Mini *does* have a clean T≈11.74 modular clock embedded in its residual stream at L25 — it was just buried under 98.5% superposition noise from other circuits. The high CV in previous experiments was entirely a superposition artifact.

#### Step 3: Causal Phase-Shift

| Case Type | Success Rate | Examples |
|---|---|---|
| **No-carry** | **0/20 (0.0%)** | 45+12 rot+1: pred=7, exp=8 ✗; 22+20 rot+1: pred=2, exp=3 ✗ |
| **Carry** | **0/20 (0.0%)** | 5+15 rot+2: pred=0, exp=2 ✗; 18+26 rot+1: pred=4, exp=5 ✗ |

> **Despite the clean clock face, causal intervention fails completely.** The model again outputs the original correct answer every time, ignoring the rotation. The T≈11.74 clock exists geometrically but is not on the causal path for ones-digit computation. The SVD reading plane (U[:, 0:2] of W_OV at L25 H28) is not the circuit the model actually uses to produce its answer.

#### Phi-3 Mini Final Summary

| Metric | Value | Interpretation |
|---|---|---|
| T=11.74 variance explained | 1.5% | Very faint signal |
| CV after isolation | **0.102** | ✅ Clean circle after isolation |
| Phase after isolation | **0.940** | ✅ Strong mod-T alignment |
| Causal no-carry | 0.0% | ❌ Not causally linked |
| Causal carry | 0.0% | ❌ Not causally linked |

**Overall Verdict: 🔶 A clean T≈11.74 clock face EXISTS in Phi-3 Mini (confirmed by Fourier isolation — CV=0.102, Phase=0.940), but it is NOT causally responsible for the model's arithmetic output. The clock is a geometric structure in the weight space that produces a clean circular encoding of numbers, but the model does not read from this circle to compute its answer. The causal arithmetic circuit lies elsewhere.**

---

### Final Pipeline — Cross-Model Comparison

| Model | Layer | Period | R² | CV (isolated) | Phase (isolated) | Circle? | Causal? |
|---|---|---|---|---|---|---|---|
| **Gemma 7B** | L21 | T=10.0 | 9.2% | 0.443 | 0.647 | ❌ No | ❌ 0% |
| **Phi-3 Mini** | L25 | T=11.74 | 1.5% | **0.102** | **0.940** | ✅ Yes | ❌ 0% |

### Final Pipeline — Key Findings

- **Phi-3 Mini has a confirmed clean clock face** — CV=0.102, Phase=0.940, T=11.74 after Fourier isolation. This is the first ✅ CLEAN result in the entire study. The high CV in all previous experiments was caused by **superposition** with other signals (98.5% of variance). Once isolated, the T≈11.74 component forms a near-perfect circle.
- **Gemma 7B's clock face is intrinsically noisy** — Fourier isolation does not help (CV stays 0.443). The T=10 signal at L21 explains 9.2% of variance but does not form a clean circle even in isolation. The noise is not from superposition; it's intrinsic to how Gemma encodes the periodic component.
- **Neither clock is causally linked to the output.** Both models achieve **0% success** on causal phase-shift tests. Rotating the activation in the SVD reading plane does not shift the ones digit. The models output the correct original answer every time, meaning the arithmetic computation flows through a different circuit.
- **The SVD helix is a representational structure, not a computational one.** The OV matrix geometry creates a structured encoding of numbers (confirmed across 8 models), but the model does not *read from* this particular 2D plane to produce its arithmetic answer. The causal arithmetic circuit likely involves:
  - Higher-dimensional subspaces (not just 2D)
  - MLP layers downstream of the attention head
  - Multiple heads collaborating rather than a single head's OV plane
- **Superposition is the key distinction:** Phi-3 Mini's clock is real but hidden (1.5% of variance); Gemma 7B's "clock" is noisy even in isolation. Architecture matters — Phi-3's instruction-tuned design produces a cleaner geometric encoding.
- **Period T≈11.74 ≠ 10** — Even Phi-3's confirmed clock uses T≈11.74, not T=10. This suggests the model may be encoding a slightly different modular structure than pure mod-10 arithmetic, or the period is stretched by the continuous nature of the residual stream.
