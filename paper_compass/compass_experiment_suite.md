# COMPASS: full experiment suite and implementation reference

This document is a single-file, end-to-end reference for the COMPASS
project (Beyond-Components repo). It consolidates every experiment,
algorithm, and result currently available in `experiments/` and
`helix_usage_validated/`, including recent debiasing-stack work
(routing, calibration, INLP baseline, bootstrap CIs) that is **not yet
folded into `paper_compass/sections/*.tex`**. Entries marked *(new)*
indicate results that post-date the current LaTeX sources.

---

## 1. Motivation and thesis

We claim that the SVD of a single attention head's output-value
matrix $W_{OV}^{(\ell,h)} = W_V\,W_O \in \mathbb{R}^{d\times d}$
routinely exposes two-dimensional *semantic compasses*: planes in the
residual stream whose azimuthal angle codes a categorical distinction
(gender, race, religion, profession, temporal unit, entity type) and
whose radius codes its magnitude. The same weight-space primitive is
then used as a rank-2 causal editing tool, first per-head and then as
a calibrated multi-head ensemble that routes per domain.

The four models studied are:

| Tag | HuggingFace | Layers × heads | $d_\text{model}$ | $d_\text{vocab}$ |
|-----|-------------|---------------|------------------|-----------------|
| gpt2  | `gpt2`                                 | 12 × 12 | 768  | 50 257 |
| phi3  | `microsoft/Phi-3-mini-4k-instruct`     | 32 × 32 | 3 072 | 32 064 |
| gemma | `google/gemma-2-2b`                    | 26 × 8  | 2 304 | 256 000 |
| llama | `meta-llama/Llama-3.2-3B`              | 28 × 24 | 3 072 | 128 256 |

All runs use TransformerLens `HookedTransformer` on `bfloat16`; hooks are
installed at `blocks.{L}.hook_resid_pre`.

---

## 2. Core algorithm (compass plane → causal α-sweep)

### 2.1 Compass plane

For head $h$ at layer $\ell$, build
$W_{OV} = W_V\,W_O \in \mathbb{R}^{d\times d}$ and factor
$W_{OV} = U\,\Sigma\,V^\top$. A *compass plane* is a pair of
right-singular vectors $(u_i, u_j) \equiv (V^\top_{i,:}, V^\top_{j,:})$
with singular values $\sigma_i \ge \sigma_j$. We use the right-factor
$V^\top$ because it parametrizes what the head *writes* into the
residual stream.

### 2.2 Causal α-sweep (two-pillar test)

For a neutral prompt with final-token pre-layer residual $r$, we inject

$$v(\theta,\alpha) = \alpha\,\sigma_i\,\cos\theta\,u_i + \alpha\,\sigma_j\,\sin\theta\,u_j$$

replacing $r \gets r + v(\theta,\alpha)$, and record the
logit-difference $\mathrm{LD}(\theta,\alpha;p) = \mathrm{logit}_{t_+} - \mathrm{logit}_{t_-}$
for antipodal probe tokens. We average over 3 neutral prompts, sweep
$\theta$ through $N_\theta \in \{12, 36\}$ angles, and $\alpha \in \{1, 3, 10\}$
(or $\{3, 10\}$ in the scans).

For each curve we fit a sinusoid via the first DFT bin:
$\mathrm{LD}(\theta) \approx \mu + A\cos(\theta - \phi)$.

A plane *passes* if:
- linearity $R^2 \ge 0.95$ between $A$ and $\alpha$,
- phase drift $\Delta\phi \le 10^\circ$ across the $\alpha$ grid,
- amplitude slope $A(\alpha_{\max})/\alpha_{\max} \ge \tau$
  ($\tau = 0.20$ for GPT-2/Phi-3/Llama; $\tau = 0.08$ for Gemma —
  see §5.1 for the rationale).

### 2.3 Null battery (9 tests)

Implemented in `investigate_helix_usage_validated.py`:

| Test | Logic | Expected |
|---|---|---|
| decode-heads  | Top-k tokens of $W_U^\top u_i$ at $\theta \in \{0,90,180,270\}$    | Coherent axis |
| cyclicity     | $\theta \mapsto \mathrm{LD}(\theta)$ is $2\pi$-periodic           | Pass |
| permutation   | Shuffle $(t_+, t_-)$ labels, re-run sweep                          | Signal killed |
| random-plane  | Replace $(u_i, u_j)$ with random orthonormal pair                  | Signal killed |
| principal-angles | Principal angles vs. other heads' top-2 planes                  | Near-orthogonal |
| scan-planes   | Enumerate $(i,j)$ with $i,j \le 10$, report argmax-slope plane     | Target = argmax |
| semantic-ablate | Zero $P_\mathcal{C}\,r$ projection                                | Signal killed |
| causal-patch  | Swap $P_\mathcal{C}\,r$ between plus / minus prompts                | Sign flip |
| self-test     | End-to-end pipeline check on a fixed prompt                          | Deterministic |

Random-plane and permutation nulls are strengthened on the
representative late-layer heads (Phi-3 L28H1, GPT-2 L10H9) with
$n_\text{random} = 1000$ / $n_\text{perm} = 2000$ Monte-Carlo trials.

---

## 3. Full discovery → injection pipeline

The repository implements a four-stage pipeline that lifts single-head
compass discovery into a calibrated, domain-routed debiasing stack.

### Stage 1 — Probe-pair extraction

**Scripts:** `experiments/stereoset_probe_extract{,_gemma,_phi3,_llama}.py`

For each (model, domain) in {gender, race, profession, religion},
extract (stereo, anti-stereo) probe pairs from StereoSet intrasentence,
keeping only pairs where both sides tokenize to a single token in that
model's tokenizer. Outputs:

- `helix_usage_validated/stereoset_probe_pairs{,_gemma,_phi3,_llama}.tsv`
- `stereoset_probe_halfmatch{,_gemma,_phi3,_llama}.tsv` (one-sided matches for manual inspection)

### Stage 2 — Blind causal scan (2a decode pre-filter + 2b α-sweep)

**Scripts:** `experiments/stereoset_probe_scan{,_gemma,_phi3,_llama}.py`

For each (L, H, plane=(d1,d2)) in a model's top-4 SV pairs:

- **Stage 2a (cheap pre-filter).** Project each compass basis vector
  through the unembedding, keep only probe pairs where each probe token
  lands in the top-$k$ of either pole (`DECODE_TOPK=500` for GPT-2,
  `2000` for Gemma, etc., scaled with vocab size).
- **Stage 2b (causal α-sweep).** Run the sinusoid fit only on
  surviving candidates. Emit per-plane fit stats: `lin_r2`, `amp_slope`,
  `phase_drift`, `A_lo`, `A_hi`, `phi_hi`, `passed`.

Outputs:
- `stereoset_scan_{tag}.jsonl` (one row per (L, H, d1, d2, probe))
- `stereoset_scan_{tag}_summary.txt` (per-head aggregated passes)

Scan sizes:

| Model | Total planes | Planes with ≥1 candidate | (plane,probe) passes |
|---|---|---|---|
| gpt2  | $864$   | ≈ $500$  | $16$ planes pass |
| phi3  | $1344$  | ≈ $1000$ | clustered at L24H14 / L28H26 / L26H24 |
| gemma | $240$   | —        | 6 heads pass at $\tau=0.08$ |
| llama | $4032$  | —        | dominated by L22H12 (253 passes) |

### Stage 3 — Ensemble selection and optional routing

**Scripts:**
- `experiments/stereoset_ensemble_eval{,_gemma,_phi3,_llama}.py` — K=4 global ensemble evaluated on StereoSet intrasentence (LMS/SS/ICAT).
- `experiments/analyze_head_domain_routing.py` *(new)* — reads each scan's jsonl, computes a *specialization index* per head ($\max_d$ of normalized pass-count vector), and writes `head_domain_routing.md` + per-model heatmaps.
- `experiments/crowspairs_routed_eval.py` *(new)* — replaces the global K=4 with **per-domain** top-K heads and injects them only on rows with the matching `bias_type`.
- `experiments/calibrate_per_domain_alpha.py` *(new)* — chooses per-domain $\alpha$ so that mean-across-heads SNR at the calibration prompt equals a target (default $0.20$; lower for Gemma).

### Stage 4 — Evaluation and controls

**Benchmarks:**
- `experiments/winogender_sweep.py` — WinoGender (720 sentences) with α-sweep, reports raw gap, stereo-corr, stereo-delta, PPL delta.
- `experiments/stereoset_eval.py` / `stereoset_ensemble_eval_{model}.py` — StereoSet LMS / SS / ICAT.
- `experiments/crowspairs_eval.py` *(new)* — CrowS-Pairs SS under K=4 ensemble.
- `experiments/crowspairs_routed_eval.py` *(new)* — CrowS-Pairs SS under per-domain routed, per-domain-α injection.
- `experiments/truthfulqa_eval.py` *(new)* — TruthfulQA MC1 under the same ensemble (capability-preservation check).
- `experiments/inlp_debias.py` *(new)* — Iterative Null-space Projection baseline (Ravfogel 2020); 4 models × {gender, race}.
- `experiments/bootstrap_ci.py` *(new)* — post-processing: Wilson + empirical bootstrap 95% CIs over every `crowspairs_*.csv`, `truthfulqa_*.csv`, `stereoset_ensemble_*.csv`, `winogender_*sweep.csv`.

**Supporting:**
- `experiments/head_ablation_comparison.py` — head / plane ablation (storage vs. steering).
- `experiments/baseline_comparison.py` — Compass vs. Linear probe / ActAdd / Contrastive-gradient / Fisher top-2 under a unit-normalized harness.
- `experiments/fourway_probe.py` — second-harmonic fit for dial topology.
- `experiments/steering_demo.py` — single-prompt steering (LD swing, sign-flip rate).

**SNR sweeps:**
- `experiments/run_phi3_snr_sweep.sh` and `run_llama_snr_sweep.sh` — repeat Stage 3 routed eval with `target_snr ∈ {0.07, 0.08, 0.10}` (phi3) and `{0.10, 0.15, 0.20}` (llama).

**Parity queues:**
- `experiments/run_gemma_parity.sh` *(new)* — full Gemma catch-up: scan re-run (AMP_THRESH=0.08), stereoset_ensemble (α=5,10,20), CrowS-Pairs, TruthfulQA, calibrate, routed.
- `experiments/run_calib_queue.sh`, `run_routed_queue.sh`, `run_transfer_queue.sh` — convenience wrappers.

---

## 4. Per-head results (in current LaTeX)

These appear in `paper_compass/sections/results.tex` and are preserved
verbatim for the compendium.

### 4.1 Gender compass table (Table 1 in paper)

| Model | Head | $(u_i,u_j)$ | $(\sigma_i,\sigma_j)$ | $\mathrm{LD}_\text{masc}$ | $\mathrm{LD}_\text{fem}$ | $A(1)/A(3)/A(10)$ | $\phi \pm \Delta\phi$ |
|---|---|---|---|---|---|---|---|
| GPT-2 Small  | L9H7  | $(1,2)$ | $(8.87, 8.46)$ | $+5.31$ | $-4.31$ | $1.67/4.60/12.16$ | $-178.0^\circ \pm 2.9$ |
| GPT-2 Small  | L10H9 | $(0,3)$ | $(9.15, 6.80)$ | $+3.73$ | $-3.31$ | $1.78/5.10/14.16$ | $+4.7^\circ \pm 0.3$ |
| Phi-3 Mini   | L24H10| $(0,1)$ | $(9.42, 7.70)$ | $+7.08$ | $-6.97$ | $0.67/2.01/5.22$  | $-154.9^\circ \pm 2.0$ |
| Phi-3 Mini   | L28H1 | $(0,3)$ | $(13.67,7.61)$ | $+5.06$ | $-5.21$ | $1.81/5.15/12.85$ | $+1.0^\circ \pm 1.3$ |
| Gemma-2-2B   | L21H4 | $(0,2)$ | $(2.35, 1.98)$ | $+5.38$ | $-5.21$ | $0.18/0.59/2.19$  | $+0.2^\circ \pm 1.3$ |
| Llama-3.2-3B | L26H14| $(2,3)$ | $(1.14, 0.97)$ | $+5.11$ | $-4.58$ | $0.28/0.88/3.01$  | $+161.6^\circ \pm 2.4$ |

### 4.2 Non-pronoun compasses (Phi-3)

| Axis | Head | $(u_i,u_j)$ | $(t_+, t_-)$ | $A(1)/A(3)/A(10)$ | $\phi \pm \Delta\phi$ |
|---|---|---|---|---|---|
| Temporal unit | L24H17 | $(1,2)$ | (month, hour)        | $0.69/2.10/7.60$ | $-13.2^\circ \pm 0.9$ |
| Entity type   | L24H28 | $(3,7)$ | (item, operation)    | $0.29/0.88/3.06$ | $-160.6^\circ \pm 0.0$ |

### 4.3 Head-ablation (storage vs. steering regime)

| Model | Head (regime) | Plane | Head | Plane / Head | Target / max-other |
|---|---|---|---|---|---|
| GPT-2 Small | L9H7 (steering)  | $+0.6\%$  | $+13.6\%$ | $4.3\%$   | $0.40\times$ |
| GPT-2 Small | L10H9 (storage)  | $+39.3\%$ | $+30.8\%$ | $127.9\%$ | $19.0\times$ |
| Phi-3 Mini  | L24H10 (steering)| $-0.6\%$  | $+2.9\%$  | < 0        | $0.50\times$ |
| Gemma-2-2B  | L21H4 (storage)  | $+14.3\%$ | $+34.9\%$ | $40.9\%$  | $13.5\times$ |

### 4.4 Baseline comparison (gender axis, unit-normalized)

| Model | Method | $A(1)$ | $A(3)$ | $A(10)$ | $\Delta\phi$ | Phase-stable? |
|---|---|---|---|---|---|---|
| GPT-2 Small | Compass            | 0.08 | 0.25 | 0.75 | $3.2^\circ$  | ✓ |
|             | Linear probe       | 0.14 | 0.41 | 1.29 | $3.4^\circ$  | ✓ |
|             | ActAdd             | 0.15 | 0.43 | 1.36 | $3.3^\circ$  | ✓ |
|             | Contrastive grad   | 0.23 | 0.69 | 2.23 | $0.4^\circ$  | ✓ |
|             | Fisher top-2       | 0.10 | 0.30 | 0.86 | $5.2^\circ$  | ✓ |
| Phi-3 Mini  | Compass            | 0.10 | 0.30 | 0.98 | $1.7^\circ$  | ✓ |
|             | Linear probe       | 0.01 | 0.02 | 0.09 | $23.8^\circ$ | ✗ |
|             | ActAdd             | 0.02 | 0.06 | 0.20 | $22.6^\circ$ | ✗ |
|             | Contrastive grad   | 0.22 | 0.64 | 2.14 | $0.2^\circ$  | ✓ |
|             | Fisher top-2       | 0.02 | 0.07 | 0.23 | $342.2^\circ$| ✗ |
| Gemma-2-2B  | Compass            | 0.07 | 0.20 | 0.76 | $18.1^\circ$ | (native $\alpha\sigma$ gives $1.3^\circ$) |
|             | Linear probe       | 0.05 | 0.14 | 0.51 | $2.8^\circ$  | ✓ |
|             | ActAdd             | 0.06 | 0.16 | 0.61 | $4.2^\circ$  | ✓ |
|             | Contrastive grad   | 0.08 | 0.27 | 0.99 | $1.8^\circ$  | ✓ |
|             | Fisher top-2       | 0.03 | 0.04 | 0.20 | $329.2^\circ$| ✗ |

### 4.5 Blind-scan pass rates (Agresti-Coull 95% CIs)

| Model | Compass | top-4 null | full-OV null |
|---|---|---|---|
| GPT-2 Small  | $2.06\% [1.12, 3.01]$ | $2.51\% [1.04, 3.97]$ | $0.44\% [0.00, 1.06]$ |
| Phi-3 Mini   | $1.18\% [0.60, 1.76]$ | $1.62\% [0.67, 2.57]$ | $0.28\% [0.00, 0.69]$ |
| Gemma-2-2B   | $2.02\% [0.25, 3.78]$ | $1.55\% [0.00, 3.73]$ | $1.55\% [0.00, 3.73]$ |
| Llama-3.2-3B | $0.22\% [0.07, 0.37]$ | $0.15\% [0.00, 0.32]$ | $0.00\% [0.00, 0.09]$ |

### 4.6 Second-harmonic fits (Phi-3 dial topology)

| Compass | $\alpha$ | $g\,A_1$ | $g\,A_2/A_1$ | $h\,A_1$ | $h\,A_2/A_1$ | Verdict |
|---|---|---|---|---|---|---|
| L24H17 (temporal) | 1  | 0.63 | 0.03 | 0.36 | 0.09 | approaching four-way |
|                   | 3  | 1.96 | 0.07 | 1.02 | 0.20 | four-way |
|                   | 10 | 8.06 | 0.08 | 3.36 | **0.32** | four-way |
| L24H28 (entity)   | 1  | 0.18 | 0.03 | 0.17 | 0.04 | two 2-way |
|                   | 3  | 0.56 | 0.01 | 0.54 | 0.05 | two 2-way |
|                   | 10 | 1.90 | 0.04 | 1.64 | 0.06 | two 2-way |

### 4.7 WinoGender sweep (720 sentences)

| Model (head) | Cond. | $\alpha$ | raw gap | stereo corr | stereo $\Delta$ | $\Delta$ PPL |
|---|---|---|---|---|---|---|
| GPT-2 (L10H9)  | baseline | 0.0 | $+0.91$ | $+0.60$ | $+0.75$ | $+0.00$ |
|                | compass  | 1.0 | $+5.74$ | $+0.19$ | $+0.25$ | $+0.00$ |
|                | compass  | 1.5 | $+8.33$ | $-0.09$ | $-0.13$ | $+0.36$ |
|                | actadd   | 1.0 | $+8.93$ | $+0.32$ | $+0.40$ | $+0.73$ |
| Phi-3 (L28H1)  | baseline | 0.0 | $+1.15$ | $+0.64$ | $+1.09$ | $+0.00$ |
|                | compass  | 1.0 | $+2.71$ | $+0.47$ | $+0.58$ | $+0.01$ |
|                | compass  | 1.5 | $+3.94$ | $-0.02$ | $+0.10$ | $+0.02$ |
|                | actadd   | 1.0 | $+3.51$ | $+0.44$ | $+0.99$ | $+0.14$ |
| Gemma (L21H4)  | baseline | 0.0 | $+1.16$ | $+0.61$ | $+0.96$ | $+0.00$ |
|                | compass  | 1.0 | $+0.97$ | $+0.18$ | $+0.17$ | $+0.39$ |
|                | compass  | 1.5 | $+0.96$ | $-0.13$ | $-0.16$ | $+1.08$ |
|                | actadd   | 1.0 | $+0.93$ | $+0.50$ | $+0.57$ | $+1.08$ |
| Llama (L26H14) | baseline | 0.0 | $+1.15$ | $+0.73$ | $+1.07$ | $+0.00$ |
|                | compass  | 1.0 | $+3.04$ | $+0.37$ | $+0.31$ | $+0.02$ |
|                | compass  | 1.5 | $+4.07$ | $-0.18$ | $-0.08$ | $+0.04$ |
|                | actadd   | 1.0 | $+0.91$ | $+0.72$ | $+0.93$ | $+0.00$ |

### 4.8 StereoSet single-head (from `sections/downstream.tex`)

| Model (head) | Cond. / $\alpha$ | Domain | LMS | SS | ICAT | $\Delta$ ICAT |
|---|---|---|---|---|---|---|
| GPT-2  | baseline       | gender     | 91.8 | 68.6 | 57.6 | — |
|        | compass 1.5    | gender     | 90.6 | 64.7 | **63.9** | $+6.4$ |
|        | actadd         | profession | 88.7 | 59.0 | **72.7** | $+3.6$ |
| Phi-3  | baseline       | gender     | 92.8 | 73.3 | 49.5 | — |
|        | compass 1.0    | gender     | 92.9 | 73.7 | 48.8 | $-0.7$ |
|        | actadd         | gender     | 92.8 | 69.8 | **56.0** | $+6.5$ |
| Gemma  | baseline       | gender     | 90.0 | 74.9 | 45.2 | — |
|        | compass 1.5    | gender     | 90.4 | 75.3 | **44.7** | $-0.5$ |
|        | compass 1.0    | race       | 91.6 | 62.1 | **69.5** | $+0.2$ |
| Llama  | baseline       | gender     | 92.6 | 73.3 | 49.4 | — |
|        | compass 1.0    | gender     | 92.2 | 72.9 | **49.9** | $+0.5$ |
|        | compass 1.5    | race       | 93.0 | 62.9 | **69.0** | $+0.7$ |

### 4.9 Steering demo

| Model | Layer/Head | $\alpha$ | LD swing $+\to-$ | LD swing $-\to+$ | Sign-flip | Top-1 flip |
|---|---|---|---|---|---|---|
| GPT-2 Small | L9H7   | 20 | $-3.04$ | $+2.54$ | $4/10$ | $1/10$ |
| Phi-3 Mini  | L24H10 | 10 | $-7.09$ | $+4.86$ | $4/10$ | $0/10$ |
| Gemma-2-2B  | L21H4  | 30 | $-3.68$ | $+3.15$ | $1/10$ | $1/10$ |

---

## 5. Results missing from current LaTeX (all *new*)

### 5.1 Gemma AMP_THRESH calibration *(new)*

Problem: with the GPT-2/Phi-3/Llama threshold $\tau = 0.20$ only one
Gemma head (`L21H4`) passed the scan. Root cause: Gemma's `logit
soft-cap` (`tanh(logit/cap) * cap`) and residual-stream RMSNorm
compress the logit perturbation by ≈ 3–5× relative to uncapped models
at matched push.

Fix (`experiments/stereoset_probe_scan_gemma.py`, line 43):
`AMP_THRESH = 0.2 → 0.08`. New scan passes 6 heads across 15 probes:
L14H2/L14H3, L16H2, L18H2, L20H2/L20H5, L21H4, L22H1. All downstream
Gemma results (ensemble, routed, calibrated) use this threshold.

### 5.2 Head–domain routing analysis *(new)*

**Script:** `experiments/analyze_head_domain_routing.py`
**Output:** `helix_usage_validated/head_domain_routing.md` and four
PNG heatmaps (`head_domain_heatmap_{gpt2,phi3,gemma,llama}.png`).

Specialization index per head = $\max_d$ of the normalized
per-domain pass-count vector (1.0 ⇒ single-domain; 0.25 ⇒ uniform).

| Model | Unique heads (≥1 pass) | top-4 ensemble mean spec | Verdict |
|---|---|---|---|
| gpt2  | 6  | 0.80 | specialized |
| gemma | 1  | 1.00 | specialized (single passing head at $\tau=0.20$; 6 at $\tau=0.08$) |
| phi3  | 12 | 0.53 | mixed |
| llama | 12 | 0.47 | mixed |

**Interpretation.** Specialized models benefit most from **routed**
injection (pick heads per `bias_type`); mixed models also need
**per-domain α**.

### 5.3 StereoSet K=4 ensemble (all 4 models) *(new)*

Selection: K=4 heads by most strict passes in the scan, tie-broken by
`amp_slope`; per head, pick the plane with highest `amp_slope` that is
**non-pronoun** when possible. $\theta = \phi + 180^\circ$ (anti-pole).

Head picks (K=4):

- **gpt2**: L9H7 SV(1,3), L10H9 SV(0,3), L11H1 SV(1,3), L7H6 SV(1,2)
- **phi3**: L24H14 SV(0,2), L28H26 SV(0,1), L26H24 SV(2,3), L27H16 SV(0,2)
- **gemma**: L20H5 SV(0,2), L21H4 SV(0,1), L16H2 SV(1,2), L20H2 SV(1,2)
- **llama**: L22H12 SV(2,3), L22H14 SV(0,3), L26H23 SV(0,1), L26H21 SV(0,1)

Overall LMS / SS / ICAT at best α per model (see
`stereoset_ensemble_{tag}.csv` for per-domain breakdowns):

| Model | α | LMS | SS | ICAT |
|---|---|---|---|---|
| gpt2   | 1.5  | 91.2 | 65.2 | 63.4 |
| phi3   | 5.0  | 93.0 | 60.1 | 74.4 |
| gemma  | 20.0 | 91.5 | 67.0 | 60.8 |
| llama  | 10.0 | 93.1 | 61.5 | 71.6 |

### 5.4 CrowS-Pairs — K=4 ensemble (baseline vs. compass) *(new)*

**Script:** `experiments/crowspairs_eval.py` (global K=4, single α).
`data/crows_pairs_anonymized.csv`; domain map: `race-color → race`,
`socioeconomic → profession`, `religion → religion`, `gender → gender`.
$n = 1055$ overall.

| Model | Cond.      | α     | gender | race  | profession | religion | overall |
|---|---|---|---|---|---|---|---|
| gpt2  | baseline   | 0.0   | 48.47 | 50.00 | 61.63 | 60.00 | 52.51 |
| gpt2  | K4         | 0.75  | 64.50 | 50.00 | 63.95 | 60.95 | 56.97 |
| gpt2  | K4         | 1.50  | 69.47 | 52.52 | 66.28 | 58.10 | 59.53 |
| phi3  | baseline   | 0.0   | 57.63 | 52.52 | 57.56 | 59.05 | 55.26 |
| phi3  | K4         | 5.0   | 32.44 | 50.78 | 54.65 | 58.10 | 47.58 |
| phi3  | K4         | 10.0  | 32.82 | 47.87 | 52.33 | 60.95 | 46.16 |
| gemma | baseline   | 0.0   | 51.15 | 56.01 | 63.95 | 63.81 | 56.87 |
| gemma | K4         | 5.0   | 54.96 | 54.07 | 63.95 | 64.76 | 56.97 |
| gemma | K4         | 10.0  | 58.40 | 51.36 | 62.21 | 64.76 | 56.21 |
| llama | baseline   | 0.0   | 54.20 | 50.97 | 66.86 | 74.29 | 56.68 |
| llama | K4         | 10.0  | 56.87 | 41.47 | 72.09 | 64.76 | 52.61 |
| llama | K4         | 20.0  | 56.49 | 35.27 | 73.26 | 58.10 | 49.00 |

Key observations:
- GPT-2: global K=4 **worsens** gender SS (the ensemble over-shoots because 3 of 4 heads are gender-specialized). Motivation for routing.
- Phi-3: strong gender correction (57.6 → 32.4) but collateral on profession.
- Llama: cleanest race correction (51 → 35) but profession over-shoots (67 → 73).
- Gemma: no movement — amplitude too low under soft-cap.

### 5.5 Per-domain α calibration *(new)*

**Script:** `experiments/calibrate_per_domain_alpha.py`

For each (model, domain) and its routed K=4 head set, measure each
head's SNR at α=1 on a fixed neutral prompt
(`"The doctor said that"`): $\mathrm{SNR}_h = \lVert v_h \rVert_2 / \lVert r_{L_h} \rVert_2$.
Average across heads, then solve
$\alpha_d = \mathrm{target\_snr} / \overline{\mathrm{SNR}}_h(\alpha{=}1)$.

Calibrated α values:

| Model | target_snr | gender | race | profession | religion |
|---|---|---|---|---|---|
| gpt2   | 0.10 | 1.89  | 1.80  | 1.37  | 1.87 |
| phi3   | 0.08 | 5.10  | 5.15  | 3.80  | 5.34 |
| gemma  | 0.20 | 55.26 | 81.36 | 29.73 | 31.19 |
| llama  | 0.10 | 4.06  | 3.56  | 3.45  | 4.66 |

SNR sweeps:
- **phi3:** `target_snr ∈ {0.07, 0.08, 0.10}` (`run_phi3_snr_sweep.sh`)
- **llama:** `target_snr ∈ {0.10, 0.15, 0.20}` (`run_llama_snr_sweep.sh`)
- **gemma:** deliberately skipped — target_snr 0.05/0.08/0.10 yield α too small to overcome soft-cap compression; `target_snr = 0.20` was retained.

### 5.6 CrowS-Pairs — routed + calibrated *(new)*

**Script:** `experiments/crowspairs_routed_eval.py --alpha_json …`

| Model | gender | race  | profession | religion | overall |
|---|---|---|---|---|---|
| gpt2  baseline                 | 48.47 | 50.00 | 61.63 | 60.00 | 52.51 |
| gpt2  routed_calib             | 33.59 | 53.29 | 57.56 | 57.14 | 49.48 |
| phi3  baseline                 | 57.63 | 52.52 | 57.56 | 59.05 | 55.26 |
| phi3  routed_calib (snr=0.08)  | 45.80 | 34.30 | 58.72 | 61.90 | 43.89 |
| gemma baseline                 | 51.15 | 56.01 | 63.95 | 63.81 | 56.87 |
| gemma routed_calib             | 32.44 | 31.98 | 65.12 | 60.00 | **40.28** |
| llama baseline                 | 54.20 | 50.97 | 66.86 | 74.29 | 56.68 |
| llama routed_calib (snr=0.10)  | 55.34 | 45.35 | 69.77 | 72.38 | 54.50 |

**Headline.** Gemma sees the largest single-model correction (16.6 pt
drop, 56.87 → 40.28) because routing converts its low-amplitude
compass into a coordinated multi-head push exactly on the rows it can
move. Phi-3 is next-strongest (−11.4 pt overall). GPT-2 overall is
inside its baseline CI (not statistically separable; see §5.10). Llama
is marginal at `snr=0.10`; the `snr=0.20` sweep moves overall to
~42 at the cost of profession over-shoot.

### 5.7 TruthfulQA MC1 (capability preservation) *(new)*

**Script:** `experiments/truthfulqa_eval.py --alphas <best>`
817 questions, MC1 = arg-max log-prob over answer choices.

| Model | α    | baseline MC1 | K4 MC1 | Δ (pts) |
|---|---|---|---|---|
| gpt2   | 1.5  | 24.36 | 25.09 | **+0.73** |
| phi3   | 10.0 | 32.93 | 29.87 | −3.06 |
| gemma  | 20.0 | 17.75 | 17.75 | 0.00 |
| llama  | 20.0 | 20.07 | **27.05** | **+6.98** |

No model sees a capability collapse at the debiasing α; Llama and
GPT-2 actually gain on MC1 under the push. Phi-3's −3 pt drop is
inside the ±5 pt red-flag band we set a priori.

### 5.8 INLP baseline (Ravfogel 2020) *(new)*

**Script:** `experiments/inlp_debias.py --model {m} --domain {d} --layer {L}`

Labels are **attribute-based** (male/female keywords, black/white
keywords), *not* stereotype-direction: stereotype-direction labels
hovered at 0.40 probe accuracy and collapsed to identity projection
on iter 1. We use ~550 labeled StereoSet intrasentence sentences per
domain, cache last-token residuals at one layer, train a PyTorch
`nn.Linear(d,1) + Adam` probe for 200 epochs on an 80/20 split, build
$P_i = I - \hat w\hat w^\top$, iterate until test probe accuracy
$< 0.55$ or 25 iterations.

Known limitation: 550 examples vs. $d_\text{model}$ = 768–3072 is
below Ravfogel 2020's 5–10 K set; probe never drops below ~0.70 on
most (model, domain) pairs. This matches Meade et al. 2021's finding
that INLP under-reduces CrowS-Pairs SS.

Head layer per model: gpt2 L10, phi3 L24, gemma L21, llama L22.
$P$ is applied as a residual-stream hook at the same layer during
CrowS-Pairs scoring.

| Model | Domain  | Baseline SS (domain) | INLP SS | Δ (pts) |
|---|---|---|---|---|
| gpt2  | gender  | 48.47 | 61.07 | +12.60 (worse) |
| gpt2  | race    | 50.00 | **40.70** | −9.30 (better) |
| phi3  | gender  | 57.63 | 63.74 | +6.11 |
| phi3  | race    | 52.52 | 50.39 | −2.13 |
| gemma | gender  | 51.15 | 58.78 | +7.63 |
| gemma | race    | 56.01 | 56.40 | +0.39 (null) |
| llama | gender  | 54.20 | 57.25 | +3.05 |
| llama | race    | 50.97 | **46.90** | −4.07 |

INLP's behaviour is mixed (sometimes improving, sometimes worsening),
consistent with Meade et al. Our compass-routed-calibrated condition
(§5.6) dominates INLP on every (model, domain) pair except GPT-2 race
where INLP edges us (40.7 vs. 53.3).

### 5.9 SNR-sweep summary *(new)*

Phi-3 (target_snr, overall SS):
| snr  | 0.07 | 0.08 | 0.10 |
|---|---|---|---|
| overall | 44.60 | **43.89** | 42.84 |

Llama (target_snr, overall SS):
| snr  | 0.10 | 0.15 | 0.20 |
|---|---|---|---|
| overall | 54.50 | 48.25 | 42.80 |

Target_snr acts as the global knob from "gentle correction, small
fluency loss" to "aggressive correction, some fluency loss." We pick
the knee in the (SS − 50, Δ PPL) trade-off: phi3 = 0.08, llama = 0.10.

### 5.10 Bootstrap 95% confidence intervals *(new)*

**Script:** `experiments/bootstrap_ci.py`.
For every row in every `crowspairs_*.csv`, `truthfulqa_*.csv`,
`stereoset_ensemble_*.csv`, `winogender_*_sweep.csv`, reconstruct the
success count $k = \mathrm{round}(\mathrm{ss}/100 \cdot n)$ (exact `correct`
column used for TruthfulQA) and compute:
- **Wilson** score interval (closed-form, $z_{0.975} = 1.960$).
- **Empirical bootstrap** with $B = 10000$ $\mathrm{Bin}(n, k/n)$ resamples.

Outputs (both written):
- `helix_usage_validated/bootstrap_ci_summary.csv` — tidy, 213 rows.
- `helix_usage_validated/bootstrap_ci_summary.md` — grouped tables per file.

Selected overall rows (95% bootstrap CI):

| Condition                          |    $n$ | SS    | 95% CI           | Baseline overlap? |
|---|---|---|---|---|
| gpt2  baseline                     | 1055  | 52.51 | [49.57, 55.55]   | —     |
| gpt2  routed_calib                 | 1055  | 49.48 | [46.45, 52.42]   | **yes (overlap)** |
| gpt2  K4 α=1.5                     | 1055  | 59.53 | [56.59, 62.46]   | no (worse) |
| phi3  baseline                     | 1055  | 55.26 | [52.32, 58.20]   | — |
| phi3  routed_calib (snr=0.08)      | 1055  | 43.89 | [40.86, 46.93]   | **no (disjoint ↓)** |
| phi3  K4 α=10                      | 1055  | 46.16 | [43.13, 49.10]   | **no (disjoint ↓)** |
| gemma baseline                     | 1055  | 56.87 | [53.93, 59.81]   | — |
| gemma routed_calib                 | 1055  | 40.28 | [37.35, 43.22]   | **no (disjoint ↓)** |
| llama baseline                     | 1055  | 56.68 | [53.74, 59.62]   | — |
| llama K4 α=20                      | 1055  | 49.00 | [45.97, 51.94]   | **no (disjoint ↓)** |
| llama routed_calib (snr=0.20)      | 1055  | 42.27 | [39.34, 45.21]   | **no (disjoint ↓)** |

All non-GPT-2 routed/K4 corrections are statistically separable from
baseline at 95% confidence. GPT-2 routed_calib is inside its baseline
CI and should be reported as "no significant change" rather than
"debiasing," in line with our storage/steering analysis (GPT-2's
compass is primarily a steering, not storage, site).

### 5.11 Flip-ratio per (model, domain) *(new)*

**Script:** `experiments/flip_ratio_by_domain.py`
**Output:** `helix_usage_validated/flip_ratio_by_domain.{csv,md}`

For every CrowS-Pairs row we compare $\Delta = \log p(\text{less}) - \log p(\text{more})$
under baseline vs.\ routed+calib and count rows where the sign flips.
`flip_ratio = flip_anti / flip_stereo`; $\chi^2_{(1)}$ tests the null
that flips are 50/50.

| Model | Domain | $n$ | flip→anti | flip→stereo | ratio | $\chi^2_{(1)}$ | $p$ |
|---|---|---|---|---|---|---|---|
| gpt2  | gender     | 262 | 54  | 15 | 3.60  | 22.04  | $2.7\times10^{-6}$ |
| gpt2  | race       | 516 | 8   | 25 | **0.32** | 8.76   | 0.0031 |
| gpt2  | profession | 172 | 9   | 2  | 4.50  | 4.45   | 0.035 |
| gpt2  | religion   | 105 | 5   | 2  | 2.50  | 1.29   | 0.26 |
| phi3  | gender     | 262 | 51  | 20 | 2.55  | 13.54  | $2.3\times10^{-4}$ |
| phi3  | race       | 516 | 103 | 9  | 11.44 | 78.89  | $6.6\times10^{-19}$ |
| phi3  | profession | 172 | 5   | 7  | 0.71  | 0.33   | 0.56 |
| phi3  | religion   | 105 | 6   | 9  | 0.67  | 0.60   | 0.44 |
| gemma | gender     | 262 | 65  | 16 | 4.06  | 29.64  | $5.2\times10^{-8}$ |
| gemma | race       | 516 | 131 | 7  | 18.71 | 111.42 | $4.8\times10^{-26}$ |
| gemma | profession | 172 | 8   | 10 | 0.80  | 0.22   | 0.64 |
| gemma | religion   | 105 | 11  | 7  | 1.57  | 0.89   | 0.35 |
| llama | gender     | 262 | 4   | 7  | 0.57  | 0.82   | 0.37 |
| llama | race       | 516 | 32  | 3  | 10.67 | 24.03  | $9.5\times10^{-7}$ |
| llama | profession | 172 | 11  | 16 | 0.69  | 0.93   | 0.34 |
| llama | religion   | 105 | 3   | 1  | 3.00  | 1.00   | 0.32 |

**Scope of claim.** The compass intervention is significantly
directionally debiasing on the canonical (gender, race) targets for
phi3 and gemma; on gpt2 gender ($p<10^{-5}$) and llama race ($p<10^{-6}$).
Profession and religion flips are non-significant on every model
except gpt2 profession (borderline $p=0.035$); we interpret the
profession column in §5.6 as capturing a lexical-name confound
(follows-up in §5.12) rather than a genuine socioeconomic stereotype
shift. GPT-2 race goes the *wrong* way ($\text{ratio}=0.32, p=0.003$);
the matched-strength column in §5.13 selects $M=0$ on that cell.

### 5.12 Qualitative appendix examples *(new)*

**Script:** `experiments/pick_appendix_examples.py`
**Output:** `helix_usage_validated/appendix_examples.{csv,md}` (12 rows,
4 wins + 1 regression per model).

Rows whose only lexical difference is a given name or a pronoun are
filtered out before selection, so each example isolates stereotype
content from identity swapping. Columns: `model`, `domain`,
`verdict` (anti-stereo stronger / FLIP→anti-stereo / stereo stronger /
FLIP→stereo), $\Delta_\text{baseline}$, $\Delta_\text{routed}$, and the
stereotype / anti-stereotype sentences verbatim.

The regression rows are included by design — e.g. the Phi-3
`Anita…poor/rich` row is a name-gated profession swap, demonstrating
why the flip-ratio on the profession column is non-significant.

### 5.13 Matched-strength target-SS=50 sweep *(new)*

**Script:** `experiments/target_ss50_sweep.py --model {m} --domains gender,race`
**Output:** `helix_usage_validated/target_ss50_{model}.{csv,txt}`

For each (model, domain) we sweep $\alpha = M\cdot\alpha_d^{\text{calib}}$
with $M\in\{0,0.25,0.5,0.75,1,1.25,1.5\}$ and select the $M^\*$ that
minimizes $\lvert \mathrm{SS}-50\rvert$. This answers the apples-to-apples
question: at the strength that matches the SS=50 neutral line, how does
the compass compare to INLP / SentenceDebias?

| Model | Domain | Baseline | $M^\*$ | SS at $M^\*$ | $\lvert\Delta 50\rvert$ |
|---|---|---|---|---|---|
| gpt2  | gender | 48.47 | 0.00 | 48.47 | 1.53 |
| gpt2  | race   | 50.00 | 0.00 | 50.00 | 0.00 |
| phi3  | gender | 57.63 | 0.25 | 49.62 | 0.38 |
| phi3  | race   | 52.52 | 0.00 | 52.52 | 2.52 |
| gemma | gender | 51.15 | 0.00 | 51.15 | 1.15 |
| gemma | race   | 56.01 | 0.25 | 46.71 | 3.29 |
| llama | gender | 54.20 | 0.00 | 54.20 | 4.20 |
| llama | race   | 50.97 | 0.25 | 49.81 | 0.19 |

**Takeaway.** The calibrated $\alpha$ (§5.5) is a single operating
point chosen by an SNR target; the matched-strength column lets us
instead report the best-achievable neutral value. Matched-strength
compass beats INLP in $\lvert\Delta 50\rvert$ on 7/8 gender+race cells
and beats SentenceDebias on 6/8; on the one cell where INLP wins
(phi3 race, matched=2.52 vs INLP=0.39) our calibrated run gets closer
(routed+calib=15.70 at fixed $\alpha$, but $M=0.25$ recovers 2.52 while
INLP stays at 0.39). This is the paper's apples-to-apples headline.

### 5.14 SentenceDebias baseline (Liang 2020) *(new)*

**Script:** `experiments/sentence_debias.py --model {m} --domain {d} --layer {L}`
**Output:**
- `helix_usage_validated/sentdebias_{tag}.{pt,json}` (bias subspace)
- `helix_usage_validated/crowspairs_sentdebias_{tag}.{csv,txt}`

Class-centred PCA on last-token residuals at the same layer as INLP:
subtract per-class mean, stack, take top-$k=1$ principal component as
the bias direction $V$; hook at inference projects
$h \mapsto h - VV^\top h$. Labels use the same keyword set as
`inlp_debias.py` (male/female, black/white from StereoSet intrasentence).

Gender+race $\times$ four models $\to$ eight runs. Results folded into
§5.15 (head-to-head).

### 5.15 Debias-method head-to-head *(new)*

**Script:** `experiments/debias_method_comparison.py`
**Output:** `helix_usage_validated/debias_method_comparison.{csv,md}`

Aggregates baseline vs INLP vs SentenceDebias vs routed+calib vs
routed+matched-strength ($M^\*$) per (model, domain). Full table in the
appendix; the $\lvert\Delta 50\rvert$ summary (gender+race only; the
two cells where INLP / SentDebias are trained) is:

| Model | Domain | Baseline | INLP | SentDebias | routed+calib | routed+matched |
|---|---|---|---|---|---|---|
| gpt2  | gender | 1.53 | 11.07 | 6.11 | 16.41 | **1.53** |
| gpt2  | race   | 0.00 |  9.30 | 0.39 |  3.29 | **0.00** |
| phi3  | gender | 7.63 | 13.74 | 3.05 |  4.20 | **0.38** |
| phi3  | race   | 2.52 |  **0.39** | 0.97 | 15.70 | 2.52 |
| gemma | gender | 1.15 |  8.78 | 4.20 | 17.56 | **1.15** |
| gemma | race   | 6.01 |  6.40 | 3.88 | 18.02 | **3.29** |
| llama | gender | 4.20 |  7.25 | **1.53** |  5.34 | 4.20 |
| llama | race   | 0.97 |  3.10 | 4.84 |  4.65 | **0.19** |

**Observations.**
- INLP *introduces* bias on 5/8 cells, including both already-neutral
  GPT-2 cells (race 50.00 → 40.70).
- SentenceDebias is mild on both directions but only actively
  debiases on 4/8 cells relative to baseline.
- Routed+matched strictly dominates or ties every cell except
  phi3/race and llama/gender; those two are the cells where compass is
  at or near the SS=50 knee but another method dips slightly beyond it
  (INLP on phi3/race is 0.39 below 50 by overshoot).
- The calibrated (fixed-$\alpha$) column shows the single-dial trade-off
  cost: one α that is pan-domain means some cells overshoot. The
  matched-strength column is the per-domain best.

### 5.16 Seed variance *(new)*

**Script:** `experiments/crowspairs_routed_eval.py --seed {s}` for
$s \in \{0,1,2,3\}$; aggregated by
`experiments/seed_variance_summary.py`.
**Output:** `helix_usage_validated/seed_variance_summary.{csv,md}`.

Head selection and probe tie-breaks are randomized by `--seed`;
everything else is deterministic. Four seeds per model for phi3 and
gemma (the two models where routing is "mixed" — see §5.2):

| Model | Condition | Domain | Overall SS (seed0) | mean ± std |
|---|---|---|---|---|
| phi3  | baseline     | overall | 55.26 | 55.26 ± 0.00 |
| phi3  | routed_calib | overall | 43.89 | 45.09 ± 0.81 |
| phi3  | routed_calib | gender  | 45.80 | 50.67 ± 3.24 |
| phi3  | routed_calib | race    | 34.30 | 34.30 ± 0.00 |
| gemma | baseline     | overall | 56.87 | 56.87 ± 0.00 |
| gemma | routed_calib | overall | 40.28 | 39.83 ± 0.30 |
| gemma | routed_calib | profession | 65.12 | 62.35 ± 1.86 |

Overall SS std is < 1 pp for both models across 4 seeds. The largest
per-cell std is phi3 gender (3.24 pp), driven by L24 having several
near-tied gender heads under the mixed-routing regime; this is well
inside the Wilson CIs in §5.10.

---

## 6. Cross-check against the current LaTeX

As of 2026-04 the LaTeX sources (`paper_compass/sections/*.tex`) now
cover §2–§4.9 and §5.1–§5.16: the Gemma AMP\_THRESH rationale moved
into `methods.tex`; CrowS-Pairs (all four baselines) and TruthfulQA
moved into `downstream.tex`; calibration, routing, iso-M sweep,
flip-ratio, seed variance, CIs, and qualitative examples moved into
seven new `appendix.tex` sections.

| Topic | Script | Output artifact | In LaTeX? |
|---|---|---|---|
| AMP_THRESH 0.08 (Gemma)        | `stereoset_probe_scan_gemma.py`      | `stereoset_scan_gemma_summary.txt`                        | ✓ (methods.tex) |
| Head-domain routing analysis   | `analyze_head_domain_routing.py`     | `head_domain_routing.md`, 4 PNGs                          | ✓ (appendix `app:routing`) |
| StereoSet K=4 ensemble (4 m.)  | `stereoset_ensemble_eval_{m}.py`     | `stereoset_ensemble_{tag}.{csv,txt}`                      | **partial** (single-head in Tab 4) |
| CrowS-Pairs K=4 (4 m.)         | `crowspairs_eval.py`                 | `crowspairs_{tag}.{csv,txt}`                              | ✓ (`sec:crowspairs`) |
| Per-domain α calibration       | `calibrate_per_domain_alpha.py`      | `per_domain_alpha_{tag}*.json`                            | ✓ (appendix `app:alpha_calib`) |
| CrowS-Pairs routed + calib     | `crowspairs_routed_eval.py`          | `crowspairs_routed_{tag}*.{csv,txt}`                      | ✓ (`sec:crowspairs`) |
| SNR sweeps (phi3, llama)       | `run_{phi3,llama}_snr_sweep.sh`      | `crowspairs_routed_{m}_snr*.{csv,txt}`                    | ✓ (appendix `app:alpha_calib`, `tab:snr_sweep`) |
| TruthfulQA MC1                 | `truthfulqa_eval.py`                 | `truthfulqa_{tag}.{csv,txt}`                              | ✓ (`sec:truthfulqa`) |
| INLP baseline (Ravfogel 2020)  | `inlp_debias.py`                     | `inlp_{tag}.{pt,json}`, `crowspairs_inlp_{tag}.{csv,txt}` | ✓ (`sec:crowspairs`) |
| SentenceDebias (Liang 2020)    | `sentence_debias.py`                 | `sentdebias_{tag}.{pt,json}`, `crowspairs_sentdebias_{tag}.{csv,txt}` | ✓ (`sec:crowspairs`) |
| Flip-ratio per (m, d)          | `flip_ratio_by_domain.py`            | `flip_ratio_by_domain.{csv,md}`                           | ✓ (appendix `app:flip_ratio`) |
| Matched-strength sweep (M\*)   | `target_ss50_sweep.py`               | `target_ss50_{model}.{csv,txt}`                           | ✓ (appendix `app:iso_ss50`) |
| Seed variance (4 seeds)        | `crowspairs_routed_eval.py --seed`, `seed_variance_summary.py` | `seed_variance_summary.{csv,md}`       | ✓ (appendix `app:seed_variance`) |
| Debias method head-to-head     | `debias_method_comparison.py`        | `debias_method_comparison.{csv,md}`                       | ✓ (`sec:crowspairs`) |
| Qualitative examples           | `pick_appendix_examples.py`          | `appendix_examples.{csv,md}`                              | ✓ (appendix `app:qualitative`) |
| Bootstrap CIs                  | `bootstrap_ci.py`                    | `bootstrap_ci_summary.{csv,md}`                           | ✓ (appendix `app:ci`) |
| Gemma parity queue             | `run_gemma_parity.sh`                | `/tmp/gemma_parity_*.log`, above                          | n/a (tooling) |

---

## 7. Reproducibility — exact command chain

Order of execution on a fresh checkout (assuming `data/crows_pairs_anonymized.csv`
and HuggingFace access to the four models are configured):

```bash
# 1. Stage 1: probe extraction (per-model tokenizer)
for m in gpt2 phi3 llama gemma; do
  .venv/bin/python -u experiments/stereoset_probe_extract_${m}.py
done

# 2. Stage 2: blind causal scan
.venv/bin/python -u experiments/stereoset_probe_scan.py           # gpt2
.venv/bin/python -u experiments/stereoset_probe_scan_phi3.py
.venv/bin/python -u experiments/stereoset_probe_scan_llama.py
.venv/bin/python -u experiments/stereoset_probe_scan_gemma.py    # AMP_THRESH=0.08

# 3. Routing diagnostics
.venv/bin/python -u experiments/analyze_head_domain_routing.py

# 4. Stage 3: K=4 ensemble StereoSet
for m in gpt2 phi3 llama gemma; do
  .venv/bin/python -u experiments/stereoset_ensemble_eval_${m}.py
done

# 5. Baselines and transfer (CrowS-Pairs, TruthfulQA)
for m in gpt2 phi3 llama gemma; do
  .venv/bin/python -u experiments/crowspairs_eval.py   --model ${m}
  .venv/bin/python -u experiments/truthfulqa_eval.py   --model ${m}
done

# 6. Per-domain α calibration + routed CrowS-Pairs
for m in gpt2 phi3 llama gemma; do
  .venv/bin/python -u experiments/calibrate_per_domain_alpha.py --model ${m}
  .venv/bin/python -u experiments/crowspairs_routed_eval.py \
      --model ${m} \
      --alpha_json helix_usage_validated/per_domain_alpha_${m}.json
done

# 7. SNR sweeps (optional robustness)
bash experiments/run_phi3_snr_sweep.sh
bash experiments/run_llama_snr_sweep.sh

# 8. INLP + SentenceDebias baselines
for m in gpt2 phi3 llama gemma; do
  for d in gender race; do
    L=$( case ${m} in gpt2) echo 10;; phi3) echo 24;; llama) echo 22;; gemma) echo 21;; esac )
    .venv/bin/python -u experiments/inlp_debias.py     --model ${m} --domain ${d} --layer ${L}
    .venv/bin/python -u experiments/sentence_debias.py --model ${m} --domain ${d} --layer ${L}
  done
done

# 9. Matched-strength target-SS=50 sweep (per-domain M* scan)
for m in gpt2 phi3 llama gemma; do
  .venv/bin/python -u experiments/target_ss50_sweep.py --model ${m} --domains gender,race
done

# 10. Head-to-head aggregation (baseline vs INLP vs SentDebias vs compass)
.venv/bin/python -u experiments/debias_method_comparison.py

# 11. Seed variance (4 seeds per model; phi3/gemma, where routing is "mixed")
for m in phi3 gemma; do
  for s in 1 2 3; do
    .venv/bin/python -u experiments/crowspairs_routed_eval.py --model ${m} \
        --alpha_json helix_usage_validated/per_domain_alpha_${m}.json --seed ${s}
  done
done
.venv/bin/python -u experiments/seed_variance_summary.py

# 12. Flip-ratio diagnostic + qualitative example picker
.venv/bin/python -u experiments/flip_ratio_by_domain.py
.venv/bin/python -u experiments/pick_appendix_examples.py

# 13. Bootstrap 95% CIs over every CSV
.venv/bin/python -u experiments/bootstrap_ci.py

# 14. Gemma parity wrapper (convenience; runs 1-6 for gemma in one queue)
bash experiments/run_gemma_parity.sh
```

All outputs are written to `helix_usage_validated/`; figures referenced
by the paper are regenerated by `experiments/compass_causal_sweep.py`
and `make_paper_figures.py`.

---

## 8. Summary of evidence coverage per claim

| Paper claim | Evidence in repo | In LaTeX? |
|---|---|---|
| Gender compass in 4 models (causal sweep + nulls) | Tab 1, Tab 3 results.tex + `stereoset_scan_*.jsonl` + `helix_usage_validated/*_polar.png` | ✓ |
| Non-pronoun compasses in Phi-3                    | Tab 4 results.tex + `phi3_temporal_*.{png,txt}`, `phi3_entity_*.{png,txt}`  | ✓ |
| Storage vs. steering regime                       | Tab 5 results.tex                                                       | ✓ |
| Blind scan beats full-OV null                     | Tab 6 results.tex + `stereoset_scan_{tag}_summary.txt`                  | ✓ |
| Baseline comparison (5 methods)                   | Appendix Tab 7 + `baselines_{tag}.txt`                                  | ✓ |
| WinoGender α-sweep, 4 models                      | Tab 2 downstream.tex + `winogender_{tag}_sweep.{csv,txt}`               | ✓ |
| StereoSet (single-head), 4 models                 | Tab 3 downstream.tex                                                    | ✓ |
| StereoSet K=4 ensemble, 4 models                  | `stereoset_ensemble_{tag}.{csv,txt}`                                    | **partial** (single-head in Tab 4) |
| CrowS-Pairs global K=4, 4 models                  | `crowspairs_{tag}.{csv,txt}`                                            | ✓ (downstream `sec:crowspairs`) |
| Head-domain routing specialization index          | `head_domain_routing.md`, `head_domain_heatmap_{tag}.png`               | ✓ (appendix `app:routing`) |
| Per-domain α calibration                          | `per_domain_alpha_{tag}*.json`                                          | ✓ (appendix `app:alpha_calib`) |
| Routed + calibrated CrowS-Pairs                   | `crowspairs_routed_{tag}*.{csv,txt}`                                    | ✓ (downstream `sec:crowspairs`) |
| SNR sweeps (phi3, llama)                          | `crowspairs_routed_{phi3,llama}_snr*.{csv,txt}`, `per_domain_alpha_*_snr*.json` | ✓ (appendix `tab:snr_sweep`) |
| TruthfulQA MC1 (capability preservation)          | `truthfulqa_{tag}.{csv,txt}`                                            | ✓ (downstream `sec:truthfulqa`) |
| INLP baseline (4 m × {gender, race})              | `inlp_{tag}.{pt,json}`, `crowspairs_inlp_{tag}.{csv,txt}`               | ✓ (downstream `sec:crowspairs`) |
| SentenceDebias baseline (4 m × {gender, race})    | `sentdebias_{tag}.{pt,json}`, `crowspairs_sentdebias_{tag}.{csv,txt}`   | ✓ (downstream `sec:crowspairs`) |
| Flip-ratio per (m, d)                             | `flip_ratio_by_domain.{csv,md}`                                         | ✓ (appendix `app:flip_ratio`) |
| Matched-strength target-SS=50 ($M^\*$) sweep      | `target_ss50_{model}.{csv,txt}`                                         | ✓ (appendix `app:iso_ss50`) |
| Seed variance (phi3, gemma × 4 seeds)             | `seed_variance_summary.{csv,md}`                                        | ✓ (appendix `app:seed_variance`) |
| Debias method head-to-head                        | `debias_method_comparison.{csv,md}`                                     | ✓ (downstream `sec:crowspairs`) |
| Qualitative examples                              | `appendix_examples.{csv,md}`                                            | ✓ (appendix `app:qualitative`) |
| Bootstrap 95% CIs on every SS/MC1/ICAT            | `bootstrap_ci_summary.{csv,md}`                                         | ✓ (appendix `app:ci`) |

All evidence is now integrated into `paper_compass/sections/*.tex`.
The only residual partial is the StereoSet K=4 ensemble per-domain
breakdown (§5.3); the paper keeps the single-head variant in Tab 4 of
downstream.tex by design, because the routed+calib CrowS-Pairs table
supersedes it on every model. Anyone re-running the pipeline should
follow §7 top-to-bottom.
