# Geometric Compass — Implementation Cookbook

This is the reference bible for the **Geometric Compass** half of the
`Beyond-Components` repo. The repo contains two papers; this cookbook is
about the second one (`paper_compass/`). The first one (`paper/`,
arithmetic circuit) is out of scope here.

The contract:

- Every paper claim → every script that produces it → every output
  artifact it lands in.
- Every command shown is executable as written from the repo root with
  the project venv activated (`.venv/bin/python ...`).
- Known repo-state inconsistencies are documented in §10. They are
  flagged at the point of use.

> **One-line citation note.** Stage-1 of the compass paper (learnable-mask
> direction discovery on the Gender Pronoun task) is method and
> infrastructure from **Beyond Components (Ahmad et al.)**. The compass
> paper extends that line of work; it does not claim direction discovery
> as its own. See `paper_compass/sections/results.tex` §3.1 (currently
> needs a citation pass — flagged for the next paper revision).

---

## Table of Contents

1. [Quickstart — five commands that regenerate Figure 2](#1-quickstart)
2. [Math core — the five formulas](#2-math-core)
3. [File map — L0 through L8](#3-file-map)
4. [Run recipes](#4-run-recipes)
5. [Artifact catalog](#5-artifact-catalog)
6. [Paper ↔ code mapping table](#6-paper--code-mapping)
7. [Canonical model/head/SVD configurations](#7-canonical-configurations)
8. [Known gotchas](#8-known-gotchas)
9. [What is NOT in the implementation](#9-what-is-not-in-the-implementation)
10. [Extending — adding a model / domain / compass](#10-extending)

---

## 1. Quickstart

Five commands. Each regenerates one artifact you can hand to a reviewer.

```bash
# 1. Regenerate Figure 2 of the paper (GPT-2 L9H7 gender compass).
.venv/bin/python experiments/compass_causal_sweep.py \
    --model gpt2 --layer 9 --head 7 --dims 1 2 \
    --tok_plus " he" --tok_minus " she" \
    --prompt_neutral "The person said that" \
    --prompt_neutral "Then they said that" \
    --prompt_neutral "Afterwards, the speaker said that" \
    --prompt_plus "The man laced up his boots because" \
    --prompt_plus "The father waved to the crowd and" \
    --prompt_plus "The king announced that" \
    --prompt_minus "The woman laced up her boots because" \
    --prompt_minus "The mother waved to the crowd and" \
    --prompt_minus "The queen announced that" \
    --out_prefix gpt2_quickstart
# Outputs: helix_usage_validated/gpt2_quickstart_{curves,polar,linearity}.png + .txt

# 2. Decode a head's compass dictionary (Table of decoded SVD axes).
.venv/bin/python experiments/compass_dictionary.py
# Outputs: helix_usage_validated/compass_dict_{gpt2,phi3,gemma}.txt + compass_dict_all.md

# 3. Blind causal scan (Fig. 3 heatmap, Table 3 pass-rates).
.venv/bin/python experiments/compass_scan.py \
    --model gpt2 --tok_plus " he" --tok_minus " she" \
    --top_svs 4 --alphas 3 10 --n_angles 12 \
    --null_mode top4 --null_seeds 3 --out_prefix gpt2_quickstart_scan
# Outputs: helix_usage_validated/gpt2_quickstart_scan_{scan.txt,heatmap.png}

# 4. Nine-test falsification battery on the default head.
.venv/bin/python investigate_helix_usage_validated.py --test all-must-have gpt2
# Outputs: helix_usage_validated/workshop_suite_results.json
#          + per-test artifacts

# 5. Routed CrowS-Pairs at calibrated alpha (Table 7 of paper).
.venv/bin/python experiments/crowspairs_routed_eval.py \
    --model gpt2 \
    --alpha_json helix_usage_validated/per_domain_alpha_gpt2.json
# Outputs: helix_usage_validated/crowspairs_routed_gpt2_calib.{csv,txt}
```

If all five run cleanly, your environment reproduces the paper.

---

## 2. Math core

The compass is five formulas. Once you know these, the rest is plumbing.

### 2.1 The compass plane

For an attention head `(L, H)`:
```
W_OV = W_V @ W_O   ∈ R^(d_model × d_model)        # combined OV map
U, Σ, V^T = svd(W_OV)                              # full SVD
```
A **compass plane** is two columns of `U` (or, equivalently for the
implementation, two rows of `V^T`):
```
plane(L, H, i, j) = span(u_i, u_j),  i ≠ j
```
The standalone scripts use **`Vt[d1, :]` and `Vt[d2, :]`** (rows of
`V^T`); the class methods sometimes use `U[:, d_i]`. Both are write-side
directions because of how `W_OV` factorizes.

**Source of truth:** `experiments/compass_causal_sweep.py:71-79`
(the `Vt[d1]` / `Vt[d2]` form), `paper_compass/sections/methods.tex` §3.1
(the `(u_i, u_j)` form).

### 2.2 The causal injection

```
v(θ, α) = α · σ_i · cos(θ) · u_i  +  α · σ_j · sin(θ) · u_j         (Eq. 1)
```
`α` is the **scale**, `θ` is the **angle**. `σ_i, σ_j` are the singular
values that pair with the chosen columns.

### 2.3 Where it gets injected

Hook `blocks.{L}.hook_resid_pre`, **last token only**:
```python
def patch_hook(act, hook):
    act[0, -1, :] += v(theta, alpha).to(act.dtype)
    return act
```
The remainder of the forward pass is untouched. This is the only
intervention site in the entire compass pipeline.

**Source of truth:** `experiments/compass_causal_sweep.py:96-109`,
`compass_scan.py:70-80`.

### 2.4 The probe metric

For two antipodal probe tokens `(t+, t−)`:
```
LD(θ, α; p) = logit(t+) − logit(t−)          # at last position
```
Averaged over **3 prompts per condition**: neutral, plus-context,
minus-context.

### 2.5 The sinusoid fit (1st DFT bin)

```
LD(θ) ≈ μ + A · cos(θ − φ)
```
fit by:
```python
mu = y.mean()
c = ((y - mu) * cos(θ)).sum() * 2.0 / N
s = ((y - mu) * sin(θ)).sum() * 2.0 / N
A = hypot(c, s)
phi = degrees(atan2(s, c))
```
**Source of truth:** `experiments/compass_scan.py:54-62` (most-cited
form), `compass_causal_sweep.py:159-163`.

### 2.6 The two-pillar pass criterion

A plane is a compass iff:
```
R²(A vs α through origin) ≥ 0.95          # amplitude linearity
|φ(α_hi) − φ(α_lo)|       ≤ 10°            # phase stability
A(α_hi) / α_hi            ≥ amp_threshold # 0.20 default; 0.08 for Gemma
```
**Source of truth:** `experiments/compass_scan.py:247-250`. Threshold
defaults at `compass_scan.py:145-147`.

These six items are the **entire mathematical content** of the paper.
Everything below is algorithms, evaluation, and bookkeeping built on
top of them.

---

## 3. File map

The pipeline has eight conceptual stages (L0–L8). Every script in the
repo lives in exactly one.

### L0 — Math + hooks (shared infrastructure)

| File | Lines | Role |
|---|---:|---|
| `experiments/compass_causal_sweep.py` (math fragment) | 71–109 | Reference impl of Eq. 1 + the hook |
| `experiments/compass_scan.py` (math fragment) | 54–121 | Same Eq. 1 + sinusoid fit + pass criterion |

These are the source of truth for the math; everything else calls or
inlines these formulas.

### L1 — Discovery (find candidate compasses)

| File | Lines | Role |
|---|---:|---|
| **L1a Decode-driven** | | |
| `experiments/compass_dictionary.py` | 159 | Decode top-K SVD axes through mean-centered `W_U`; multi-model |
| `experiments/compass_dictionary_single.py` | 136 | Single-model variant; consumed by `run_llama_suite.sh` |
| **L1b Blind causal scan (gender)** | | |
| `experiments/compass_scan.py` | 345 | Scans every `(L, H, top-4 SV-pair)`; two null modes |
| `experiments/run_llama_suite.sh` | — | End-to-end Llama scan + downstream |
| `experiments/run_gemma_parity.sh` | — | Gemma scan with `AMP_THRESH=0.08` |
| **L1c Multi-domain probe scan** | | |
| `experiments/stereoset_probe_extract.py` (+ `_phi3/_gemma/_llama`) | 187 | Mine single-token (stereo, anti) probe pairs from StereoSet |
| `experiments/stereoset_probe_scan.py` (+ `_phi3/_gemma/_llama`) | 349 | Decode pre-filter + causal sweep over survivors |
| **L1d Routing analysis (post-scan)** | | |
| `experiments/analyze_head_domain_routing.py` | 176 | Per-head specialization index; cross-model verdict |
| `experiments/scan_ci_summary.py` | 96 | Agresti-Coull 95% CIs from scan logs |
| **L1e Coherence sweep (workshop-only)** | | |
| `investigate_helix_usage_validated.py` `decode_coherence_sweep` (line 2520), `decode_head_compasses` (line 2736) | — | Pair-scoring by decode coherence; not used in main paper figures |

### L2 — Validation (α-sweep on a single head)

| File | Lines | Role |
|---|---:|---|
| `experiments/compass_causal_sweep.py` | 287 | Production driver: 36 angles × {1,3,10}; 3 plots + log |
| `investigate_helix_usage_validated.py` `validate_concept_compass` (line 1771) | — | Class-method equivalent (chainable from battery) |
| `investigate_helix_usage_validated.py` `causal_compass_patch` (line 2919) | — | In-plane rotation Δ ∈ {±180, ±90, 0}; cheaper sanity check |

### L3 — Falsification (nine-test battery)

| Test | Class method (line) | Standalone | CLI flag |
|---|---|---|---|
| 1. decode-heads | `subspace_vocab_projection` (622) | `compass_dictionary.py` | `--projection-only` / `--test decode-heads` |
| 2. cyclicity | `cyclicity_check` (2840), `cyclicity_all_heads` (2706) | — | `--test cyclicity` / `--test cyclicity-all` |
| 3. permutation null | `compass_permutation_test` (2296) | — | `--test permutation` |
| 4. random-plane null | `compass_random_plane_baseline` (2230) | (also baked into `compass_scan.py:220-244`) | `--test random-plane` |
| 5. principal-angles | `principal_angles_between_heads` (2789) | `l10h9_tests.py`, `phi3_l28h1_tests.py` | `--test principal-angles` |
| 6. scan-planes | `scan_good_random_planes` (2377) | `compass_scan.py` | `--test scan-planes` |
| 7. semantic-ablate | `semantic_task_ablation` (3075) | — | `--test semantic-ablate` |
| 8. causal-patch | `downstream_patch_decode` (2605), `causal_compass_patch` (2919) | — | `--test downstream-patch` / `--test causal-patch` |
| 9. self-test | `run_self_tests` (3240) | — | `--self-test` / `--test hook-check` |

Driver: `run_workshop_suite` (line 3361), CLI `--test all-must-have`.
Runs tests **4, 3, 5, 2, 8, 7** in that order (random-plane, permutation,
principal-angles, cyclicity, causal-patch, semantic-ablate). Skips
tests 1, 6, 9 — run those separately.

### L4 — Geometry (storage vs steering, harmonics, plane comparisons)

| File | Lines | Role |
|---|---:|---|
| `experiments/head_ablation_comparison.py` | 241 | Plane ablation for storage vs. steering classification |
| `experiments/second_harmonic_fit.py` | 124 | Post-hoc 2-harmonic fit on saved `.txt` sweeps |
| `experiments/fourway_probe.py` | 181 | Fresh 4-token probe with full 2-harmonic fit |
| `experiments/l10h9_tests.py` | 230 | GPT-2 L10H9 vs L9H7 — principal angles + decode + α-sweep |
| `experiments/phi3_l28h1_tests.py` | 117 | Phi-3 L28H1 vs L24H10 — principal angles + decode |
| `experiments/introspect_compass_scale.py` | 97 | Cross-model SNR introspection (`||resid_pre||`, σ₁, α-budget) |

### L5 — Downstream (does the compass actually steer behavior?)

| File | Lines | Role |
|---|---:|---|
| **L5a WinoGender** | | |
| `experiments/winogender_eval.py` | 413 | raw_gap + stereo_corr + ΔPPL; baseline / ActAdd / compass |
| `experiments/winogender_sweep.py` | 497 | α-sweep over {0.25, 0.5, 1.0, 1.5, 2.0} |
| **L5b StereoSet** | | |
| `experiments/stereoset_eval.py` | 337 | Single-head LMS/SS/ICAT |
| `experiments/stereoset_ensemble_eval.py` (+ `_phi3/_gemma/_llama`) | 323 | K=4 ensemble |
| **L5c CrowS-Pairs** | | |
| `experiments/crowspairs_eval.py` | 322 | Global K=4 ensemble (one alpha for all domains) |
| `experiments/crowspairs_routed_eval.py` | 374 | Per-domain routing — the headline downstream experiment |
| **L5d Prior baselines** | | |
| `experiments/inlp_debias.py` | 350 | Iterative null-space projection (Ravfogel 2020) |
| `experiments/sentence_debias.py` | 322 | PCA-of-class-centered (Liang 2020) |
| `experiments/debias_method_comparison.py` | 223 | Aggregator: routed+calib vs INLP vs SentDebias |
| **L5e Matched-strength** | | |
| `experiments/target_ss50_sweep.py` | 169 | Sweep multiplier M, pick M* minimizing \|SS−50\| |
| **L5f Capability** | | |
| `experiments/truthfulqa_eval.py` | 277 | TruthfulQA MC1 at the operating α |
| **L5g Steering demo** | | |
| `experiments/steering_demo.py` | 250 | True in-plane rotation (vs additive); single-prompt |

### L6 — Calibration + multi-seed robustness

| File | Lines | Role |
|---|---:|---|
| `experiments/calibrate_per_domain_alpha.py` | 207 | Compute `α_d = τ / mean(snr_h)` per domain |
| `experiments/seed_variance_summary.py` | 123 | Aggregate seed-{0,1,2,3} routed runs → mean ± std |
| `experiments/bootstrap_ci.py` | 219 | 10k bootstrap + Wilson CIs across all bias CSVs |

### L7 — Plot/figure generation (post-hoc)

| File | Lines | Role |
|---|---:|---|
| `experiments/generate_paper_plots.py` | 482 | Build figures cited in `paper_compass/figures/` |
| `experiments/make_paper_figures.py` | 348 | Alt figure builder |
| `experiments/generate_missing_plots.py` | 402 | Fill figure gaps |
| `experiments/pick_appendix_examples.py` | 163 | Qualitative examples → `appendix_examples.{md,csv}` |
| `experiments/qualitative_spotcheck.py` | 333 | Generation samples; uses `per_domain_alpha_*_snr*.json` |

### L8 — Production drivers (shell)

| File | Role |
|---|---|
| `experiments/run_llama_suite.sh` | End-to-end Llama-3.2-3B: scan → top head → WinoGender → StereoSet → dict |
| `experiments/run_gemma_parity.sh` | Gemma full bias suite at AMP_THRESH=0.08 |
| `experiments/run_stereoset_queue.sh` | Phi-3 then Llama: probe-extract → probe-scan → ensemble-eval |
| `experiments/run_stereoset_queue_after_gemma.sh` | Same, gated on Gemma completion |
| `experiments/run_routed_queue.sh` | Routed CrowS-Pairs across models (no calibration) |
| `experiments/run_calib_queue.sh` | Calibrated routed CrowS-Pairs (consumes `per_domain_alpha_*.json`) |
| `experiments/run_transfer_queue.sh` | Stage A: CrowS-Pairs at two αs; Stage B: TruthfulQA MC1 |
| `experiments/run_phi3_snr_sweep.sh` | Phi-3 SNR sweep over τ ∈ {0.07, 0.08, 0.10} |
| `experiments/run_llama_snr_sweep.sh` | Llama SNR sweep over τ ∈ {0.10, 0.15, 0.20} |

---

## 4. Run recipes

Each recipe is the minimum command set to reproduce one paper claim.

### 4.1 The α-sweep table (Table 1 of paper, `tab:gender`)

For each row of the table, run one invocation of `compass_causal_sweep.py`:

```bash
# GPT-2 L9H7 (1, 2)
.venv/bin/python experiments/compass_causal_sweep.py \
    --model gpt2 --layer 9 --head 7 --dims 1 2 \
    --tok_plus " he" --tok_minus " she" \
    --prompt_neutral "The person said that" \
    --prompt_plus "The man laced up his boots because" \
    --prompt_minus "The woman laced up her boots because" \
    --out_prefix gpt2

# GPT-2 L10H9 (0, 3)
.venv/bin/python experiments/compass_causal_sweep.py \
    --model gpt2 --layer 10 --head 9 --dims 0 3 \
    [same probe + prompts] --out_prefix gpt2_l10h9

# Phi-3 Mini L24H10 (0, 1)
.venv/bin/python experiments/compass_causal_sweep.py \
    --model microsoft/Phi-3-mini-4k-instruct \
    --layer 24 --head 10 --dims 0 1 \
    [same probe + prompts] --out_prefix phi3

# Phi-3 Mini L28H1 (0, 3)
[same, --layer 28 --head 1 --dims 0 3, --out_prefix phi3_l28h1_causal]

# Gemma-2-2B L21H4 (0, 2)
.venv/bin/python experiments/compass_causal_sweep.py \
    --model google/gemma-2-2b --layer 21 --head 4 --dims 0 2 \
    --alphas 5 10 20 \                                      # bigger α budget
    [same probe + prompts] --out_prefix gemma

# Llama-3.2-3B L26H14 (2, 3)
.venv/bin/python experiments/compass_causal_sweep.py \
    --model meta-llama/Llama-3.2-3B --layer 26 --head 14 --dims 2 3 \
    --alphas 5 10 20 \
    [same probe + prompts] --out_prefix llama32_3b
```

The full prompt list for each condition is in `paper_compass/sections/appendix.tex`
`app:prompts`. Three prompts per condition; same prompts for all four
gender models.

For non-pronoun compasses (Phi-3 only):

```bash
# Phi-3 temporal (month vs hour)
.venv/bin/python experiments/compass_causal_sweep.py \
    --model microsoft/Phi-3-mini-4k-instruct \
    --layer 24 --head 17 --dims 1 2 \
    --tok_plus " month" --tok_minus " hour" \
    --prompt_neutral "The event took about one" \
    --prompt_plus "The rent is due every" \
    --prompt_minus "The meeting lasts exactly one" \
    [+ 2 more prompts per condition from app:prompts] \
    --out_prefix phi3_temporal

# Phi-3 entity (item vs operation)
[--layer 24 --head 28 --dims 3 7, item/operation, prompts from app:prompts]
```

### 4.2 The blind scan heatmap (Fig. 3, `tab:scan`)

```bash
# GPT-2: 864 planes, 12 layers × 12 heads × C(4,2) = 6 SV-pairs
.venv/bin/python experiments/compass_scan.py \
    --model gpt2 --tok_plus " he" --tok_minus " she" \
    --top_svs 4 --alphas 3 10 --n_angles 12 \
    --null_mode top4 --null_seeds 3 \
    --out_prefix gpt2_scan_gender

# Same with full_ov null
.venv/bin/python experiments/compass_scan.py \
    --model gpt2 ... --null_mode full_ov \
    --out_prefix gpt2_scan_gender_nullfullov

# Phi-3: 1344 planes
.venv/bin/python experiments/compass_scan.py \
    --model microsoft/Phi-3-mini-4k-instruct \
    --top_svs 4 --alphas 3 10 \
    --null_mode top4 --null_seeds 3 \
    --out_prefix phi3_scan_gender

# Gemma: 240 planes — IMPORTANT: lower amp threshold
.venv/bin/python experiments/compass_scan.py \
    --model google/gemma-2-2b \
    --amp_thresh 0.08 \                                     # 0.20 default; Gemma needs lower
    --top_svs 4 --alphas 3 10 \
    --null_mode top4 --null_seeds 3 \
    --out_prefix gemma_scan_gender

# Llama: 4032 planes — long-running (~hours)
.venv/bin/python experiments/compass_scan.py \
    --model meta-llama/Llama-3.2-3B \
    --top_svs 4 --alphas 3 10 \
    --null_mode top4 --null_seeds 3 \
    --out_prefix llama32_3b_scan_gender
# OR use the wrapped version that picks the top head and chains downstream:
HF_TOKEN=hf_xxx bash experiments/run_llama_suite.sh

# Confidence intervals on pass rates (Agresti-Coull)
.venv/bin/python experiments/scan_ci_summary.py \
    helix_usage_validated/{gpt2,phi3,gemma,llama32_3b}_scan_gender_scan.txt
```

### 4.3 Nine-test falsification battery

```bash
# Run all 6 main battery tests in sequence
.venv/bin/python investigate_helix_usage_validated.py --test all-must-have gpt2

# Or run individual tests
.venv/bin/python investigate_helix_usage_validated.py --test random-plane gpt2
.venv/bin/python investigate_helix_usage_validated.py --test permutation gpt2
.venv/bin/python investigate_helix_usage_validated.py --test principal-angles gpt2
.venv/bin/python investigate_helix_usage_validated.py --test cyclicity gpt2
.venv/bin/python investigate_helix_usage_validated.py --test causal-patch gpt2
.venv/bin/python investigate_helix_usage_validated.py --test semantic-ablate gpt2
.venv/bin/python investigate_helix_usage_validated.py --test scan-planes gpt2
.venv/bin/python investigate_helix_usage_validated.py --test decode-heads gpt2
.venv/bin/python investigate_helix_usage_validated.py --self-test gpt2

# Override the default head per model
.venv/bin/python investigate_helix_usage_validated.py \
    --test all-must-have --layer 10 --head 9 --dims 0 3 gpt2
```

For Phi-3 / Gemma / Llama, replace the model arg:
- `microsoft/Phi-3-mini-4k-instruct`
- `google/gemma-2-2b`
- `meta-llama/Llama-3.2-3B`

### 4.4 Storage vs. steering (head ablation, `tab:headshare`)

```bash
.venv/bin/python experiments/head_ablation_comparison.py \
    --model gpt2 --layer 9 --head 7 --dims 1 2 \
    --tok_plus " he" --tok_minus " she" \
    --prompt_plus "The man laced up his boots because" \
    --prompt_plus "The father waved to the crowd and" \
    --prompt_plus "The king announced that" \
    --prompt_minus "The woman laced up her boots because" \
    --prompt_minus "The mother waved to the crowd and" \
    --prompt_minus "The queen announced that" \
    --other_layers 8 10 \
    --out_prefix head_ablation_gpt2_l9h7

# Repeat for L10H9 to compare storage (L10H9) vs steering (L9H7)
.venv/bin/python experiments/head_ablation_comparison.py \
    --model gpt2 --layer 10 --head 9 --dims 0 3 ...
```

### 4.5 Two-harmonic fit (Phi-3 4-way vs 2-way; `tab:harmonic`)

```bash
# Re-fit on saved sweep .txt files (post-hoc, no model load)
.venv/bin/python experiments/second_harmonic_fit.py
# Reads phi3_compass_causal.txt, phi3_temporal_compass_causal.txt,
#       phi3_entity_compass_causal.txt
# Writes helix_usage_validated/second_harmonic_fit.txt

# Or fresh 4-token probe (Phi-3 temporal example)
.venv/bin/python experiments/fourway_probe.py \
    --model microsoft/Phi-3-mini-4k-instruct \
    --layer 24 --head 17 --dims 1 2 \
    --tokens month upon hour year \
    --prompt "The event took about one " \
    --prompt "They waited nearly one " \
    --prompt "The project finished within one " \
    --out_prefix phi3_temporal_fourway
```

### 4.6 Two-head subspace tests (`sec:phi3_two`)

```bash
# GPT-2: principal angles between L10H9 and L9H7
.venv/bin/python experiments/l10h9_tests.py

# Phi-3: principal angles between L28H1 and L24H10
.venv/bin/python experiments/phi3_l28h1_tests.py
```

### 4.7 Cross-model α/SNR introspection (informs `tab:alpha_calib`)

```bash
.venv/bin/python experiments/introspect_compass_scale.py
# Prints per-(model, head) SNR table; informs why GPT-2 uses α=1
# but Gemma needs α=20+
```

### 4.8 Multi-domain probe scan (for routed CrowS-Pairs)

Two-stage pipeline. Per-model variants exist because tokenizers differ.

```bash
# Stage 1: extract single-token probe pairs (one-time)
.venv/bin/python experiments/stereoset_probe_extract.py        # GPT-2 default
.venv/bin/python experiments/stereoset_probe_extract_phi3.py
.venv/bin/python experiments/stereoset_probe_extract_gemma.py
.venv/bin/python experiments/stereoset_probe_extract_llama.py

# Stage 2: scan top-4 OV-SVD planes against extracted probes
.venv/bin/python experiments/stereoset_probe_scan.py
.venv/bin/python experiments/stereoset_probe_scan_phi3.py
.venv/bin/python experiments/stereoset_probe_scan_gemma.py
.venv/bin/python experiments/stereoset_probe_scan_llama.py

# Or use the wrapped queue
bash experiments/run_stereoset_queue.sh

# Routing analysis
.venv/bin/python experiments/analyze_head_domain_routing.py
```

### 4.9 Per-domain α calibration

```bash
.venv/bin/python experiments/calibrate_per_domain_alpha.py --model gpt2
.venv/bin/python experiments/calibrate_per_domain_alpha.py --model phi3
.venv/bin/python experiments/calibrate_per_domain_alpha.py --model gemma
.venv/bin/python experiments/calibrate_per_domain_alpha.py --model llama
# Outputs: helix_usage_validated/per_domain_alpha_<model>[_snr<TAU>].json
```

### 4.10 Routed CrowS-Pairs at calibrated α (`tab:crowspairs`)

```bash
# Single-shot:
bash experiments/run_calib_queue.sh
# Internally invokes:
#   crowspairs_routed_eval.py --model gpt2  --alpha_json per_domain_alpha_gpt2.json
#   crowspairs_routed_eval.py --model phi3  --alpha_json per_domain_alpha_phi3_snr0.08.json
#   crowspairs_routed_eval.py --model llama --alpha_json per_domain_alpha_llama_snr0.10.json
# Gemma is NOT in this queue (it has only 1 passing head, nothing to route).
# For Gemma, use:
.venv/bin/python experiments/crowspairs_routed_eval.py --model gemma \
    --alpha_json helix_usage_validated/per_domain_alpha_gemma.json

# Multi-seed runs
.venv/bin/python experiments/crowspairs_routed_eval.py \
    --model phi3 --seed 1 \
    --alpha_json helix_usage_validated/per_domain_alpha_phi3_snr0.08.json
# Repeat for seed=2, 3
.venv/bin/python experiments/seed_variance_summary.py
# → helix_usage_validated/seed_variance_summary.{csv,md}
```

### 4.11 Prior debiasing baselines

```bash
# INLP (Ravfogel 2020)
.venv/bin/python experiments/inlp_debias.py --model gpt2 --domain gender --layer 10
.venv/bin/python experiments/inlp_debias.py --model gpt2 --domain race   --layer 10
.venv/bin/python experiments/inlp_debias.py --model phi3 --domain gender --layer 24
.venv/bin/python experiments/inlp_debias.py --model gemma --domain gender --layer 21
.venv/bin/python experiments/inlp_debias.py --model llama --domain gender --layer 22
# (and --domain race for each)

# SentenceDebias (Liang 2020)
.venv/bin/python experiments/sentence_debias.py --model gpt2  --domain gender --layer 10
# (per-(model, domain) same as above)

# Aggregate INLP + SentDebias + routed+calib + matched into one table
.venv/bin/python experiments/debias_method_comparison.py
# → helix_usage_validated/debias_method_comparison.{csv,md}
```

### 4.12 Matched-strength (iso-SS=50) sweep

```bash
.venv/bin/python experiments/target_ss50_sweep.py --model phi3
.venv/bin/python experiments/target_ss50_sweep.py --model gemma
# → helix_usage_validated/target_ss50_<model>.{csv,txt}
```

### 4.13 TruthfulQA capability check (`tab:truthfulqa`)

```bash
.venv/bin/python experiments/truthfulqa_eval.py --model gpt2  --alphas 1.5
.venv/bin/python experiments/truthfulqa_eval.py --model phi3  --alphas 10.0
.venv/bin/python experiments/truthfulqa_eval.py --model gemma --alphas 20.0
.venv/bin/python experiments/truthfulqa_eval.py --model llama --alphas 20.0

# Or wrapped:
bash experiments/run_transfer_queue.sh    # CrowS-Pairs + TruthfulQA chained
```

### 4.14 Single-prompt steering demo (`tab:steering`)

```bash
.venv/bin/python experiments/steering_demo.py \
    --model google/gemma-2-2b --layer 21 --head 4 --dims 0 2 \
    --theta_he 180 --theta_she 0 \
    --boost 3.0 \
    --out_prefix steering_gemma
```

### 4.15 Bootstrap CIs across all bias eval CSVs

```bash
.venv/bin/python experiments/bootstrap_ci.py
# Reads every helix_usage_validated/*.csv with a known metric column
# Writes helix_usage_validated/bootstrap_ci_summary.{csv,md}
```

---

## 5. Artifact catalog

Every output filename pattern in `helix_usage_validated/`, keyed by
producer.

### From `compass_dictionary.py`

| Pattern | Content |
|---|---|
| `compass_dict_<short>.txt` | Per-model decode log (top-K SVD axes × ±poles × top-15 tokens) |
| `compass_dict_all.md` | Combined paper-ready markdown table |
| `compass_dict_run.log` | Driver log |

`<short>` ∈ {`gpt2`, `phi3`, `gemma`, `llama32_3b`}.

### From `compass_scan.py`

| Pattern | Content |
|---|---|
| `<prefix>_scan.txt` | Per-plane scores + null table; full ranked listing |
| `<prefix>_heatmap.png` | (L, H) heatmap of passing planes |

`<prefix>` ∈ {`gpt2_scan_gender`, `phi3_scan_gender`, `gemma_scan_gender`,
`llama32_3b_scan_gender`} for the canonical runs, plus `_nullfullov` and
`_calib` variants.

### From `compass_causal_sweep.py`

| Pattern | Content |
|---|---|
| `<prefix>.txt` | Per-(α, θ) LD table for neutral / plus / minus |
| `<prefix>_curves.png` | Three LD-vs-θ panels (one per α) |
| `<prefix>_polar.png` | Polar dial at the middle α with ± sectors |
| `<prefix>_linearity.png` | Fit amplitude A vs α |

Canonical `<prefix>` (paper figures): `gpt2_compass_causal`, `phi3_compass_causal`,
`gemma2b_compass_causal`, `phi3_temporal_compass_causal`, `phi3_entity_compass_causal`,
`phi3_l28h1_causal`. Sanity / robustness variants: `*_fixed_v{1,2,3}`,
`*_fixed_fp32`, `*_incontext`, `*_ref_check`.

### From `stereoset_probe_extract*.py`

| Pattern | Content |
|---|---|
| `stereoset_probe_pairs.tsv` | (stereo, anti) pairs both side-tokenize to one token |
| `stereoset_probe_halfmatch.tsv` | One-side-only matches (manual review) |

### From `stereoset_probe_scan*.py`

| Pattern | Content |
|---|---|
| `stereoset_scan_<tag>.jsonl` | Per (L, H, plane, probe, domain) row |
| `stereoset_scan_<tag>_summary.txt` | Aggregated head pass-counts |

`<tag>` ∈ {`gpt2`, `phi3`, `gemma`, `llama`}.

### From `analyze_head_domain_routing.py`

| Pattern | Content |
|---|---|
| `head_domain_heatmap_<tag>.png` | Per-model (L, H) × domain heatmap |
| `head_domain_routing.md` | Cross-model specialization-index table + verdict |

### From `compass_dictionary.py` ➝ paper figure provenance

`compass_dict_all.md` is the source for the decode tables in
`paper_compass/sections/results.tex` §`sec:gender` and the appendix.

### From `calibrate_per_domain_alpha.py`

| Pattern | Content |
|---|---|
| `per_domain_alpha_gpt2.json` | GPT-2 calibrated α per domain (target SNR baked in JSON) |
| `per_domain_alpha_gemma.json` | Gemma calibrated α |
| `per_domain_alpha_phi3_snr<TAU>.json` | Phi-3, one file per τ ∈ {0.07, 0.08, 0.10} |
| `per_domain_alpha_llama_snr<TAU>.json` | Llama, one file per τ ∈ {0.10, 0.15, 0.20} |

**Canonical SNR target per model:** GPT-2 = 0.20, Gemma = 0.08, Phi-3 =
0.08, Llama = 0.10. See `paper_compass/sections/appendix.tex` Table 9.

### From `crowspairs_routed_eval.py`

| Pattern | Content |
|---|---|
| `crowspairs_routed_<model>.csv` | Grid of α (no calibration); seed=0 |
| `crowspairs_routed_<model>_calib.csv` | Calibrated α from JSON; seed=0 (GPT-2, Gemma) |
| `crowspairs_routed_<model>_calib_seed<N>.csv` | Same with non-zero seed (head-selection variance) |
| `crowspairs_routed_<model>_snr<TAU>.csv` | SNR-sweep variants (Phi-3, Llama) |

**The "seed-0 baseline" filename is non-uniform across models.** See
`experiments/seed_variance_summary.py:25-29`:
```
phi3:  crowspairs_routed_phi3_snr0.08.csv
gemma: crowspairs_routed_gemma_calib.csv
llama: crowspairs_routed_llama_snr0.10.csv
gpt2:  crowspairs_routed_gpt2_calib.csv
```

### From `inlp_debias.py` / `sentence_debias.py`

| Pattern | Content |
|---|---|
| `inlp_<model>_<domain>_L<layer>.{json,pt}` | Trained projector |
| `crowspairs_inlp_<model>_<domain>_L<layer>.{csv,txt}` | INLP-conditioned CrowS-Pairs eval |
| `sentdebias_<model>_<domain>_L<layer>.{json,pt}` | Trained projector |
| `crowspairs_sentdebias_<model>_<domain>_L<layer>.{csv,txt}` | SentDebias eval |

### From `winogender_*.py`

| Pattern | Content |
|---|---|
| `winogender_<model>_<setup>.csv` | Per-occupation logit-gap table |
| `winogender_<tag>_sweep.csv` | α-sweep variant |

### From `stereoset_eval.py` and `stereoset_ensemble_eval*.py`

| Pattern | Content |
|---|---|
| `stereoset_<tag>.{csv,txt}` | LMS / SS / ICAT |
| `stereoset_ensemble_<model>.{csv,txt}` | K=4 ensemble variant |

### From `truthfulqa_eval.py`

| Pattern | Content |
|---|---|
| `truthfulqa_<model>.{csv,txt}` | MC1 accuracy under routed ensemble |

### From post-hoc aggregators

| Pattern | Producer | Content |
|---|---|---|
| `bootstrap_ci_summary.{csv,md}` | `bootstrap_ci.py` | 10k bootstrap + Wilson CIs |
| `seed_variance_summary.{csv,md}` | `seed_variance_summary.py` | Mean ± std across seeds |
| `flip_ratio_by_domain.{csv,md}` | (manual / queries) | Per-(model, domain) flip-anti vs flip-stereo + χ² |
| `debias_method_comparison.{csv,md}` | `debias_method_comparison.py` | Cross-method head-to-head table |
| `target_ss50_<model>.{csv,txt}` | `target_ss50_sweep.py` | Multiplier sweep with M* selection |
| `scan_pass_rate_ci.txt` | `scan_ci_summary.py` | Agresti-Coull CIs from scan logs |
| `appendix_examples.{csv,md}` | `pick_appendix_examples.py` | Qualitative steering examples |
| `qualitative_spotcheck_<model>.md` | `qualitative_spotcheck.py` | Generation samples at operating α |
| `workshop_suite_results.json` | `run_workshop_suite` | Combined battery results |

### Top-head pickers

| Pattern | Producer | Content |
|---|---|---|
| `<tag>_top_head.json` | `run_llama_suite.sh` (inline Python) | Best (L, H, d1, d2) by amp_slope |

---

## 6. Paper ↔ code mapping

Every figure and table in `paper_compass/sections/`, mapped to the
script that produces it and the artifact it lands in.

### Tables

| Paper label | Section | Producer script | Artifact |
|---|---|---|---|
| `tab:directions` | results §3.1 | `src/models/masked_transformer_circuit.py` (Beyond Components method) | (Stage-1 mask training output) |
| `tab:bc_full` | appendix `app:bc_results` | Stage-1 sweep + `experiments/evaluation/comprehensive_metrics_table.py` | (Stage-1 GP eval) |
| `tab:fem_calib` | appendix `app:fem_calib` | Stage-2 sweep on L10H9 fem-pole | (Stage-2 calibration log) |
| `tab:gender` | results §`sec:gender` | `experiments/compass_causal_sweep.py` ×6 | `<prefix>.txt` per row |
| `tab:nonpronoun` | results §`sec:other` | `compass_causal_sweep.py` (temporal, entity) | `phi3_{temporal,entity}_compass_causal.txt` |
| `tab:scan` | results §`sec:scan` | `compass_scan.py` ×4 + `scan_ci_summary.py` | `*_scan.txt` + `scan_pass_rate_ci.txt` |
| `tab:nulls` | results §3.3 | `investigate_helix_usage_validated.py --test all-must-have` | `workshop_suite_results.json` |
| `tab:harmonic` | appendix `app:harmonic` | `second_harmonic_fit.py` | `second_harmonic_fit.txt` |
| `tab:headshare` | results §`sec:headshare` | `head_ablation_comparison.py` ×4 | `head_ablation_<tag>.txt` |
| `tab:baselines` | appendix `app:baselines` | (5-method eval at unit-norm injection) | (run logs in `helix_usage_validated/`) |
| `tab:winogender` | downstream §`sec:winogender` | `winogender_eval.py` ×4, `winogender_sweep.py` | `winogender_*.csv` |
| `tab:stereoset` | downstream §`sec:stereoset` | `stereoset_eval.py` + `stereoset_ensemble_eval_*.py` | `stereoset_*.csv` |
| `tab:crowspairs` | downstream §`sec:crowspairs` | `crowspairs_routed_eval.py` + `inlp_debias.py` + `sentence_debias.py` + `target_ss50_sweep.py` | `crowspairs_*_calib*.csv` + `crowspairs_inlp_*.csv` + `crowspairs_sentdebias_*.csv` + `target_ss50_*.csv`; aggregated by `debias_method_comparison.py` |
| `tab:truthfulqa` | downstream §`sec:truthfulqa` | `truthfulqa_eval.py` ×4 | `truthfulqa_*.csv` |
| `tab:steering` | downstream §`sec:steering` | `steering_demo.py` ×3 | `steering_<model>.txt` |
| `tab:alpha_calib` | appendix `app:alpha_calib` | `calibrate_per_domain_alpha.py` ×4 | `per_domain_alpha_*.json` |
| `tab:snr_sweep` | appendix `app:alpha_calib` | `run_phi3_snr_sweep.sh` + `run_llama_snr_sweep.sh` | `crowspairs_routed_*_snr*.csv` |
| `tab:routing_spec` | appendix `app:routing` | `analyze_head_domain_routing.py` | `head_domain_routing.md` |
| `tab:iso_ss50` | appendix `app:iso_ss50` | `target_ss50_sweep.py` | `target_ss50_*.csv` |

### Figures

| Paper figure | Section | Producer | Source artifact |
|---|---|---|---|
| `fig:mask_progression` | results §3.1 | (Stage-1 mask training) | Beyond-Components figures |
| `fig:bc_sweep` | results §3.3 | (Stage-1 sweep) | `bc/causal_sweep_curves.png`, `bc/sigma_amplification.png` |
| `fig:gender_polar` | results §`sec:gender` | `compass_causal_sweep.py:_plot_polar` | `<prefix>_polar.png` |
| `fig:gender_linearity` (appendix) | appendix `app:linearity` | `compass_causal_sweep.py:_plot_linearity` | `<prefix>_linearity.png` |
| `fig:other_compasses` | results §`sec:other` | `compass_causal_sweep.py` (Phi-3 temporal/entity) | `phi3_{temporal,entity}_polar.png`, `_linearity.png` |
| `fig:scan_heatmap` | results §`sec:scan` | `compass_scan.py:main` (heatmap) | `gpt2_scan_heatmap.png` |
| `fig:llama_scan_heatmap` | results §`sec:scan` | `compass_scan.py` (Llama) | `llama32_3b_scan_gender_heatmap.png` |
| `fig:gpt2_curves` (appendix) | appendix `app:curves` | `compass_causal_sweep.py:_plot_curves` | `<prefix>_curves.png` |
| `fig:transfer` | downstream §`sec:transfer` | (downstream sweep figures) | `bc/bias_score_curves.png`, `bc/flip_rates.png` |
| `fig:calib` | downstream §`sec:calib` | (calibration figures) | `bc/top2_bend_comparison.png`, `bc/endpoint_tradeoffs.png` |

### Sections without dedicated tables/figures

| Section | What's there | Producer |
|---|---|---|
| `sec:phi3_two` | Principal-angle numbers in prose | `phi3_l28h1_tests.py`, `l10h9_tests.py` |
| `sec:downstream` overview | Cross-cutting commentary | (no single producer; aggregates the above) |
| `sec:discussion` | Limitations | (none) |
| `sec:related` | Citations | (none) |

---

## 7. Canonical configurations

### Per-model compass heads (paper §3.4 + Table 1)

| Model | HF name | Compass head(s) | (d_i, d_j) | (σ_i, σ_j) | Discovered by |
|---|---|---|---|---|---|
| GPT-2 Small | `gpt2` | L9H7 (steering) | (1, 2) | (8.87, 8.46) | decode (L1a) |
| GPT-2 Small | `gpt2` | L10H9 (storage) | (0, 3) | (9.15, 6.80) | scan (L1b) |
| Phi-3 Mini | `microsoft/Phi-3-mini-4k-instruct` | L24H10 (steering) | (0, 1) | (9.42, 7.70) | decode (L1a) |
| Phi-3 Mini | same | L28H1 (storage) | (0, 3) | (13.67, 7.61) | scan (L1b) |
| Phi-3 Mini | same | L24H17 (temporal) | (1, 2) | — | decode |
| Phi-3 Mini | same | L24H28 (entity) | (3, 7) | — | decode |
| Gemma-2-2B | `google/gemma-2-2b` | L21H4 (storage) | (0, 2) | (2.35, 1.98) | decode |
| Llama-3.2-3B | `meta-llama/Llama-3.2-3B` | L26H14 | (2, 3) | (1.14, 0.97) | scan only |

### Per-model α budgets (`introspect_compass_scale.py`)

| Model | Working α range | Compass α-sweep | Calibrated routed α (canonical) |
|---|---|---|---|
| GPT-2 | {0.5, 1.0, 1.5} | {1, 3, 10} | gender=1.89, race=1.80, prof=1.37, religion=1.87 |
| Phi-3 | {0.5, 1.0, 1.5} | {1, 3, 10} | gender=5.10, race=5.15, prof=3.80, religion=5.34 |
| Gemma | {5, 10, 20} | {1, 3, 10} (or with `--alphas 5 10 20`) | gender=55.26, race=81.36, prof=29.73, religion=31.19 |
| Llama | {5, 10, 20} | {1, 3, 10} (or with `--alphas 5 10 20`) | gender=4.06, race=3.56, prof=3.45, religion=4.66 |

Working α refers to the WinoGender/StereoSet operating points
(`introspect_compass_scale.py:94-97`). Compass α-sweep is the validation
sweep. Calibrated α is for the routed CrowS-Pairs (Table 9).

### Per-model INLP / SentenceDebias layer (Table 7)

| Model | Layer (L*) |
|---|---|
| GPT-2 | 10 |
| Phi-3 | 24 |
| Gemma | 21 |
| Llama | 22 |

### Threshold conventions

| Threshold | Default | Override | Reason |
|---|---|---|---|
| `lin_thresh` (R²) | 0.95 | — | Two-pillar test |
| `phase_thresh` (Δφ) | 10° | — | Two-pillar test |
| `amp_thresh` (slope) | 0.20 | **0.08 for Gemma** (`run_gemma_parity.sh`) | Gemma's residual is normalized more aggressively after pre-norm; same plane writes at smaller magnitude |

---

## 8. Known gotchas

These are real repo-state facts that bit the implementation. None of
them invalidate the paper, but a new collaborator will hit them.

### 8.1 `run_calib_queue.sh` JSON filename mismatch (FIXED)

Previously the script referenced `per_domain_alpha_phi3.json` and
`per_domain_alpha_llama.json`, neither of which exists. Fixed in this
cookbook's commit to point at the canonical SNR-tagged files
(`per_domain_alpha_phi3_snr0.08.json`, `per_domain_alpha_llama_snr0.10.json`).
GPT-2 and Gemma JSONs are SNR-untagged because their calibration was run
once at a single τ. Phi-3 and Llama have full SNR families because
`run_phi3_snr_sweep.sh` / `run_llama_snr_sweep.sh` regenerate per-τ files.

### 8.2 Llama tag inconsistency

Output filenames use **two different tags** for Llama-3.2-3B:
- Canonical scan: `llama32_3b_scan_gender_*.{txt,png}`
- Routing analysis (StereoSet domains): `<tag>_*` where `<tag> = "llama"`
- Sanity / pre-merge files: `llama_scan_gender_heatmap.png`,
  `llama_sanity_{heatmap.png,scan.txt}` (1-plane debug runs)

When in doubt:
- Gender scan + Llama suite → `llama32_3b_*`
- Multi-domain (CrowS, StereoSet) → `llama_*`
- Anything saying "sanity" → discard

### 8.3 Seed-0 baseline filename is non-uniform

`seed_variance_summary.py:25-29` knows this:
```python
SEED0_FILENAME = {
    "phi3":  "crowspairs_routed_phi3_snr0.08.csv",
    "gemma": "crowspairs_routed_gemma_calib.csv",
    "llama": "crowspairs_routed_llama_snr0.10.csv",
    "gpt2":  "crowspairs_routed_gpt2_calib.csv",
}
```
Don't try to predict the seed-0 filename pattern from the seed-{1,2,3}
pattern (`crowspairs_routed_<model>_calib_seed<N>.csv`). Look it up.

### 8.4 Gemma uses a different `amp_thresh`

Default `amp_thresh = 0.20` (`compass_scan.py:147`). Gemma uses
**`amp_thresh = 0.08`** because its OV map writes at a uniformly smaller
magnitude. The `run_gemma_parity.sh` driver bakes this in. If you scan
Gemma with the default 0.20, you get 0 passing planes.

This is documented in `paper_compass/sections/methods.tex:106-118`. Not a
heuristic; a true property of how Gemma normalizes after pre-norm.

### 8.5 `investigate_helix_usage_validated.py` is two papers in one file

The 3,770-line class file mixes:
- **Compass methods** (lines 1771+): `validate_concept_compass`,
  `compass_random_plane_baseline`, `compass_permutation_test`,
  `scan_good_random_planes`, `decode_coherence_sweep`,
  `decode_head_compasses`, `principal_angles_between_heads`,
  `cyclicity_check`, `cyclicity_all_heads`, `causal_compass_patch`,
  `downstream_patch_decode`, `semantic_task_ablation`, `run_self_tests`,
  `run_workshop_suite`.
- **Arithmetic-paper methods** (lines 288–1664): `analyze_ov_helix`,
  `analyze_residual_helix`, `test_fourier_isolation`,
  `causal_phase_shift_test`, `_verify_number_token_ordering`,
  `mlp_translation_lens`, `trace_neurons_to_vocab`,
  `causal_mlp_ablation`.

`run_validated_investigation` runs the **arithmetic** pipeline (Tests 1–6
in lines 3422–3475 are all helix). For compass work, use
`run_workshop_suite` or `--test <name>`.

### 8.6 `run_workshop_suite` test order

Order in lines 3372–3384 is:
1. `compass_random_plane_baseline` (test #4)
2. `compass_permutation_test` (test #3)
3. `principal_angles_between_heads` (test #5)
4. `cyclicity_check` (test #2)
5. `causal_compass_patch` (test #8)
6. `semantic_task_ablation` (test #7)

It **skips** test #1 (decode-heads), test #6 (scan-planes), and test #9
(self-test). Run those separately or use the subset CLI flags.

### 8.7 `phi3_scan_gender_l29_30_*` exists but isn't in the paper

`helix_usage_validated/phi3_scan_gender_l29_30_{scan.txt,heatmap.png}`
is an extended scan over Phi-3 layers 29 and 30 (beyond the paper's
focus on L24 and L28). It's not cited. Decide before next paper revision:
either fold the negative result into a footnote (strengthens the
"concentration" claim) or delete the files.

### 8.8 Plot variants from sanity sweeps

`gpt2_fixed_v1_*`, `phi3_fixed_v{1,2,3}_*`, `phi3_fixed_fp32_*` are
robustness sanity sweeps (alternate prompts, fp32 vs bf16). Not cited
in the paper. Either fold a sentence into the appendix
("Verified under fp32 and across three prompt sets; raw outputs in
`helix_usage_validated/`") or delete.

### 8.9 Stage-1 directions are from Beyond Components

`paper_compass/sections/results.tex` §3.1 reproduces Beyond Components'
learnable-mask method on the Gender Pronoun task. The compass paper's
**novel** Stage-1 contribution is the scalar steering + σ×2.0
amplification (Tables 4-fem-calib, Figure bc_sweep), not the discovery.
Citation to Ahmad et al. is currently missing — flagged for the next
paper revision.

---

## 9. What is NOT in the implementation

To set honest expectations:

- **No per-token injection.** All causal injections target the last
  token only.
- **No multi-head simultaneous α-sweep for fitting amplitude/phase.**
  Ensembles exist for downstream eval, not for the validation step.
- **No inter-layer compass tracking** (asking whether the same plane
  carries the same dial across multiple layers within one model).
- **No paraphrase robustness eval.** Acknowledged as a limitation in
  `paper_compass/sections/discussion.tex`.
- **No unit tests for the compass pipeline.** The `tests/` directory is
  arithmetic-paper tests. Compass code uses output-file diffing
  for regression detection.
- **No version-pinned environment.** `requirements.txt` exists but the
  paper's appendix doesn't pin commit hashes or pip versions.
- **No master "regenerate all paper figures" driver.** The closest are
  `run_llama_suite.sh` (one model, end-to-end) and the `run_*_queue.sh`
  family (one stage, multi-model). Combining them into a one-command
  reproducer is open work.

---

## 10. Extending

### 10.1 Adding a new model

1. Add to `MODEL_SPECS` dict in:
   - `experiments/crowspairs_routed_eval.py:56`
   - `experiments/inlp_debias.py:59`
   - `experiments/sentence_debias.py:60`
   - `experiments/calibrate_per_domain_alpha.py:47`
   - `experiments/truthfulqa_eval.py` (similar dict)
2. Run discovery: `compass_dictionary.py` (decode) + `compass_scan.py`
   (blind). Pick top-pass head by amp_slope.
3. Add to `MODEL_HEADS` in `compass_dictionary.py:27`.
4. Add to `primary_head_spec` in `introspect_compass_scale.py:27`.
5. Run α-sweep on the chosen head:
   `compass_causal_sweep.py --model <hf> --layer L --head H --dims i j ...`
6. If passing, add to Table 1 and Figure 2 of the paper.

### 10.2 Adding a new bias domain

1. Extract probe pairs:
   - Add domain to `DOMAINS` in `stereoset_probe_extract.py:31`.
   - Run `stereoset_probe_extract*.py` (per model).
2. Scan probes:
   - Run `stereoset_probe_scan*.py` to populate scan JSONL with new domain.
3. Routed CrowS-Pairs:
   - Add domain to `CROWS_DOMAINS` in `crowspairs_routed_eval.py:50`.
   - Add to `pick_heads_per_domain` domain list (line 88).
4. Calibrate:
   - `calibrate_per_domain_alpha.py` will pick up the new domain
     automatically from the JSONL.
5. Add row to Table 7.

### 10.3 Adding a new compass to an existing model

1. Identify candidate via `compass_scan.py` (filter by `domain == X`)
   or `compass_dictionary.py` (manual decode inspection).
2. Run α-sweep:
   `compass_causal_sweep.py --model <hf> --layer L --head H --dims i j --tok_plus + --tok_minus -`
3. Run two-harmonic fit if you suspect 4-way structure: `fourway_probe.py`
4. Add row to Table 2 (`tab:nonpronoun`) or as a new model row in
   Table 1.
5. Run battery: `investigate_helix_usage_validated.py --test all-must-have`
   with `--layer/--head/--dims` overrides.

### 10.4 Adding a new prior baseline (e.g. CDA, Bolukbasi)

1. Mirror the structure of `inlp_debias.py`:
   - Train on StereoSet intrasentence labels at `L*`.
   - Save projector to `<method>_<model>_<domain>_L<layer>.{json,pt}`.
   - Eval on CrowS-Pairs split.
2. Output: `crowspairs_<method>_<model>_<domain>_L<layer>.{csv,txt}`.
3. Add column to `debias_method_comparison.py` aggregator.
4. Add column to Table 7.

---

## Appendix: One-line command index

For when you've already read this once and just need the right command.

```bash
# DISCOVERY
compass_dictionary.py                                               # decode
compass_scan.py --model X --tok_plus T+ --tok_minus T-              # blind
stereoset_probe_extract*.py && stereoset_probe_scan*.py             # multi-domain

# VALIDATION
compass_causal_sweep.py --model --layer --head --dims               # α-sweep

# FALSIFICATION
investigate_helix_usage_validated.py --test all-must-have <model>   # 6-of-9 battery
investigate_helix_usage_validated.py --test <name> <model>          # single test

# GEOMETRY
head_ablation_comparison.py                                         # storage vs. steering
second_harmonic_fit.py                                              # post-hoc 2-harmonic
fourway_probe.py                                                    # fresh 4-token probe
introspect_compass_scale.py                                         # cross-model SNR table
l10h9_tests.py / phi3_l28h1_tests.py                               # two-head subspace

# DOWNSTREAM
winogender_eval.py / stereoset_eval.py                              # bench
crowspairs_routed_eval.py --alpha_json ...                          # CrowS-Pairs routed
inlp_debias.py / sentence_debias.py                                 # baselines
truthfulqa_eval.py                                                  # capability
steering_demo.py                                                    # single-prompt demo
target_ss50_sweep.py                                                # matched-strength

# CALIBRATION
calibrate_per_domain_alpha.py                                       # produce α JSON
seed_variance_summary.py                                            # ±std across seeds
bootstrap_ci.py                                                     # 10k bootstrap

# DRIVERS
bash experiments/run_llama_suite.sh                                 # end-to-end, one model
bash experiments/run_calib_queue.sh                                 # routed, all models
bash experiments/run_transfer_queue.sh                              # CrowS + TQA
```
