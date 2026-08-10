# Arithmetic Circuit Discovery — Execution Pipeline

## Overview
Complete pipeline for discovering and validating the Fourier arithmetic circuit in a new transformer language model. Covers encoding characterization, causal validation, component attribution, computation mechanism verification, and generalization testing.

**Models tested so far**: Gemma 2B, Phi-3 Mini, LLaMA 3.2-3B
**All scripts live in**: `experiments/`
**All results go to**: `mathematical_toolkit_results/`

---

# QUICK REFERENCE — Adding a New Model

### Step 0: Register the Model

Add your model to `MODEL_MAP` in `experiments/arithmetic_circuit_scan_updated.py`:
```python
MODEL_MAP = {
    "phi-3":       "microsoft/Phi-3-mini-4k-instruct",
    "gemma-2b":    "google/gemma-2-2b",
    "llama-3b":    "meta-llama/Llama-3.2-3B",
    "your-model":  "org/model-name",   # ← ADD HERE
}
```

Also add default layer configs in scripts that use them (e.g., `eigenvector_dft.py`):
```python
readout_defaults = {"gemma-2b": 25, "phi-3": 31, "llama-3b": 27, "your-model": XX}
comp_defaults    = {"gemma-2b": 19, "phi-3": 26, "llama-3b": 20, "your-model": YY}
```

### Step 0b: Determine Teacher-Forced vs Direct-Answer Mode

- **Teacher-forced** (default): Prompt = `"Calculate 13 + 8 = 2"`, model predicts next token `1`. Works when single-digit answer tokens exist (0-9).
- **Direct-answer**: Prompt = `"a + b = "`, model predicts full answer as one token. Required if model tokenizes numbers 0-198 as single tokens (e.g., LLaMA 3.2-3B).

Test: run a few prompts manually. If model gets >90% on `"a + b = "` format → use `--direct-answer`. Add `--direct-answer` flag to ALL subsequent commands.

### Key Layer Parameters

You need two layer indices:
- **comp-layer**: Where the main arithmetic computation happens (identified by Step 1)
- **readout-layer**: Last layer before output (usually `n_layers - 1`)

These are auto-detected for known models, but must be specified via `--comp-layer` for new models after Step 1.

---

# FULL PIPELINE — 14 Steps in Execution Order

---

## ═══════════════════════════════════════════════════════
## PHASE A: DISCOVERY — Find the Circuit
## ═══════════════════════════════════════════════════════

### Step 1: Layer Scan + Unembed Patching (Find comp-layer and readout-layer)

**Script**: `experiments/arithmetic_circuit_scan_updated.py`
**Purpose**: Identify which layers carry arithmetic information and at what dimensionality
**THIS MUST RUN FIRST** — determines comp-layer for all subsequent experiments

```bash
python experiments/arithmetic_circuit_scan_updated.py \
    --model your-model --device mps --n-per-digit 100 --n-test 150
# For direct-answer models:
# --direct-answer
```

**Key functions**:
- `run_layer_scan()` → sweeps all layers, patches activations between digit-pairs, measures transfer rate
- `compute_unembed_basis()` / `compute_unembed_basis_direct_answer()` → SVD of W_U digit columns → 9D unembed-aligned basis
- `run_patching_experiment()` → patches only a subspace (unembed, Fisher, or random) at each layer
- `compute_fisher_matrix()` → standard Fisher information matrix from gradients
- `compute_contrastive_fisher()` → per-digit-class Fisher for digit-discriminative directions
- `filter_correct_teacher_forced()` / `filter_correct_direct_answer()` → keep only problems the model solves correctly

**Output**: `mathematical_toolkit_results/arithmetic_scan_<model>.json`
**What to look for**:
- Layer scan: find the layer where transfer rate first exceeds 80% → this is your **comp-layer**
- The last layer with ~100% transfer → your **readout-layer**
- Unembed patching: 9D should capture most transfer at readout (100% for teacher-forced)
- Fisher patching: effective dimensionality of the arithmetic subspace

**Runtime**: ~30-60 min per model on MPS

---

### Step 2: Eigenvector DFT (Verify Fourier Structure)

**Script**: `experiments/eigenvector_dft.py`
**Purpose**: Check if the digit encoding at comp-layer and readout-layer forms a **perfect Fourier basis** of ℤ/10ℤ

```bash
python experiments/eigenvector_dft.py \
    --model your-model --comp-layer YY --device mps
```

**Key functions**:
- `analyze_layer()` → DFT of each SVD direction's 10-element digit score vector
- `collect_per_digit_means()` → per-digit mean activations at a layer
- Checks W_U SVD, computation layer SVD, and readout layer SVD

**Output**: `mathematical_toolkit_results/eigenvector_dft_<model>.json`
**What to look for**:
- "PERFECT FOURIER BASIS" = each frequency k=1..4 gets exactly 2 directions, k=5 gets 1 (total 9)
- Mean purity > 50% → strong Fourier structure
- Dominant frequency assignments: which k dominates (k=5 = parity, k=1 = ordinal)

**Critical gate**: If NOT a perfect Fourier basis → the model may not use Fourier encoding. Check data balance and try more samples before concluding.

**Runtime**: ~15-30 min

---

### Step 3: Fourier Layer Sweep (Track Encoding Across Layers)

**Script**: `experiments/fourier_decomposition.py`
**Purpose**: Decompose the digit subspace into Fourier components at every layer, track how Fourier energy builds up

```bash
python experiments/fourier_decomposition.py \
    --model your-model --layer-sweep "5,6,7,...,N" --device mps
```

**Key functions**:
- `build_fourier_basis_functions()` → canonical DFT basis for ℤ/10ℤ (9 functions)
- `fourier_decomposition()` → project digit-conditional means onto Fourier basis, measure per-frequency energy
- `per_neuron_fourier_analysis()` → per-MLP-neuron frequency tuning and purity
- `run_fourier_at_layer()` → full analysis at one layer including optional patching

**Output**: `mathematical_toolkit_results/fourier_decomposition_<model>_L<range>.json`
**What to look for**:
- Fourier energy should build up from early layers to comp-layer
- Per-frequency energy profile: which frequencies dominate at which layers
- CRT score (Chinese Remainder Theorem alignment)

**Runtime**: ~1-2 hours for full sweep

---

## ═══════════════════════════════════════════════════════
## PHASE B: CAUSAL VALIDATION — Prove the Circuit Matters
## ═══════════════════════════════════════════════════════

### Step 4: Causal Fourier Knockout (Necessity Test)

**Script**: `experiments/fourier_knockout.py`
**Purpose**: Zero out the 9D Fourier subspace at each layer → measure accuracy damage. Proves the subspace is **causally necessary**.

```bash
python experiments/fourier_knockout.py \
    --model your-model --comp-layer YY --device mps
```

**Key functions**:
- `evaluate_accuracy()` → hook-based ablation at a single layer, measures accuracy
- `evaluate_accuracy_multi_layer()` → simultaneous ablation across multiple layers
- `make_random_orthonormal_basis()` → random 9D control (should cause 0% damage)
- Also runs per-frequency ablation (k=1..5 individually) and progressive ablation (top-1..9 directions)

**Output**: `mathematical_toolkit_results/fourier_knockout_<model>.json`
**What to look for**:
- Full 9D ablation should cause significant accuracy damage (11-46% observed)
- Random 9D ablation should cause ~0% damage (MUST be near zero)
- Multi-layer ablation (comp→readout) should approach chance level (~10%)
- Individual frequency ablation reveals redundancy structure

**Runtime**: ~30-60 min

---

### Step 5: Fisher/PCA Phase Shift (Sufficiency Test)

**Script**: `experiments/fisher_phase_shift.py`
**Purpose**: Rotate activations within the Fisher/PCA subspace by digit-shift amounts. Tests if the subspace is **sufficient** to control digit output.

```bash
python experiments/fisher_phase_shift.py \
    --model your-model --layers "YY,ZZ" --device cpu
# NOTE: Requires gradients → use CPU (MPS gradient bugs)
```

**Key functions**:
- `compute_fisher_eigenvectors()` → Fisher information matrix from gradients, returns top eigenvectors
- `compute_pca_directions()` → PCA/SVD control directions from digit-conditional means
- `run_causal_intervention_suite()` → scale-sweep of Fisher/PCA subspace interventions
- `run_phase_shift_experiment()` → rotate within Fisher subspace, measure digit shift

**Output**: `mathematical_toolkit_results/fisher_phase_shift_<model>.json`
**What to look for**:
- Fisher 9D knockout transfer rate: how much arithmetic info is in Fisher directions
- Phase rotation: does rotating within the subspace shift the predicted digit?

**Runtime**: ~1-2 hours (CPU, gradients)

---

### Step 6: Fourier Phase Rotation (Steering Test)

**Script**: `experiments/fourier_phase_rotation.py`
**Purpose**: Apply coherent Fourier rotation (shift all frequencies by j positions) and test if the output digit shifts by j.

```bash
python experiments/fourier_phase_rotation.py \
    --model your-model --layers "YY,ZZ" --device mps --logit-lens
```

**Key functions**:
- `compute_digit_fourier_basis()` → hybrid SVD+DFT basis construction (best method)
- `compute_rotation_delta()` → rotation matrix for shift j in Fourier space
- `run_fourier_phase_shift()` → apply rotation, measure exact_mod10, target_rank, Δlogit
- `run_logit_lens_analysis()` → project rotated activations through LN_final + W_U to check logit-space effect (enabled by `--logit-lens`)
- `sanity_check_basis()` → 6 automated checks on basis quality (orthonormality, freq purity, etc.)

**Output**: `mathematical_toolkit_results/fourier_phase_rotation_<model>.json`
**What to look for**:
- exact_mod10 > 10% (chance) means rotation works
- Large j (backward shifts) often work better than small j (forward)
- Logit-lens accuracy at comp-layer tells how "ready" the answer is at that layer

**Runtime**: ~20-40 min

---

### Step 7: Steering Improvements (W_U-Informed Steering)

**Script**: `experiments/steering_improvements.py`
**Purpose**: Bridge the encoding-readout gap with W_U-informed steering vectors

```bash
python experiments/steering_improvements.py \
    --model your-model --layer YY --device mps \
    --scales 1,2,3,5,10 --wu-scales 1,3,5,10,20
```

**Key functions**:
- `compute_wu_projector()` → project Fourier basis onto W_U column space
- `compute_wu_steering_vectors()` → ideal direction = (w_target - w_orig) projected onto Fourier subspace
- Three methods: `coherent_xN`, `wu_proj_xN`, `wu_steer_xN`

**Output**: `mathematical_toolkit_results/steering_improvements_<model>.log`
**What to look for**:
- wu_steer should dramatically improve exact_mod10 (16→70% for Gemma, 14→84% for Phi-3)
- If plain coherent already works well (>60%), wu_steer may not add much (LLaMA case)

**Runtime**: ~20-30 min

---

## ═══════════════════════════════════════════════════════
## PHASE C: COMPONENT ATTRIBUTION — Who Does What
## ═══════════════════════════════════════════════════════

### Step 8: Fourier Head Attribution (Component-Level Analysis)

**Script**: `experiments/fourier_head_attribution.py`
**Purpose**: For every attention head and MLP in the model, measure (1) how much Fourier power it writes, (2) whether it's causally necessary, (3) which frequencies it handles.

```bash
python experiments/fourier_head_attribution.py \
    --model your-model --comp-layer YY --device mps
```

**Key functions (3 phases)**:
- **Phase 1** — `compute_writing_scores()`: Direct Linear Attribution (DLA) — measures signed + unsigned Fourier writing for each component
- **Phase 2** — `causal_patch_top_components()`: Ablates Fourier content from each top component, measures accuracy damage
- **Phase 3** — `frequency_resolved_attribution()`: Decomposes each component's Fourier writing into per-frequency contributions (k=1..5)

**Sanity checks built in**:
- S1: Residual stream decomposition (components sum to total)
- S3: Max writing fractions (no single component dominates unreasonably)
- S4: Random subspace baseline (DLA ≈ 9/d_model)
- S5: DLA-damage correlation

**Output**: `mathematical_toolkit_results/fourier_head_attribution_<model>.json`
**What to look for**:
- Which components are causally necessary (damage > 0% when ablated)
- MLP-dominated vs mixed circuit (Gemma/LLaMA = pure MLP, Phi-3 = mixed with L21H30)
- Per-frequency specialization (e.g., MLP at comp-layer writes k=5 parity)

**Runtime**: ~1-2 hours

---

### Step 9: Neuron Trigonometric Analysis (Per-Neuron Decomposition)

**Script**: `experiments/neuron_trig_analysis.py`
**Purpose**: DFT analysis of individual residual stream dimensions and MLP neurons to identify frequency-tuned neurons.

```bash
python experiments/neuron_trig_analysis.py \
    --model your-model --layer YY --device mps
```

**Key functions (5 parts)**:
- **Part 1** — `collect_digit_means()` + `compute_dimension_dft()`: DFT of each residual stream dimension
- **Part 2** — `collect_mlp_neuron_means()` + `report_mlp_neuron_spectra()`: DFT of each MLP neuron (post-nonlinearity)
- **Part 3** — `analyze_phase_clustering()`: Circular statistics on phase angles (Rayleigh test)
- **Part 4** — `compute_component_attribution()`: Fourier power written by each head/MLP
- **Part 5** — `sparse_steering_test()`: Steering restricted to top-K dimensions by Fourier power

**Output**: `mathematical_toolkit_results/neuron_trig_<model>_L<layer>.log`
**What to look for**:
- How many neurons are high-purity (>80%) at each frequency
- Phase clustering: diffuse (distributed encoding) vs clustered (localized encoding)
- Sparse steering: how few dimensions are needed for effective rotation

**Runtime**: ~30-60 min

---

## ═══════════════════════════════════════════════════════
## PHASE D: COMPUTATION MECHANISM — How Addition Works
## ═══════════════════════════════════════════════════════

### Step 10: CP Tensor Decomposition (Trig Identity Verification)

**Script**: `experiments/cp_tensor_decomposition.py`
**Purpose**: Verify that the model's computation implements the trigonometric addition identity: cos(k(a+b)) = cos(ka)cos(kb) - sin(ka)sin(kb)

```bash
python experiments/cp_tensor_decomposition.py \
    --model your-model --comp-layer YY --device mps
```

**Key functions**:
- `build_fourier_matrices()` → theoretical cos⊗cos, sin⊗sin outer-product matrices for each frequency
- `fit_trig_identity()` → fit constrained trig model to each 10×10 activation matrix (→ trig_score)
- `cp_decompose_and_analyze()` → CP decomposition of the (10,10,9) activation tensor
- `validate_on_synthetic()` → validate algorithm on known trig tensor first (MUST PASS)
- `build_ones_digit_tensor()` → construct (10,10,9) tensor from per-(a,b) activations projected onto 9D basis
- `collect_activations_and_filter()` → collect activations for all (a,b) pairs at comp-layer

**Output**: `mathematical_toolkit_results/cp_tensor_<model>.json`
**What to look for**:
- σ²-weighted trig score > 0.8 → strong angle addition structure
- Anti-diagonal R² ≈ 1.0 for signal tensor (activations depend on (a+b)%10 only)
- CP rank-9 fit quality and Fourier factor matching
- Trig constraint satisfaction: CC ≈ -SS, SC ≈ CS

**Runtime**: ~30-60 min

---

### Step 11: Carry Stratification (Carry vs No-Carry Analysis)

**Script**: `experiments/carry_stratification.py`
**Purpose**: Split problems into carry (a+b≥10) and no-carry (a+b<9), analyze if the Fourier structure differs.

```bash
python experiments/carry_stratification.py \
    --model your-model --comp-layer YY --device mps
```

**Key functions**:
- `compute_basis_from_means()` → 9D Fourier basis from per-digit means (for each carry group)
- `analyze_group_dft()` → DFT analysis on SVD directions, frequency assignment, purity
- `principal_angles()` → principal angles between carry and no-carry subspaces
- `analyze_carry_directions()` → carry direction decomposition in Fourier subspace, per-digit projections

**Output**: `mathematical_toolkit_results/carry_stratification_<model>.json`
**What to look for**:
- Subspace alignment between carry and no-carry bases (high = shared encoding)
- Whether carry-conditioned bases are still perfect Fourier
- k=5 frequency energy difference between carry and no-carry

**Runtime**: ~15-30 min

---

## ═══════════════════════════════════════════════════════
## PHASE E: GENERALIZATION & VISUALIZATION
## ═══════════════════════════════════════════════════════

### Step 12: Generalization Tests (Subtraction, Substitution, Multi-Digit)

**Script**: `experiments/generalization_tests.py`
**Purpose**: Test if the Fourier basis transfers to related tasks

```bash
python experiments/generalization_tests.py \
    --model your-model --comp-layer YY --device mps --test all
```

**Key functions**:
- `run_substitution_test()` → swap 9D Fourier projection between problems with different answer digits, check if output shifts to donor digit
- `run_subtraction_test()` → compute Fourier basis from subtraction problems, measure alignment with addition basis
- `run_multidigit_test()` → check if tens-digit also has Fourier structure at comp-layer

**Output**: `mathematical_toolkit_results/generalization_tests_<model>.log`
**What to look for**:
- Substitution transfer > 10% (chance) → Fourier subspace encodes digit identity
- Subtraction alignment with addition basis → shared vs separate circuits

**Runtime**: ~30-60 min

---

### Step 13: UMAP Visualization

**Script**: `experiments/fourier_umap.py`
**Purpose**: 2D/3D UMAP of 9D Fourier projections colored by digit class → visual confirmation of circular structure

```bash
python experiments/fourier_umap.py \
    --model your-model --comp-layer YY --device mps
```

**Output**: `mathematical_toolkit_results/fourier_umap_<model>.png`
**What to look for**: Clean circular arrangement of digits 0-9 in UMAP space

**Runtime**: ~10-20 min

---

### Step 14 (Advanced): Multi-Layer Frequency Ablation

**Script**: `experiments/multilayer_freq_ablation.py`
**Purpose**: Ablate individual frequencies across multiple layers simultaneously — resolves the "individual frequency paradox" (single-layer k=5 ablation causes 0% damage due to redundancy)

```bash
python experiments/multilayer_freq_ablation.py \
    --model your-model --comp-layer YY --readout-layer ZZ --device mps
```

**Output**: `mathematical_toolkit_results/multilayer_freq_ablation_<model>.json`
**What to look for**:
- Multi-layer k=5 ablation should cause significant damage (unlike single-layer)
- Reveals which frequency channels are most critical when redundancy is blocked

**Runtime**: ~30-60 min

---

## ═══════════════════════════════════════════════════════
## PHASE F: MULTI-DIGIT CIRCUIT (Gemma-specific, extensible)
## ═══════════════════════════════════════════════════════

### Step 15 (Optional): Multi-Digit Circuit Discovery

**Script**: `experiments/multidigit_circuit.py`
**Purpose**: Analyze how the model handles tens-digit computation, carry routing, and operand decomposition

```bash
python experiments/multidigit_circuit.py \
    --model your-model --device mps --test A D C B F G H
```

**Sub-experiments** (selectable via `--test`):
- **A**: Carry-conditioned tens-digit Fourier basis
- **B**: Operand digit subspace decomposition (a₀, b₀, a₁, b₁ separate bases)
- **C**: Carry router head identification (attention to ones-digit operands)
- **D**: Carry signal causal intervention (add/remove/flip carry direction)
- **F**: Carry router head ablation (causal)
- **G**: Tens-digit-native carry direction sweep
- **H**: End-to-end causal chain validation

**Output**: `mathematical_toolkit_results/multidigit_circuit_<model>.json`
**This is the most complex experiment** — run F, G, H after A-D to build on findings.

**Runtime**: ~2-4 hours for all sub-experiments

---

# EXECUTION ORDER SUMMARY

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE A: DISCOVERY                                         │
│  Step 1:  arithmetic_circuit_scan_updated.py  (MUST BE 1ST) │
│  Step 2:  eigenvector_dft.py                                │
│  Step 3:  fourier_decomposition.py --layer-sweep            │
├─────────────────────────────────────────────────────────────┤
│  PHASE B: CAUSAL VALIDATION                                 │
│  Step 4:  fourier_knockout.py                               │
│  Step 5:  fisher_phase_shift.py          (needs CPU)        │
│  Step 6:  fourier_phase_rotation.py      --logit-lens       │
│  Step 7:  steering_improvements.py                          │
├─────────────────────────────────────────────────────────────┤
│  PHASE C: COMPONENT ATTRIBUTION                             │
│  Step 8:  fourier_head_attribution.py    (longest step)     │
│  Step 9:  neuron_trig_analysis.py                           │
├─────────────────────────────────────────────────────────────┤
│  PHASE D: COMPUTATION MECHANISM                             │
│  Step 10: cp_tensor_decomposition.py                        │
│  Step 11: carry_stratification.py                           │
├─────────────────────────────────────────────────────────────┤
│  PHASE E: GENERALIZATION & VIZ                              │
│  Step 12: generalization_tests.py                           │
│  Step 13: fourier_umap.py                                   │
│  Step 14: multilayer_freq_ablation.py                       │
├─────────────────────────────────────────────────────────────┤
│  PHASE F: MULTI-DIGIT (OPTIONAL)                            │
│  Step 15: multidigit_circuit.py                             │
└─────────────────────────────────────────────────────────────┘
```

### Dependencies Between Steps

```
Step 1 ──→ ALL (provides comp-layer, readout-layer)
Step 2 ──→ Steps 3-14 (confirms Fourier structure exists)
Steps 3-7: Independent of each other (can run in parallel)
Step 8:    Independent (but benefits from Step 2 results)
Step 9:    Independent
Step 10:   Independent (but uses comp-layer from Step 1)
Step 11:   Independent
Steps 12-14: Independent
Step 15:   Benefits from Steps 8 (carry router heads), 4 (knockout)
```

### Parallel Execution Groups (if you have multiple GPUs/terminals)

- **Group 1**: Steps 1 → 2 (sequential, fast)
- **Group 2**: Steps 3, 4, 6, 7, 9, 10, 11, 12, 13 (all on MPS, independent)
- **Group 3**: Steps 5 (CPU-only, gradients)
- **Group 4**: Step 8 (longest, MPS)
- **Group 5**: Step 14, 15 (after confirming basic results)

---

# TOTAL RUNTIME ESTIMATE (Single Model, Sequential)

| Phase | Steps | Time |
|-------|-------|------|
| A: Discovery | 1-3 | ~2-3 hours |
| B: Causal | 4-7 | ~2-4 hours |
| C: Attribution | 8-9 | ~2-3 hours |
| D: Mechanism | 10-11 | ~1-1.5 hours |
| E: Generalization | 12-14 | ~1-2 hours |
| F: Multi-digit | 15 | ~2-4 hours |
| **Total** | | **~10-17 hours** |

---

# RESULTS CHECKLIST — What You Need for a Complete Analysis

For each model, confirm these key results:

- [ ] **Perfect Fourier basis** at comp-layer and readout-layer (Step 2)
- [ ] **Progressive rotation**: transfer rate increases from comp→readout (Step 1)
- [ ] **9D ablation causes damage**, random 9D causes 0% (Step 4)
- [ ] **Multi-layer ablation → near chance** (Step 4)
- [ ] **Fourier rotation steers digits** above chance (Step 6/7)
- [ ] **Causally necessary components identified** (Step 8 Phase 2)
- [ ] **Per-frequency specialization** of key MLPs (Step 8 Phase 3)
- [ ] **Trig identity verified** (σ²-weighted trig score > 0.8) (Step 10)
- [ ] **Carry stratification** shows shared encoding (Step 11)
- [ ] **Cross-model consistency** in circuit architecture (compare with other models)

---

# EXISTING MODEL REFERENCE — Layer Configurations

| Model | HuggingFace ID | comp-layer | readout-layer | Mode | Circuit Type |
|-------|---------------|------------|---------------|------|-------------|
| Gemma 2B | google/gemma-2-2b | 19 | 25 | teacher-forced | Pure MLP |
| Phi-3 Mini | microsoft/Phi-3-mini-4k-instruct | 26 | 31 | teacher-forced | Mixed (L21H30 + MLPs) |
| LLaMA 3.2-3B | meta-llama/Llama-3.2-3B | 20 | 27 | direct-answer | Pure MLP |

---

# SUPPLEMENTARY SCRIPTS — Not in Main Pipeline but Useful

These scripts provide alternative analyses or theoretical validation. They are **not required** for the main pipeline but can be run for deeper investigation.

### S1: Fisher Patching (Standalone)

**Script**: `experiments/fisher_patching.py`
**Purpose**: Focused Fisher subspace patching with contrastive v3 (teacher-forced ones-digit targets). More detailed than the Fisher analysis in Step 1.

```bash
python experiments/fisher_patching.py \
    --model your-model --layers "YY,ZZ" --n-per-digit 100 --device cpu
# --direct-answer for direct-answer models
# --standard-only to skip contrastive Fisher
```

**Key functions**:
- `compute_fisher_eigenvectors()` → standard Fisher eigenvectors
- `compute_contrastive_fisher_v3()` → contrastive Fisher with teacher-forced ones-digit targets (best variant)
- `run_patching_experiment()` → Fisher subspace patching at multiple dimensionalities (2D, 5D, 9D, 10D, 20D, 50D)

**When to use**: After Step 1, to get more detailed Fisher dimensionality analysis at specific layers. Provides contrastive Fisher v3 which is better than the v1 in Step 1.

---

### S2: CRT Sanity Check

**Script**: `experiments/crt_sanity_check.py`
**Purpose**: Validate that the Fisher subspace has freq-2 and freq-5 structure consistent with Chinese Remainder Theorem (mod-2 × mod-5 = mod-10).

```bash
python experiments/crt_sanity_check.py \
    --model your-model --layer YY --device cpu
```

**When to use**: After Step 5, to verify CRT-aware rotation predictions (e.g., rotating in freq-2 plane by 2π/5 should shift digit by +6 mod 10).

---

### S3: Probe Steering v2 (Difference-in-Means)

**Script**: `experiments/probe_steering_v2.py`
**Purpose**: DIM (difference-in-means) steering — simplest possible steering approach using statistical difference between digit class activations.

```bash
python experiments/probe_steering_v2.py \
    --model your-model --layers "YY,ZZ" --device mps
```

**When to use**: As a baseline comparison for Step 7's W_U-informed steering. DIM steering doesn't assume any geometry — purely statistical.

---

### S4: Modular Arithmetic Transformer

**Script**: `experiments/modular_arithmetic.py`
**Purpose**: Train a small transformer on (a + b) mod p, then analyze its Fourier structure. Theoretical reference showing that the trig identity mechanism is the *correct* solution.

```bash
python experiments/modular_arithmetic.py
```

**When to use**: For theoretical grounding — demonstrates the same Fourier/trig mechanism in a controlled setting where it can be verified exactly.

---

### S5: Fourier Circuit Analysis (Old Pipeline)

**Script**: `experiments/analyze_fourier_circuits.py`
**Purpose**: Phase 4a/4b analysis from the older mask-learning pipeline. Performs SVD direction cos/sin fitting and MLP neuron trig identity analysis.

**Prerequisite**: Requires a trained `MaskedTransformerCircuit` checkpoint from the old pipeline. **Not directly usable** with the current pipeline without adaptation.

**Key functions**:
- `fit_cosine_sine()` → fit cos/sin to individual SVD directions
- `analyze_frequency_groups()` → group directions by frequency, check phase coherence
- `analyze_neuron_trig_identity()` → per-neuron product-of-trig analysis
- `collect_mlp_hidden_activations()` → collect MLP hidden states for trig analysis

**When to use**: Only if you have a MaskedTransformerCircuit checkpoint. The current pipeline's Step 9 + Step 10 provide equivalent analyses without the mask-learning dependency.

---

# EXPERIMENT COMPLETION STATUS (as of April 2025)

| Experiment | Gemma 2B | Phi-3 Mini | LLaMA 3.2-3B |
|-----------|----------|-----------|--------------|
| Step 1: Layer Scan + Unembed | ✅ | ✅ | ✅ |
| Step 2: Eigenvector DFT | ✅ | ✅ | ✅ |
| Step 3: Fourier Layer Sweep | ✅ | ✅ | ✅ |
| Step 4: Fourier Knockout | ✅ | ✅ | ✅ |
| Step 5: Fisher/PCA Phase Shift | ✅ | ✅ | ✅ |
| Step 6: Fourier Phase Rotation | ✅ | ✅ | ✅ |
| Step 7: Steering Improvements | ✅ | ✅ | ✅ |
| Step 8: Fourier Head Attribution | ✅ | ✅ | ✅ |
| Step 9: Neuron Trig Analysis | ✅ | ✅ | ✅ |
| Step 10: CP Tensor Decomposition | ✅ | ✅ | ✅ |
| Step 11: Carry Stratification | ✅ | ✅ | ✅ |
| Step 12: Generalization Tests | ✅ | ✅ | ✅ |
| Step 13: UMAP Visualization | ✅ | ✅ | ✅ |
| Step 14: Multi-Layer Freq Ablation | ✅ | — | — |
| Step 15: Multi-Digit Circuit | ✅ | ✅ | ✅ |
| S1: Fisher Patching (standalone) | ✅ | ✅ | ✅ |

---
---

# ARCHIVED: Original Implementation Plan (Historical)

*The original plan below is preserved for historical context. The actual pipeline executed is documented above.*

## Original Goal
Identify the complete mechanistic circuit for integer addition in transformer language models: which components participate, which singular directions carry the computation, and how MLPs transform helical number representations into the answer.

## Original Methodology
Combines three approaches:
1. **Our SVD scan** (online_svd_scanner.py) — bottom-up geometric discovery
2. **Helix paper** (arXiv 2502.00873) — helical representation fitting + activation patching
3. **Beyond Components paper** (arXiv 2511.20273) — SVD directional masking for circuit discovery

## Model Ladder

| Phase | Model | Params | Device | Rationale |
|-------|-------|--------|--------|-----------|
| 0 | GPT-2 Small | 124M | M1 Max | Pipeline validation (replicate IOI results) |
| 1-2 | Pythia-1.4B | 1.4B | M1 Max | Single-token numbers, same family as helix paper's Pythia-6.9B |
| 3-5 | GPT-J (6B) or Pythia-6.9B | 6-7B | Cloud GPU | Full experiment on helix paper's primary model |

**Why Pythia-1.4B over GPT-2 Medium**: Same tokenizer as Pythia-6.9B (helix paper verified helical structure there). Provides a natural scaling ladder. GPT-2 Medium was never tested for helical structure.

---

## What Already Exists (Verified)

| Component | Location | Status |
|-----------|----------|--------|
| Augmented SVD (QK, OV, MLP_in, MLP_out) | `masked_transformer_circuit.py` lines 333-406 | ✅ Done |
| QK asymmetry (no complementary term) | `masked_transformer_circuit.py` lines 543-546 | ✅ Done |
| Selective mask training | `train_masks=['OV']` or `['MLP_in','MLP_out']` | ✅ Done |
| SVD disk caching | `svd_cache/` directory | ✅ Done |
| Training loop (KL + L1) | `masked_transformer_circuit.py` lines 1284-1565 | ✅ Done |
| Clean/corrupted patching (OV) | `masked_transformer_circuit.py` lines 851-882 | ✅ Done |
| Clean/corrupted patching (MLP_in) | `masked_transformer_circuit.py` lines 938-965 | ✅ Done |
| Clean/corrupted patching (MLP_out) | `masked_transformer_circuit.py` lines 980-1011 | ✅ Done |
| Sparsity statistics | `masked_transformer_circuit.py` lines 1596-1739 | ✅ Done |
| IOI config | `configs/ioi_config.yaml` | ✅ Done |
| IOI data | `data_main/` (CSVs extracted) | ✅ Done |
| Online SVD scanner (Fourier, geometry) | `online_svd_scanner.py` | ✅ Done |
| Training script | `experiments/train.py` | ✅ Done |
| Direction-level intervention | `experiments/ablation/intervention.py` | ✅ Done |

## What Needs to Be Built

| Component | Effort | Phase |
|-----------|--------|-------|
| Arithmetic data generator (clean + corrupted pairs) | Small | 3 |
| Arithmetic config YAML | Small | 3 |
| Arithmetic accuracy adapter in `find_KL_divergence` | Small | 3 |
| `discover_number_geometry()` function | Medium | 1 |
| `identify_arithmetic_circuit()` function | Medium | 2 |
| Geometric interpreter (direction-level Fourier analysis) | Medium | 4a |
| Neuron-level trig identity analyzer | Medium | 4b |
| Direction-level scalar swapping for arithmetic | Medium | 5 |

---

## Phase 0: Validate Existing Pipeline on IOI

**Goal**: Confirm `MaskedTransformerCircuit` + `experiments/train.py` work end-to-end before touching arithmetic.

**Steps**:
1. Run: `python experiments/train.py --config configs/ioi_config.yaml`
2. Verify results match Beyond Components paper benchmarks

**Success criteria**:
- KLD < 0.5
- Relative sparsity > 85%
- Name Mover heads (9.6, 9.9, 10.0) show high mask activations
- S-Inhibition heads (7.3, 7.9, 8.6, 8.10) show high QK mask activations

**Device**: M1 Max — GPT-2 Small (~0.5GB) is trivial.
**Time**: ~2 hours.

**Gate**: Do NOT proceed to Phase 1 until Phase 0 succeeds. Any failures here indicate pipeline bugs that would corrupt all downstream results.

---

## Phase 1: Bottom-Up Fourier Discovery

**Goal**: Discover the natural periodic structure of number representations without assuming T=[2,5,10,100].

**What to build**: A function `discover_number_geometry(model, device)` that:
1. Runs each single-token number `a ∈ [0, N]` through the model
2. Collects `resid_post` after layer 0 at the number token position
3. Fourier decomposes each residual stream dimension as a function of `a`
4. Returns the dominant periods (averaged across dimensions)
5. For each attention head: SVD on W_OV, project number embeddings onto each Vh row, FFT each projection → map `(layer, head, Vh_direction) → period`

**Reuse**: `online_svd_scanner.py` already has `map_svd_to_frequencies` (FFT on SVD dimensions). Adapt for augmented matrices.

**Model**: Pythia-1.4B on M1 Max (~3GB). Single-token range is [0, ~557].

**Success criteria**:
- Clear Fourier peaks at T ∈ {2, 5, 10} (matching helix paper's findings on Pythia-6.9B)
- At least 3 attention heads show linearity > 0.9 when projecting onto specific Vh directions
- If NO clear peaks found: try Pythia-2.8B before concluding the model is too small

**Output**: A table of discovered periods and a map of `(layer, head, Vh_direction) → frequency`.
**Time**: ~1 day.

---

## Phase 2: Circuit Identification via Activation Patching

**Goal**: Find which heads and MLPs participate in arithmetic. Quick pre-screening to scope Phase 3.

**What to build**: A function `identify_arithmetic_circuit(model, device)` that:
1. Generates clean/corrupted pairs:
   - A-corruption: `(a + b =, a' + b =)` where `a ≠ a'`
   - B-corruption: `(a + b =, a + b' =)` where `b ≠ b'`
2. For each component (attention head, MLP), activation-patches from clean into corrupted
3. Measures logit difference recovery for the correct answer token
4. Ranks components by total effect (TE) and direct effect (DE)

**Reuse**: TransformerLens's `model.run_with_cache()` + manual patching. Only use prompts where the model gets the clean answer right.

**Model**: Same Pythia-1.4B as Phase 1.

**Success criteria**:
- Clear separation between high-TE and low-TE components (top ~20 heads account for >80% of total effect)
- Identifiable clustering: early attention heads (layers 5-10) with high indirect effect = **a,b movers**; middle MLPs (layers 10-15) = **builders**; late MLPs (layers 15+) = **readers**
- Results broadly consistent with helix paper's GPT-J findings (a,b heads early, builder MLPs mid, reader MLPs late)

**Output**: Ranked component list, categorized by role. Defines which layers to mask in Phase 3.
**Time**: ~1 day.

---

## Phase 3: Direction-Level Mask Learning on Arithmetic

**Goal**: For each circuit component from Phase 2, find the sparse set of singular directions that carry the arithmetic computation.

**What to build**:
1. **Arithmetic data generator**: produces clean + corrupted tensors in the format `MaskedTransformerCircuit` expects. Output format must match existing data loader interface (input_ids, attention_mask, sequence_lengths, answer_token_ids).
2. **Arithmetic config YAML**: `configs/arithmetic_config.yaml`
3. **Arithmetic accuracy adapter**: In `find_KL_divergence`, replace `indirect_object_index` comparison with `answer_token_id` comparison. The KL divergence itself is already task-agnostic (line 1105).

**Execution — three sequential runs**:

### Run 1: OV masks only
```yaml
masking:
  train_masks: ['OV']
  mask_init_value: 0.99
```
Identifies which OV directions in which heads matter.

### Run 2: MLP masks (builder layers only)
```yaml
masking:
  train_masks: ['MLP_in', 'MLP_out']
  # Only mask layers identified as builders in Phase 2
```
Identifies which MLP directions in builder layers carry the helix transformation.

### Run 3: MLP masks (reader layers only)
Same structure, targeting late-layer MLPs that read helix(a+b) and write to logits.

### Cross-referencing A-corruption vs B-corruption

Run each configuration with BOTH corruption types, then compare:

| High in A-corruption | High in B-corruption | Interpretation |
|---------------------|---------------------|----------------|
| Yes | No | Encodes operand a |
| No | Yes | Encodes operand b |
| Yes | Yes | Encodes the sum (or both) |

**Model**: Pythia-1.4B on M1 Max for prototyping. GPT-J on cloud GPU for the real experiment.

**Success criteria**:
- KLD < 0.5 with relative sparsity > 80% (most directions pruned)
- ~5-15 high-mask directions per circuit component
- Cross-referencing reveals clear operand-vs-sum encoding pattern

**Output**: For each circuit component, a sparse set of important directions labeled by operand encoding.
**Time**: ~2-3 days (3 runs × optimization time).

---

## Phase 4a: SVD Direction Fourier Analysis

**Goal**: Determine what Fourier component each important direction reads from and writes to the residual stream.

**What to build**: A function `interpret_directions(circuit, mask_results, number_embeddings)` that for each high-mask direction from Phase 3:

1. **Left singular vector (U column)**: project number embeddings → FFT → identify what this direction reads
2. **Right singular vector (Vh row)**: project through unembedding → FFT over number tokens → identify the logit receptor pattern
3. **Subspace-level helix fitting (fallback)**: If individual directions don't show clean Fourier signatures:
   - Collect all high-mask directions for a given component
   - Project number embeddings into their joint subspace
   - Fit generalized helix using periods discovered in Phase 1
   - This handles the case where helical structure is distributed across multiple SVD directions

**Reuse**: `online_svd_scanner.py` has `map_svd_to_frequencies`, Fourier analysis infrastructure, and plotting functions.

**Success criteria**:
- At least some individual directions show clear Fourier peaks (T=10 for units digit)
- OR the subspace fallback shows high linearity (>0.9) when fitting a helix
- Late-layer MLP_out directions, projected through unembedding, show periodic vocabulary preferences with T=10

**Output**: A map of `direction → Fourier frequency` for each circuit component.
**Time**: ~1-2 days.

---

## Phase 4b: Neuron-Level Trig Identity Analysis

**Goal**: Determine HOW builder MLPs transform input helices into the output helix. This is the mechanism the helix paper couldn't find.

**Why this is separate from 4a**: The MLP computation is `output = act_fn(input @ W_in + b_in) @ W_out + b_out`. The activation function (GELU) introduces nonlinearity BETWEEN W_in and W_out. SVD directions of W_in and W_out are NOT directly connected — the connection goes through individual neurons. Phase 4a tells you WHAT the MLP reads/writes; Phase 4b tells you HOW.

**What to build**: For each builder MLP identified in Phase 3:

1. **Identify top neurons**: Use attribution patching (gradient-based approximation of activation patching) to find the ~1% of neurons most important for arithmetic (following helix paper Section 5.4).

2. **Analyze neuron preactivations**: For each top neuron n in layer l:
   - Compute `N_n^l(a,b) = x @ W_up[n]` for all `(a,b)` pairs
   - Fourier decompose `N_n^l` as a function of `a`, `b`, and `a+b`
   - Fit: `N_n^l(a,b) ≈ Σ c_{T,t} · trig(2π·t/T)` where `t ∈ {a, b, a+b}` and `T ∈ {discovered periods}`

3. **Check for trig identity**: If neuron n reads `cos(2πa/10)·cos(2πb/10)`, this is evidence of `cos(a+b) = cos(a)cos(b) - sin(a)sin(b)` decomposition.

4. **Analyze neuron outputs**: For each top neuron, check `W_down[:, n]` — what direction does it write to? Does it align with helix(a+b) components?

**Success criteria**:
- Top neurons' preactivations are well-modeled by Fourier components (R² > 0.8)
- Early builder neurons read from `a` and `b` Fourier components
- Late builder neurons read from `a+b` Fourier components
- Evidence of multiplicative interaction (product of trig functions of a and b)

**Output**: A mechanistic account of how builder MLP neurons transform helix(a) + helix(b) → helix(a+b).
**Time**: ~2-3 days.

---

## Phase 5: Causal Validation

**Goal**: Prove the discovered mechanism is causally responsible for the model's arithmetic behavior.

### Primary: Direction-Level Scalar Swapping
For each important direction identified in Phase 3:
- Run clean prompt `a + b =`, record scalar activation `s_clean = input · U_k · σ_k`
- Run corrupted prompt `a' + b =`, record `s_corrupt`
- Swap: patch `s_clean` into corrupted run at direction k only
- Measure logit recovery for correct answer

This is the approach validated by both papers. It does NOT depend on clean circular geometry in live activations.

### Secondary: Subspace Helical Patching
- Fit helix(a) at each layer using Phase 1's discovered periods
- Replace the full helical subspace from clean into corrupted run
- Compare logit recovery with the helix paper's results (Fig. 4)

### Tertiary (Stretch Goal): Angular Rotation
- Only attempt if Phase 4a shows clean circular structure in live activations (linearity > 0.9)
- Use corrected `test_causal_phase_shift` with measured period from Phase 1
- **NOTE**: This approach failed on Phi-3 (linearity=0.13 in live L24 residual). May fail here too.

**Success criteria**:
- Scalar swapping of top-5 directions recovers >50% of logit difference
- Subspace patching approaches the effect of full activation patching
- Combined evidence from Phases 3-5 provides a complete mechanistic story

**Output**: Causal validation metrics, confirming the discovered circuit is functionally responsible.
**Time**: ~1-2 days.

---

## Total Timeline (Original Plan)

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 0 (IOI validation) | 2 hours | 2 hours |
| Phase 1 (Fourier discovery) | 1 day | 1.5 days |
| Phase 2 (Activation patching) | 1 day | 2.5 days |
| Phase 3 (Mask learning) | 2-3 days | 5 days |
| Phase 4a (Direction Fourier) | 1-2 days | 7 days |
| Phase 4b (Neuron trig identity) | 2-3 days | 10 days |
| Phase 5 (Causal validation) | 1-2 days | 12 days |

**Total**: ~12 working days for the full pipeline.

---

## Key Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Pythia-1.4B lacks helical structure | Medium | Fall back to Pythia-2.8B, then jump to cloud GPU for 6.9B |
| MPS gradient bugs during mask optimization | Medium | Run mask optimization on CPU (slower but correct), or move to cloud GPU early |
| Helix is distributed across many SVD directions | High | Subspace-level helix fitting (Phase 4a fallback) |
| Trig identity is not cleanly separable in neurons | Medium | This is a genuine discovery — document as negative result if it doesn't work |
| GPT-J too heavy for Phase 3 on M1 Max | High | Prototype on Pythia-1.4B, then move to cloud for GPT-J |

---

## Files to Create

```
configs/arithmetic_config.yaml          # Phase 3
src/data/arithmetic_data.py             # Phase 3 (data generator)
src/analysis/fourier_discovery.py       # Phase 1
src/analysis/circuit_identification.py  # Phase 2
src/analysis/geometric_interpreter.py   # Phase 4a
src/analysis/neuron_analyzer.py         # Phase 4b
experiments/arithmetic_validation.py    # Phase 5
```

---
---

# Part 2: Circuit Analysis — Completed Work & Next Steps

*Updated: April 2025*

A parallel investigation was conducted using causal activation patching, linear probing,
and direct logit attribution across 3 models. This section documents what was found and
what remains.

---

## Completed Experiments

### Models Tested

| Model | Architecture | Layers | Heads | Baseline Accuracy | Stages Run |
|-------|-------------|--------|-------|-------------------|------------|
| **Phi-3 Mini** | Phi-3 | 32 | 32 | 100% | Stages 1-3 + Exp 1-5 |
| **Gemma 2B** | Gemma | 18 | 8 | 100% | Stages 1-3 + Exp 2-3 |
| **Pythia 1.4B** | GPT-NeoX | 24 | 16 | ~40% | Stages 1-2 (patching only) |

### Universal Findings (All 3 Models)

1. **Three-Phase Circuit**: Route → Crossover → Compute, with proportional layer scaling
2. **MLP-Dominated Computation**: Top-5 MLPs have 0.2-0.6 recovery; no head exceeds 0.21
3. **Near-Deterministic A→+ Copy Head**: Phi-3 L1H19 (0.75), Gemma L10H4 (0.77)
4. **Carry Peaks in Compute Zone**: Linear probes reach 91-97% at compute MLP layers
5. **No Individual MLP Writes the Answer**: Distributed superposition across 5-7 MLPs
6. **Extreme Routing Redundancy**: 20 heads ablated = 0% accuracy drop (Phi-3)
7. **MLP Activations Are Superposed**: PCA separability <0.5; not clusterable

### Results Location

| Directory | Contents |
|-----------|----------|
| `arithmetic_circuit_results/` | Phi-3 Mini results (Stages 1-3, Experiments 1-5) |
| `arithmetic_circuit_results/gemma-2b/` | Gemma 2B results (Stages 1-3, Experiments 2-3) |
| `arithmetic_circuit_results/pythia-1.4b/` | Pythia 1.4B results (Stages 1-2) |
| `experiments/circuit_synthesis.md` | Full synthesis with 3-model comparison tables |

### Codebase

| File | Purpose |
|------|--------|
| `experiments/arithmetic_circuit_discovery.py` | Stages 1-3: Logit Lens, Layer/Component Patching, Ablation |
| `experiments/circuit_analysis.py` | Experiments 1-5: MLP Unembedding, Operand B Hunt, Carry Probe, PCA, Ensemble Patching |
| `tests/test_arithmetic_circuit_discovery.py` | 38 tests for discovery pipeline |
| `tests/test_circuit_analysis.py` | 14 tests for analysis experiments |

---

## Recommended Next Steps

### Tier A: High Impact, Low-to-Moderate Effort

#### Step A1: Causal Knockout of A→+ Copy Head

**Effort**: ~2 hours (30 min code + 90 min runtime)  
**Target venue impact**: Converts strongest correlational claim to causal  

**What to do**:
- Zero out L1H19 (Phi-3) and L10H4 (Gemma) specifically
- Measure: accuracy drop, logit degradation, and whether the crossover zone shifts
- Given extreme routing redundancy, the key question is *what compensates*

**Implementation**:
- Add `run_targeted_head_knockout()` to `experiments/circuit_analysis.py`
- Hook `blocks.{L}.attn.hook_z` and zero dimension H for the target head
- Run 200 problems, compare against baseline
- Also test: knock out top-3 copy heads simultaneously

**Expected outcome**: Single head knockout likely survives (redundancy). Multi-head
knockout may cause degradation. The compensation mechanism itself is interesting.

---

#### Step A2: 3-Digit Operand Scaling

**Effort**: ~1 day  
**Target venue impact**: Addresses strongest reviewer objection ("could be memorized")  

**What to do**:
- Run full Phi-3 + Gemma pipeline with operands 100-999
- Verify prompt format and tokenization work for 3-digit numbers (may be multi-token)
- Compare: do the same MLPs activate? Does the crossover zone shift?
- Run carry probe: does multi-digit carry (e.g., 999+1 with carry chain) require more layers?

**Key test cases**:
- Standard: 123 + 456 = 579 (no carry)
- Single carry: 157 + 268 = 425 (carry in ones)
- Chain carry: 999 + 1 = 1000 (triple carry chain)
- Large: 500 + 499 = 999 (boundary)

**Risk**: Tokenization — 3-digit numbers may be multi-token in some models.
Need to verify `get_answer_token_id()` still works or adapt for multi-token answers.

---

#### Step A3: Subtraction Circuit Comparison

**Effort**: ~2-3 days  
**Target venue impact**: "Does addition and subtraction share a circuit?" is a compelling question  

**What to do**:
- Modify `generate_arithmetic_prompts()` to support `a - b` (ensure a > b for positive results)
- Run Stages 1-2 on Phi-3 + Gemma for subtraction
- Compare: which MLPs have high recovery? Are they the same as addition?
- Run carry/borrow probe: does "borrowing" use the same layers as "carrying"?

**Possible outcomes**:
- **Shared circuit** → evidence for a general arithmetic module
- **Different circuit** → evidence for operation-specific computation
- **Partially shared** → shared routing, different compute (most likely)

All outcomes are publishable.

---

### Tier B: Highest Scientific Value, High Effort

#### Step B1: Sparse Autoencoder on Compute MLPs

**Effort**: 4-6 weeks  
**Target venue impact**: Transforms paper from "circuit geography" to "algorithm discovery"  
**Required for**: Top-venue paper (NeurIPS / ICML / ICLR)  

**What to do**:
- Train SAEs on L20-L22 MLP activations (Phi-3) and L13-L14 (Gemma)
- Collect activations from ~10K arithmetic problems
- Train with standard SAE architecture (encoder-decoder with L1 sparsity)
- Analyze learned features: look for interpretable features like:
  - "carry = 1" (fires when ones digits sum > 9)
  - "ones digit = 7" (fires for specific digit values)
  - "tens digit computation" (fires during tens-place processing)
- Validate features causally: ablate individual SAE features and measure accuracy drop

**Infrastructure needed**:
- SAE training library (e.g., `sae-lens` or custom)
- GPU recommended for training (CPU feasible but slow)
- Hyperparameter sweep: dictionary size (512-4096), L1 coefficient, learning rate

**This is the single biggest gap** in the current work. We know WHERE computation
happens but not HOW. SAE features would answer the "how" question.

---

#### Step B2: ACDC / Edge Activation Patching

**Effort**: 2-3 weeks  
**Target venue impact**: Publication-standard minimal circuit diagram  
**Required for**: Matching IOI paper's presentation standard  

**What to do**:
- Use ACDC (Conmy et al., 2023) or EAP (Syed et al., 2023) to automatically
  enumerate all causal edges in the arithmetic circuit
- Produces a clean graph: which heads write to which MLPs, which MLPs write to the output
- Our manual patching found the big components; ACDC finds all edges

**Implementation options**:
- Integrate `acdc` Python library (https://github.com/ArthurConmy/Automatic-Circuit-Discovery)
- Or implement EAP from scratch (simpler, gradient-based approximation)

---

### Tier C: Statistical Hardening & Additional Models

#### Step C1: Statistical Rigor

**Effort**: ~1 week  

- Scale all experiments to 200+ problems (currently 30-40)
- Add bootstrap confidence intervals on all recovery scores
- Add null-distribution baseline for carry probe (random labels → ~50%)
- Report effect sizes and p-values for key claims

---

#### Step C2: 4th Model — Llama 3.1 8B or Mistral 7B

**Effort**: ~1 day runtime (needs GPU or patient CPU)  

- Validates universality claim at larger scale (8B vs current max 3.8B)
- Different architecture family strengthens cross-model argument
- Requires GPU or very patient CPU execution

---

## Publication Pathway

| Target | What's Needed | Timeline |
|--------|--------------|----------|
| **Workshop paper** (MI@NeurIPS, ATTRIB@ICML) | Current work + writeup | 2-3 weeks |
| **Findings paper** (ACL/EMNLP Findings) | + Steps A1-A3 + C1 | 6-8 weeks |
| **Top conference** (ICLR / NeurIPS main) | + Steps B1 + B2 + C2 | 3-4 months |

### Recommended Execution Order

```
Week 1:    A1 (copy head knockout)     — 2 hours
           A2 (3-digit scaling)         — 1 day
           C1 (statistical hardening)   — parallel with A2

Week 2-3:  A3 (subtraction comparison)  — 2-3 days
           Workshop paper writeup       — remainder

Week 4-9:  B1 (SAE on compute MLPs)    — 4-6 weeks
           B2 (ACDC) can overlap        — 2-3 weeks

Week 10:   C2 (4th model, Llama 8B)    — 1 day
           Full paper writeup           — remainder
```
