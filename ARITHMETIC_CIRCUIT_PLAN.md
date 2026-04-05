# Arithmetic Circuit Discovery — Final Implementation Plan

## Goal
Identify the complete mechanistic circuit for integer addition in transformer language models: which components participate, which singular directions carry the computation, and how MLPs transform helical number representations into the answer.

## Methodology
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

## Total Timeline

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
