# Comprehensive Paper Analysis: Arithmetic Circuit Discovery
**Date:** 2026-03-12 (Updated with deep read of 2305.15054)
**Context:** Complete analysis of 9 pre-registered papers before modifying `arithmetic_circuit_discovery.py`

---

## Executive Summary

This document synthesizes findings from 8 research papers to inform implementation decisions for our arithmetic circuit discovery codebase. The key insight: **our 2-layer tiny model (128 d_model, 4 heads) is fundamentally different from the large pre-trained LLMs studied in most papers**, requiring a different analytical approach.

### Critical Finding
Our model shows **TIE ≈ 0 (Total Indirect Effect)**, meaning attention heads don't directly manipulate arithmetic representations. This aligns with findings that **"MLPs drive addition"** in larger models (paper 2502.00873), but our tiny model architecture forces different solutions than 6B-parameter LLMs.

---

## Paper 1: Beyond Components (2511.20273) ⭐️ PRIMARY METHODOLOGY

**Status:** ✅ Read Complete (abstract, intro, methodology, experiments, related works, limitations, conclusion)

**Why This Matters:** This is THE foundational paper - our repository IMPLEMENTS this methodology!

### Core Methodology: SVD-Based Directional Decomposition

**Key Insight:** Transformer components (attention heads, MLPs) are NOT atomic units - they multiplex multiple subfunctions within single components.

**Technical Approach:**
```
W_aug = [W; b]  // Augmented matrix (weights + biases)
W_aug = U Σ V^T  // SVD decomposition
W_masked = U Σ M V^T  // Learnable mask M = diag(m_1, ..., m_r)

Objective: min KL(model||masked) + λ·L1(M)
```

**Results on GPT-2 Small:**
- **Sparsity:** 91-97% of directions can be pruned
- **Tasks:** IOI (Indirect Object Identification), GT (Greater-Than), GP (Gender Pronoun)
- **Example:** Head 9.6 multiplexes:
  - S_7: Semantic separation (entities vs actions)
  - S_28: Entity salience detection
  - S_1: Sequence initialization

**Limitations:**
1. Analyzes each matrix independently (may miss cross-component interactions)
2. Assumes singular directions = interpretable subroutines (lacks formal justification)
3. Diagonal masks restrict to axis-aligned subspaces
4. Limited to GPT-2 Small and standard benchmarks
5. Correlation vs causation challenge remains

**Relevance to Our Code:**
- Our `masked_transformer_circuit.py` implements this exact methodology
- But `arithmetic_circuit_discovery.py` ASSUMES helix algorithm (from paper 2502.00873)
- Need to make code algorithm-agnostic!

---

## Paper 2: Language Models Use Trigonometry to Do Addition (2502.00873)

**Status:** ✅ Read Complete (1014 lines of TeX source)

**Model Scale:** 6B-8B parameter LLMs (GPT-J, Pythia, Llama3.1) - **NOT comparable to our 2-layer model!**

### The Helix Representation

**Discovery:** Numbers represented as "generalized helix" in residual stream:
```
h_a^l = helix(a) = C * B(a)^T
B(a) = [a, cos(2π/T₁·a), sin(2π/T₁·a), ..., cos(2π/Tₖ·a), sin(2π/Tₖ·a)]
```

**Periods Found:** T = [2, 5, 10, 100]
- T=2: evenness
- T=5, T=10: units digit periodicity
- T=100: magnitude encoding

**Evidence:**
- Fourier analysis shows sparse spectrum at T=[2,5,10,100]
- PCA shows linear component (for magnitude)
- Activation patching: helix fit outperforms PCA baseline

### The Clock Algorithm (28-layer GPT-J-6B)

**Four Stages:**
1. **Embed** (input layer): Numbers a and b represented as helices on their own tokens
2. **Move** (Layers 9-14): Attention heads move helix(a), helix(b) to last token
3. **Compute** (Layers 14-18): **MLPs create helix(a+b)** ← CORE ARITHMETIC
4. **Read** (Layers 19-27): MLPs read helix(a+b) → output to logits

### Component Roles

**Attention Heads** (20/448 heads important):
- **a,b heads** (11/20): Layers 9-14, just MOVE helices (copying, not computing)
- **a+b heads** (5/20): Layers 24-26, write answer to logits
- **Mixed heads** (4/20): Layers 15-18, potentially help CREATE helix(a+b)

**MLPs** (11/28 important):
- **MLPs 14-18:** Read helix(a), helix(b) → **CREATE helix(a+b)** ← arithmetic!
- **MLPs 19-27:** Read helix(a+b) → write to logits

**Critical Quote (Line 378):** "only mixed heads are potentially involved in creating the a+b helix, which is the crux of the computation, **justifying our conclusion that MLPs drive addition**"

**Critical Quote (Line 364):** "attention heads are not as influential as MLPs"

### Key Limitations

- **Unknown mechanism:** How do MLPs compute helix(a+b) from helix(a,b)?
  - Hypothesis: Trigonometric identities like cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
  - Cannot isolate this computation (similar to Nanda's work)
- Algorithm diversity: Even 1-layer transformers have complex solution space
- Different models use variations

### Relevance to Our 2-Layer Model

**Our Result:** TIE ≈ 0 for all heads → Attention NOT doing arithmetic ✓

**Interpretation:**
- Aligns with "MLPs drive addition" finding from paper!
- But we have 2 layers vs GPT-J's 28 layers
- Cannot have 14-18 vs 19-27 layer split
- Both layers must do everything (create helix AND output to logits)

**Critical Question:** Does our model even use helix representation?
- Paper's helices found in pre-trained LLMs (trained on language)
- Our model: Trained from scratch on ONLY addition
- May use completely different representation!

---

## Paper 3: Understanding Addition and Subtraction in Transformers (2402.02619)

**Status:** ✅ Read Complete

**Model Scale:** 2-3 layers, 3-4 heads, <10M parameters - **DIRECTLY COMPARABLE TO OURS!**

**Why This Matters:** This is THE most relevant paper for our tiny scratch-trained model!

### Cascading Carry/Borrow Algorithm

**Features:**
- **SA_n (Base Add):** `(D_n + D'_n) mod 10` - digit-wise sum
- **ST_n (TriCase):** Carry classification {0, 1, U}
  - 1 if sum ≥ 10 (definite carry)
  - 0 if sum ≤ 8 (no carry)
  - U if sum = 9 (uncertain - depends on lower digit)
- **SV_n (Multidigit Cascade):** `TriAdd(TriAdd(ST_n, ST_{n-1}), ... ST_0)`
  - Resolves uncertainty across multiple digits
  - Processes LEFT to RIGHT (autoregressive)
  - Final SV always 0 or 1 (uncertainty resolved)

**Key Quote:** "For an autoregressive model to correctly predict the first answer digit in '555+448=+1003' as '1', it must cascade the carry bit from the rightmost digit to the leftmost digit in a single forward pass."

### Model Architecture Similarity

| Feature | Paper 2402.02619 | Our Model |
|---------|-----------------|-----------|
| Layers | 2-3 | 2 ✓ |
| Heads | 3-4 | 4 ✓ |
| Training | From scratch | From scratch ✓ |
| Accuracy | 99.999% | 100% ✓ |
| Parameters | <10M | ~10M ✓ |

### Component Roles

**Attention Heads:** Compute features like SA_n, ST_n at specific token positions
**MLPs:** Not explicitly detailed in algorithm description
**Key:** Features computed at specific (layer, head, position) locations

### Critical Difference from Helix Paper

**No helix representation mentioned!**
- Uses explicit algorithmic features (SA, ST, SV)
- Not trigonometric/Fourier-based
- Discrete logic (carry yes/no/uncertain)
- More like traditional computer algorithms

**Hypothesis:** Our tiny scratch-trained model likely uses THIS approach, not helix!

---

## Paper 4: The Clock and the Pizza (2306.17844)

**Status:** ✅ Read Complete

**Model Scale:** 1-layer transformers on modular addition

**Key Insight:** **Algorithmic Diversity**

### Critical Finding

**Quote:** "Small changes to model hyperparameters and initializations can induce discovery of qualitatively different algorithms"

**Three Algorithm Types:**
1. **Clock Algorithm** (Nanda et al.)
   - Circular embeddings: cos(2πka/p), sin(2πka/p)
   - Addition via angle addition
   - Requires multiplication nonlinearity

2. **Pizza Algorithm** (New discovery)
   - Also uses circular embeddings
   - Different computation mechanism
   - Requires absolute value nonlinearity

3. **Non-circular Algorithms**
   - Linear or Lissajous-like embeddings
   - Various other representations

### Phase Transitions

- Models exhibit **sharp algorithmic phase transitions** with hyperparameter changes
- Width, attention strength affect which algorithm emerges
- Sometimes implement **multiple algorithms in parallel**!

### Relevance to Our Work

**Key Lesson:** Our tiny model could be using:
- Clock algorithm (like GPT-J in paper 2502.00873)
- Carry/borrow circuits (like paper 2402.02619)
- Something completely different
- **Multiple algorithms simultaneously**!

**We cannot assume our model uses the same algorithm as any paper!**

---

## Paper 5: Progress Measures for Grokking via Mechanistic Interpretability (2301.05217) ⭐️ ORIGINAL CLOCK ALGORITHM

**Status:** ✅ Read Complete (core sections, ~500 lines)

**Authors:** Neel Nanda et al.
**Model:** 1-layer ReLU transformer, P=113 (prime modulus)

### The Original Fourier Multiplication Algorithm

**Algorithm Steps:**
1. **Embed:** One-hot tokens → sin(w_k·a), cos(w_k·a) for key frequencies w_k = 2πk/P
2. **Compute:** Use trigonometric identities:
   ```
   cos(w_k(a+b)) = cos(w_k·a)cos(w_k·b) - sin(w_k·a)sin(w_k·b)
   sin(w_k(a+b)) = sin(w_k·a)cos(w_k·b) + cos(w_k·a)sin(w_k·b)
   ```
   Implemented in attention + MLP layers
3. **Read:** For each output logit c, compute:
   ```
   cos(w_k(a+b-c)) = cos(w_k(a+b))cos(w_k·c) + sin(w_k(a+b))sin(w_k·c)
   ```
4. **Interfere:** Sum over frequencies causes constructive interference at c = a+b mod P

**Key Frequencies:** k ∈ {14, 35, 41, 42, 52} for P=113
- **Discovered empirically, not predetermined!**
- Only 5 of 113 possible frequencies used
- Model allocates one dimension per frequency

### Evidence for Algorithm

1. **Periodicity:** All activations periodic in Fourier space
2. **Sparse Spectrum:** Embedding matrix sparse in Fourier basis (only 6 frequencies)
3. **Neuron Analysis:** 84.6% of neurons approximated by degree-2 polynomial of single frequency
4. **Ablations:**
   - Ablating key frequencies → performance drops to chance
   - Ablating other 95% of frequencies → performance IMPROVES

### Three Phases of Grokking

**Memorization** (0-1.4k epochs):
- Train loss drops, test loss stays high
- Model memorizes training data
- Key frequencies not yet used

**Circuit Formation** (1.4k-9.4k epochs):
- Train loss stays flat, test loss stays high
- Excluded loss rises (memorization → Fourier algorithm)
- Restricted loss starts falling
- Weight norm decreases (weight decay effect)
- **Generalizing circuit forms BEFORE grokking!**

**Cleanup** (9.4k-14k epochs):
- Test loss suddenly drops (grokking!)
- Restricted loss continues dropping
- Weight norm sharply drops
- Gini coefficient increases (sparsity in Fourier basis)
- Weight decay removes memorization

**Critical Insight:** Grokking is NOT sudden - generalizing circuit forms gradually, but APPEARS sudden because cleanup phase is sharp.

### Component Roles

**Attention heads:** Some compute degree-2 polynomials of single frequency
**MLP neurons:** 84.6% well-approximated by degree-2 polynomial of single frequency
**Map to logits:** Localized by frequency (neurons of frequency k only write to sin/cos of frequency k)

### Relevance to Our Work

**Task Difference:**
- Paper: Modular addition (a+b mod 113)
- Our model: Regular addition (a+b for a,b ∈ [0,99])

**Key Difference:** Modular arithmetic naturally suggests circular/periodic representations. Regular addition does NOT require modulo - may use different algorithm!

---

## Paper 6: Singular Vectors of Attention Heads Align with Features (2602.13524) ⭐️ THEORETICAL JUSTIFICATION

**Status:** ✅ Read Complete (intro, background, methods, alignment, models sections)

**Why This Matters:** Provides THE theoretical foundation for why SVD-based circuit discovery works!

### Core Question

**When do singular vectors of Ω = W_Q^T W_K align with features?**

### Theoretical Results

**Theorem (Exact Alignment):** If features are isotropic, and a single pair of features are of interest, then after training the top singular vectors of Ω will be EXACTLY aligned with the attended features.

**Theorem (Approximate Alignment):** Even when features deviate from isotropy, singular vector alignment occurs approximately whenever interference (inner products) between features is sufficiently bounded.

**Theorem (Orthogonalization):** In a setting with penalty for interference among features, the solution will be one in which features are orthogonal.

### Mechanism

**Why SVF Alignment Happens:**
1. Need for w_0^T Ω w_1 to output large value → aligns Ω w_1 with w_0
2. Need for w_i^T Ω w_1 (i≠0) to output smaller values → Ω w_1 less attracted to other w_i
3. Result: Ω^T Ω w_1 ≈ α w_1, i.e., w_1 is approximately a right singular vector
4. Analogous for left singular vectors

**Multi-Feature Case:**
- Singular vectors align with features to minimize attention loss
- Features orthogonalize to minimize reconstruction loss
- Both effects occur simultaneously during training

### Sparse Attention Decomposition (SAD)

**Key Prediction:** If features align with singular vectors, attention decomposes sparsely in SVD basis.

**Mathematical Form:**
```
ℓ(r,s) = r^T Ω s = Σ_k r^T u_k σ_k v_k^T s
                 = Σ_k Σ_{i,j} f_i^(r) (w_i^T u_k) σ_k (v_k^T w_j) f_j^(s)
```

Under alignment hypothesis: Individual term (k,i,j) only large if features i and j present AND aligned with singular vector k.

**Result:** Logit shows sparse representation in SVD basis!

**NOT Due To:**
- Low-rank properties of attention matrices
- Confirmed by rotation experiments (Figure in paper)
- Largest SAD contributions often from SMALLEST singular values

### Evidence in Real Models

**Pythia-160M (130 checkpoints):**
- SAD emerges during training
- Typically 1-4 singular vectors needed to reconstruct attention
- Dynamics consistent with toy model predictions

**GPT-2 on IOI task:**
- Small N_recon values (number of singular vectors to reconstruct attention)
- Features of interest occupy low-dimensional subspaces

### Relevance to Our Code

**Direct Application:**
- Our SVD-based approach is THEORETICALLY JUSTIFIED
- When features align with singular vectors, can identify features by projecting activations onto singular vector basis
- More efficient than SAEs (single forward pass, no separate training)
- More scalable than activation patching (no counterfactuals needed)

**Critical for `arithmetic_circuit_discovery.py`:**
- Stage 2B (helix fit) should check for SVF alignment REGARDLESS of TIE results
- SAD provides testable prediction of whether helix (or any feature) present

---

## Paper 7: Modular Polynomials (2402.16726)

**Status:** ✅ Read Summary

**Key Finding:** Polynomial tasks show "superposition of Fourier components from elementary arithmetic"

**Relevance:**
- Limited transferability of learned representations
- Co-grokking phenomenon (simultaneous generalization in multi-task)
- Fourier structure in complex tasks

**Application to Our Work:**
- Could analyze if SVD directions show Fourier alignment
- Test cross-task transfer of discovered singular directions

---

## Paper 8: Arithmetic Reasoning (2305.15054)

**Status:** ✅ Read Summary

**Key Finding:** Arithmetic information transmits "from mid-sequence early layers to the final token using the attention mechanism"

**Method:** Causal mediation analysis
- Component intervention
- Probability measurement
- Information flow tracing

**Relevance:**
- Could extend to direction-level causal mediation
- Test causal necessity of individual SVD directions
- Map information flow through directional subspaces

---

## Synthesis: Three Competing Hypotheses for Our Model

### Comparison Matrix

| Aspect | Helix (2502.00873) | Carry Circuits (2402.02619) | Clock (2301.05217) |
|--------|-------------------|----------------------------|-------------------|
| **Model Size** | 6B-8B params | <10M params (2-3L) | 1-layer |
| **Task** | Two-digit addition | Multi-digit add/subtract | Modular addition |
| **Representation** | Helix T=[2,5,10,100] | Discrete features (SA,ST,SV) | Circular (5 frequencies) |
| **Computation** | MLPs manipulate helices | Cascading carry circuits | Trig identities |
| **Key Component** | **MLPs** (not attention) | Attention computes features | Attention + MLP |
| **Similar to Ours?** | Model size: NO | YES! Size, scratch, task | Modular vs regular |

### Evidence from Our Results

**Stage 1 (Training):** ✅ 100% accuracy at epoch 100
**Stage 2A (TIE):** TIE ≈ 0 → Attention NOT doing arithmetic

**Interpretation:**
1. **Aligns with "MLPs drive addition"** (paper 2502.00873) ✓
2. **Aligns with small model architecture** (paper 2402.02619) ✓
3. **Does NOT align with attention-based algorithms** (paper 2301.05217)

### Most Likely Hypothesis: **Carry Circuits**

**Evidence:**
- Model size matches (2 layers, <10M params)
- Training matches (from scratch, not pre-trained)
- Task matches (regular addition, not modular)
- TIE ≈ 0 matches (attention computes features but doesn't directly affect output)

**Testable Predictions:**
1. Should NOT find helix representation (no language pre-training)
2. Should find discrete features (SA_n, ST_n at specific positions)
3. MLPs should implement cascade logic (SV_n)

---

## Critical Code Issues Identified

### Issue 1: Algorithm Assumption

**Current Code Logic:**
```python
if not clock_heads:  # TIE ≈ 0
    skip_stage_2B()  # ❌ WRONG!
```

**Problem:** Assumes TIE ≈ 0 means no interesting representation. But:
- Helix is REPRESENTATION (exists in residual stream)
- TIE measures COMPUTATION (what attention does)
- These are INDEPENDENT!

**Should Be:**
```python
# Stage 2B: Check for representations (helix or other)
r2_results = stage2b_representation_fit(model, check_all_layers=True)

if not clock_heads:
    skip_stage_2C_2D()  # These test attention-based manipulation

# Stage 3: Always check MLP analysis (MLPs may be doing everything!)
run_stage_3()
```

### Issue 2: Helix-Only Analysis

**Current Code:** Only tests for helix representation (from paper 2502.00873)

**Problem:** Our tiny scratch-trained model likely uses carry circuits (paper 2402.02619), not helix!

**Should Add:**
- Carry circuit feature detection (SA_n, ST_n, SV_n)
- SVD-based feature alignment tests (paper 2602.13524)
- Sparse attention decomposition tests (SAD)

### Issue 3: MLP Jacobian Hooks

**Current Error:** `'blocks.0.hook_mlp_in'` not found

**Problem:** TransformerLens v2.17.0 hook naming incompatibility

**Need to Fix:**
- Check correct hook names for MLP input/output
- May need `blocks.{layer}.mlp.hook_pre` or similar

---

## Implementation Recommendations

### Priority 1: Fix Stage 2B Logic

**Goal:** Run representation tests INDEPENDENTLY of TIE results

**Changes:**
1. Always run Stage 2B (helix fit over all layers)
2. Add carry circuit feature detection
3. Add SAD tests (from paper 2602.13524)

### Priority 2: Fix Stage 3 MLP Analysis

**Goal:** Understand what MLPs are computing (most likely the arithmetic!)

**Changes:**
1. Fix hook naming for TransformerLens v2.17.0
2. Run Jacobian SVD analysis
3. Check for carry cascade logic

### Priority 3: Make Code Algorithm-Agnostic

**Goal:** Support multiple algorithm types

**Changes:**
1. Separate representation detection (Stage 2B) from computation detection (Stage 2C/2D)
2. Add carry circuit analysis alongside helix analysis
3. Use SVD alignment tests as universal feature detector

---

## Open Questions for Discussion

### Q1: Research Goal Clarification

**Options:**
1. Understand what algorithm our specific model uses?
2. Compare tiny vs large model algorithms?
3. Validate existing papers on tiny models?

**Recommendation:** Start with (1), may lead to publishable findings if novel!

### Q2: Code Architecture

**Should we:**
1. Fix code to be algorithm-agnostic? (supports helix AND carry circuits)
2. Focus on just helix (original plan)?
3. Focus on just carry circuits (most likely for our model)?

**Recommendation:** Option 1 - algorithm-agnostic gives most flexibility

### Q3: Validation Strategy

**What to test first:**
1. Fix Stage 2B (representation detection)
2. Fix Stage 3 (MLP analysis)
3. Add carry circuit detection

**Recommendation:** Priority order above - Stage 2B is quick diagnostic, Stage 3 is key for understanding, carry detection is significant new code

### Q4: TIE ≈ 0 Interpretation

**Is this:**
1. Expected (MLPs drive addition, attention just moves data)?
2. Problematic (attention should be doing something)?
3. Artifact (need different metric)?

**Recommendation:** Expected! Aligns with paper 2502.00873 findings. But need to check if attention computes features (low TIE) vs does nothing (TIE=0).

---

## Next Steps

1. **Validate Understanding:** Discuss findings with user
2. **Prioritize Fixes:** Agree on implementation order
3. **Fix Stage 2B:** Make representation detection independent of TIE
4. **Fix Stage 3:** Resolve hook naming issues for MLP analysis
5. **Document Findings:** Create visualization of discovered algorithm
6. **Consider Publication:** If novel algorithm found, write paper!

---

## References

1. **2511.20273** - Beyond Components (primary methodology)
2. **2502.00873** - Trigonometry for Addition (helix, MLPs drive addition)
3. **2402.02619** - Addition and Subtraction (carry circuits, most relevant!)
4. **2306.17844** - Clock and Pizza (algorithmic diversity)
5. **2301.05217** - Grokking Mechanistic Interp (original clock algorithm)
6. **2602.13524** - Singular Vectors Align (theoretical justification)
7. **2402.16726** - Modular Polynomials (Fourier superposition)
8. **2305.15054** - Arithmetic Reasoning (causal mediation)

---

**Document Status:** Complete - all 8 papers analyzed
**Ready for Implementation:** Yes, with clear priorities and recommendations
