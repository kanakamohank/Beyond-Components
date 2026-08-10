# A Mechanistic Interpretation of Arithmetic Reasoning in Language Models using Causal Mediation Analysis

**arxiv ID:** 2305.15054
**URL:** https://arxiv.org/abs/2305.15054
**Authors:** Alessandro Stolfo (ETH Zürich), Yonatan Belinkov (Technion), Mrinmaya Sachan (ETH Zürich)
**Published:** May 2023 (EMNLP 2023)
**Models Studied:** GPT-J (6B), Pythia 2.8B, LLaMA 7B, Goat 7B

---

## TL;DR
Using causal mediation analysis, this paper traces how information flows through transformers when answering arithmetic questions. **Key finding:** Information about operands is transmitted from mid-sequence early layers to the final token via attention, then processed by late MLP modules (layers 19-20 in GPT-J) which generate result-related information incorporated into the residual stream.

## Problem
**Limited Understanding:** While LLMs show impressive math performance, there's limited mechanistic understanding of how they internally process and store arithmetic information.

**Key Questions:**
1. Which model components mediate arithmetic predictions?
2. How does information flow through the network for arithmetic tasks?
3. Are activation patterns specific to arithmetic vs other tasks?

## Method

### Causal Mediation Analysis Framework

**Core Idea:** Model as causal graph where components (MLPs, attention) are mediators between input and output.

**Intervention Procedure:**
```
1. Sample two operand sets: N, N'
   → Generate queries: p₁ = p(N, f_O), p₂ = p(N', f_O)
   → Results: r = f_O(N), r' = f_O(N')

2. Forward pass p₁: Store activations
   m̄ˡₜ := MLP^l(h^(l-1)_t)
   āˡₜ := A^l(h^(l-1)_1, ..., h^(l-1)_t)

3. Forward pass p₂ WITH intervention:
   Replace activations at position t, layer l with m̄ˡₜ, āˡₜ

4. Measure probability changes:
   IE(z) = ½[ΔP(r)/P(r) + ΔP(r')/P_z*(r')]
```

**Indirect Effect (IE):** Measures how much component z shifts probability mass from r' to r.

**Relative Importance (RI):** For subset of MLPs M*:
```
RI(M*) = Σ_{m∈M*} log(IE(m)+1) / Σ_{m∈M} log(IE(m)+1)
```

### Experimental Setup

**Tasks:**
- **2-operand arithmetic:** "How much is n₁ plus n₂?" (4 operators: +, -, ×, ÷)
- **3-operand arithmetic:** "What is the difference between n₁ and the ratio of n₂ and n₃?"
- **Number retrieval:** "Paul has n₁ apples and n₂ oranges. How many apples does Paul have?"
- **Factual knowledge:** LAMA benchmark (e.g., "Paris is the capital of ___")

**Operand range:** S = {1, 2, ..., 300} (larger numbers split into multiple tokens)
**Prompting:** 2-shot examples (same operation as query)
**Models:** GPT-J (6B), Pythia 2.8B, LLaMA 7B, Goat 7B (LLaMA fine-tuned on arithmetic)

## Key Results

### 1. Information Flow for Arithmetic (GPT-J 6B)

**Four Primary Activation Sites:**

1. **Early MLPs at operand tokens (Layer 1)**
   - IE: High at tokens containing n₁, n₂
   - Function: Encode operand representations

2. **Mid-late Attention at last token (Layers 11-18)**
   - IE: High at final token position
   - Function: **Move operand/operator information to end of sequence**

3. **Late MLPs at last token (Layers 19-20)**
   - IE: **Highest** at final token position
   - Function: **Process information and generate result**
   - RI = 40.2% (result-varying), 4.4% (result-fixed)

4. **Last layer MLP (Layer 28)**
   - IE: Moderate
   - Function: Final output projection

**Key Insight:** Attention mechanism acts as **information transporter**, MLPs act as **computation engines**.

### 2. Operand vs Result Information

**Experiment:** Fix result r = r' while varying operands

**Findings:**
- **Early MLPs (Layer 1):** High IE in BOTH settings → encode operand info
- **Attention (Layers 11-18):** Similar IE in BOTH settings → move operand info
- **Late MLPs (Layers 19-20):**
  - High IE when r ≠ r' → generate result info
  - Low IE when r = r' → NOT just operand carriers
  - **RI drops from 40.2% → 4.4%** when result fixed

**Conclusion:** Layers 19-20 MLPs specifically encode **result-related information**, not just operands!

### 3. Three-Operand Queries & Fine-Tuning

**Pre-trained Pythia 2.8B:**
- Accuracy: 0.9% (essentially random)
- Only last layer MLP shows effect

**Fine-tuned Pythia 2.8B:**
- Accuracy: 39.7% (40x improvement!)
- **Mid-late MLP activation site emerges** (similar to 2-operand case)
- RI = 24.7% (result-varying), 13.6% (result-fixed)

**Interpretation:** Fine-tuning causes model to **develop arithmetic circuit** with characteristic information flow pattern.

### 4. Task Specificity

**Number Retrieval Task:**
- Main site: Early MLPs at entity tokens (expected - just retrieval)
- Secondary: Late MLPs at last token (RI = 8.7%)
- **10% less RI than arithmetic** → late MLPs do computation, not just numerical prediction

**Factual Knowledge (LAMA):**
- Main site: Early MLPs at subject tokens (Layers 0-5)
- Consistent with prior work (Meng et al. 2022)
- RI = 4.2% for late MLPs
- **Completely different pattern from arithmetic!**

**Arithmetic uniqueness confirmed:**
- Late MLPs (19-20) critical for arithmetic
- NOT critical for number retrieval or factual knowledge
- Activation patterns are **task-specific**

### 5. Neuron-Level Analysis (Layer 19)

**Top 400 neurons (10% of 4096 dimensions):**

| Task Pair | Overlap |
|-----------|---------|
| Arithmetic (Arabic) ↔ Arithmetic (Words) | **50%** |
| Arithmetic ↔ Number Retrieval | **22-23%** |
| Arithmetic ↔ Factual Knowledge | **9-10%** (random!) |

**Interpretation:**
- Same neurons handle arithmetic regardless of number representation
- Different circuits for arithmetic vs retrieval (even though both numerical!)
- **Essentially no overlap** between arithmetic and factual circuits

### 6. Cross-Model Validation

**Consistent patterns across all models:**

| Model | RI (result-varying) | RI (result-fixed) |
|-------|---------------------|-------------------|
| GPT-J 6B | 40.2% | 4.4% |
| Pythia 2.8B | 43.2% | 5.8% |
| LLaMA 7B | 36.1% | 7.5% |
| Goat 7B | 33.5% | 7.4% |
| GPT-J (Words) | 27.8% | 4.5% |

**All models show:**
- Late MLP activation site at last token
- Large drop in RI when result fixed
- Information flow: early layers → attention → late MLPs

## Limitations & Open Questions

1. **Operator Scope:** Only studies +, -, ×, ÷ (not exponentials, roots, etc.)
2. **Synthetic Queries:** Not real-world math word problems
3. **Attention Heads:** Treats attention module as whole (doesn't analyze individual heads)
4. **Scalability:** Component-level analysis impractical for very large models
5. **Mechanism:** Shows WHERE computation happens, not HOW (internal algorithm)

## Relevance & Application Ideas

### For Arithmetic Circuit Discovery

**Critical Connections to Our Work:**

1. **Information Flow Pattern:**
   ```
   Operands (early layers) → Attention (move to last token)
                           → MLPs (compute result) → Output
   ```
   **Our TIE ≈ 0:** Attention NOT computing arithmetic ✓ (just moving info!)
   **MLPs do arithmetic:** Aligns with paper 2502.00873 "MLPs drive addition"

2. **Layer Specificity:**
   - GPT-J (28 layers): Layers 19-20 for arithmetic
   - Pythia 2.8B: Similar relative position (~70% through model)
   - **Our 2-layer model:** Both layers must do everything!

3. **Component Roles:**
   - **Attention:** Information routing (copying, moving)
   - **MLPs:** Computation (arithmetic processing)
   - This MATCHES findings in papers 2502.00873 and 2402.02619

### Practical Implications for `arithmetic_circuit_discovery.py`

**1. TIE ≈ 0 Interpretation:**
```python
# Our result: TIE ≈ 0 for attention heads
# This paper: Attention moves info, doesn't compute
# Conclusion: EXPECTED! Not a bug!
```

**2. Where to Look for Arithmetic:**
```python
# DON'T look in attention (just routing)
# DO look in MLPs (especially late layers)
# For 2-layer model: MLP at layer 1 (late 50% of model)
```

**3. Result vs Operand Information:**
```python
def distinguish_computation_from_routing(model):
    # Test with r = r' (same result, different operands)
    # High IE when r = r' → routing/encoding
    # High IE when r ≠ r' but low when r = r' → computation

    ie_varying = measure_ie(operands_vary=True, results_vary=True)
    ie_fixed = measure_ie(operands_vary=True, results_vary=False)

    if ie_varying >> ie_fixed:
        return "Computation site (generates result)"
    else:
        return "Routing site (moves operands)"
```

**4. Fine-Tuning Analysis:**
```python
# Pythia: Circuit emerges ONLY after fine-tuning
# Pre-trained: No arithmetic circuit (RI = 13.5%)
# Fine-tuned: Arithmetic circuit appears (RI = 24.7%)
# Implication: Our scratch-trained model should show circuit from start
```

### Implementation Extensions

**Add to `arithmetic_circuit_discovery.py`:**

1. **Result-Fixed Intervention:**
```python
def test_result_encoding(model, dataset):
    """Test if component encodes result vs just operands."""
    # Generate pairs with same result
    pairs = [(12+7, 10+9), (15+8, 20+3), ...]  # All sum to same

    for layer in range(model.n_layers):
        ie_varying = compute_ie(layer, result_varies=True)
        ie_fixed = compute_ie(layer, result_varies=False)

        if ie_varying / ie_fixed > 5:
            print(f"Layer {layer}: Result computation site!")
```

2. **Relative Importance Metric:**
```python
def compute_relative_importance(model, component_subset):
    """Compute RI following paper's methodology."""
    total_effect = sum(np.log(ie + 1) for ie in all_components)
    subset_effect = sum(np.log(ie + 1) for ie in component_subset)
    return subset_effect / total_effect
```

3. **Cross-Task Comparison:**
```python
def compare_circuits(model):
    """Compare arithmetic vs factual knowledge circuits."""
    arithmetic_neurons = identify_top_neurons(arithmetic_task)
    factual_neurons = identify_top_neurons(factual_task)

    overlap = len(set(arithmetic_neurons) & set(factual_neurons))
    print(f"Circuit overlap: {overlap/len(arithmetic_neurons)*100:.1f}%")
    # Expect: ~10% (random) if circuits are separate
```

### Connection to Other Papers

**Paper 2502.00873 (Helix/MLPs Drive Addition):**
- THIS paper: Late MLPs (19-20) generate arithmetic results
- 2502.00873: MLPs (14-18) create helix(a+b), MLPs (19-27) read it
- **Alignment:** Both find MLPs do arithmetic, attention does routing ✓

**Paper 2402.02619 (Carry Circuits):**
- THIS paper: Attention moves operands to last token
- 2402.02619: Attention computes SA_n, ST_n features
- **Distinction:** 2402.02619 has attention computing features, not just moving

**Paper 2602.13524 (SVD Alignment):**
- THIS paper: Task-specific neurons (9-10% overlap between arithmetic and factual)
- 2602.13524: Features align with singular vectors when of interest to head
- **Connection:** Low overlap → features in different singular directions!

**Paper 2209.10652 (Superposition):**
- THIS paper: 50% neuron overlap between Arabic and word representations
- 2209.10652: Same features can be represented in superposition
- **Interpretation:** Arithmetic features partially shared across representations

### What We Now Understand

**Information Flow for Arithmetic:**
```
Input: "What is 12 + 7?"
  ↓
Layer 0-1: Encode operands (12, 7) in early MLPs
  ↓
Layers 11-18: Attention moves operand info to last token
  ↓
Layers 19-20: MLPs compute result (19) at last token
  ↓
Layer 28: Final projection to vocabulary
  ↓
Output: "19"
```

**Our 2-Layer Model Implication:**
```
Input: "What is 12 + 7?"
  ↓
Layer 0: MLPs encode operands + attention may start routing
  ↓
Layer 1: MLPs compute result + attention moves to last token
  ↓
Output: "19"
```
Everything compressed into 2 layers!

## Tags
`causal-mediation` `arithmetic-reasoning` `information-flow` `mechanistic-interpretability` `component-analysis` `intervention-analysis` `mlp-computation` `attention-routing`

---

## Key Equations

**Indirect Effect:**
```
IE(z) = ½[ΔP(r)/P(r) + ΔP(r')/P_z*(r')]
```

**Relative Importance:**
```
RI(M*) = Σ_{m∈M*} log(IE(m)+1) / Σ_{m∈M} log(IE(m)+1)
```

**Transformer Computation:**
```
h^l_t = h^(l-1)_t + a^l_t + m^l_t
where:
  a^l_t = A^l(h^(l-1)_1, ..., h^(l-1)_t)    [Attention]
  m^l_t = W_proj^l σ(W_fc^l h^(l-1)_t)      [MLP]
```

---

## Critical Quotes

*"Our experimental results indicate that LMs process the input by transmitting the information relevant to the query from mid-sequence early layers to the final token using the attention mechanism. Then, this information is processed by a set of MLP modules, which generate result-related information that is incorporated into the residual stream."*

*"The large IE observed at mid-sequence early MLPs [for factual knowledge] represents a difference in the information flow with respect to the arithmetic scenario, where the modules with the highest influence on the model's prediction are located at the end of the sequence."*

*"The size of the neuron overlap between arithmetic queries and number retrieval is considerably lower (22% and 23%), even though both tasks involve the prediction of numerical quantities."* [Circuits are task-specific, not just domain-specific!]

---

## Experimental Details

**Accuracy Results:**
- GPT-J: 67.8% overall (69.3% add, 78.0% subtract, 82.8% multiply, 40.8% divide)
- LLaMA 7B: 97.2% overall (best performer)
- Pythia 2.8B: 59.9% overall
- Goat 7B: 85.6% overall (fine-tuned on arithmetic)

**Templates Used:** 6 templates per operator (e.g., "How much is n₁ plus n₂?", "What is the sum of n₁ and n₂?")

**Sample Sizes:** 50 pairs per operator × 6 templates × 4 operators = 1200 intervention pairs
