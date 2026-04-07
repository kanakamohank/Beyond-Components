# Arithmetic Circuit Discovery in Phi-3 Mini — Synthesis

## Model
**microsoft/Phi-3-mini-4k-instruct** (32 layers, 32 heads/layer, d_model=3072)

## Task
Two-operand addition: `a + b =` where a,b ∈ [1, 50]. Two-shot prompting with worked examples.
Baseline accuracy: **100%** (greedy generation, 50 problems).

---

## 1. Information Flow Map (Position-Aware Layer Patching)

Position-aware activation patching reveals a clear three-phase information flow:

```
              operand_a    final(=)
L 0-L 8:     ~1.0-1.2     ~0.05      ← Info lives at operand token
L10-L12:      0.42-0.73    0.37-0.75  ← CROSSOVER: routing happening
L16-L19:      0.22-0.31    0.62-0.87  ← Computation at answer position
L20+:         ~0.00        ~1.0       ← Done, answer fully at = position
```

**Key insight**: The crossover at L10-L12 marks where operand information leaves
the operand token position and arrives at the answer (`=`) position. By L20 the
answer is fully computed at the final position.

## 2. The Three-Phase Circuit

### Phase 1: Routing (L8-L15) — Attention Heads

Attention heads copy operand values from their token positions to the `=` position.

**Top routing heads** (attention from `=` → operand_a):

| Head     | →op_a  | →op_b  | →plus  | Role              |
|----------|--------|--------|--------|-------------------|
| L15 H7   | 0.276  | 0.012  | 0.099  | Primary A-router  |
| L16 H30  | 0.249  | 0.009  | 0.056  | Primary A-router  |
| L12 H29  | 0.188  | 0.006  | 0.023  | A-router          |
| L13 H0   | 0.162  | 0.006  | 0.170  | A + context       |
| L16 H20  | 0.143  | 0.044  | 0.099  | Dual-operand      |
| L19 H20  | 0.141  | 0.024  | 0.086  | Late A-router     |
| L15 H28  | 0.131  | 0.043  | 0.291  | Context router    |
| L9 H28   | 0.124  | 0.026  | 0.143  | Early router      |

**Causal confirmation** (operand-position head patching, recovery when patching at op_a):

| Head     | Recovery | Role         |
|----------|----------|--------------|
| L9 H30   | 0.146    | Early router |
| L13 H8   | 0.146    | Mid router   |
| L8 H0    | 0.140    | Early router |
| L5 H13   | 0.114    | Very early   |

**Critical finding**: Ablating 20 attention heads simultaneously causes **0% accuracy drop**.
Routing is massively redundant — the model has many alternative paths to move operand
information to the answer position.

### Phase 2: Computation (L18-L27) — MLP Layers

MLP layers perform the actual addition computation. This is the core of the circuit.

**Component-level patching recovery** (patching clean MLP output into corrupted run):

| MLP  | Recovery | Role            |
|------|----------|-----------------|
| L31  | 0.477    | Output + compute |
| L26  | 0.300    | Core compute    |
| L22  | 0.299    | Core compute    |
| L24  | 0.250    | Core compute    |
| L23  | 0.240    | Core compute    |
| L21  | 0.240    | Core compute    |
| L19  | 0.211    | Early compute   |
| L18  | 0.169    | Early compute   |

### Phase 3: Output (L31) — MLP + Unembedding

L31 MLP has the highest single-component patch recovery (0.477), combining
final computation with output formatting for the unembedding matrix.

## 3. Necessity Testing — Cumulative Ablation

### Single-component ablation
Ablating ANY single component (head or MLP) → **0% accuracy drop**.
The circuit is highly redundant at the individual component level.

### Cumulative MLP ablation (the key experiment)

| K ablated | Accuracy | Drop   | Layers ablated                    |
|-----------|----------|--------|-----------------------------------|
| 1         | 100%     | 0%     | L31                               |
| 2         | 96%     | 4%     | L31, L26                          |
| **3**     | **58%**  | **42%**| **L31, L26, L22**                 |
| **4**     | **22%**  | **78%**| **L31, L26, L22, L24**            |
| **5**     | **8%**   | **92%**| **L31, L26, L22, L24, L23**       |
| 6         | 2%       | 98%    | + L21                             |
| **7**     | **0%**   | **100%**| **+ L19**                        |

**The arithmetic computation is distributed across 7 MLP layers (L19-L27 + L31).**
No single MLP is necessary, but 3 together account for 42% of the capability,
and 7 are sufficient to completely ablate arithmetic.

### Cumulative head ablation

| K ablated | Accuracy | Drop |
|-----------|----------|------|
| 5         | 100%     | 0%   |
| 10        | 100%     | 0%   |
| 15        | 100%     | 0%   |
| 20        | 100%     | 0%   |

**Heads are completely dispensable.** Even ablating 20 heads simultaneously has no effect.

### Layer-range MLP ablation

Ablating ALL MLPs in any contiguous range of 8+ layers → 0% accuracy.
This includes L0-L15, confirming even early MLPs contribute (likely to operand encoding).

## 4. Logit Lens

The standard logit lens shows 0% top-1 accuracy at ALL layers — the "unembedding trap."
The answer rank drops steadily: L0 (18k) → L11 (959) → L17 (39) → **L21 (6.9, best)** → L30 (9.2).
The representation encodes the answer but never aligns perfectly with W_U until generation.

## 5. Relationship to Prior Helix/SVD Work

| Experiment | Finding | Reconciliation |
|------------|---------|----------------|
| SVD mask pipeline (Phases 2-5) | Output layers L28-31 important | Confirmed: L31 MLP has highest single recovery (0.477). But the pipeline missed L18-27 because KL-divergence was biased toward output formatting. |
| OV helix (L24 H28) | Helical structure in OV matrix | L24 is part of the core computation zone (MLP recovery 0.250). The helix may encode number-magnitude features used by L24's MLP for computation. |
| Goldilocks Layer (L24) | Semantic separation peaks at L24 | L24 is indeed where MLPs actively compute on operand representations. The "semantic peak" is the computation representation, not output formatting. |
| LayerNorm angle preservation | Angular encoding is an architectural necessity | Consistent with distributed MLP computation — LayerNorm preserves the angular features that MLPs read. |

## 6. Key Conclusions

1. **Arithmetic in Phi-3 is MLP-dominated.** Attention heads route information but are individually and collectively dispensable. MLPs perform the actual computation.

2. **The computation is highly distributed.** No single component is necessary. The model uses ~7 MLP layers (L19-L27 + L31) with graceful degradation — a "distributed ensemble" rather than a single "calculator module."

3. **Three phases**: Routing (heads, L8-15, redundant) → Computation (MLPs, L18-27, distributed) → Output (L31, formatting).

4. **The position crossover at L10-12** marks the handoff from operand-position processing to answer-position processing.

5. **Base-10 addition does NOT use a single Fourier/helical circuit.** Unlike modular arithmetic in grokking models, standard addition in a pretrained LLM uses distributed MLP computation, consistent with learned lookup/interpolation rather than algorithmic Fourier decomposition.

## 7. Open Questions

- **Path patching**: Which specific routing heads feed into which computation MLPs? (Node patching done; path patching needed for the full circuit graph.)
- **MLP internals**: What do the L22/L26 MLPs actually compute? Do they implement carrying, digit-level lookup, or something else?
- **Operand B routing**: Most routing heads preferentially attend to operand A. How does operand B information reach the computation MLPs?
- **Scaling**: Does this distributed MLP pattern hold in larger models (Phi-3 Medium, Llama 3.1 70B)?
