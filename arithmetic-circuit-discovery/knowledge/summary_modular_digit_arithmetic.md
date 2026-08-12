# Modular Arithmetic: Language Models Solve Math Digit by Digit

**arxiv ID:** 2508.02513
**URL:** https://arxiv.org/html/2508.02513
**Authors:** Tanja Baeumel, Daniil Gurgurov, Yusser Al Ghussin, Josef van Genabith, Simon Ostermann (DFKI / Saarland University / CERTAIN)
**Published:** August 2025
**Models Studied:** LLaMA 3 8B, LLaMA 3 70B, OLMo 2 7B (multi-digit tokenization), Gemma 2 9B (single-digit tokenization)
**Code:** https://github.com/tbaeumel/transformer-digit-arithmetic

---

## TL;DR
Causal evidence that base LLMs solve **3-digit no-carry addition and subtraction** by running **three position-specific MLP "circuits" in parallel** — one for the units digit, one for the tens, one for the hundreds. Circuits are identified with a **Fisher Score** over MLP-neuron activations and validated via **interchange interventions** that selectively flip exactly one digit of the answer. Reconciles the prior "bag of heuristics" view (Nikankin et al. 2024) with the dual-pathway view (Lindsey et al. 2025): heuristics may be *neuron-level fragments* of larger digit-position circuits.

## Problem
Two prior camps disagree on how LLMs do arithmetic:
1. **Heuristic view** (Nikankin et al. 2024): a "bag of heuristics" — many narrow pattern-matchers, no coherent algorithm.
2. **Dual-pathway view** (Lindsey et al. 2025): magnitude estimation + units-digit pathway in Claude-3.5 Haiku.
3. **Probing view** (Levy & Geva 2024; Gould et al. 2023): digit values are linearly readable from the residual stream — but reading is not solving.

**Central question:** do LLMs *internally* exploit digit-wise representations to *solve* arithmetic, or just to encode it?

**Main claim:** for `347 + 231`, separate MLP subgroups compute `7+1`, `4+3`, and `3+2` in parallel, with little overlap.

## Method

### Step 1 — Fisher Score feature selection (correlational)

For each MLP neuron `i` (per layer), each digit position `d ∈ {hundreds, tens, units}`, group inputs by class `c ∈ {00…99}` formed from the concatenated operand digits at position `d`.

**Class-conditional statistics:**
- `μ_{i,c,d} = (1/|X_{c,d}|) Σ_{x ∈ X_{c,d}} a_i(x)`
- `σ²_{i,c,d} = (1/|X_{c,d}|) Σ (a_i(x) - μ_{i,c,d})²`
- `μ_{i,d} = (1/Σ_c |X_{c,d}|) Σ_c |X_{c,d}| μ_{i,c,d}`

**Fisher Score:**
```
F_{i,d} = [Σ_c |X_{c,d}| (μ_{i,c,d} − μ_{i,d})²]
          ─────────────────────────────────────────
            [Σ_c |X_{c,d}| σ²_{i,c,d}]
```

Numerator = between-class variance (signal). Denominator = within-class variance (noise). Threshold `t ∈ {0.4,…,1.0}`. Neurons with `F_{i,d} ≥ t` form circuit `C_{m,o,d,t}` for model `m`, operation `o`, position `d`. Search restricted to layers *after operand injection* (LLaMA 3 8B: layers ~15–24).

### Step 2 — Interchange interventions (causal)

200 paired prompts per condition (`D_add,op1`, `D_add,op2`, `D_sub,op1`, `D_sub,op2`). Base and source share one operand and differ in *every* digit of the other.

Using **pyvene** (Wu et al. 2024), MLP activations of digit-position circuit neurons in the **base** prompt at the **final token** are replaced with those from a **source** prompt across selected layers.

Outcomes labeled by which digits of the answer match base (`b`) vs. source (`s`): one of `bbb, bbs, bsb, sbb, bss, sbs, ssb, sss` (hundreds-tens-units).

**Prediction:** patching the units circuit shifts probability mass into `bbs` and *only* `bbs`. Tens → `bsb`. Hundreds → `sbb`.

## Datasets
- `D_add`, `D_sub`: 1000 prompts each, format `"157 o 431 = 588; A o B = "` with `A, B ∈ [100, 999]`, **no carries**, results also 3-digit.
- 200 intervention pairs per condition.

Baseline accuracy: LLaMA 3 8B 100/100, LLaMA 3 70B 100/100, OLMo 2 7B 99.0/99.5, Gemma 2 9B 98.5/99.5 — failures are not a confound.

## Key Results

### Circuit statistics (LLaMA 3 8B, addition)
- Average ~60.3% of MLP neurons per layer are flagged for *some* digit-position circuit
- Pairwise overlap between digit circuits **<2%** at thresholds ≥0.7 → mostly disjoint despite size
- LDA classification confirms selected neurons are sufficient for digit-pair classification (especially mid-late layers)

### Headline intervention numbers (Δ probability of target variant, optimal `t*`)

**LLaMA 3 8B, `D_add,op2`:**

| Circuit | t* | Target Δ | Baseline `bbb` Δ |
|---|---|---|---|
| Units | 0.6 | bbs **+30.93%** | −78.89% |
| Tens | 0.5 | bsb **+22.76%** | −57.29% |
| Hundreds | 0.9 | sbb **+45.56%** | −77.01% |

**LLaMA 3 8B, `D_sub,op2`:** units +28.50, tens +14.40, hundreds +36.27

**LLaMA 3 70B (addition):** units +23.16, tens +17.71, hundreds +29.68
**OLMo 2 7B (addition):** units +27.81, tens +42.57, hundreds +37.37

Off-target buckets (`bss, sbs, ssb, sss`) change minimally → specificity confirmed.

### Flip rates (top-1 prediction actually changes from `bbb` to target variant)

| Model / Op | Units | Tens | Hundreds |
|---|---|---|---|
| LLaMA 3 8B add | 51% | 33% | 68.5% |
| OLMo 2 7B sub | 64% | 53% | 54.5% |

### Single-digit tokenization (Gemma 2 9B)
Dominant **hundreds** circuit (70–90% of MLP neurons in many layers). Smaller tens, effectively no units (since the model emits one token at a time, the next token *is* the hundreds digit). Hundreds intervention nearly fully flips: **+91.55%** on `s` (addition, t*=0.6).

### Supporting analyses
- **Add vs. sub circuit overlap (top-100 neurons):** units 19%, tens 9.2%, hundreds 19.8% → largely distinct circuits per operation
- **Cosine similarity within circuits** materially exceeds random: e.g., layer 15, units (t=0.6): 0.84 vs random 0.72 ± 0.08
- **Carry handling NOT localized in digit circuits.** When tested on prompts that require carries, intervention still moves mass into `bbs/bsb`, not into carry-adjusted buckets `bb_{+1}s` / `b_{+1}sb`

### Heuristics vs. structured processing
Qualitative inspection of top-Fisher-score neurons (LLaMA 3 8B, layers 15–24):
- Neuron N_{19,136} (units circuit) implements **result mod 2** (parity)
- Neuron N_{23,2705} (hundreds circuit) implements **result range 900–999**

The heuristic *type* aligns with which digit circuit it belongs to. The authors argue heuristics may be neuron-level fragments of digit-position circuits — i.e. the two views are compatible.

## Limitations & Open Questions
1. Only addition and subtraction — multiplication/division left to future work
2. **Carry propagation is not in the digit circuits** — the modular story breaks where it gets hard
3. How digit-level results get composed into the final answer is not analyzed
4. Focus on MLPs; attention heads only treated implicitly
5. Fisher Score is univariate/linear — neurons in superposition or with multiplicative interactions could be missed entirely

## Key Equations Recap

**Fisher Score (digit-position selectivity):**
```
F_{i,d} = Σ_c |X_{c,d}| (μ_{i,c,d} − μ_{i,d})²  /  Σ_c |X_{c,d}| σ²_{i,c,d}
```

**Intervention measurement (per outcome bucket β ∈ {bbb,…,sss}):**
```
Δ P(β) = P_intervened(β) − P_base(β)
```

## Critical Quotes

*"The average number of MLP neurons per layer responsible for one of the digit-position specific circuits is 60.3% of all MLP neurons."*

*"LLMs solve arithmetic tasks in a far more organized way than previously thought."* — main claim

*"The bag of heuristics view and modular circuits are compatible — heuristics may be neuron-level fragments of digit-position circuits."*

## Tags
`mechanistic-interpretability` `arithmetic-reasoning` `digit-position-circuits` `fisher-score` `interchange-intervention` `causal-validation` `mlp-neurons` `bag-of-heuristics` `pyvene`

---

## Honest Critical Assessment

### Weakest assumption
**No-carry arithmetic is the load-bearing simplification.** The cleanness of the modular story depends on every digit position being algebraically independent in the inputs. The carry experiment (§4.2) shows that as soon as positions interact, the modular circuits do *not* explain the behavior. The title "LLMs solve math digit by digit" is true for *the easy half* of arithmetic; the mechanism for the hard half is acknowledged as unknown.

### Methodological gaps
- **Threshold selection is per-position-tuned to maximize the intervention effect.** Not strictly circular (Fisher precedes intervention), but reported `t*` values vary substantially (units 0.5–0.6, hundreds 0.9), and the threshold sweep is not the headline.
- **"Off-target buckets change minimally"** is asserted qualitatively. A tighter quantitative bound would strengthen specificity.
- **Tens-position effect on subtraction (+14.40)** is notably weaker than units/hundreds; the paper does not dwell on why.
- **Flip rates of 33%** (LLaMA 3 8B tens) mean that two-thirds of the time, patching the tens circuit *fails* to flip the predicted tens digit — the framing is stronger than the numbers.
- **Fisher Score is univariate.** Neurons in superposition or with multiplicative interactions are invisible to it. The paper acknowledges this.

### Likely failure modes
- Multi-digit numbers beyond 3 digits (untested)
- Multiplication/division (out of scope)
- Carry-heavy arithmetic (acknowledged negative result)
- Instruction-tuned models or chain-of-thought (all results on base, no-CoT)
- Word problems where digit injection layers shift

### Overhyped relative to evidence
The framing *"LLMs solve arithmetic tasks in a far more organized way than previously thought"* is stronger than the data warrants. What the data show: for no-carry 3-digit add/sub in base models, large MLP subgroups behave like position-specific circuits, and patching them produces directionally specific but partial flips. The reconciliation with heuristics is presented in the conclusion as a plausible hypothesis, not a result.

### Counter-intuitive finding worth flagging
Calling 60% of MLP neurons per layer a "circuit" challenges the intuition that interpretable circuits should be sparse. This matters when designing sparsity-based interp tools: sparsity ≠ modularity here.

---

## Why This Matters for the Beyond-Components Repository

Direct relevance to **`paper/main.tex` (The Fourier Basis of Digit Arithmetic)** and **`ARITHMETIC_CIRCUIT_PLAN.md`** — see `comparison_modular_digit_vs_repo.md` for the full deep comparison and critique.

**Short version:**
- Both papers find that base LLMs solve digit arithmetic via position-specific MLP-dominated mechanisms with little cross-position overlap
- 2508.02513 stops at *which neurons*; the repo's paper goes further and characterizes the **subspace structure** (perfect Fourier basis of ℤ/10ℤ) and the **mechanism** (trig-identity angle addition via CP tensor decomposition)
- 2508.02513's "60% of neurons in a circuit" paradox dissolves under the repo's framing: those neurons project onto a **9-D subspace**, and the actionable circuit is the subspace, not the neuron set
- The repo's `k=5` paradox (a strongly encoded but causally inert frequency) is a more rigorous version of the same observational-vs-causal gap that 2508.02513's Fisher Score-only analysis cannot diagnose
