# Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits

*Ahmad, Joshi, Modi (IIT Kanpur). arXiv:2511.20273*

---

## 1. Overview & Motivation

Mechanistic interpretability has converged on a working unit: the *component* (an attention head, an MLP layer). Prior work identifies, e.g., a "name mover head" in GPT-2 Small for the Indirect Object Identification (IOI) task and treats that head as a single functional primitive. This paper argues that treatment is too coarse: a single attention head's QK or OV matrix often **superposes several unrelated sub-computations along orthogonal directions** of its weight-space SVD. The core claim is that you can take a frozen pretrained transformer, decompose each head/MLP via SVD into singular directions, and *learn a diagonal mask* over those directions that (a) keeps almost all of the model's behavior on a task and (b) reveals each retained direction as a distinct, human-interpretable sub-function (e.g., one direction in head 9.6 detects sequence-initial tokens, another marks named entities). On IOI in GPT-2 Small they retain ~9% of the directions while preserving KL of 0.21 and exact-match accuracy of 0.77.

---

## 2. Key Prerequisites

**(a) Residual-stream view of transformers.** Every block reads from and writes to a single high-dimensional residual stream. Each head's output is `OV @ attention-weighted inputs`, projected back into the residual via `W_O`. Heads "communicate" by writing into low-rank subspaces of this stream — that's what makes a directional analysis natural.

**(b) The QK/OV factoring (Elhage et al. 2021).** An attention head decomposes cleanly into two bilinear forms: `W_QK = W_Q W_K^T` (which token attends to which) and `W_OV = W_V W_O` (what content is moved when an attention edge fires). This paper does *not* analyze `W_Q`/`W_K`/`W_V`/`W_O` individually — it analyzes the *composed* matrices, because those are what causally affect the residual stream.

**(c) SVD as a basis for "directions in weight space".** For any matrix `M = U Σ V^T`, the columns of `V` are orthogonal *input* directions, the columns of `U` are orthogonal *output* directions, and σᵢ tells you how strongly the matrix couples the i-th input direction to the i-th output direction. The intuition: if a head's `W_OV` is rank-r, it can read at most r independent "features" from the stream and write at most r independent "concepts" back. SVD lets you enumerate them.

**(d) Faithfulness via masking + KL.** To check whether a sub-circuit is doing the work, interpretability work commonly *ablates* parts of the model and measures the KL divergence between the masked and original output distribution. Low KL = faithful. The "circuit" is then the minimal set of components that keeps KL low. This paper does the same thing but at the *direction* level instead of the component level.

---

## 3. The Method — Step by Step

### Step 1: Reformulate every component as a single linear map

Attention heads and MLPs are written as standalone matrices that include their biases by augmenting inputs with a constant `1`:

- For each head h, build `W_aug^(QK)` (a `(1+d_model)×(1+d_model)` matrix that absorbs `b_Q`, `b_K`) and `W_aug^(OV)` (`(1+d_model)×d_model`, absorbing `b_V` and a `1/|H|` share of `b_O`).
- For MLPs, `W_aug^(in)` and `W_aug^(out)` similarly absorb `b_in` and `b_out`.

This is just bookkeeping — it makes every component a single linear operator (with no separate bias term to manage during SVD).

### Step 2: SVD each augmented matrix

`W_aug = U Σ V^T`. Effective ranks reported: QK = 64, OV = 65 (one extra direction for the bias row), MLPs keep their full nonzero spectrum. Crucially, all weights are **frozen** — only the mask is learned.

### Step 3: Insert a learnable diagonal mask in the singular spectrum

Replace `W_aug` with `Ũ = U Σ M V^T` where `M = diag(m_1, …, m_r)` and each `m_i ∈ [0,1]`. `m_i = 1` keeps direction i intact; `m_i = 0` deletes it.

For QK matrices specifically, they keep only the masked term and discard the complement (`U Σ (I − M) V^T`) because softmax is non-linear — superimposing two QK kernels through softmax is not equivalent to summing their attention maps. For OV and MLP, the masked + complement decomposition is mathematically exact (linear), so they could be split additively, but only the masked piece is used in the forward pass.

### Step 4: Train masks with KL + L1, using clean/corrupted pairs

For each task they have a clean prompt `x` and a corrupted prompt `x_corrupt` (e.g., names swapped). For each direction k, the input fed through that direction is a *gated mixture*: `m_k · x + (1 − m_k) · x_corrupt`. So if `m_k → 0`, that direction contributes the corrupted activation — it's effectively "ablated to the counterfactual mean", which is the standard activation-patching protocol but applied per singular direction.

Loss: `L = KL(p(y|x) ‖ p_M(y|x)) + λ ||diag(M)||_1`. KL keeps task behavior, L1 pushes masks toward zero. AdamW, λ = 1.5e-4, lr = 1e-2, 15 epochs (up to 150 with early stopping).

### Step 5: Inspect the surviving directions

For each direction with `m_k` significantly above 0, look at:
- *Input activations* `ν^T u_k` per token (which tokens excite this direction?)
- *Output projection* `v_k^T W_U` — the "logit receptor": a fixed direction in vocabulary space that this singular component always pushes toward (the magnitude is set by the input-dependent scalar `ν^T u_k σ_k`).

This last move is the interpretability payoff: each surviving OV direction is a *constant* vocabulary vector scaled by an input-dependent gain, which makes it directly inspectable.

---

## 4. Mathematics — Go Deep

### 4.1 The unified linear view

**Attention scores (standard):**

$$\alpha_{ij}^{(h)} = \mathrm{Softmax}_j\!\left(\frac{q_i^{(h)} \cdot k_j^{(h)\top}}{\sqrt{d_{\text{head}}}}\right)$$

with `q_i = x_i W_Q + b_Q`, `k_j = x_j W_K + b_K`. Symbols: `x_i` is the residual-stream vector at position i, `W_Q, W_K ∈ ℝ^{d_model × d_head}`, biases are head-specific. The dot product `q_i · k_j^T` is the bilinear form `x_i W_Q W_K^T x_j^T`, plus bias terms.

**Augmented QK matrix:**

$$W_{\text{aug}}^{(QK)} = \begin{pmatrix} b_Q b_K^\top & b_Q W_K^\top \\ W_Q b_K^\top & W_Q W_K^\top \end{pmatrix}$$

In plain English: append a constant `1` to every input vector, and `[1, x_i] · W_aug^(QK) · [1, x_j]^T` reproduces the full bilinear form including all four bias-cross-terms. This turns "QK + four bias terms" into one matrix you can SVD.

**OV augmented matrix:**

$$W_{\text{aug}}^{(OV)} = \begin{pmatrix} b_V W_O^{(h)} + \tfrac{1}{|H|} b_O \\ W_V W_O^{(h)} \end{pmatrix} \in \mathbb{R}^{(1+d_{\text{model}}) \times d_{\text{model}}}$$

The `1/|H|` term distributes the shared output bias `b_O` across heads. (The paper doesn't justify why the *equal* split is the right attribution — see §6.)

**MLP:** Just stacks bias on top of weight: `W_aug^(in) = [b_in; W_in]`, similarly for output. The nonlinearity `f` (GELU in GPT-2) sits between them and is *not* touched by the decomposition.

### 4.2 The masking and faithful split

$$\widetilde{W}_{\text{aug}} = U \Sigma \mathcal{M} V^\top, \qquad \mathcal{M} = \mathrm{diag}(m_1,\ldots,m_r),\ m_i\in[0,1]$$

The exact additive identity:

$$U\Sigma V^\top = U\Sigma\mathcal{M} V^\top + U\Sigma(\mathcal{I}-\mathcal{M})V^\top$$

Intuition: every direction either lives in the "kept" subspace or the "ablated" subspace, and the original weight is the sum. For OV/MLP the residual stream sees a linear sum so this split is exact at the matrix level. For QK they keep only `M` and drop `(I−M)` because the softmax destroys linearity — running both kernels through one softmax does not equal the convex combination of their attention maps. **Caveat:** this means QK ablation is not bit-equivalent to "running both halves separately"; the paper handles this by faithfulness loss but doesn't prove a tighter equivalence.

### 4.3 The training objective

$$\mathcal{L}_{\mathcal{M}} = \mathrm{KL}\!\left[p(y|x)\,\|\,p_{\mathcal{M}}(y|x)\right] + \lambda\,\|\mathrm{diag}(\mathcal{M})\|_1$$

with the masked forward pass

$$p_{\mathcal{M}}(y|x) = f\!\left(\sum_k\bigl(m_k x + (1-m_k)x_{\text{corrupt}}\bigr)\,\sigma_k u_k v_k^\top\right)$$

Symbols: `p(y|x)` is the original model's next-token distribution, `p_M` is the masked model's, `σ_k, u_k, v_k` are the k-th SVD components, `f` collapses the rest of the forward pass.

In words: every singular direction independently "decides" (via its scalar `m_k`) whether it sees the clean or the corrupted input. KL forces directions critical for the task to keep `m_k ≈ 1`; L1 punishes any direction not pulling its weight.

**Underspecified bits.** The paper writes the masked forward pass as a sum over directions, suggesting all directions of all components are masked simultaneously, but it does not spell out: (i) whether one global mask is trained per layer/head/MLP or one mask per matrix per task, (ii) how `x_corrupt` is paired token-by-token with `x` when prompts have different tokenizations, and (iii) what "f" is — formally, the forward pass is not a simple function of one component's output because of LayerNorm and downstream blocks. The clean/corrupted mixture is therefore a per-direction *input substitution*, not an output ablation, which is a meaningfully different operation than standard activation patching.

### 4.4 Logit receptors

For an attention output post-`W_O`, projected to the unembedding `W_U`:

$$y_{lh} W_U = \sum_k \sigma_{lhk}\,(\nu_{lh}^\top u_{lhk})\,(v_{lhk}^\top W_U)$$

Symbols: `l` layer, `h` head, `k` singular index, `ν_{lh}` is the head's input (post-attention-weighted), `u_{lhk}` and `v_{lhk}` are left/right singular vectors, `σ_{lhk}` the singular value.

The factor `v_{lhk}^T W_U` is **fixed per direction** — it's just a vector in vocabulary space — and the only input dependence is the scalar `ν_{lh}^T u_{lhk}`. So each direction is *a fixed vocabulary push, gated by a scalar feature detector*. This is the cleanest statement of why singular-direction analysis is interpretable: each direction is a one-feature → one-logit-vector circuit.

### 4.5 Causal intervention (Algorithm 2)

For OV direction k, given an input where the token-mean activation is `ν^T u_k`, replace it with `a'_i` (the mean activation under the *opposite* gender) and amplify:

$$\Delta R = (a'_i - \nu^\top u_i)\,\sigma_i\,v_i^\top$$

Add `ΔR` to the residual stream and re-decode. With `σ_scale = 20` they get 100% pronoun flip rates in the GP task. This is the strongest causal claim in the paper — directions identified by the mask aren't just *correlationally* sufficient, they're *causally* steerable.

---

## 5. Experiments & Results

### Setup
- Frozen GPT-2 Small (124M).
- Three classic mech-interp tasks: IOI (1k train), Greater-Than (2k), Gender Pronoun (1k).
- Single A40 (48GB).
- Effective ranks: QK=64, OV=65, MLPs full.

### Headline numbers (Table 1)

| Task | Direction sparsity | KL(masked‖orig) | Acc. masked / full | Exact match |
|---|---|---|---|---|
| IOI | 91.3% pruned | 0.21 | 0.70 / 0.79 | 0.77 |
| GT  | 95.2% pruned | 0.23 | — | 0.33 |
| GP  | 96.8% pruned | 0.13 | 0.75 / 0.77 | 0.86 |

So 3–9% of singular directions retain near-original behavior across three tasks. Note: GT has no masked accuracy reported (only exact-match 0.33, vs no full-model baseline given), which makes that row hard to interpret.

### Functional decomposition of head 9.6 on IOI

The "name mover" head decomposes into at least three distinct sub-functions along orthogonal directions:
- **S₁** (mask 0.53): activations 20–25× higher on the first token of a sequence — sequence-initialization detector.
- **S₇** (mask 0.64): entities +3.52±1.42 vs actions −4.44±0.68 — entity/action axis.
- **S₂₈** (mask 0.97): named entities (e.g. "Kevin": 5.22) ≫ function words ("the": 0.50) — entity salience.

This is the strongest qualitative result: a head previously summarized as one functional role splits into orthogonal, interpretable axes.

### Logit receptors on GP

For pronoun resolution, three OV directions stand out:
- L9.H7.SV1: mask 1.00, σ=8.87, ν^T u differs by **+0.568** between "he" and "she" contexts → masculine receptor.
- L10.H9.SV0: mask 1.00, σ=9.15, diff **−0.925** → feminine receptor.
- L11.H7.SV0: mask **1.3e-5** (zeroed out), σ=22.07 (largest) — high variance, but `ν^T u` is identical (+0.755 vs +0.774) across genders, so the optimization correctly prunes a direction whose σ is large but discriminatively useless.

**This is the key ablation-style finding:** singular value is *not* a good importance ranking on its own — discriminative variance is what matters.

### Causal interventions (GP)

| Intervention | Prompt | Baseline Δlogit | Intervened Δlogit | Flip rate |
|---|---|---|---|---|
| Swap all gender directions | "he" | +2.53 | −42.31 | 100% → she |
| Swap all | "she" | +2.84 | −40.82 | 100% → he |
| Swap masc-only | "he" | +2.53 | −18.15 | 100% → she |
| Swap fem-only | "she" | +2.84 | −27.59 | 100% → he |

Even single-axis interventions flip 100% of predictions, which is strong causal evidence that each direction independently encodes a usable gender signal.

### Do the results support the claims?
- **R1 (few directions suffice):** Yes — 3–9% retention with low KL.
- **R2 (alignment with prior circuits):** Partially shown via head 9.6 matching prior IOI literature; cross-task circuit alignment is not systematically tabulated.
- **R3 (decompose known heads):** Demonstrated for one head (9.6) on IOI and a few on GP. Not shown across all heads or tasks.
- **R4 (discover new axes):** Claim is made (S₁ "sequence initialization") but evidence is qualitative.

---

## 6. Honest Critical Assessment

**Weakest assumption: SVD axes = interpretable axes.** The paper assumes that singular directions of `W_aug` correspond to causally meaningful sub-functions. But SVD is invariant to nothing about transformers' learned structure — it's a generic matrix factorization that *happens* to give orthogonal directions. There's no a priori reason a head trained with SGD has its sub-functions axis-aligned with the singular basis of its weight matrix. The paper itself flags this in Limitations 2 and 3 (the diagonal mask "restricts optimization to axis-aligned subspaces"). The interpretability win on head 9.6 may partly be a self-fulfilling prophecy: the optimization keeps directions that happen to be SVD-aligned and interpretable, while non-axis-aligned sub-functions are simply invisible to this method.

**Cherry-picking risks.**
- **One head, three directions.** Head 9.6 is the only deep-dive functional decomposition. We don't know whether other heads on IOI yield equally clean 3-axis stories or chaotic mixtures.
- **GT is under-reported.** Greater-Than is included for sparsity (95.2%) and KL (0.23) but the paper gives only "exact match 0.33" with no full-model baseline. This row could be cleaning up noise rather than preserving capability.
- **Logit-receptor results focus on three GP directions.** The non-discriminative high-σ direction L11.H7.SV0 makes a great point, but we're not told how many other directions had similar patterns or how the three were selected from the surviving set.

**Likely failure modes.**
- **Polysemantic non-axis-aligned directions.** If a head's two sub-functions live in a 2D subspace at 45° to the SVD axes, the diagonal mask cannot disentangle them — both will be kept or both dropped.
- **MLP interpretability is asserted, not demonstrated.** The method extends to MLPs but the qualitative analysis is entirely about attention heads. With GELU between `W_in` and `W_out`, the singular-direction story is much weaker for MLPs and the paper does not show a single interpretable MLP direction.
- **Scaling.** All experiments are GPT-2 Small (124M). Modern models are 100–10000× larger with much wider heads; whether 3% of directions still suffice — and remain interpretable — is open.
- **Forward-pass dependence on LayerNorm and downstream blocks.** Masking one component changes downstream activations non-linearly; the KL objective handles this in aggregate but the per-direction "this direction does X" stories are conditioned on every other direction being unchanged. Compositional claims are not validated.

**Overhyped vs evidence.**
- The framing "beyond components" is somewhat oversold: prior work (Merullo et al., Gao et al., Cunningham et al., cited) has already explored low-rank/SVD analyses. The novelty is the *learned diagonal mask in singular space* with a clean/corrupted forward pass — a concrete and useful contribution, but more incremental than the title suggests.
- The "discovers new functional axes" claim rests on qualitative inspection of activation values per token. There's no quantitative test (e.g., probe accuracy, transfer to new prompts) that S₁ generalizes beyond the IOI training set.

**Underspecified.** As noted in §4: the corrupted-input pairing protocol, the mask granularity (per-component vs global), the full forward-pass equation when many components are masked simultaneously, and the choice of `1/|H|` allocation for `b_O` are all worth a clearer treatment.

---

## 7. Takeaways

- **Read this if you do mech-interp on small/medium transformers.** The augmented-matrix-then-SVD-then-learn-a-mask recipe is concrete, easy to implement, and gives finer-grained results than head-level circuit analysis.
- **The "logit receptor" formulation (§4.4) is the most reusable idea.** Decomposing `OV @ W_U` into fixed vocabulary directions × scalar feature detectors is a clean primitive that other interpretability tools could adopt.
- **Take the head-9.6 decomposition as a proof-of-concept, not a general result.** It convincingly shows multiple sub-functions exist within one head; it does not show this is universal or that SVD always finds the right axes.
- **Causal interventions on GP (100% flip rate) are the strongest evidence in the paper.** If you're skeptical that direction-level analyses are causally meaningful, that table is what to look at first.
- **Skip if you need anything beyond GPT-2 Small.** No scaling experiments, no instruction-tuned models, no MLP-level qualitative results — generalization is entirely future work.

---

## 8. Is This a Gender-Debiasing Paper?

**Short answer: No.** Gender is used only as a *case study* for the interpretability method.

- The **Gender Pronoun (GP) task** (resolving "he"/"she" in tag-question prompts like *"So David is a really great friend, isn't __"*) is one of three test beds — alongside IOI and Greater-Than — used to validate the **singular-direction decomposition method**.
- The contribution is an **interpretability framework**: decompose attention heads / MLPs via SVD, learn a diagonal mask, and identify which singular directions matter for a behavior.
- The "gender directions" they find (e.g., L9.H7.SV1 = masculine receptor, L10.H9.SV0 = feminine receptor) are presented as *evidence that distinct sub-functions live on orthogonal singular axes* — not as a debiasing artifact.
- The **causal intervention** (Algorithm 2, the 100% pronoun flip rate) is offered as **causal validation** that the identified directions are real, not as a debiasing technique. It steers a pronoun prediction in a controlled prompt; it does not address bias in any harm/fairness sense, doesn't measure stereotype associations, and is not evaluated on debiasing benchmarks (e.g., StereoSet, CrowS-Pairs, WinoBias).

**Could the technique be repurposed for debiasing?** Potentially — direction-level ablation/steering is conceptually adjacent to methods like INLP or concept-erasure. But the paper makes no such claim, runs no debiasing evaluation, and does not discuss fairness implications. Treating it as a debiasing paper would be reading something into it that isn't there.