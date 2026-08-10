# Deep Comparison: Two External Arithmetic Papers vs. The Repo's Fourier Arithmetic Work

**Papers under comparison:**
- **External A:** Zhou, Fu, Sharan, Jia 2024, *Pre-trained Large Language Models Use Fourier Features to Compute Addition* (arXiv:2406.03445, NeurIPS 2024). See `summary_fourier_features_addition.md`.
- **External B:** Baeumel et al. 2025, *Modular Arithmetic: Language Models Solve Math Digit by Digit* (arXiv:2508.02513). See `summary_modular_digit_arithmetic.md`.
- **Internal:** `paper/main.tex` — *The Fourier Basis of Digit Arithmetic: Mechanistic Interpretability of Addition Circuits in Language Models*. Plus `ARITHMETIC_CIRCUIT_PLAN.md` and the implementation under `experiments/` and `mathematical_toolkit_results/`.
- **Adjacent context:** `summary_arithmetic_reasoning.md` (Stolfo et al. 2305.15054), `summary_trigonometry_addition.md` (Kantamneni & Tegmark 2502.00873 "helix"), `summary_arithmetic_circuits.md` (2402.02619), `summary_grokking_mechanistic.md` (Nanda et al. 2301.05217).

---

## 1. Bird's-Eye View

| Axis | Zhou et al. (2406.03445) | Baeumel et al. (2508.02513) | Repo paper (Fourier basis) |
|---|---|---|---|
| **Question** | *What computational primitive does addition use?* | *Which neurons own which digit position?* | *What subspace and what algorithm encodes a digit?* |
| **Granularity** | Per-frequency Fourier components in logit space | MLP neuron sets per (model, op, position) | 9-D Fourier subspace of `ℤ/10ℤ` per layer |
| **Discovery tool** | DFT of per-module logits (`W^U · Attn^{(ℓ)}`, `W^U · MLP^{(ℓ)}`) | Fisher Score on neuron activations (univariate) | SVD of per-digit means + DFT labeling (subspace) |
| **Causal tool** | Fourier-band filter `F(h; Γ)` (low-pass / high-pass per module) | Interchange intervention via pyvene | Subspace patching (Fisher / unembed-aligned / random) + Fourier knockout + phase rotation |
| **Models** | GPT-2-XL primary, GPT-2-small, GPT-J, Phi-2, GPT-3.5/4, PaLM-2 | LLaMA 3 8B/70B, OLMo 2 7B, Gemma 2 9B | Gemma 2B, Phi-3 Mini, LLaMA 3.2-3B |
| **Operands** | Single-token integers, `a, b ≤ 260`, no carry framing | 3-digit, **no carry** | 1-2 digit, ones digit only (with separate carry/multi-digit experiments) |
| **Operations** | Addition (multiplication in appendix) | Addition + subtraction | Addition (subtraction tested as generalization) |
| **Headline finding** | **Approximation/MLP/low-freq** vs. **classification/Attn/high-freq** division of labor | Three position-specific MLP circuits (~60% of neurons each, <2% overlap) | A 9-D Fourier subspace, identical to irreducible reps of `ℤ/10ℤ`, that is necessary, sufficient, and specific |
| **Mechanism claim** | Two-mechanism logit construction (smooth bump × comb) | Modular *parallelism* across positions | Modular parallelism + **trig-identity angle addition** (CP tensor decomposition, σ²-weighted score 0.964 on Gemma) |
| **Causal triangle** | Frequency-band ablation establishes necessity per (module, band); no specificity control | Necessity-only (intervention shifts probabilities) | Necessity (multi-layer knockout → ≤19% accuracy) + Sufficiency (Fisher 9-D transfers ≥83%) + Specificity (random 9-D = 0% effect) |
| **Origin claim** | Fourier features inherited from pre-trained `W^E`; freezing it alone gives 100% on GPT-2-small | Not addressed | Cross-architecture convergence; pre-training origin not directly tested |
| **Steering / control** | Not attempted | Not attempted | Fourier phase rotation + W_U-informed steering: 70–88% exact digit shift |

All three papers point at *the same underlying phenomenon* — Fourier-domain digit representation in pretrained LLMs — at three different levels of abstraction. The repo paper sits at the **subspace/mechanism** level and is strictly more granular than either external paper.

---

## 2. Where They Agree (and Why That Convergence Matters)

### 2.1 Position-specific, MLP-dominated processing
Both find that ones-digit information lives primarily in **MLPs** at *post-injection* mid-late layers, with attention acting more as router than computer. This is consistent with `summary_arithmetic_reasoning.md` (late MLPs at last token, RI = 40.2% for arithmetic vs. 4.4% when result is fixed) and with `summary_trigonometry_addition.md` (builder MLPs at middle layers).

The convergence across **two independent papers, six different models, three architecture families, two granularities** (neuron sets vs. SVD subspace) is strong evidence the finding is real and not a methodological artifact.

### 2.2 Modularity of digit positions
- 2508.02513: <2% pairwise overlap between hundreds/tens/units neuron circuits
- Repo paper: separate Fourier substructures emerge at separate computation layers; multi-digit experiments (Phase F, Step 15) show tens-digit Fourier basis distinct from ones-digit basis.

### 2.3 Carry handling is *not* in the modular story
Both papers find that carry propagation is **outside the clean modular circuit**:
- 2508.02513 §4.2: digit-position interventions on carry prompts still produce non-carry-adjusted outputs.
- Repo paper Step 11 (`carry_stratification.py`): carry vs. no-carry subspaces are aligned but not identical; the repo paper's discussion explicitly leaves "multi-digit carry propagation circuits unexplored" as a limitation.

This is the same negative result, found independently, with two completely different methodologies. That convergence makes it credible.

### 2.4 Single-token vs. multi-token tokenization matters
- 2508.02513: Gemma 2 9B (single-digit tokenization) collapses to a dominant **hundreds** circuit only — because the model emits one token at a time and the next token *is* the hundreds digit.
- Repo paper: LLaMA 3.2-3B requires `--direct-answer` mode for the same reason; the repo handles this with mode switching.
- 2406.03445: explicitly *requires* single-token integers (`a, b ≤ 260`) for the entire mechanism to be visible — its Fourier basis is defined on ℤ/(p−1)ℤ where p = 521 is the single-token range +1.

Three independent papers converge on tokenization as a structural prerequisite. The repo paper's mode-switching pipeline is the cleanest operational handling of this.

---

## 2bis. Where Zhou et al. (2406.03445) Sits in This Picture

Zhou et al. is the **canonical reference** for "Fourier features in pretrained LLMs do arithmetic" and predates both 2508.02513 and the repo paper. Its position relative to the repo paper is different from Baeumel et al.'s — Zhou et al. is **more aligned in framework** but **less rigorous in causal evidence**.

### 2bis.1 Where Zhou et al. and the repo paper agree precisely
- **Fourier features are real.** Both papers see outlier components at periods 2, 2.5, 5, 10 in the same kinds of spectra. The repo paper formalizes this as the irreducible representations of `ℤ/10ℤ` (k = 1..5 frequencies giving 9 dimensions); Zhou et al. observes the same outliers empirically without identifying them as the rep-theoretic basis.
- **MLPs do the heavy lifting.** Zhou et al.'s ablation: removing low-freq from MLP drops 99.74 → 35.89%. Repo paper's Step 4 (`fourier_knockout.py`): multi-layer MLP-dominated Fourier ablation drops to 12–19%. Same conclusion via different methodology.
- **The mechanism lives in pre-training.** Zhou et al.'s frozen-`W^E` GPT-2-small (random other layers → 100% addition) is the cleanest "inductive bias is in the embeddings" demonstration. The repo paper's cross-architecture convergence (3 model families all show the same 9-D structure) is consistent with this; it's circumstantial vs. Zhou et al.'s direct test.

### 2bis.2 Where Zhou et al. is genuinely stronger than the repo paper
- **Pre-training origin claim is causal.** The frozen-`W^E` GPT-2-small experiment is a clean dissociation that the repo paper has no analog for. The repo paper's "different architectures converge on the same Fourier structure" is suggestive but not causal evidence about *where* the bias lives.
- **Module-level division of labor is named.** Zhou et al. crisply states: MLPs ≈ approximation (low-freq), Attention ≈ classification (high-freq). The repo paper's framing is "MLPs dominate computation, attention routes" — less mechanistic. The repo paper *could* extract a Zhou-style frequency-band-by-module ablation directly from its `fourier_head_attribution.py` (Step 8 Phase 3 already does per-frequency component attribution), but does not currently report attention-vs-MLP at the frequency-band level.
- **The error-mode prediction.** Zhou et al. predicts and verifies that low-freq-MLP-ablation produces off-by-10/50/100 errors and high-freq-Attn-ablation produces off-by-<6 errors. The repo paper's ablations measure accuracy but do not characterize *the shape of the residual errors* this way. This is a missed opportunity — would directly support the repo's mechanism story.

### 2bis.3 Where the repo paper is strictly stronger than Zhou et al.
- **Specificity control.** Zhou et al.'s Fourier filter `F(h; Γ)` zeros out a frequency band from a module — that's *necessity*. There is no random-band control showing that an *arbitrary* band of the same size has zero effect. The repo paper's matched-dimension random-subspace ablation (0% effect) is the missing piece.
- **Sufficiency.** Zhou et al. shows that *removing* a frequency band damages performance. It does not show that *patching only* that band's worth of activation transfers digit information from a clean to a corrupted run. The repo paper's Fisher 9-D patching transfer (83–100%) is the missing positive direction.
- **Subspace structure.** Zhou et al. analyzes one frequency at a time. The repo paper identifies the **9-dimensional subspace** as a single object — a perfect Fourier basis of the irreducible reps of `ℤ/10ℤ` — and treats it as the unit of intervention. This generalizes Zhou et al.'s per-frequency picture into a coherent algebraic object.
- **Mechanism: trig-identity angle addition.** Zhou et al. says low-freq + high-freq combine to produce the answer. *How* they combine is left implicit (the paper says low-freq picks magnitude, high-freq picks unit digit — this is correct but operational, not mechanistic). The repo paper's CP tensor decomposition (σ²-trig score 0.964 on Gemma) verifies that the underlying operation is `cos(α)cos(β) − sin(α)sin(β) = cos(α+β)` — angle addition in the Fourier basis. This is a strictly stronger mechanism claim.
- **Progressive rotation.** Zhou et al. observes that the model "first approximates, then refines" via the logit lens. The repo paper formalizes this as the Fourier subspace **progressively rotating** to align with `W_U` — going from 30% unembed-aligned at compute layers to 100% at readout. This converts an observation into a quantitative geometric claim.
- **Modern models, larger scale.** Zhou et al.'s primary model is GPT-2-XL (1.5B, 2019-era). Repo paper covers Gemma 2B, Phi-3 Mini, LLaMA 3.2-3B (2024-era). Zhou et al.'s extension to Phi-2 / GPT-J / closed-source is partially behavioral. The repo paper does the full subspace analysis on three modern open-weight models.

### 2bis.4 The k=5 paradox (repo) reframes the Zhou et al. picture
Zhou et al. presents the period-2/2.5/5/10 outliers as *the mechanism* — they're large in the spectrum, therefore they matter. The repo paper's k=5 paradox shows that **the largest Fourier component (k=5, parity, period 2) in Gemma is causally inert**: ablating it alone causes 0.4–1.0% damage. The load-bearing frequencies are k=1 (ordinal) and k=2 (mod-5).

This is a problem for Zhou et al.'s framing too. The high-freq attention components Zhou et al. identifies as "doing modular classification" likely include some that are observationally prominent but causally redundant. Zhou et al.'s ablation lumps all high frequencies together (`k ≥ 50`) and so cannot detect this. The repo paper's frequency-by-frequency multi-layer ablation (Step 14) is required to make the distinction.

**Practical implication:** if the repo paper rerun Zhou et al.'s style frequency-band ablation but at single-frequency granularity, it would likely find that some outlier frequencies in Zhou et al.'s GPT-2-XL spectrum are also epiphenomenal. This is a one-day experiment that would be a clean methodological finding.

### 2bis.5 Combined critique of both Zhou et al. and Baeumel et al.

Both external papers share a methodological gap: **observational prominence vs. causal use** is not properly separated.

- Zhou et al. picks "outlier" frequencies by inspecting spectra → ablates them → confirms they matter. But *outlier-ness* and *causal load* are correlated, not equivalent. The repo paper's k=5 paradox shows they can dissociate.
- Baeumel et al. picks "high Fisher Score" neurons → patches them → confirms they shift outputs. Fisher Score measures selectivity, which again correlates with but does not equal causal load. Same paradox waiting to bite.

The repo paper's full triangle (necessity + sufficiency + specificity) and frequency-by-frequency ablation are the only methodology in this corner of the literature that resists this confound.

---

## 3. Where the Repo Paper is Strictly Stronger

### 3.1 Causal Evidence Triangle vs. Necessity-Only
2508.02513 establishes only *necessity-via-shift*: patching a circuit's neurons moves probability mass toward the source-digit prediction. Even at best `t*`, **flip rates are 33–69%** — the intervention is directional, not deterministic.

The repo paper establishes:
- **Necessity:** multi-layer Fourier 9-D knockout drops accuracy to **12.1–18.8%** across three models.
- **Sufficiency:** Fisher 10-D patching at readout transfers **85–100%** of digit information; contrastive Fisher 9-D transfers **83–100%**.
- **Specificity:** matched-dimension **random** subspace ablation has exactly **0%** effect.

The "evidentiary triangle" framing is explicit and well-defended. 2508.02513 has nothing analogous to the random-subspace control.

### 3.2 Subspace vs. Neuron Set
2508.02513's headline number — *"60% of MLP neurons per layer per circuit"* — is uncomfortable. If 60% of neurons participate in a "modular circuit," in what sense is the circuit modular? The paper handwaves this as compatible with the "bag of heuristics" view: the circuit *is* the bag, mostly disjoint per position.

The repo paper resolves the paradox: those neurons project onto a **9-dimensional subspace** that is the actionable computational unit. The paradox isn't that 60% of neurons participate — the paradox is using neurons as the unit of analysis at all when the underlying object is a low-rank subspace. The repo paper's per-neuron analysis (Step 9, `neuron_trig_analysis.py`) confirms only **8.8%** of MLP neurons (Gemma L19) have >80% frequency purity — i.e. most participating neurons each contribute a bit, but the *information lives in a ninth-rank subspace of activation space*.

This is a fundamental conceptual advantage. Fisher Score, being univariate, *cannot* see subspace structure.

### 3.3 Mechanism, Not Just Geography
2508.02513 tells you **where** computation lives (which neurons in which layers) and **that** it's modular. It does not tell you **how** addition is performed.

The repo paper goes the additional mile:
- **Trig identity verification** (Step 10, `cp_tensor_decomposition.py`): σ²-weighted trig score 0.964 on Gemma confirms the Fourier-domain angle addition `cos(α)cos(β) = ½[cos(α−β) + cos(α+β)]`. This is a mechanism, not just a location.
- **Progressive rotation** (§3.3 of `paper/main.tex`): the Fourier subspace at computation layers is largely *orthogonal* to W_U; it is progressively rotated to align with W_U by readout. LLaMA 3.2-3B: 29.8% → 56.9% → 94.5% → 100% across L20→L21→L24→L27. Fisher visibility and W_U alignment are **dissociated** at intermediate layers (Gemma L21: Fisher 10-D = 85% but unembed 9-D = 30%). 2508.02513 has no analog — its single intervention site (final token, post-circuit) cannot resolve this.
- **CRT decomposition** (`ℤ/10ℤ ≅ ℤ/2ℤ × ℤ/5ℤ`): Gemma emphasizes k=5 (parity), Phi-3 emphasizes k=2 (mod-5), LLaMA balanced. Architecture-specific computational strategies, all converging to ordinal at readout.

### 3.4 The k=5 Paradox
The repo paper finds that the *most prominent* SVD direction in Gemma (`k=5`, σ=135, 71% of subspace variance) is **causally inert** — multi-layer ablation of k=5 alone causes only 0.4–1.0% damage. The model encodes parity strongly but doesn't *use* parity for ones-digit computation; the load-bearing frequencies are k=1 (ordinal) and k=2 (mod-5).

This is a non-trivial methodological warning. 2508.02513 cannot detect this kind of phenomenon at all: a high-Fisher-Score neuron is by definition selective for digit pairs, but selectivity does not imply causal use. The repo paper's frequency-by-frequency multi-layer ablation (Step 14, `multilayer_freq_ablation.py`) is the only way to dissociate strong encoding from causal use, and 2508.02513 has no equivalent.

### 3.5 Steering and Control
2508.02513 demonstrates digit-position-specific *probability shifts*. The repo paper demonstrates digit-position-specific *steering*: 70–88% exact-shift accuracy via W_U-informed Fourier phase rotation. Steering is a strictly stronger demonstration of mechanistic understanding — it's the difference between "I can knock it out" and "I can drive it to a chosen target".

### 3.6 Replication Pipeline
2508.02513 ships a single GitHub repo. The repo paper ships a 14-step pipeline (`ARITHMETIC_CIRCUIT_PLAN.md`) with explicit success/failure gates per step, threshold sweeps, sanity checks (S1–S5 in `fourier_head_attribution.py`), and per-model layer reference tables. This is operationally much more reusable.

---

## 4. Where 2508.02513 is Stronger

### 4.1 Larger-scale models
LLaMA 3 70B is in 2508.02513's lineup. The repo paper tops out at LLaMA 3.2-3B. Cross-scale validation at 70B is genuinely informative — many phenomena that look universal at 2B/3B/8B can disappear or change shape at 70B.

### 4.2 Three digit positions, not just one
The repo paper focuses on the ones digit; multi-digit and tens-digit work is in supplementary scripts (Step 15, `multidigit_circuit.py`) and not the headline. 2508.02513 makes the parallel-modular claim across all three positions in 3-digit numbers. This is a more complete answer to the modularity question, even if shallower per position.

### 4.3 Subtraction with similar rigor
The repo paper treats subtraction as a generalization test (Step 12, `generalization_tests.py`). 2508.02513 runs the full hundreds/tens/units intervention pipeline on subtraction with comparable detail and finds that subtraction circuits are largely **distinct** from addition circuits (top-100 neuron overlap: units 19%, tens 9.2%, hundreds 19.8%). This is a useful operation-level dissociation result the repo paper doesn't match.

### 4.4 Heuristics-as-fragments synthesis
2508.02513's qualitative finding that some top-Fisher-Score neurons implement specific heuristics (e.g. neuron N_{19,136} ≈ result mod 2; N_{23,2705} ≈ result range 900–999), and that the heuristic *type* aligns with the neuron's digit circuit, is a genuinely useful conceptual move. It reconciles two prior research traditions. The repo paper has no equivalent neuron-level interpretation — it stays at subspace/frequency level. Adding a heuristics-style qualitative lens to the repo paper's per-neuron analysis (Step 9 already collects per-neuron Fourier purities) would strengthen the argument that the Fourier subspace and the heuristics view are the same thing seen from different angles.

---

## 5. Where Both Papers Are Equally Vulnerable

### 5.1 Carry propagation is unsolved by both
Both papers identify this as a limitation. The repo paper's `carry_stratification.py` provides a starting point (subspace alignment between carry and no-carry conditions) but stops short of a mechanistic account of how carries are computed and propagated. 2508.02513 provides a clean negative result: digit-position circuits don't do carries.

### 5.2 Composition/readout dynamics
2508.02513 doesn't analyze how three digit-level results get composed into the final token. The repo paper has a partial answer (progressive rotation + W_U-aligned readout) but only at the ones-digit level. Multi-digit composition remains open.

### 5.3 Final-token / single-position bias
Both papers run their primary interventions at the final answer-token position. Whether digit-position circuits are causally active *during* answer generation (token-by-token) versus only *latent* at the final token is not addressed by either.

### 5.4 Distribution of training data
Both treat models as black boxes. Whether the modular structure is learned because the training data favors it, or because it's the natural representational solution to the cyclic group `ℤ/10ℤ` (the repo paper's stronger argument), is not directly testable without retraining.

---

## 6. Honest Critique of the Repo Work, Informed by 2508.02513

These are issues the repo paper should address before submission, in light of what 2508.02513 does well:

### 6.1 The "60% of neurons per circuit" framing in 2508.02513 looks like a bug to the repo's framework, but it's also a stress test
If `paper/main.tex` claims a 9-D subspace is the computational unit, but 2508.02513 reports that ~60% of MLP neurons per layer participate in a digit-specific circuit, the repo paper should explicitly compute and report what fraction of neurons have non-trivial loading on the 9-D Fourier subspace. If the answer is also ~60%, the two pictures are consistent and *the subspace framing is the cleaner version of the same story*. If the answer is much smaller, the repo paper has discovered that 2508.02513's neuron count over-states the actual computational footprint, which is itself a publishable result. **Recommend adding this comparison to the paper.**

### 6.2 The repo paper does not directly report multi-position parallelism
2508.02513's strongest empirical claim is *parallelism* — three independent circuits operating simultaneously on three positions. The repo paper's paragraph-level claim is parallelism, but the headline experiments are all on the ones digit. Step 15 (`multidigit_circuit.py`) covers multi-digit but is buried in Phase F. The paper should foreground the multi-digit results, replicating 2508.02513's three-digit interchange test using Fourier subspaces (one per position) rather than neuron sets, *with the random-subspace control 2508.02513 lacks*. This would make the repo paper a strict superset of 2508.02513.

### 6.3 The repo paper should run the no-carry vs. carry contrast more aggressively
2508.02513's negative result on carries is a clean, citable finding. The repo paper's `carry_stratification.py` produces a quantitative subspace-alignment number but doesn't make a strong claim. Either claim "carries live outside the Fourier subspace" with the same kind of intervention 2508.02513 did, or claim a positive mechanism. Either is publishable; the current middle-ground is weaker than the prior art on this specific point.

### 6.4 The repo paper should explicitly cite and contrast with 2508.02513
At paper-time `paper/references.bib` was written before 2508.02513 appeared (August 2025). The current submission needs a related-work paragraph contrasting:
- 2508.02513's neuron-set, Fisher-Score, intervention-only methodology and its 60% / <2% / 33–69% flip-rate numbers
- The repo paper's subspace, SVD+DFT, full-triangle methodology and its 9-D / 0% random / 70–88% steering numbers

Failing to engage with 2508.02513 will be a reviewer red flag — the empirical claims are too close.

### 6.5 The repo paper's "perfect Fourier basis" claim should be stress-tested at LLaMA 3 70B
2508.02513 ran at 70B; the repo paper did not. The progressive-rotation and CRT-emphasis findings are model-architecture-specific and need 70B validation to claim universality. This is a 1–2 day GPU experiment, not a months-long ablation. **Strongly recommend before submission.**

### 6.6 The repo paper's per-neuron analysis (Step 9) is under-used in the writeup
Step 9 (`neuron_trig_analysis.py`) collects per-neuron Fourier tunings and counts (e.g., 809/9216 = 8.8% high-purity neurons in Gemma L19). This is the *direct* link to 2508.02513's neuron-level analysis. The current paper buries this in §4.4 ("Per-neuron frequency tuning"). Promoting it and explicitly contrasting with 2508.02513's heuristic-neuron examples would close the methodological gap and make the synthesis claim ("heuristics are subspace fragments") explicit and provable.

### 6.7 The "k=5 is causally inert" finding deserves its own subsection
This is the repo paper's most surprising and *most useful* contribution methodologically. It's currently a §3.2 paragraph titled "The k=5 paradox" — strong enough as is, but should be promoted with a dedicated experiment showing what happens to 2508.02513's-style Fisher-Score-selected neurons when you ablate them: they should behave the same way (selectively encoded but not causally used) for the same reason.

### 6.8 Steering claim has a subtle weakness
The 70–88% exact-shift accuracy is impressive, but it's *with W_U-informed steering* — i.e., the readout direction is part of the intervention. A skeptical reviewer will say: "you've shown that you can drive the right logit by perturbing the right unembedding direction, which is nearly tautological at readout layers." The repo paper should clarify (it partially does at §3.5: "readout layers are immune") and ideally produce a steering result *purely within the Fourier subspace at computation layers* with comparable numbers, to defang this critique.

---

## 7. Cross-Paper Synthesis: What Should the Field Believe?

Pooling 2508.02513 + the repo paper + the prior literature in `knowledge/`:

1. **Base LLMs perform digit-by-digit arithmetic** in a position-specific, MLP-dominated way. This is consistent across at least 7 distinct models from 4 architecture families (LLaMA 3 8B/70B, OLMo 2 7B, Gemma 2 9B, Gemma 2 2B, Phi-3 Mini, LLaMA 3.2-3B).

2. **The computational unit is a 9-D Fourier subspace of `ℤ/10ℤ`**, not a neuron set. The "neuron set" view (2508.02513) is the projection of this subspace onto the standard basis and over-counts the participating units.

3. **The mechanism is trig-identity angle addition** — confirmed by CP tensor decomposition (repo paper σ²-trig 0.964) and consistent with 2508.02513's modular parallelism. This is the same mechanism `summary_grokking_mechanistic.md` (Nanda et al.) found in toy modular addition models, now confirmed at scale in pretrained LLMs. The "Clock" algorithm is real and universal.

4. **Different architectures emphasize different CRT components** during computation but converge to ordinal encoding at readout. This is novel to the repo paper.

5. **Carry propagation is a separate, unsolved problem** — both papers identify this as a limitation, neither solves it.

6. **Selectivity ≠ causal use.** The k=5 paradox (repo paper) is a methodological warning the entire field should heed. Fisher-Score-style selectivity analyses (2508.02513, also Stolfo et al.) systematically over-attribute computation to the most observationally prominent features.

7. **Heuristics view and modular-circuit view are reconcilable.** The most credible synthesis: a digit-position circuit is the *subspace*, and individual heuristic-implementing neurons are *low-rank approximators* of that subspace's action. This needs explicit experimental confirmation — currently a hypothesis.

---

## 8. TL;DR

2508.02513 and the repo paper tell the same story. 2508.02513 tells it with cleaner *headline experiments* on more positions and bigger models; the repo paper tells it with **strictly stronger causal evidence**, a **mechanism-level explanation** (trig identity, progressive rotation), and a **complete reproducible pipeline**. The repo paper subsumes 2508.02513 conceptually but should:

1. Add an explicit comparison to 2508.02513 in related work
2. Run the multi-digit interchange experiment in 2508.02513's style with Fourier subspaces + random controls
3. Run at LLaMA 3 70B for cross-scale validation
4. Promote the k=5 paradox as the central methodological warning
5. Add a steering-without-W_U baseline at computation layers to defang the "you steered the readout, not the computation" critique

Doing all five turns the repo paper into the definitive account of pretrained-LLM digit arithmetic. The empirical phenomenon is not in dispute; the contribution is in *which framework* the field adopts.
