# Beyond Fourier and SVD: A Mathematician's Toolkit for Cracking Arithmetic Circuits

**Date:** April 10, 2026
**Status:** Proposal + Implementation
**Goal:** Resolve the mystery of how transformers compute addition, using mathematical tools that address the specific failure modes of SVD, Fourier, and PCA.

---

## 1. The Problem Statement (Precise Mathematical Formulation)

A transformer model $\mathcal{M}$ computes $f: \mathbb{Z} \times \mathbb{Z} \to \mathbb{Z}$, specifically $f(a,b) = a + b$, through a sequence of layer computations:

$$h^{(0)} \xrightarrow{\text{Attn}^{(1)}} h^{(1)} \xrightarrow{\text{MLP}^{(1)}} \cdots \xrightarrow{\text{Attn}^{(L)}} h^{(L)} \xrightarrow{W_U} \text{logits}$$

**We know WHERE** (MLP layers 19-27 in Phi-3, distributed across 5-7 components) **but not HOW**.

**Empirical constraints on any theory:**
1. Computation is **distributed** — no single MLP writes the answer
2. Computation is **redundant** — ablating 20 attention heads causes 0% accuracy drop
3. Computation is **superposed** — PCA separability < 0.5 on MLP activations
4. The **helix is representational, not computational** — 0% causal success on phase-shift intervention
5. Fourier structure exists in weight space (T≈10) but explains only **1.5-9.2%** of activation variance

---

## 2. Why Existing Methods Fail — Five Violated Assumptions

### 2.1 SVD/PCA: The Orthogonality Trap

**Assumption:** Features are encoded along orthogonal directions.
**Reality:** Superposition packs $k$ features into $d < k$ dimensions at oblique angles.

If the carry bit $c$ and ones digit $d_1$ are encoded as:
$$h = \alpha_c \cdot \hat{e}_c + \alpha_{d_1} \cdot \hat{e}_{d_1} + \cdots$$

where $\langle \hat{e}_c, \hat{e}_{d_1} \rangle = \cos(60°) = 0.5$ (not zero!), then:
- PCA axis 1 captures a **mixture** of both features
- SVD directions do NOT align with either feature axis
- Separability is bounded by $1 - |\cos\theta|^2 = 0.75$ (matches your <0.5 observation for >2 features)

**Mathematical consequence:** No orthogonal decomposition will cleanly isolate arithmetic features.

### 2.2 Fourier Analysis: The Wrong Group

**Assumption:** The model uses $\mathbb{Z}/10\mathbb{Z}$ (cyclic group of order 10) directly.
**Reality:** By the Chinese Remainder Theorem:

$$\mathbb{Z}/10\mathbb{Z} \cong \mathbb{Z}/2\mathbb{Z} \times \mathbb{Z}/5\mathbb{Z}$$

The irreducible representations of $\mathbb{Z}/10\mathbb{Z}$ over $\mathbb{R}$ are:
- **Trivial:** $\rho_0(n) = 1$
- **Sign (mod 2):** $\rho_1(n) = (-1)^n$ — period T=2
- **Rotation (mod 5):** $\rho_2(n) = \begin{pmatrix} \cos(2\pi n/5) \\ \sin(2\pi n/5) \end{pmatrix}$ — period T=5
- **Rotation × sign:** $\rho_3(n) = (-1)^n \begin{pmatrix} \cos(2\pi n/5) \\ \sin(2\pi n/5) \end{pmatrix}$ — period T=10

The model could implement addition using **only** $\rho_1$ and $\rho_2$ (periods 2 and 5), never touching $\rho_3$ (period 10). Looking for T=10 periodicity specifically misses this decomposition.

**Evidence this is happening:** Llama 3.2 3B shows T=99 in dims 0-3 AND T=2.0 in dims 4-8 — a clear CRT split!

### 2.3 2D Interventions: Measure-Zero Perturbation

**Assumption:** The arithmetic computation lives in a 2D plane.
**Reality:** If computation is distributed across $k$ MLPs each with $d$ dimensions, the effective computational subspace has dimension $O(k \cdot r)$ where $r$ is the rank per MLP. For 5 MLPs with rank 5 each, that's a 25-dimensional subspace.

A rotation in a 2D plane perturbs $2/25 = 8\%$ of the relevant dimensions. The remaining 92% still encode the correct answer, so the model's output is unchanged.

**Mathematical consequence:** Causal intervention must target the **full computational subspace**, not a 2D slice.

### 2.4 Single-Component Patching: The Hydra Problem

**Assumption:** Ablating a causal component should degrade performance.
**Reality:** With $k$-fold redundancy, the model maintains $k$ parallel copies of the computation. Ablating any $j < k$ copies leaves $k - j$ intact copies.

This is the **matroid** structure: the set of sufficient computational subsets forms a matroid where any basis (minimal sufficient set) has the same cardinality $r$, and you must ablate at least $r$ elements to see degradation.

**Mathematical consequence:** You need **simultaneous multi-component intervention** to break through redundancy.

### 2.5 Point Statistics: Missing Distributional Structure

**Assumption:** Each (a,b) pair maps to a single activation vector.
**Reality:** The relevant information is in how the **distribution** of activations shifts across (a,b) pairs. Compression through layer norms destroys magnitude information, leaving only directional/distributional structure.

**Mathematical consequence:** Need distributional metrics (Wasserstein, Fisher) rather than point metrics (Euclidean, cosine).

---

## 3. Five Novel Mathematical Approaches

### 3.1 Fisher Information Geometry — Finding the TRUE Causal Subspace

**Core idea:** Instead of finding the geometrically prettiest subspace (SVD) or the highest-variance subspace (PCA), find the subspace that carries the most **information about the answer**.

**Definition:** For model output probability $p(y \mid h)$ where $h$ is the hidden state at a specific layer, the Fisher Information Matrix is:

$$\mathbf{F} = \mathbb{E}_{a,b}\left[\nabla_h \log p(y_{a+b} \mid h(a,b)) \cdot \nabla_h \log p(y_{a+b} \mid h(a,b))^\top\right]$$

**Properties that make this ideal:**
1. **Basis-independent:** $\mathbf{F}$ is a Riemannian metric tensor — its eigenvectors are the natural coordinate system for the computation, regardless of the model's weight parameterization.
2. **Directly causal:** High eigenvalues of $\mathbf{F}$ mean small perturbations in that direction cause large changes in output probability. Low eigenvalues mean the direction is **causally irrelevant** (even if it has high variance!).
3. **Aggregated across examples:** Unlike patching one example at a time, $\mathbf{F}$ averages over all (a,b) pairs, giving the **population-level** causal subspace.

**Algorithm:**
1. For each (a,b) pair, run forward pass and compute gradient $g = \nabla_h \log p(y_{a+b} \mid h)$ at compute-zone MLP layers
2. Accumulate $\mathbf{F} = \frac{1}{N} \sum_i g_i g_i^\top$
3. Eigendecompose $\mathbf{F} = V \Lambda V^\top$
4. The top-$k$ eigenvectors of $\mathbf{F}$ span the **Fisher subspace** — the directions that maximally affect the output
5. **THEN** look for Fourier/helical structure in this subspace (not in the SVD subspace!)

**Prediction:** The Fisher subspace will be ~10-30 dimensional (matching the distributed nature), will NOT align with SVD directions (explaining 0% causal success), and causal intervention in the Fisher subspace should achieve >50% success.

### 3.2 Independent Component Analysis — Resolving Superposition Without Training

**Core idea:** ICA finds statistically **independent** (not orthogonal) directions. This is exactly what superposition creates — multiple features packed at oblique angles that are statistically independent but geometrically overlapping.

**Mathematical foundation:** Given activation matrix $X \in \mathbb{R}^{N \times d}$ (N examples, d dimensions), ICA finds a demixing matrix $W$ such that:

$$S = X \cdot W$$

where the rows of $S$ are maximally statistically independent (measured by negentropy or kurtosis).

**Why ICA > SAE for this problem:**
- **SAE** requires training a separate neural network with hyperparameter tuning — expensive and may overfit
- **ICA** is a closed-form algorithm (FastICA converges in <100 iterations) — fast and deterministic
- **ICA** finds the **mathematically optimal** independent decomposition, not an approximate one
- Both solve the same problem (superposition), but ICA is the analytical solution

**Algorithm:**
1. Collect MLP activations for all (a,b) pairs at compute-zone layers
2. Apply FastICA to extract independent components
3. For each IC, compute correlation with:
   - $a \mod 10$ (ones digit of first operand)
   - $b \mod 10$ (ones digit of second operand)
   - $(a+b) \mod 10$ (ones digit of answer)
   - $\lfloor(a+b)/10\rfloor$ (carry bit)
   - $a \mod 2$, $b \mod 2$, etc. (CRT components)
4. Label ICs by their strongest correlate
5. **Causal test:** Zero out specific ICs and measure accuracy change

**Prediction:** ICA will find components cleanly correlated with carry, ones digit, tens digit — features that PCA could not separate because they share oblique directions.

### 3.3 CP Tensor Decomposition — Capturing Bilinear Structure

**Core idea:** The interaction between operands $a$ and $b$ in producing the answer is fundamentally a **rank-deficient tensor**, not a matrix. The trig identity for addition is literally a tensor decomposition.

**Mathematical framework:** Define the activation tensor:

$$\mathcal{T} \in \mathbb{R}^{A \times B \times D}$$

where $\mathcal{T}[a, b, d] = h_d(a, b)$ is the $d$-th dimension of the hidden state for input $(a, b)$.

The **CP decomposition** finds:

$$\mathcal{T} \approx \sum_{r=1}^{R} \lambda_r \cdot \mathbf{u}_r \otimes \mathbf{v}_r \otimes \mathbf{w}_r$$

where $\mathbf{u}_r \in \mathbb{R}^A$, $\mathbf{v}_r \in \mathbb{R}^B$, $\mathbf{w}_r \in \mathbb{R}^D$.

**Why this matters for arithmetic:**

The trig addition identity $\cos(a+b) = \cos(a)\cos(b) - \sin(a)\sin(b)$ is exactly a **rank-2 CP decomposition**:

$$T_{\cos(a+b)} = \underbrace{\cos(\cdot)}_{\mathbf{u}_1} \otimes \underbrace{\cos(\cdot)}_{\mathbf{v}_1} \otimes \underbrace{\mathbf{w}_1}_{\text{write dir}} - \underbrace{\sin(\cdot)}_{\mathbf{u}_2} \otimes \underbrace{\sin(\cdot)}_{\mathbf{v}_2} \otimes \underbrace{\mathbf{w}_2}_{\text{write dir}}$$

If the model implements this identity, the CP decomposition will **directly reveal** the $\cos(a)$, $\sin(a)$, $\cos(b)$, $\sin(b)$ factors and the write directions $\mathbf{w}$.

**Algorithm:**
1. Collect activations $h(a,b)$ for a grid of (a,b) pairs at each compute-zone MLP
2. Form tensor $\mathcal{T}[a, b, d]$
3. Compute CP decomposition for ranks R = 1, 2, ..., 20
4. For each rank-1 component: FFT the $\mathbf{u}_r$ and $\mathbf{v}_r$ factors to check for periodicity
5. Check if $\mathbf{u}_r$ and $\mathbf{v}_r$ are Fourier modes at matching frequencies (the trig identity signature)
6. The $\mathbf{w}_r$ vectors identify which activation dimensions each "channel" writes to

**Prediction:** Rank 4-8 will capture >80% of the tensor's variation. The factors will decompose into Fourier modes at T=2 and T=5 (CRT frequencies), with matching cos/sin pairs (trig identity signature).

### 3.4 Persistent Homology — Basis-Free Topology Detection

**Core idea:** If numbers form a circle (mod 10), this is a **topological invariant** — it exists regardless of the embedding basis, rotation, noise, or dimensionality. Persistent homology detects topological features (loops, voids, tori) that survive across scales.

**Mathematical framework:**

Given point cloud $\{h(n) : n = 0, 1, \ldots, 99\} \subset \mathbb{R}^d$ (activation vectors for numbers 0-99), compute the **Vietoris-Rips complex** at increasing scale $\epsilon$:

$$VR_\epsilon = \{\sigma \subseteq \{h(n)\} : \text{diam}(\sigma) \leq \epsilon\}$$

The **persistent homology** $PH_k$ tracks the birth and death of $k$-dimensional topological features:
- $PH_0$: connected components (clusters)
- $PH_1$: loops (circles) — **this is what we're looking for**
- $PH_2$: voids (spheres, tori)

A **persistent 1-cycle** (long bar in the $H_1$ barcode) indicates a robust circular structure.

**Key advantages over Fourier:**
1. **No assumed period** — detects circles of ANY period
2. **No assumed basis** — works in any dimension, any rotation
3. **No assumed embedding** — tolerates noise, superposition, non-linear warping
4. **Detects product structure** — a torus ($S^1 \times S^1$) has two persistent 1-cycles and one persistent 2-cycle

**CRT Connection:** If the model uses $\mathbb{Z}/2\mathbb{Z} \times \mathbb{Z}/5\mathbb{Z}$:
- We expect **two** persistent 1-cycles in $H_1$ (one for each factor)
- We expect **one** persistent 2-cycle in $H_2$ (the torus)
- Compare with $\mathbb{Z}/10\mathbb{Z}$ which gives **one** 1-cycle and **zero** 2-cycles

This is a **topological fingerprint** that distinguishes the mod-10 hypothesis from the CRT hypothesis!

**Algorithm:**
1. Collect activations at each layer for numbers 0-99
2. Optionally project onto Fisher subspace (from Approach 3.1) to reduce noise
3. Compute Vietoris-Rips persistent homology using Ripser
4. Count persistent 1-cycles and 2-cycles
5. At the layer where computation happens (L19-L27), check if topology changes from input (one 1-cycle for number line) to output (specific cycle structure for the answer)
6. The **layer where topology changes** is where the computation happens

**Prediction:**
- Early layers: $H_1$ shows 0 persistent cycles (numbers form a line, not a circle)
- Compute layers: $H_1$ shows 2 persistent cycles (CRT: mod-2 and mod-5 circles)
- Output layers: $H_1$ may collapse back as the model converts to logit space

### 3.5 Wasserstein (Optimal Transport) Geometry — Distributional Structure

**Core idea:** Instead of analyzing individual activation vectors, analyze how the **distribution** of activations shifts as we vary the input. Optimal transport provides a natural metric on distributions that respects the geometry of the underlying space.

**Mathematical framework:**

For each digit $d \in \{0, 1, \ldots, 9\}$, define the conditional distribution:

$$\mu_d = \{h(a, b) : (a + b) \mod 10 = d\}$$

The **Wasserstein-2 distance** between two such distributions is:

$$W_2(\mu_i, \mu_j) = \left(\inf_{\gamma \in \Gamma(\mu_i, \mu_j)} \int \|x - y\|^2 \, d\gamma(x, y)\right)^{1/2}$$

The resulting **10×10 distance matrix** $D_{ij} = W_2(\mu_i, \mu_j)$ encodes the geometry of how the model separates different ones-digits.

**Key insight:** If the model uses modular arithmetic, $D$ should have **circular structure**:

$$D_{ij} \propto \min(|i - j|, 10 - |i - j|) \quad \text{(circular metric on } \mathbb{Z}/10\mathbb{Z}\text{)}$$

If the model uses CRT decomposition, $D$ should decompose as:

$$D_{ij} \approx \alpha \cdot d_2(i, j) + \beta \cdot d_5(i, j)$$

where $d_2$ is the mod-2 distance and $d_5$ is the mod-5 circular distance.

**Algorithm:**
1. For each MLP layer in the compute zone, collect activations for all (a,b) pairs
2. Group by ones digit of the answer: $(a+b) \mod 10$
3. Compute pairwise Wasserstein-2 distances (or Sinkhorn approximation for efficiency)
4. Analyze the 10×10 distance matrix:
   - Compute MDS embedding to 2D → should show circular arrangement if mod-10
   - Test fit to circular metric vs CRT metric vs unstructured
   - Track how the distance matrix evolves across layers (where does circular structure emerge?)
5. Repeat grouping by carry bit, tens digit, CRT components

**Prediction:** The Wasserstein distance matrix will show circular structure at compute layers even though individual activations (PCA) don't cluster. The distributional structure is preserved even under superposition because OT respects the full shape of distributions, not just centroids.

---

## 4. The Unified Experiment: Implementation Plan

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│             Activation Collection (shared)                │
│  Model: Phi-3 Mini (or Gemma 2B)                         │
│  Prompts: "{a} + {b} = " for a,b ∈ [0, 49]              │
│  Layers: L18-L27 (compute zone) + L31 (output)           │
│  Positions: operand, operator, equals, final              │
└───────┬──────┬──────┬──────┬──────┬──────────────────────┘
        │      │      │      │      │
   ┌────▼──┐ ┌─▼───┐ ┌▼────┐ ┌▼───┐ ┌▼──────────┐
   │Fisher │ │ ICA │ │ CP  │ │TDA │ │Wasserstein│
   │Info   │ │     │ │Tens.│ │+CRT│ │           │
   └───┬───┘ └──┬──┘ └──┬──┘ └─┬──┘ └─────┬─────┘
       │        │       │      │           │
   ┌───▼────────▼───────▼──────▼───────────▼──────┐
   │          Cross-Validation & Synthesis          │
   │  • Do Fisher eigenvectors align with ICA?      │
   │  • Do CP factors match CRT frequencies?        │
   │  • Does TDA topology match Wasserstein metric? │
   │  • CAUSAL TEST in discovered subspaces         │
   └───────────────────────────────────────────────┘
```

### Expected Outcomes

| Approach | What it reveals | How it connects |
|----------|----------------|-----------------|
| **Fisher Info** | The 10-30D causal subspace | Defines WHERE to look with other tools |
| **ICA** | Individual superposed features (carry, digits) | Explains WHY PCA fails |
| **CP Tensor** | The bilinear algorithm (trig identities) | Reveals HOW computation works |
| **TDA + CRT** | Topological structure (circle vs torus) | Reveals WHAT group structure is used |
| **Wasserstein** | Layer-by-layer distributional geometry | Reveals WHEN computation happens |

### Cross-Validation Predictions

If our theory is correct, all five approaches should **converge**:
1. Fisher eigenvectors should span the same subspace as the top ICA components
2. CP tensor factors should be Fourier modes at CRT frequencies (T=2, T=5)
3. TDA should show a torus ($H_1$ = 2 cycles) in the Fisher subspace, not a single circle
4. Wasserstein distances should show CRT-metric structure at layers where TDA topology changes
5. Causal intervention in the Fisher/ICA subspace should succeed where SVD-plane rotation failed

---

## 5. Mathematical Novelty & Expected Contribution

This approach is novel because:

1. **First application of Fisher Information Geometry to circuit discovery** — existing work uses activation patching (discrete) or gradient-based attribution (scalar). Fisher IM gives the full Riemannian structure.

2. **First application of tensor decomposition to arithmetic circuits** — existing work decomposes weight matrices (2-way). The bilinear structure of addition demands 3-way decomposition.

3. **First topological analysis of arithmetic computation** — persistent homology has been used for analyzing word embeddings (Savle et al., 2019) but never for circuit-level computation analysis.

4. **CRT hypothesis for neural arithmetic** — a novel theoretical prediction that can be tested: the model decomposes mod-10 into mod-2 × mod-5, not a single mod-10 circle.

5. **Wasserstein metric for distributed computation** — optimal transport has been used for comparing neural network representations (Alvarez-Melis & Jaakkola, 2018) but never for tracking computation flow through layers.

---

## 6. Dependencies

```
# Core (likely already installed)
torch, numpy, scipy, scikit-learn, matplotlib

# New dependencies needed:
tensorly          # CP/Tucker tensor decomposition
ripser            # Persistent homology (fast Vietoris-Rips)
persim            # Persistence diagram utilities
pot               # Python Optimal Transport (Wasserstein)
```
