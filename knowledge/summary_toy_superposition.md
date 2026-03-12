# Toy Models of Superposition

**arxiv ID:** 2209.10652
**URL:** https://arxiv.org/abs/2209.10652
**Web:** https://transformer-circuits.pub/2022/toy_model/index.html
**Authors:** Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, Christopher Olah (Anthropic)
**Published:** September 2022

---

## TL;DR
Neural networks pack multiple features into single neurons ("polysemanticity") through a mechanism called **superposition**. This paper provides toy models where superposition can be fully understood, revealing phase changes, geometric structure (uniform polytopes), and implications for mechanistic interpretability.

## Problem
**Polysemanticity Challenge:** Individual neurons often respond to multiple unrelated concepts, making interpretability difficult. Why does this happen? When does it happen? Can we predict and control it?

**Key Question:** If models have more features than neurons (overcomplete regime), how do they compress information?

## Method

### The Toy Autoencoder Model

**Setup:**
- **Input:** n sparse features (each with probability (1-S) of being active)
- **Hidden layer:** m < n dimensions (creates bottleneck)
- **Output:** Reconstruct input features
- **Loss:** Mean squared error weighted by feature importances I_i

**Mathematical Formulation:**
```
x ∈ ℝⁿ  // Input features (sparse)
h = ReLU(W₁x)  // Hidden layer (m dimensions)
x' = ReLU(W₁ᵀh + b)  // Reconstruction
Loss = Σᵢ Iᵢ(xᵢ - x'ᵢ)²  // Weighted MSE
```

**Linear Representation Assumption:**
- Features represented as directions in activation space
- Feature fᵢ has representation direction Wᵢ
- Multiple features: x_f₁W_f₁ + x_f₂W_f₂ + ...

### Key Parameters

**Feature Sparsity (S):** Probability feature is zero
- S = 0: Dense (all features always active)
- S = 0.99: Very sparse (features rarely active)

**Feature Importance (Iᵢ):** Weight of each feature in loss
- Can follow power law: Iᵢ = 0.75ⁱ or Iᵢ = 0.8ⁱ
- Models Zipf-like distribution of concept importance

## Key Results

### 1. Phase Transitions

**Sharp transitions between regimes:**

**Disentangled Phase:** (Low sparsity or high capacity)
- Each feature gets dedicated neuron
- Features orthogonal: Wᵢᵀ Wⱼ ≈ 0 for i≠j
- ||W||²_F ≈ m (one feature per dimension)

**Superposition Phase:** (High sparsity or low capacity)
- Multiple features share neurons
- Features non-orthogonal but still separated
- ||W||²_F > m (more features than dimensions)

**Critical Finding:** "Dimensions per feature" D* = m / ||W||²_F shows sharp phase change as sparsity varies!

### 2. Geometric Structure

**Features arrange in symmetric polytopes:**
- **2D (m=2):** Regular polygons (pentagon, hexagon, etc.)
- **3D (m=3):** Platonic solids (tetrahedron, cube, dodecahedron)
- **Higher dimensions:** Uniform polytopes

**Why geometry matters:**
- Maximizes minimum angle between features
- Minimizes interference (dot products)
- Allows efficient packing in limited dimensions

**Connection to Uniform Polytopes:**
- Superposed features naturally align with vertices of symmetric polytopes
- This is NOT coincidental - it's optimal for minimizing interference
- Models discover these structures through gradient descent!

### 3. Interference and Orthogonalization

**Interference Cost:**
```
When reconstructing feature i:
Interference from feature j = (Wᵢᵀ Wⱼ)² · xⱼ²

Total interference on feature i:
Σⱼ≠ᵢ (Wᵢᵀ Wⱼ)² · xⱼ²
```

**Why Features Orthogonalize:**
- When multiple important features must coexist
- When features activate simultaneously (low sparsity)
- Orthogonalization minimizes reconstruction error
- Cost of interference > cost of dedicating dimensions

**Key Insight:** Features orthogonalize when interference becomes prohibitive!

### 4. Conditions for Superposition

**Superposition occurs when:**
1. **Feature sparsity is high:** Features rarely activate together → low interference
2. **Model capacity limited:** n >> m (more features than dimensions)
3. **Feature importance varies:** Less important features can tolerate more interference

**Mathematical Bound (from compressed sensing):**
```
m = Ω(k log(n/k))
where k = (1-S)n (number of active features)

Simplifies to: m = Ω(-n(1-S)log(1-S))
```

This means: Number of superposed features is **linear** in m but modulated by sparsity!

### 5. Computation in Superposition

**Can models compute on superposed features?**

**Example: Absolute Value**
Model learns to compute |x| for superposed features using:
```
|x| = ReLU(x) + ReLU(-x)
```

**Requirement:** Computation must be sparse (only activates for relevant features)

**Open Question:** What class of computations can be performed in superposition?

## Limitations & Open Questions

1. **How realistic are toy models?** Do they capture real model behavior?
2. **Feature counting:** How many features should we expect in superposition?
3. **Scaling:** Does superposition persist or disappear with larger models?
4. **Control:** Can we prevent superposition via regularization (L1 on activations)?
5. **Recovery:** Can we find overcomplete basis post-hoc (sparse coding problem)?
6. **Non-independence:** What if features are correlated/anti-correlated?

## Relevance & Application Ideas

### For Arithmetic Circuit Discovery

**Critical Connections:**

1. **Paper 2602.13524 Foundation:**
   - THIS paper explains **why features orthogonalize**
   - When features are "of interest" to attention head → must minimize interference
   - Paper 2602.13524's toy model DIRECTLY EXTENDS this work by adding attention!

2. **Our TIE ≈ 0 Result:**
   - Could mean features are in **superposition** (not orthogonal)
   - OR features are orthogonal but attention doesn't manipulate them
   - Need to check: Are arithmetic features sparse or dense?

3. **Why SVD Works:**
   - When features orthogonalize → align with singular vectors
   - SVD finds the orthogonal basis that maximizes variance
   - In superposition regime → SVD still finds dominant directions!

### Practical Implications

**For `arithmetic_circuit_discovery.py`:**

1. **Check Feature Sparsity:**
   - Are carry features sparse (rarely activate)?
   - Are digit features dense (always active)?
   - Sparsity determines if superposition expected

2. **Expect Phase Transitions:**
   - Small models (2 layers) may be in superposition phase
   - Large models (GPT-J 28 layers) may be disentangled
   - This explains algorithmic diversity across model scales!

3. **Interference Analysis:**
   - Measure Wᵢᵀ Wⱼ for discovered features
   - High interference → superposition
   - Low interference → orthogonal features

4. **Geometric Visualization:**
   - Project features into 2D/3D
   - Look for polytope structures
   - If found → evidence of superposition

### Connection to Other Papers

**Paper 2502.00873 (Helix):**
- Periods T=[2,5,10,100] could be superposed features!
- Sparse spectrum suggests only 5-6 key frequencies active
- Aligns with superposition theory (sparse features packed efficiently)

**Paper 2402.02619 (Carry Circuits):**
- SA_n, ST_n, SV_n could be orthogonal (high interference cost)
- Carry propagation requires simultaneous activation
- Predicts disentangled phase (features get dedicated neurons)

**Paper 2301.05217 (Clock Algorithm):**
- Key frequencies k∈{14,35,41,42,52} are sparse (5 of 113)
- Perfect candidate for superposition!
- Explains why model learns specific frequencies through training

**Paper 2306.17844 (Algorithmic Diversity):**
- Phase transitions explain algorithmic diversity!
- Hyperparameters determine if model is in superposition or disentangled phase
- Different phases → different algorithms

### Implementation Extensions

**Add to `arithmetic_circuit_discovery.py`:**

1. **Superposition Detection:**
   ```python
   def check_superposition(features):
       # Measure ||W||²_F
       frobenius_norm = torch.norm(features, 'fro')**2
       d_star = model.d_model / frobenius_norm

       if d_star < 1:
           print("Superposition detected!")
       else:
           print("Disentangled features")
   ```

2. **Interference Matrix:**
   ```python
   def compute_interference(W):
       # W: [n_features, d_model]
       interference = W @ W.T  # [n_features, n_features]
       off_diagonal = interference - torch.diag(torch.diag(interference))
       return off_diagonal.abs().mean()
   ```

3. **Phase Diagram:**
   ```python
   def plot_phase_diagram(sparsity_range, importance_curve):
       # Sweep sparsity and importance
       # Plot dimensions per feature
       # Identify phase transition boundary
   ```

4. **Polytope Visualization:**
   ```python
   def visualize_features_3d(features):
       # Project to 3D using PCA
       # Check if vertices form platonic solid
       # Measure symmetry score
   ```

## Tags
`superposition` `polysemanticity` `toy-models` `phase-transitions` `geometric-interpretability` `feature-interference` `orthogonalization` `compressed-sensing` `mechanistic-interpretability`

---

## Key Equations

**Linear Representation:**
```
x = Σᵢ xᵢ Wᵢ  // Features as weighted sum of directions
```

**Interference Cost:**
```
L_interference(i) = Σⱼ≠ᵢ (Wᵢᵀ Wⱼ)² · E[xⱼ²]
```

**Dimensions Per Feature:**
```
D* = m / ||W||²_F
where ||W||²_F ≈ Σᵢ ||Wᵢ||²
```

**Compressed Sensing Bound:**
```
m = Ω(-n(1-S)log(1-S))
where S = feature sparsity
```

---

## Critical Quote

*"Neural networks empirically seem to have linear representations. This is not a coincidence—linear representations are the natural format for neural networks to represent information in!"*

*"Superposition is a very simple idea which fundamentally challenges a naive understanding of neural networks."*

*"When features are in superposition, they arrange themselves into geometric structures like uniform polytopes to minimize interference."*
