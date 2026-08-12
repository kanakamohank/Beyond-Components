# Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits

**arxiv ID:** 2511.20273
**URL:** https://arxiv.org/abs/2511.20273
**Authors:** Areeb Ahmad, Abhinav Joshi, Ashutosh Modi
**Published:** 2025 (NeurIPS 2025)

---

## TL;DR
This paper introduces a fine-grained interpretability approach that decomposes transformer attention heads and MLPs into orthogonal singular directions, revealing that multiple independent computations coexist within single components rather than treating them as atomic units.

## Problem
Existing mechanistic interpretability methods treat attention heads and MLPs as indivisible units, potentially missing fine-grained computational structure. Current circuit discovery approaches assume functionality aligns with component boundaries, but transformer layers may multiplex multiple subfunctions within single heads or MLPs.

## Method
**Core Approach: SVD-Based Directional Decomposition**

1. **Singular Value Decomposition**: Apply SVD to augmented weight matrices:
   - Query-Key (QK) matrices: `W_QK = U_qk @ S_qk @ V_qk^T`
   - Output-Value (OV) matrices: `W_OV = U_ov @ S_ov @ V_ov^T`
   - MLP projection matrices: Similar decomposition

2. **Learnable Masking**: Learn sparse binary masks over singular directions:
   - Optimize masks to minimize KL divergence + L1 sparsity penalty
   - Identify functionally important directions within each component

3. **Directional Attribution**: Enable intervention and analysis at singular direction level rather than full component level

4. **Logit Receptors**: Discover stable directions in logit space aligned with specific tokens (e.g., " he", " she") that can be controlled via scalar interventions

## Key Results
- **Sparsity**: Achieves 40-60% parameter reduction vs standard circuit discovery while maintaining faithfulness
- **IOI Task**: "Name mover" heads contain multiple overlapping subfunctions in distinct singular directions
- **Gender Pronoun**: Specific directions consistently influence toward "he"/"she" tokens with controllable activation
- **Greater Than**: Numerical comparison computations distributed across directional subspaces
- **Validation**: Strong replication of GPT-2 behavior on canonical interpretability tasks (IOI, GP, GT)

## Limitations & Open Questions
- Some functional directions remain challenging to interpret exhaustively
- Method focused on GPT-2 small - scalability to larger models unclear
- Computational overhead of SVD decomposition and mask optimization
- Limited exploration of cross-layer directional interactions

## Relevance & Application Ideas
**Direct Application to Beyond Components Project:**
- The repository implements the exact methods described in this paper
- Code structure mirrors paper sections: SVD masking, circuit discovery, activation patching
- Configs for GP, IOI, GT tasks match paper experiments

**Research Extensions:**
- Apply directional analysis to other transformer architectures
- Investigate directional composition across multiple layers
- Use for targeted model editing at subcomponent granularity
- Extend to vision transformers or multimodal models

**Implementation Insights:**
- `MaskedTransformerCircuit` class implements the core SVD masking
- `activation_patching` enables directional interventions
- Visualization tools for singular value analysis included

## Tags
`mechanistic-interpretability` `svd-decomposition` `transformer-circuits` `activation-patching` `circuit-discovery` `directional-analysis`