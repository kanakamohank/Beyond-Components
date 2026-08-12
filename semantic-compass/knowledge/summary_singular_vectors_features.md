# Singular Vectors of Attention Heads Align with Features

**arxiv ID:** 2602.13524
**URL:** https://arxiv.org/abs/2602.13524
**Authors:** [Need to extract from source]
**Published:** 2026

---

## TL;DR
Demonstrates that singular vectors of attention matrices robustly align with feature representations in language models, providing theoretical framework for when and why such alignment occurs through sparse attention decomposition.

## Problem
Understanding when and why singular vectors of attention heads align with interpretable features - a central question for using SVD as a theoretically justified basis for feature identification in mechanistic interpretability.

## Method
**Theoretical Framework:**
1. **Alignment Analysis** - Investigate conditions for singular vector-feature alignment
2. **Sparse Attention Decomposition** - Method to recognize feature alignment patterns
3. **Robustness Testing** - Verify alignment across different model configurations
4. **Observable Model Analysis** - Demonstrate alignment in tractable settings

**Technical Approach:**
- Analyze mathematical conditions for SVD-feature correspondence
- Develop sparse decomposition techniques
- Test alignment robustness across various scenarios

## Key Results
- **Robust Alignment**: Singular vectors can robustly align with features in observable models
- **Theoretical Justification**: Provides sound theoretical basis for SVD-based feature identification
- **Sparse Decomposition**: Method successfully recognizes feature alignment patterns
- **Alignment Conditions**: Clear characterization of when SVD-feature alignment occurs
- **Mechanistic Foundation**: SVD provides principled approach to feature discovery

## Limitations & Open Questions
- Theoretical results may not generalize to all model types
- Sparse attention decomposition computational requirements
- Feature definition and identification challenges
- Alignment robustness across different training regimes

## Relevance & Application Ideas
**For Beyond Components:**
- **Direct Validation**: Confirms theoretical foundation for your SVD directional approach
- **Feature-Direction Correspondence**: Validates that discovered directions represent interpretable features
- **Alignment Testing**: Apply their robustness tests to your discovered directions
- **Theoretical Grounding**: Strengthens justification for singular vector interpretability

**Implementation Extensions:**
- Use their alignment metrics to validate discovered directions
- Apply sparse attention decomposition to your masked circuits
- Test robustness of your directional discoveries using their framework
- Combine their theoretical insights with your empirical circuit discovery

## Tags
`singular-vectors` `feature-alignment` `theoretical-interpretability` `sparse-attention` `svd-justification`