# Towards Empirical Interpretation of Internal Circuits and Properties in Grokked Transformers on Modular Polynomials

**arxiv ID:** 2402.16726
**URL:** https://arxiv.org/abs/2402.16726
**Authors:** [Need to extract from source]
**Published:** 2024

---

## TL;DR
Investigates internal representations and circuits in transformers during "grokking" on modular polynomial tasks, discovering Fourier superposition patterns and limited transferability across arithmetic operations.

## Problem
Understanding how transformers implement "Fourier representation and calculation circuits" in modular arithmetic, particularly for polynomial operations that build upon elementary arithmetic, and analyzing transferability of learned representations.

## Method
**Novel Analysis Techniques:**
1. **Fourier Frequency Density** - Measures concentration of Fourier components
2. **Fourier Coefficient Ratio** - Quantifies dominant frequency patterns
3. **Multi-task Transfer Analysis** - Tests representation sharing across tasks
4. **Co-grokking Investigation** - Studies simultaneous generalization

**Technical Approach:**
- Analyze internal representations across different modular operations
- Focus on polynomial tasks as superposition of elementary arithmetic
- Test transferability between related mathematical tasks

## Key Results
- **Superposition Discovery**: Polynomial tasks show "superposition of Fourier components from elementary arithmetic"
- **Limited Transferability**: Representation transfer only works for specific task combinations
- **Co-grokking Phenomenon**: Some multi-task training enables simultaneous generalization
- **Task-Specific Circuits**: Most learned representations are not universally transferable
- **Fourier Structure**: Complex tasks built from elementary Fourier patterns

## Limitations & Open Questions
- Transferability patterns not fully predictable
- Some multi-task mixtures fail to find optimal solutions
- Limited to modular arithmetic domain
- Computational overhead of Fourier analysis techniques

## Relevance & Application Ideas
**For Beyond Components:**
- **Fourier-SVD Hybrid**: Combine Fourier analysis with SVD directional decomposition
- **Superposition Detection**: Use SVD to identify overlapping computational patterns
- **Transfer Learning**: Apply multi-task directional analysis to GP/IOI/GT
- **Co-grokking Tracking**: Monitor simultaneous emergence of multiple directional patterns

**Implementation Extensions:**
- Analyze if SVD directions show Fourier alignment patterns
- Test cross-task transfer of discovered singular directions
- Monitor co-emergence of related directional structures
- Use Fourier metrics as additional progress measures

## Tags
`grokking` `fourier-analysis` `modular-polynomials` `transfer-learning` `multi-task` `circuit-superposition`