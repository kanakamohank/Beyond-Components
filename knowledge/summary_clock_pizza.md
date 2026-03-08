# The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks

**arxiv ID:** 2306.17844
**URL:** https://arxiv.org/abs/2306.17844
**Authors:** [Need to extract from source]
**Published:** 2023

---

## TL;DR
Explores algorithmic diversity in neural networks for modular addition, showing that small hyperparameter changes can lead to qualitatively different algorithmic implementations, motivating new tools for algorithmic phase space characterization.

## Problem
Understanding how neural networks can implement multiple distinct algorithms for the same task and how hyperparameter variations affect algorithmic discovery and implementation diversity.

## Method
**Algorithm Discovery Framework:**
1. **Hyperparameter Variation** - Systematic changes to model initialization and hyperparameters
2. **Algorithmic Classification** - Identify qualitatively different solution approaches
3. **Phase Space Mapping** - Characterize transitions between algorithmic implementations
4. **Mechanistic Analysis** - Understand internal representations of different algorithms

**Technical Approach:**
- Focus on modular addition as tractable test case
- Analyze how small changes lead to different computational strategies
- Map the "algorithmic phase space" of possible implementations

## Key Results
- **Algorithmic Diversity**: Small hyperparameter changes induce discovery of qualitatively different algorithms
- **Multiple Implementations**: Networks can implement multiple distinct algorithms simultaneously
- **Phase Transitions**: Clear boundaries between different algorithmic regimes
- **Robustness vs Diversity**: Trade-off between algorithmic robustness and diversity
- **Implementation Coexistence**: Different algorithms can coexist within single networks

## Limitations & Open Questions
- Limited to simple modular arithmetic tasks
- Algorithmic characterization methods need development
- Unclear scaling to more complex problems
- Phase space mapping computationally intensive

## Relevance & Application Ideas
**For Beyond Components:**
- **Algorithmic Direction Discovery**: Different algorithms may align with different SVD directions
- **Implementation Diversity**: Multiple algorithms within single heads via directional decomposition
- **Phase Space SVD**: Use singular directions to characterize algorithmic phase spaces
- **Robustness Analysis**: Test algorithmic diversity across different directional configurations

**Implementation Extensions:**
- Analyze if different initializations lead to different directional patterns
- Test whether multiple algorithms coexist in different singular directions
- Use directional analysis to map algorithmic phase transitions
- Develop SVD-based tools for algorithmic characterization

## Tags
`algorithmic-diversity` `phase-space` `modular-arithmetic` `neural-algorithms` `mechanistic-explanation`