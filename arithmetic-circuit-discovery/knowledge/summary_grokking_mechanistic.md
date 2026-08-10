# Progress Measures for Grokking via Mechanistic Interpretability

**arxiv ID:** 2301.05217
**URL:** https://arxiv.org/abs/2301.05217
**Authors:** Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, Jacob Steinhardt
**Published:** 2023

---

## TL;DR
Fully reverse-engineers how transformers learn modular addition through discrete Fourier transforms and trigonometric identities, revealing grokking as a gradual 3-phase process rather than sudden emergence.

## Problem
Understanding "grokking" - the phenomenon where neural networks suddenly generalize after extensive memorization, and how mechanistic interpretability can reveal the underlying training dynamics.

## Method
**Algorithm Discovery:**
- Network converts modular addition to rotation about a circle
- Uses discrete Fourier transforms: `embed → DFT → pointwise multiply → inverse DFT → unembed`
- Leverages trigonometric identity: `cos(a) + cos(b) = 2cos((a+b)/2)cos((a-b)/2)`

**Progress Measures:**
1. **Embedding SVD analysis** - measure structured vs random components
2. **Fourier basis alignment** - track emergence of trigonometric structure
3. **Circuit formation metrics** - quantify algorithm implementation

## Key Results
**Three Training Phases:**
1. **Memorization** - Random, unstructured weight patterns
2. **Circuit Formation** - Gradual emergence of Fourier structure
3. **Cleanup** - Removal of memorizing components, pure algorithm

**Grokking Insights:**
- Not sudden but gradual amplification of structured mechanisms
- Progress measures reveal continuous development before apparent "breakthrough"
- SVD decomposition shows low-rank structure emergence in embeddings

## Limitations & Open Questions
- Limited to simple modular arithmetic tasks
- Unclear how insights generalize to more complex algorithms
- Progress measures specific to Fourier-based algorithms

## Relevance & Application Ideas
**For Beyond Components:**
- **SVD Progress Tracking**: Monitor singular value evolution during training
- **Circuit Formation Metrics**: Extend progress measures to attention/MLP circuits
- **Gradual vs Sudden**: Apply to understand when SVD directions emerge
- **Algorithm Discovery**: Use directional analysis to reverse-engineer learned algorithms

**Implementation Extensions:**
- Track mask sparsity evolution as progress measure
- Monitor singular direction alignment during training
- Detect circuit formation phases in IOI/GP/GT tasks

## Tags
`grokking` `mechanistic-interpretability` `fourier-analysis` `svd-analysis` `algorithm-discovery`