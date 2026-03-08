# Understanding Addition and Subtraction in Transformers

**arxiv ID:** 2402.02619
**URL:** https://arxiv.org/abs/2402.02619
**Authors:** [Need to extract from source]
**Published:** 2024

---

## TL;DR
Small transformers can solve n-digit arithmetic with 99.999% accuracy using interpretable carry/borrow circuits, while most large LLMs fail at basic arithmetic - revealing precise mechanistic understanding of algorithmic implementation.

## Problem
Most large language models (only 7% of 180 surveyed LLMs) fail at reliable arithmetic, yet small transformers can achieve near-perfect accuracy. Understanding the mechanistic differences and circuit implementations.

## Method
**Systematic Circuit Analysis:**
1. **Unified Mechanistic Account** - Cascading carry and borrow circuits
2. **Systematic Ablations** - 49 trained models with node-level constraints
3. **Circuit Discovery** - Extension of prior "addition circuits" to subtraction
4. **Reproducible Toolkit** - Interpretability framework for arithmetic circuits

**Technical Approach:**
- Focus on small, tractable transformers that implement exact algorithms
- Map out precise carry/borrow propagation mechanisms
- Systematic ablation studies to isolate circuit components

## Key Results
- **Perfect Implementation**: Small transformers achieve 99.999% accuracy on n-digit arithmetic
- **Mechanistic Clarity**: Precise understanding of carry/borrow circuit implementation
- **Algorithmic Exactness**: Networks implement mathematically correct algorithms
- **Tractable Interpretability**: Complete circuit reverse-engineering possible

## Limitations & Open Questions
- Gap between small perfect models and large failing LLMs unclear
- Scalability of exact algorithmic implementation to larger networks
- Transfer to more complex mathematical reasoning tasks

## Relevance & Application Ideas
**For Beyond Components:**
- **Exact Algorithm Discovery**: Use SVD to find precise algorithmic implementations
- **Carry/Borrow as Directions**: Arithmetic operations may align with specific singular directions
- **Perfect Interpretability**: Benchmark for complete circuit understanding
- **Algorithmic Decomposition**: Apply directional analysis to mathematical reasoning

**Implementation Extensions:**
- Search for arithmetic directions in GP/IOI/GT tasks
- Use arithmetic as test case for SVD circuit discovery
- Benchmark directional interpretability against known exact circuits

## Tags
`arithmetic-circuits` `exact-algorithms` `carry-borrow` `mechanistic-interpretability` `circuit-discovery`