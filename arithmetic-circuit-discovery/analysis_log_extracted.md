# Comprehensive Model Analysis Log

## Analysis Overview
This document contains extracted logs from extensive transformer model analysis runs focusing on arithmetic capabilities, SVD-based circuit discovery, and mechanistic interpretability across multiple models.

## Models Analyzed
1. **google/gemma-7b** - 28 layers, focused analysis on layers 14-25
2. **microsoft/Phi-3-mini-4k-instruct** - 32 layers, focused analysis on layers 24-31
3. **google/gemma-2b** - 18 layers, focused analysis on layers 10-17
4. **meta-llama/Llama-3.2-3B** - 28 layers, partial analysis

## Key Analysis Methods

### 1. Helix Analysis (SVD-based Geometric Structure Detection)
- **Target**: Detect periodic helical structures in attention matrices (period ~10 for modular arithmetic)
- **Metrics**:
  - Radius CV (coefficient of variation, target <0.20)
  - Phase Match (target >0.90)
  - Period T (target ~10.0)
- **Results**: No clean helical structures found in any model with correct period

### 2. Arithmetic Neuron Analysis
- **Method**: Ablation-based identification of neurons critical for arithmetic
- **Metrics**: Logit drop when neuron ablated
- **Key Findings**:
  - **gemma-7b**: Strong signals found (up to 23.8 logit units in Layer 22, Neuron 429)
  - **gemma-2b**: Very strong signals (up to 28.5 logit units in Layer 10, Neuron 3481)
  - **Phi-3**: Weak/distributed signals (<0.2 logit units across all layers)

### 3. Circuit Discovery via Direct Logit Attribution (DLA)
- **Method**: Measure direct contribution of each component to correct answer logits
- **Architecture**: MLP vs Attention head contributions by layer

### 4. SVD Computational Signal Strength
- **Method**: R² analysis of top-k SVD components predicting arithmetic answers
- **Finding**: Most layers show R²≈0, indicating geometric (not linear) number representation

## Detailed Results by Model

### Google Gemma-7B Analysis

#### Helix Detection Results
```
Layer Sweep Results (Operand Position):
Layer 21: CV=0.424, Phase=0.526, T=16.1 (Best layer overall)
Layer 14: CV=0.468, Phase=0.009, T=9.9 (Close to target period)
```
**Verdict**: No clean helical structures. Phase alignment insufficient.

#### Arithmetic Neuron Analysis - Strong Causal Neurons Found
**Layer 19**: Top neuron 3695 (0.0035 prob drop)
- Function: Magnitude tracker (correlation with raw sum: -0.648)

**Layer 22**: Top neuron 429 (23.75 logit drop)
- Strongest activation pattern: Small operands (27+10, 24+12, 17+11)
- Correlation analysis: Complex heuristic, no single-variable explanation

**Layer 24**: Top neuron 20892 (13.75 logit drop)
- Correlation with carry: +0.147
- Promotes tokens: 'increa', 'affor', 'inconce'

#### Circuit Discovery Results
```
Zone 1 — REPRESENTATION: Layers 0-11 (R²=0, geometric storage)
Zone 2 — COMPUTATION: Layers 23-26 (R² rises to 0.98)
Zone 3 — TRANSLATION: Layer 27 (DLA=351.4, final answer assembly)
```

**Key Computation Layer**: Layer 25
- **Head-level analysis**: Heads 2, 4, 8, 12 show high DLA + high R²
- **Ablation results**: 1.56% accuracy drop when all key heads ablated (synergistic interaction)

### Microsoft Phi-3-mini-4k-instruct Analysis

#### Helix Detection Results
```
Layer 24: CV=0.448, Phase=0.652, T=12.7 (Wrong period but best structure)
```
**Verdict**: Some angular structure exists but period≠10, phase alignment insufficient.

#### Arithmetic Neuron Analysis - Distributed/Weak Signals
- **Layer 20**: Strongest neuron 3330 (0.195 logit drop) - Very weak compared to Gemma
- **Layer 21**: Neuron 4128 identified as carry detector (correlation +0.656)
- **Overall**: Highly distributed computation, no single neurons dominate

#### Circuit Discovery Results
```
Zone 1 — REPRESENTATION: Layers 0-23 (R²=0, geometric storage)
Zone 2 — COMPUTATION: Layers 24-25 (R² rises to 0.525)
Zone 3 — TRANSLATION: Layers 26-31 (High DLA, answer projection)
```

**Key Computation Layer**: Layer 25
- **Head-level analysis**: No single head dominates (highest DLA=2.683 in Head 1)
- **Distribution**: Arithmetic distributed across many heads

### Google Gemma-2B Analysis

#### Arithmetic Neuron Analysis - Strongest Signals Found
**Layer 10**: Neuron 3481 (28.5 logit drop) - Strongest across all models
- **Activation pattern**: Fires strongest for medium sums (13-17)
- **Deep characterization**: Shows complex dependence on operand combinations
- **Tokens promoted**: Unusual/rare tokens ('thut', 'Hæ', 'impractica')

**Layer 11**: Neuron 9767 (22.6 logit drop)
- **Pattern**: Relatively uniform activation across operand combinations
- **Tokens promoted**: Literary/archaic terms ('McLaugh', 'Shakspeare')

#### Circuit Discovery Results
```
Zone 1 — REPRESENTATION: Layers 0-14 (R²=0, geometric storage)
Zone 2 — COMPUTATION: Layers 15-16 (R² rises to 0.924)
Zone 3 — TRANSLATION: Layer 17 (DLA=1252.9, massive final assembly)
```

### Vector Translation Hypothesis Testing

#### Hypothesis
Linear vector translation mechanism: answer = operand_representation + step_vector

#### Test Results Across All Models
- **Step vector consistency**: Near-zero or negative cosine similarity
- **Causal injection success**: 0% across all models
- **Verdict**: **Vector Translation hypothesis DISPROVEN**
- **Conclusion**: Arithmetic manifolds are curved, not linear

## Key Research Findings

### 1. No Helical Structures for Modular Arithmetic
Despite extensive searching across layers and models, no clean helical structures with period≈10 were found. This challenges the geometric approach to understanding modular arithmetic in transformers.

### 2. Model-Specific Computational Strategies

**Gemma Models (2B/7B)**:
- Concentrated computation in specific neurons
- Clear 3-zone architecture (representation → computation → translation)
- Strong individual neuron effects (up to 28.5 logit units)

**Phi-3**:
- Highly distributed computation
- Weaker individual neuron effects (<0.2 logit units)
- More resilient to single-neuron ablations

### 3. Circuit Architecture Pattern
All models show consistent 3-zone architecture:
1. **Representation Zone**: Early layers, geometric number storage (R²≈0)
2. **Computation Zone**: Middle-late layers, arithmetic crystallization (R²→0.5-0.9)
3. **Translation Zone**: Final layers, answer→token projection (high DLA)

### 4. Vector Translation Mechanism Refuted
Systematic testing across models shows:
- Step vectors are inconsistent across different numbers
- Causal injection fails even at proven computational layers
- Arithmetic manifolds are fundamentally non-linear

### 5. Attention Head Redundancy
Key computational attention heads show synergistic redundancy:
- Individual ablations cause minimal damage
- Combined ablation reveals circuit dependencies
- Suggests distributed backup mechanisms

## Technical Methods Summary

### Experimental Pipeline
1. **Model Loading**: HookedTransformer integration with careful device management
2. **Dataset Generation**: Controlled arithmetic pairs (single-token answers prioritized)
3. **Activation Caching**: Strategic caching of computational layers only
4. **Attribution Analysis**: Direct logit attribution (DLA) for component importance
5. **Ablation Studies**: Systematic neuron/head removal with accuracy measurement
6. **Geometric Analysis**: SVD decomposition and helical structure detection
7. **Causal Testing**: Vector injection experiments for mechanistic validation

### Key Metrics
- **DLA (Direct Logit Attribution)**: Component contribution to correct answer
- **R² (SVD Predictive Power)**: Linear decodability of arithmetic answers
- **Logit Drop**: Accuracy loss from component ablation
- **Radius CV**: Geometric structure quality metric
- **Phase Match**: Alignment quality for periodic structures

## Files Generated
- Various visualization outputs (geometry plots, FFT spectra)
- Neuron characterization data
- Circuit ablation results
- Vector injection test results

## Usage Notes
This analysis provides comprehensive empirical data on transformer arithmetic capabilities but challenges several theoretical frameworks (helical representations, vector translation). The findings suggest arithmetic computation is more complex and model-specific than previously hypothesized.