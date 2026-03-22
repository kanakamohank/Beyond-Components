# Research Roadmap: Geometric SVD Framework for Arithmetic Circuit Discovery
## Complete Mechanistic Analysis of Pre-trained Language Models

**Created:** 2026-03-17
**Status:** Ready for Implementation
**Goal:** First rigorous geometric framework for discovering and validating arithmetic circuits in pre-trained LLMs

---

## Executive Summary

### Research Goal
Develop and validate the first comprehensive geometric framework for discovering arithmetic circuits in pre-trained language models, providing both observational and causal proof of trigonometric computation mechanisms across complete transformer blocks.

### Core Innovation
6-phase SVD methodology that rigorously identifies, analyzes, and causally validates trigonometric arithmetic computation in real LLMs, bridging the gap between toy model insights and production model understanding.

### Expected Impact
- **Mechanistic Interpretability**: First complete pipeline analysis of arithmetic computation in pre-trained models
- **Methodological Contribution**: Reusable geometric framework for circuit discovery
- **Theoretical Advancement**: Mathematical understanding of why arithmetic emerges at ~75% model depth

---

## Problem Statement & Novelty

### Current Literature Gap
Existing work falls into two categories:
1. **Toy Model Analysis**: Clean geometric insights (Kantamneni & Tegmark) but limited to trained-from-scratch models
2. **Pre-trained Model Analysis**: Component identification (Quirke et al.) but no geometric characterization

### Our Unique Contribution
**Bridge**: Apply rigorous geometric analysis to pre-trained models with full transformer block coverage
**Innovation**: Complete pipeline from component localization → geometric validation → causal intervention
**Impact**: First mechanistically complete understanding of arithmetic in production LLMs

---

## Methodology: 6-Phase Geometric SVD Framework

### Phase 1: Component Localization via Total Indirect Effect (TIE)
**Goal**: Isolate attention heads responsible for mathematical routing
**Method**: Causal activation patching using minimal operand perturbation

For addition prompt $P = \text{"What is } a + b \text{?"}$ and corrupted $P' = \text{"What is } a + (b+1) \text{?"}$:

$$TIE^{(l,h)} = \mathbb{E}_{a,b} \left[ P(y_{clean} \mid P', do(A^{(l,h)} = A_{clean}^{(l,h)})) - P(y_{clean} \mid P') \right]$$

**Success Criteria**: Heads with TIE > 0.05 classified as "Clock Heads"

### Phase 2: OV Circuit Extraction and SVD
**Goal**: Decompose attention computation matrices
**Method**: Extract combined OV circuit and apply SVD

$$W_{OV}^{(l,h)} = W_V^{(l,h)} W_O^{(l,h)} = U \Sigma V^T$$

Where $V^T$ represents "reading" directions and $U$ represents "writing" directions.

### Phase 3: Input Plane Geometric Testing
**Goal**: Test if attention head reads trigonometric number representations
**Method**: Project cached residual stream onto $V^T$ singular vector pairs

For coordinates $c_{n} = (h_{pre}^{(l)}(n) \cdot v_i, h_{pre}^{(l)}(n) \cdot v_j)$:

**Geometric Criteria**:
- **Constant Radius**: CV of $||c_n|| < 0.2$
- **Linear Angular Progression**: $|r(\theta_n, n)| > 0.9$ where $\theta_n = \text{atan2}(c_{n,j}, c_{n,i})$

### Phase 4: Output Plane Computation Testing
**Goal**: Verify attention head writes trigonometric computation results
**Method**: Project post-attention residual stream onto $U$ singular vector pairs

Apply identical geometric criteria to output projections with target sum $(a+b)$.
**Proves**: Head performs computation, not just data routing.

### Phase 5: MLP Interaction Pipeline Analysis
**Goal**: Characterize complete transformer block arithmetic pipeline
**Method**: Analyze MLP transformations of trigonometric attention outputs

**Key Analyses**:
1. **Subspace Alignment**: Measure cosine similarity between attention output directions and MLP input directions
2. **Transformation Characterization**: Test if MLPs apply trigonometric identities (cos addition, sin addition)
3. **Pipeline Validation**: Verify end-to-end trigonometric computation flow

**Success Criteria**:
- High alignment (>0.7) between attention output and MLP input subspaces
- MLP transformations preserve/enhance trigonometric structure
- Complete pipeline implements clock algorithm

### Phase 6: Causal Verification via Phase-Shift Intervention
**Goal**: Definitively prove geometric planes drive arithmetic computation
**Method**: Direct rotation in discovered 2D trigonometric plane

Intervened hidden state:
$$h'_{pre} = h_{pre} - (c_1 v_1 + c_2 v_2) + (c_1 \cos\theta - c_2 \sin\theta)v_1 + (c_1 \sin\theta + c_2 \cos\theta)v_2$$

Where $\theta = \frac{2\pi \cdot \delta}{T}$ for arithmetic shift $\delta$ and period $T$.

**Success Criteria**: Model output shifts by $\delta$ with >80% accuracy
**Stratification**: Separate analysis for carry vs no-carry cases

---

## Experimental Validation Plan

### Target Models & Discovered Locations
Based on preliminary geometric analysis:
- **GPT-2 Small (117M)**: Layer 9, Head 9 (75% depth, Period ≈ 74.2)
- **GPT-2 Medium (355M)**: Layer 18, Head 15 (75% depth, Period ≈ 55.4)

### Core Experimental Protocol

#### Experiment 1: Complete Framework Validation - GPT-2 Small L9H9
**Scope**: Apply full 6-phase framework to validate trigonometric computation pipeline
**Expected Results**:
- Phase 1: TIE > 0.05 (arithmetic head confirmation)
- Phase 3: Geometric criteria met (CV < 0.2, |r| > 0.9)
- Phase 4: Output plane confirms computation vs routing
- Phase 5: High MLP-attention alignment (>0.7), trigonometric transformations
- Phase 6: Causal intervention accuracy >80%

#### Experiment 2: Cross-Model Validation - GPT-2 Medium L18H15
**Scope**: Replicate complete analysis across model scales
**Expected Results**:
- Consistent 75% depth pattern maintained
- Improved geometric precision (higher correlation coefficients)
- Robust causal control across both models
- Validation of architectural consistency

#### Experiment 3: Statistical Robustness & Generalization
**Scope**: Test framework across multiple conditions
**Variables**:
- Multiple arithmetic prompts (50+ variations)
- Different random seeds for consistency
- Various number ranges (0-99, 0-999)
- Addition vs subtraction operations

**Success Criteria**:
- Consistent geometric signatures across conditions
- Robust causal intervention (>80% accuracy maintained)
- Statistical significance of findings (p < 0.01)

### Theoretical Analysis Extensions

#### Analysis 1: 75% Depth Principle
**Research Question**: Why do arithmetic circuits emerge at ~75% model depth?
**Method**: Information flow analysis, representational capacity measurements
**Deliverable**: Theoretical framework for optimal arithmetic circuit placement

#### Analysis 2: Period Adaptation Mechanism
**Research Question**: How do different model scales adapt trigonometric periods?
**Method**: Compare GPT-2 Small (74.2) vs Medium (55.4) period differences
**Deliverable**: Mathematical model for period-scale relationships

---

## Implementation Timeline

### Week 1-2: Framework Development
- **Deliverable**: Complete 6-phase methodology implementation
- **Tasks**:
  - TIE computation infrastructure
  - SVD extraction and geometric testing
  - Causal intervention mechanism
- **Milestone**: Framework ready for model analysis

### Week 3: GPT-2 Small L9H9 Analysis
- **Deliverable**: Complete validation of trigonometric computation
- **Tasks**: Apply Phases 1-6 to Layer 9, Head 9
- **Milestone**: Proof-of-concept successful

### Week 4: GPT-2 Medium L18H15 Cross-Validation
- **Deliverable**: Cross-model consistency demonstration
- **Tasks**: Replicate analysis on larger model
- **Milestone**: Scalability confirmed

### Week 5: Statistical Robustness & Extensions
- **Deliverable**: Comprehensive validation across conditions
- **Tasks**: Multiple prompts, seeds, theoretical analysis
- **Milestone**: Publication-ready results

### Week 6: Analysis & Documentation
- **Deliverable**: Complete research manuscript
- **Tasks**: Result synthesis, theoretical framework development
- **Milestone**: Submission-ready paper

---

## Success Metrics & Publication Criteria

### Minimum Viable Contribution
- [ ] 6-phase framework implemented and validated
- [ ] Successful trigonometric computation proof in both GPT-2 models
- [ ] Causal intervention demonstrates arithmetic control
- [ ] Complete transformer block analysis (attention + MLP)

### Strong Publication Case
- [ ] Statistical robustness across multiple conditions (>80% success rate)
- [ ] Theoretical framework for 75% depth emergence principle
- [ ] Cross-model consistency validation (Small vs Medium)
- [ ] Comparison with existing mechanistic interpretability methods
- [ ] Reusable methodology applicable to other arithmetic tasks

### Publication Venues
**Primary Target**: NeurIPS (Mechanistic Interpretability Track)
**Alternative Venues**: ICML, ICLR (Interpretability Workshops)
**Rationale**: Novel methodology + rigorous validation + practical impact

---

## Resource Requirements

### Computational Resources
- **Models**: GPT-2 Small (117M), GPT-2 Medium (355M) inference
- **Hardware**: GPU with 16GB+ memory for efficient batch processing
- **Storage**: ~50GB for cached activations and intermediate results

### Software Infrastructure
- **Base**: TransformerLens for model access and activation caching
- **Extensions**: Existing SVD analysis infrastructure
- **New Components**: TIE computation, geometric testing, causal intervention

### Personnel & Expertise
- **Primary Researcher**: 1 person, 6 weeks full-time
- **Skills Required**: Mechanistic interpretability, linear algebra, PyTorch
- **Supervision**: Weekly progress reviews with faculty advisor

---

## Risk Assessment & Mitigation

### Technical Risks

**Risk 1**: Geometric criteria too strict, no heads pass validation
**Mitigation**: Adaptive thresholds based on empirical distributions
**Backup Plan**: Relaxed criteria with statistical significance testing

**Risk 2**: Causal intervention doesn't work on pre-trained models
**Mitigation**: Multiple intervention strategies (different rotation amounts)
**Backup Plan**: Focus on observational validation with theoretical analysis

**Risk 3**: MLP analysis reveals no clear patterns
**Mitigation**: Multiple analysis approaches (SVD, activation patching, attention flow)
**Backup Plan**: Focus on attention-only analysis with future work discussion

### Strategic Risks

**Risk 1**: Reviewer criticism of limited scope (only arithmetic)
**Mitigation**: Emphasize reusable methodology, discuss extensions
**Response**: Position as foundational framework for broader circuit discovery

**Risk 2**: Comparison with existing toy model work
**Mitigation**: Clear differentiation of pre-trained model complexity
**Response**: Emphasize bridge between toy insights and real-world applications

---

## Expected Contributions & Impact

### Methodological Contributions
1. **First rigorous geometric framework** for arithmetic circuit discovery in pre-trained models
2. **Complete transformer block analysis** methodology (attention + MLP pipeline)
3. **Causal validation framework** with mathematical precision
4. **Reusable infrastructure** for other mathematical reasoning tasks

### Scientific Contributions
1. **Mechanistic understanding** of trigonometric computation in production LLMs
2. **Architectural insights** about 75% depth arithmetic circuit emergence
3. **Cross-model validation** of geometric computation principles
4. **Bridge between theory and practice** in mechanistic interpretability

### Practical Impact
1. **Model interpretability** tools for arithmetic reasoning analysis
2. **Architecture insights** for designing more interpretable models
3. **Validation methodology** for other mechanistic interpretability research
4. **Foundation** for understanding more complex mathematical reasoning

---

This roadmap provides a complete, rigorous framework ready for professor review and implementation.