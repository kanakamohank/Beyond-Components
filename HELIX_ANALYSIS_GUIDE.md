# Helix-Based Arithmetic Circuit Discovery: Complete Methodology Guide

This comprehensive guide covers both the experimental methodology and practical usage for helix-based arithmetic circuit discovery in transformer models, implementing the methodology from Neel and Tegmark/Kattamaneni's research on trigonometric number representations in language models.

## Table of Contents

1. [Experimental Methodology Design](#experimental-methodology-design)
2. [Practical Usage Guide](#practical-usage-guide)
3. [Analysis Framework](#analysis-framework)
4. [Validation Protocols](#validation-protocols)

---

# Part I: Experimental Methodology Design

## 1. Research Framework and Objectives

### 1.1 Primary Research Questions
- **RQ1**: Which transformer architectures exhibit helical arithmetic encoding?
- **RQ2**: How do helix parameters correlate with arithmetic performance?
- **RQ3**: What is the relationship between model scale and geometric encoding emergence?
- **RQ4**: How do different training objectives affect helical circuit formation?

### 1.2 Experimental Hypotheses
1. **Architectural Hypothesis**: Specific transformer designs predispose models to helical encoding
2. **Scale Hypothesis**: Helix structures emerge above critical parameter thresholds
3. **Depth Hypothesis**: Arithmetic circuits form consistently at ~75% model depth
4. **Task Specificity Hypothesis**: Different arithmetic tasks activate distinct helix patterns

## 2. Multi-Factor Experimental Design

### 2.1 Experimental Matrix

| Factor | Levels | Sample Models | Rationale |
|--------|--------|---------------|-----------|
| **Architecture** | GPT-2, GPT-Neo, GPT-J, T5 | gpt2, gpt-neo-2.7B, gpt-j-6B | Test architectural influence |
| **Scale** | Small (<1B), Medium (1-5B), Large (>5B) | 117M, 2.7B, 6B | Investigate scale effects |
| **Task Type** | Counting, Addition, Modular | See task templates | Task specificity analysis |
| **Number Range** | 0-20, 0-50, 0-100 | Various ranges | Range dependency testing |
| **Prompt Format** | Natural, Symbolic, CoT | Multiple templates | Input format effects |

### 2.2 Controlled Variables
- **Hardware**: Consistent GPU/memory configuration (Apple M-series MPS)
- **Precision**: FP32 for numerical stability
- **Random Seed**: Fixed for reproducibility (seed=42)
- **Tokenization**: Model-specific standard tokenizers

## 3. Statistical Analysis Framework

### 3.1 Primary Statistical Tests
```python
STATISTICAL_ANALYSIS_PLAN = {
    'descriptive_statistics': ['mean', 'std', 'median', 'iqr', 'confidence_intervals'],
    'hypothesis_tests': {
        'architecture_effect': 'one_way_anova',
        'scale_correlation': 'pearson_correlation',
        'depth_consistency': 'chi_square_goodness_of_fit'
    },
    'effect_size_measures': ['cohens_d', 'eta_squared', 'correlation_coefficient'],
    'multiple_comparison_correction': 'bonferroni',
    'significance_level': 0.05
}
```

### 3.2 Quality Metrics Framework
```python
HELIX_QUALITY_METRICS = {
    'radius_cv': {
        'definition': 'Coefficient of variation of radii',
        'threshold': '< 0.2 for helix classification',
        'interpretation': 'Lower = more circular/consistent'
    },
    'angle_linearity': {
        'definition': 'Correlation between angles and numbers',
        'threshold': '> 0.85 for helix classification',
        'interpretation': 'Higher = more predictable progression'
    },
    'period_estimation': {
        'definition': '2π / angular_rate_of_change',
        'validation': 'Must be finite and reasonable (2-1000)',
        'interpretation': 'Indicates mathematical period of encoding'
    }
}
```

## 4. Validation and Reproducibility Protocol

### 4.1 Multi-Stage Validation
```python
VALIDATION_STAGES = {
    'internal_validation': {
        'cross_validation': 'k-fold within dataset (k=5)',
        'bootstrap_validation': 'Bootstrap resampling (n=1000)',
        'robustness_testing': 'Noise injection and subset analysis'
    },
    'external_validation': {
        'cross_hardware': 'GPU vs CPU vs MPS validation',
        'cross_implementation': 'Independent algorithm implementation',
        'literature_reproduction': 'Reproduce known results (GPT-2 Medium)'
    },
    'control_experiments': {
        'negative_controls': 'Random activations, shuffled numbers',
        'positive_controls': 'Synthetic helix data, known helix models',
        'methodology_validation': 'Inter-rater reliability testing'
    }
}
```

### 4.2 Reproducibility Requirements
- **Environment Documentation**: Complete package versions, hardware specs
- **Code Version Control**: Git commits for all analysis code
- **Random State Management**: All seeds documented and controllable
- **Data Provenance**: Complete audit trail of data collection/processing
- **Parameter Logging**: All hyperparameters saved with results

## 5. Quality Assurance Framework

### 5.1 Quality Control Checklist
- [ ] **Statistical Power Analysis**: Adequate sample sizes for planned tests
- [ ] **Multiple Comparisons**: Bonferroni correction applied where appropriate
- [ ] **Effect Size Reporting**: Practical significance assessed alongside statistical
- [ ] **Confidence Intervals**: 95% CIs reported for all estimates
- [ ] **Assumption Testing**: Statistical test assumptions verified
- [ ] **Outlier Analysis**: Systematic outlier detection and handling
- [ ] **Missing Data Protocol**: Explicit handling of missing/failed analyses

### 5.2 Risk Mitigation Strategies
```python
RISK_MITIGATION = {
    'computational_risks': {
        'memory_overflow': 'Progressive analysis with checkpointing',
        'hardware_failure': 'Cloud backup and distributed computing',
        'numerical_instability': 'Regularized SVD with condition monitoring'
    },
    'methodological_risks': {
        'false_discovery_rate': 'FDR correction and replication requirements',
        'selection_bias': 'Pre-registered analysis plans',
        'overfitting': 'Strict holdout validation',
        'confounding_variables': 'Randomized experimental design'
    }
}
```

## 6. Task Design and Data Collection Protocol

### 6.1 Arithmetic Task Battery
```python
EXPERIMENTAL_TASK_BATTERY = {
    'basic_counting': {
        'template': 'The number {n}',
        'range': range(0, 50),
        'expected_pattern': 'linear_angle_progression',
        'validation_metric': 'angle_linearity > 0.85'
    },
    'sequential_addition': {
        'template': '{n} + 1 =',
        'range': range(10, 40),
        'expected_pattern': 'phase_shift_by_one',
        'validation_metric': 'phase_shift ≈ 2π/period'
    },
    'modular_arithmetic': {
        'template': '{n} mod 10 =',
        'range': range(0, 100),
        'expected_pattern': 'periodic_wrapping',
        'validation_metric': 'period ≈ 10'
    },
    'multiplication_tables': {
        'template': '{n} × 2 =',
        'range': range(1, 21),
        'expected_pattern': 'scaled_rotation',
        'validation_metric': 'angle_scaling_factor ≈ 2'
    }
}
```

### 6.2 Data Collection Standards
- **Sample Size**: Minimum 30 numbers per task for statistical power
- **Activation Position**: Last token position for consistency
- **Model State**: Inference mode (no gradients) for memory efficiency
- **Quality Control**: Automatic outlier detection and removal
- **Missing Data**: Explicit protocol for handling failed extractions

### 6.3 Experimental Controls
```python
CONTROL_CONDITIONS = {
    'negative_controls': {
        'random_gaussian': 'Random Gaussian activations (μ=0, σ=1)',
        'shuffled_numbers': 'Randomly permuted number sequences',
        'non_arithmetic': 'Color names, animal names (no numerical content)'
    },
    'positive_controls': {
        'synthetic_helix': 'Artificially generated perfect helix data',
        'known_models': 'GPT-2 Medium L18H15 (confirmed helix)',
        'mathematical_functions': 'Exact trigonometric functions'
    }
}
```

---

# Part II: Practical Usage Guide

The helix approach discovers that language models encode numbers as points on helical structures in high-dimensional space, and perform arithmetic operations through trigonometric transformations. This provides a geometric interpretation of mathematical reasoning that complements traditional SVD-based circuit discovery.

### Key Concepts

1. **Helical Number Encoding**: Numbers are represented as angles on circular/helical structures
2. **Clock Algorithm**: Arithmetic operations work by rotating positions on the helix
3. **Trigonometric Computation**: Mathematical reasoning uses trigonometric functions
4. **SVD Integration**: Helix structures align with specific SVD directions

## Quick Start

### Basic Usage

```bash
# Run helix analysis on GPT-2 small
python run_helix_analysis.py --model gpt2-small --output_dir helix_results/

# Quick analysis with default settings
python run_helix_analysis.py --quick

# Analysis with custom parameters
python run_helix_analysis.py --cv_threshold 0.15 --linearity_threshold 0.85
```

### Programmatic Usage

```python
from src.models.helix_circuit_discovery import HelixArithmeticCircuit, quick_helix_analysis
from transformer_lens import HookedTransformer

# Load model
model = HookedTransformer.from_pretrained("gpt2-small")

# Quick analysis
helix_circuit = quick_helix_analysis(model, output_dir="my_analysis")

# Or detailed analysis
helix_circuit = HelixArithmeticCircuit(model)
found_heads = helix_circuit.find_arithmetic_heads()
report = helix_circuit.generate_comprehensive_report("detailed_results")
```

## How It Works

### 1. Helix Detection Process

The analysis follows these steps:

1. **Activation Collection**: Extract activations for number tokens across various prompts
2. **SVD Projection**: Project activations onto pairs of SVD directions
3. **Geometric Analysis**: Test for helical structure using:
   - **Radius Consistency**: Low coefficient of variation in radii
   - **Angle Linearity**: Strong correlation between angles and number values
   - **Period Estimation**: Determine the mathematical period of the helix

4. **Quality Assessment**: Rank helix candidates by geometric quality

### 2. Helix Quality Metrics

- **Radius CV** (lower is better): Measures how consistent the radius is across numbers
  - Good helix: CV < 0.2
  - Perfect circle: CV ≈ 0

- **Angle Linearity** (higher is better): Correlation between unwrapped angles and number values
  - Good helix: correlation > 0.9
  - Perfect helix: correlation = 1.0

- **Period**: The mathematical period of the helix (e.g., 10 for modular arithmetic)

### 3. Arithmetic Task Templates

The system tests various arithmetic patterns:

```python
arithmetic_tasks = {
    'number_recognition': {
        'template': 'The number {n} is',
        'range': range(0, 50)
    },
    'simple_addition': {
        'template': '{n} + 1 =',
        'range': range(10, 30)
    },
    'modular_arithmetic': {
        'template': '{n} mod 10 =',
        'range': range(10, 50)
    }
}
```

## Output Files

### Generated Visualizations

The analysis produces several types of visualizations:

#### 1. Helix Quality Heatmap
- `helix_quality_heatmap.png`: Shows helix quality metrics across all heads
- Helps identify which layers and heads exhibit strong helical structure

#### 2. Individual Head Analysis
For each helix head found:
- `helix_2d_L{layer}H{head}.png`: 2D projection showing the helix
- `helix_3d_L{layer}H{head}.html`: Interactive 3D helix visualization
- `comparison_L{layer}H{head}.png`: Comparison with standard SVD approach

#### 3. Analysis Reports
- `helix_analysis_report.json`: Complete numerical results
- `README.md`: Human-readable summary
- `operation_analysis.html`: Interactive analysis of arithmetic operations

### Understanding the Visualizations

#### 2D Helix Projection
- **Left panel**: Numbers plotted in 2D SVD space
- **Right panel**: Angle vs. number value showing linearity
- **Colors**: Represent numerical values
- **Ideal pattern**: Points form a circle (left) and straight line (right)

#### 3D Interactive Visualization
- **Helix structure**: Numbers spiral in 3D space
- **Red dashed line**: Ideal helix fit
- **Projection**: Gray points show 2D projection
- **Interactive**: Rotate and zoom to explore structure

#### Comparison Plots
- **Top row**: Standard SVD approach (masks, singular values)
- **Bottom row**: Helix approach (quality metrics, projection, phase)
- **Purpose**: Compare geometric vs. algebraic interpretability

## Advanced Usage

### Custom Arithmetic Tasks

Define your own arithmetic patterns:

```python
custom_tasks = {
    'fibonacci': {
        'template': 'Fibonacci: {n}',
        'range': [1, 1, 2, 3, 5, 8, 13, 21]  # Custom sequence
    },
    'powers_of_two': {
        'template': '2^n = {n}',
        'range': [1, 2, 4, 8, 16, 32, 64]
    }
}

helix_circuit = HelixArithmeticCircuit(model)
found_heads = helix_circuit.find_arithmetic_heads(custom_tasks)
```

### Phase Shift Analysis

Analyze how arithmetic operations correspond to rotations:

```python
# Analyze +1 operation
results = helix_circuit.analyze_arithmetic_operations(
    base_numbers=range(10, 20),
    operations={'add_1': 1, 'add_2': 2, 'subtract_1': -1}
)
```

### Comparison with Standard SVD

Compare helix discoveries with traditional circuit analysis:

```bash
python run_helix_analysis.py --compare_svd --svd_checkpoint path/to/trained/circuit.pt
```

## Integration with Existing Framework

The helix approach builds on the existing `MaskedTransformerCircuit` framework:

### Using with Trained Circuits

```python
from src.models.masked_transformer_circuit import MaskedTransformerCircuit
from src.models.helix_circuit_discovery import HelixArithmeticCircuit

# Load existing trained circuit
base_circuit = MaskedTransformerCircuit(model, cache_svd=True)
# ... training code ...

# Analyze with helix approach
helix_circuit = HelixArithmeticCircuit(model, base_circuit=base_circuit)
comparison = helix_circuit.generate_comprehensive_report("comparison_results")
```

### SVD Direction Analysis

The helix approach uses the same SVD decomposition but analyzes geometric structure:

```python
# Access SVD components
head_key = f'differential_head_{layer}_{head}'
U_ov, S_ov, Vh_ov, _ = base_circuit.svd_cache[f"{head_key}_ov"]

# Test helix structure in top SVD directions
direction_1 = Vh_ov[0]  # First principal component
direction_2 = Vh_ov[1]  # Second principal component
```

## Interpretation Guide

### What Makes a Good Helix?

1. **Low Radius CV (< 0.2)**: Numbers lie on a consistent circle
2. **High Angle Linearity (> 0.9)**: Angles increase linearly with numbers
3. **Reasonable Period**: Mathematical period makes sense (e.g., 10 for decimal)
4. **High Singular Values**: Important directions in the SVD

### Common Patterns

- **Addition Heads**: Period ≈ 10 (decimal arithmetic)
- **Modular Heads**: Period = modulus (e.g., 12 for mod 12)
- **Counting Heads**: Period ≈ sequence length
- **Position Heads**: Period related to sequence position

### Troubleshooting

#### No Helix Found
- Try lowering `cv_threshold` (e.g., 0.3)
- Try lowering `linearity_threshold` (e.g., 0.8)
- Check if model performs arithmetic (some models may not)
- Try different arithmetic tasks

#### Poor Quality Helix
- Model may use different encoding strategy
- Try different layers (arithmetic often in middle layers)
- Check tokenization (number tokens must be consistent)

## Research Applications

### Extending the Analysis

1. **New Arithmetic Operations**: Test multiplication, division, exponentiation
2. **Different Number Bases**: Binary, hexadecimal representations
3. **Complex Numbers**: 2D complex plane representations
4. **Mathematical Functions**: Trigonometric, logarithmic functions

### Integration with Training

```python
# Use helix structure to guide mask learning
helix_directions = helix_circuit.find_best_directions(layer, head)
custom_masks = create_helix_guided_masks(helix_directions)
```

## Citation

If you use this helix analysis in your research, please cite:

```bibtex
@article{helix_arithmetic,
  title={Language Models Use Trigonometry to Do Addition},
  author={[Authors from paper]},
  journal={arXiv preprint},
  year={2025}
}

@article{clock_pizza,
  title={The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks},
  author={[Authors from paper]},
  journal={arXiv preprint},
  year={2023}
}
```

## Example Results

### Typical Output

```
🔬 Starting Helix-Based Arithmetic Circuit Analysis
==============================================================

Testing task: numbers
  Layer 6:
    Head 3: ✓ HELIX FOUND!
      CV: 0.156, Linearity: 0.943, Period: 9.8
    Head 7: ✓ HELIX FOUND!
      CV: 0.203, Linearity: 0.891, Period: 12.1

Testing task: addition
  Layer 8:
    Head 2: ✓ HELIX FOUND!
      CV: 0.089, Linearity: 0.967, Period: 10.2

Found helix structures in 3 heads
✓ Analysis complete! Results saved to helix_analysis_results
```

### Key Files Generated

```
helix_analysis_results/
├── helix_analysis_report.json          # Complete results
├── README.md                           # Human-readable summary
├── helix_quality_heatmap.png           # Overview across all heads
└── visualizations/                     # Individual head analysis
    ├── L6H3/
    │   ├── helix_2d_L6H3.png          # 2D helix projection
    │   ├── helix_3d_L6H3.html         # Interactive 3D helix
    │   └── comparison_L6H3.png        # SVD vs helix comparison
    └── L8H2/
        └── ...
```

This helix approach provides a powerful geometric lens for understanding arithmetic reasoning in language models, complementing the algebraic perspective of traditional SVD-based circuit discovery.