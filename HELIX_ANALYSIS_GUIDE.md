# Helix-Based Arithmetic Circuit Discovery

This guide explains how to use the helix approach for discovering arithmetic circuits in transformer models, implementing the methodology from Neel and Tegmark/Kattamaneni's research on trigonometric number representations in language models.

## Overview

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