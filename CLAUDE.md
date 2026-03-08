# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research repository for "Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits". The project implements SVD-based circuit discovery in transformer models, focusing on fine-grained interpretability at the subcomponent level within attention heads and MLP layers.

## Key Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install as package with dev dependencies
pip install -e .[dev]
```

### Data Preparation
```bash
# Extract the provided dataset archive
cd data/
unzip data_main.zip
cd ..
```

### Training Circuit Discovery Models
```bash
# Train on Gender Pronoun task
python run_train.py --config configs/gp_config.yaml

# Train on Indirect Object Identification task
python run_train.py --config configs/ioi_config.yaml

# Train on Greater-Than task
python run_train.py --config configs/gt_config.yaml

# Alternative: Direct training script
python experiments/train.py --config configs/gp_config.yaml
```

### Running Intervention Experiments
```bash
# Run intervention experiments (after training)
python run_ablation.py

# Alternative: Direct ablation script
python experiments/ablation/intervention.py
```

### Generating Evaluation Results
```bash
# Generate comprehensive metrics table
python experiments/evaluation/comprehensive_metrics_table.py

# Generate intervention results table
python experiments/evaluation/generate_sigma_table.py

# Run comprehensive sigma tests
python experiments/ablation/comprehensive_sigma_test.py
```

## Architecture

### Core Model: `MaskedTransformerCircuit`
- **Location**: `src/models/masked_transformer_circuit.py`
- **Purpose**: Main circuit discovery model that performs SVD decomposition on attention matrices and learns sparse masks
- **Key Features**:
  - SVD decomposition of QK (Query-Key) and OV (Output-Value) matrices
  - Learnable masks on singular value components
  - Optional MLP masking for both input and output matrices
  - SVD caching system for performance
  - Activation patching capabilities

### Data Pipeline
- **Location**: `src/data/data_loader.py`
- **Supported Tasks**:
  - **GP (Gender Pronoun)**: Predict gender pronouns in context
  - **IOI (Indirect Object Identification)**: Identify indirect objects in sentences
  - **GT (Greater-Than)**: Compare numerical values
- **Key Functions**: `load_gp_dataset()`, `load_ioi_dataset()`, `load_gt_dataset()`

### Experiment Framework
- **Training**: `experiments/train.py` - Main training loop with W&B logging and checkpointing
- **Ablations**: `experiments/ablation/intervention.py` - Swap activations along identified directions
- **Evaluation**: `experiments/evaluation/` - Generate sparsity vs accuracy analysis

## Configuration System

Configuration files are in YAML format in the `configs/` directory:
- `gp_config.yaml` - Gender Pronoun task configuration
- `ioi_config.yaml` - Indirect Object Identification configuration
- `gt_config.yaml` - Greater-Than task configuration

Key parameters:
- `training.learning_rate`: Mask learning rate (default: 2.0e-2)
- `training.l1_weight`: Sparsity penalty weight (default: 1.95e-4)
- `masking.mask_init_value`: Initial mask values (default: 0.99)
- `masking.sparsity_threshold`: Threshold for "active" components (default: 1e-3)

## Key Technical Concepts

### SVD Decomposition
The model decomposes attention matrices into singular vectors:
- **W_QK matrices**: Query-Key attention computation matrices
- **W_OV matrices**: Output-Value projection matrices
- **MLP matrices**: Optional masking of MLP input/output matrices

### Mask Learning
- Learnable parameters that gate singular value components
- L1 regularization encourages sparsity
- Sigmoid activation ensures masks are in [0,1] range
- Threshold-based binary decisions for circuit identification

### Activation Patching
- Interventional technique to test discovered circuits
- Swap activations along identified directions with empirically observed values
- Validates functional importance of discovered subspaces

## Dataset Structure

After extracting `data/data_main.zip`, datasets are organized as:
- **GP**: `train_1k_gp.csv`, `val_gp.csv`, `test_gp.csv`
- **IOI**: `train_1k_ioi.csv`, `train_5k_ioi.csv`, `val_ioi.csv`, `test_ioi.csv`
- **GT**: `train_gt_1k.csv`, `train_gt_2k.csv`, `train_gt_3k.csv`, `val_gt.csv`, `test_gt.csv`

## Output Organization

Training outputs are saved to:
- `logs/` - Training logs and visualizations
- `checkpoints/` - Model checkpoints (created during training)
- `svd_cache/` - Cached SVD decompositions for performance

## Development Notes

- The codebase uses TransformerLens for mechanistic interpretability
- SVD computations are cached to disk for efficiency
- Training supports both L1 and L0 regularization schemes
- Visualization utilities generate publication-quality plots
- W&B integration for experiment tracking (optional)

## Model Dependencies

- **Base Model**: GPT-2 small (via TransformerLens)
- **Key Libraries**: transformer-lens, torch, numpy, pandas, matplotlib, seaborn
- **Optional**: wandb for experiment tracking, plotly for interactive visualizations