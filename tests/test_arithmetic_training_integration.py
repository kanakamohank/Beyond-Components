"""
Phase 2e: Integration tests for the arithmetic training pipeline.

Tests the full pipeline from config loading → data loading → label preparation
→ head color dispatch → loader kwargs construction, WITHOUT loading a real model.
These tests ensure the arithmetic data type is correctly wired into all the
train.py integration points that were adapted from the IOI pipeline.
"""

import pytest
import yaml
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from torch.utils.data import DataLoader

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.arithmetic_dataset import (
    ArithmeticDataset,
    load_arithmetic_dataset,
)
from src.utils.utils import (
    get_data_column_names,
    get_label_column_names,
    get_data_label_column_names,
    get_indirect_objects_and_subjects,
)

# Import train.py helpers — they live under experiments/
from experiments.train import (
    _arithmetic_loader_kwargs,
    _prepare_label_strings,
    get_head_color,
    get_head_color_arithmetic,
    ARITHMETIC_HEAD_CATEGORIES,
    extract_corrupted_activations,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def arithmetic_config():
    """Minimal arithmetic config matching configs/arithmetic_pythia_config.yaml."""
    return {
        'experiment_name': 'test_arithmetic',
        'seed': 42,
        'use_wandb': False,
        'data_type': 'arithmetic',
        'max_pad_length': 16,
        'model': {
            'key': 'pythia-1.4b',
            'name': 'pythia-1.4b',
            'pretrained_cache_dir': 'cache_dir',
        },
        'arithmetic': {
            'operand_range_start': 0,
            'operand_range_end': 5,
            'operation': 'add',
            'prompt_template': '{a} + {b} =',
        },
        'training': {
            'batch_size': 25,
            'num_epochs': 1,
            'device': 'cpu',
            'learning_rate': 2e-2,
            'weight_decay': 0.0,
            'l1_weight': 2e-3,
            'temperature': 1.0,
            'eval_interval': 50,
            'patience': 5,
            'min_lr': 1e-6,
            'lr_factor': 0.5,
            'early_stopping': True,
            'save_checkpoints': False,
            'validation_interval': -1,
        },
        'masking': {
            'mask_init_value': 0.5,
            'cache_svd': True,
            'warmup_steps': 0,
            'max_iterations_per_batch': 5,
            'sparsity_threshold': 1e-3,
        },
        'train_masks': ['OV', 'MLP_in', 'MLP_out'],
        'visualization': {
            'plot_interval': 100,
            'log_gradients': False,
            'save_visualizations': False,
            'visualization_dir': 'visualizations',
        },
        'log_dir': 'logs',
    }


@pytest.fixture
def ioi_config():
    """Minimal IOI config for cross-checking."""
    return {
        'data_type': 'ioi',
        'arithmetic': None,
    }


# ======================================================================
# _arithmetic_loader_kwargs
# ======================================================================

class TestArithmeticLoaderKwargs:
    """Tests for the config → DataLoader kwargs builder."""

    def test_returns_correct_keys(self, arithmetic_config):
        kwargs = _arithmetic_loader_kwargs(arithmetic_config)
        assert set(kwargs.keys()) == {'operand_range', 'operation', 'prompt_template'}

    def test_operand_range_from_config(self, arithmetic_config):
        kwargs = _arithmetic_loader_kwargs(arithmetic_config)
        assert kwargs['operand_range'] == range(0, 5)

    def test_operation_from_config(self, arithmetic_config):
        kwargs = _arithmetic_loader_kwargs(arithmetic_config)
        assert kwargs['operation'] == 'add'

    def test_prompt_template_from_config(self, arithmetic_config):
        kwargs = _arithmetic_loader_kwargs(arithmetic_config)
        assert kwargs['prompt_template'] == '{a} + {b} ='

    def test_returns_empty_for_non_arithmetic(self, ioi_config):
        kwargs = _arithmetic_loader_kwargs(ioi_config)
        assert kwargs == {}

    def test_returns_empty_when_no_arithmetic_section(self):
        config = {'data_type': 'arithmetic'}
        kwargs = _arithmetic_loader_kwargs(config)
        assert kwargs == {}

    def test_defaults_when_keys_missing(self):
        config = {
            'data_type': 'arithmetic',
            'arithmetic': {},
        }
        kwargs = _arithmetic_loader_kwargs(config)
        assert kwargs['operand_range'] == range(0, 10)
        assert kwargs['operation'] == 'add'
        assert kwargs['prompt_template'] == '{a} + {b} ='


# ======================================================================
# _prepare_label_strings
# ======================================================================

class TestPrepareLabelStrings:
    """Tests for the label string preparation function."""

    def test_arithmetic_preserves_space_prefix(self):
        labels = [' 3', ' 7', ' 0']
        result = _prepare_label_strings(labels, 'arithmetic')
        assert result == [' 3', ' 7', ' 0']

    def test_ioi_adds_space_prefix(self):
        labels = ['Mary', 'John']
        result = _prepare_label_strings(labels, 'ioi')
        assert result == [' Mary', ' John']

    def test_arithmetic_handles_tensor_input(self):
        labels = torch.tensor([3, 7, 0])
        result = _prepare_label_strings(labels, 'arithmetic')
        # Tensor items get str()'d — for arithmetic, no extra space
        assert result == ['3', '7', '0']

    def test_ioi_handles_tensor_input(self):
        labels = torch.tensor([3, 7])
        result = _prepare_label_strings(labels, 'ioi')
        assert result == [' 3', ' 7']

    def test_arithmetic_list_of_strings(self):
        labels = [' 8', ' 1']
        result = _prepare_label_strings(labels, 'arithmetic')
        assert all(isinstance(s, str) for s in result)


# ======================================================================
# Head color dispatch for arithmetic
# ======================================================================

class TestHeadColorArithmetic:
    """Tests for arithmetic head color assignment."""

    def test_strong_periodic_head_has_correct_color(self):
        color, category = get_head_color_arithmetic(7, 9)
        assert category == 'Strong Periodic (>8x)'
        assert color == '#E63946'

    def test_moderate_periodic_head(self):
        color, category = get_head_color_arithmetic(14, 10)
        assert category == 'Moderate Periodic (6-8x)'
        assert color == '#FF6B35'

    def test_weak_periodic_head(self):
        color, category = get_head_color_arithmetic(18, 9)
        assert category == 'Weak Periodic (4-6x)'
        assert color == '#F4A261'

    def test_other_head_fallback(self):
        color, category = get_head_color_arithmetic(0, 0)
        assert category == 'Other Heads'
        assert color == '#CCCCCC'

    def test_dispatch_via_get_head_color(self):
        color, category = get_head_color(7, 9, 'arithmetic')
        assert category == 'Strong Periodic (>8x)'

    def test_dispatch_ioi_via_get_head_color(self):
        color, category = get_head_color(9, 6, 'ioi')
        assert category == 'Name Mover Heads'

    def test_all_strong_heads_are_registered(self):
        strong = ARITHMETIC_HEAD_CATEGORIES['Strong Periodic (>8x)']['heads']
        assert (7, 9) in strong
        assert (23, 9) in strong
        assert (15, 13) in strong
        assert (10, 15) in strong


# ======================================================================
# DataLoader integration: arithmetic data flows through train.py pipeline
# ======================================================================

class TestDataLoaderIntegration:
    """Test that arithmetic DataLoader produces data compatible with train.py."""

    def test_loader_via_getattr(self):
        """train.py discovers the loader via getattr(data_loader, f'load_{data_type}_dataset')."""
        from src.data import data_loader as local_data_loader
        fn = getattr(local_data_loader, 'load_arithmetic_dataset', None)
        assert fn is not None, "load_arithmetic_dataset not re-exported from data_loader"

    def test_loader_with_extra_kwargs(self, arithmetic_config):
        """The loader accepts the kwargs produced by _arithmetic_loader_kwargs."""
        kwargs = _arithmetic_loader_kwargs(arithmetic_config)
        loader = load_arithmetic_dataset(
            batch_size=arithmetic_config['training']['batch_size'],
            train=True,
            **kwargs,
        )
        assert isinstance(loader, DataLoader)
        batch = next(iter(loader))
        assert len(batch[ArithmeticDataset.COL_CLEAN]) == 25  # 5*5 = 25

    def test_batch_columns_match_utils(self):
        """Batch column names must match what get_data_column_names returns."""
        clean_col, corrupt_col = get_data_column_names('arithmetic')
        label_col, wrong_col = get_label_column_names('arithmetic')

        loader = load_arithmetic_dataset(batch_size=5, train=True, operand_range=range(0, 3))
        batch = next(iter(loader))

        assert clean_col in batch, f"Missing '{clean_col}' in batch"
        assert corrupt_col in batch, f"Missing '{corrupt_col}' in batch"
        assert label_col in batch, f"Missing '{label_col}' in batch"
        assert wrong_col in batch, f"Missing '{wrong_col}' in batch"

    def test_full_batch_mode(self, arithmetic_config):
        """full_batch=True should give a single batch with all samples."""
        kwargs = _arithmetic_loader_kwargs(arithmetic_config)
        loader = load_arithmetic_dataset(full_batch=True, train=True, **kwargs)
        batches = list(loader)
        assert len(batches) == 1
        assert len(batches[0][ArithmeticDataset.COL_CLEAN]) == 25

    def test_validation_loader_deterministic(self, arithmetic_config):
        """Validation loader should be deterministic (not shuffled)."""
        kwargs = _arithmetic_loader_kwargs(arithmetic_config)
        loader1 = load_arithmetic_dataset(batch_size=25, validation=True, **kwargs)
        loader2 = load_arithmetic_dataset(batch_size=25, validation=True, **kwargs)
        b1 = next(iter(loader1))[ArithmeticDataset.COL_CLEAN]
        b2 = next(iter(loader2))[ArithmeticDataset.COL_CLEAN]
        assert b1 == b2


# ======================================================================
# Column name consistency
# ======================================================================

class TestColumnNameConsistency:
    """Ensure arithmetic column names are consistent across all utils functions."""

    def test_data_columns(self):
        clean, corrupt = get_data_column_names('arithmetic')
        assert clean == 'arithmetic_input'
        assert corrupt == 'corr_arithmetic_input'

    def test_label_columns(self):
        correct, wrong = get_label_column_names('arithmetic')
        assert correct == 'arithmetic_answer'
        assert wrong == 'arithmetic_answer_wrong'

    def test_data_label_columns(self):
        correct, wrong = get_data_label_column_names('arithmetic')
        assert correct == 'arithmetic_answer'
        assert wrong == 'arithmetic_answer_wrong'

    def test_indirect_objects_alias(self):
        correct, wrong = get_indirect_objects_and_subjects('arithmetic')
        assert correct == 'arithmetic_answer'
        assert wrong == 'arithmetic_answer_wrong'

    def test_dataset_class_constants_match(self):
        assert ArithmeticDataset.COL_CLEAN == 'arithmetic_input'
        assert ArithmeticDataset.COL_CORRUPTED == 'corr_arithmetic_input'
        assert ArithmeticDataset.COL_ANSWER == 'arithmetic_answer'
        assert ArithmeticDataset.COL_ANSWER_WRONG == 'arithmetic_answer_wrong'


# ======================================================================
# Config YAML loads correctly
# ======================================================================

class TestConfigLoading:
    """Verify the actual YAML config file parses correctly."""

    @pytest.fixture
    def loaded_config(self):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'configs', 'arithmetic_pythia_config.yaml'
        )
        if not os.path.exists(config_path):
            pytest.skip("Config file not found")
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_data_type_is_arithmetic(self, loaded_config):
        assert loaded_config['data_type'] == 'arithmetic'

    def test_train_masks_skip_qk(self, loaded_config):
        masks = loaded_config.get('train_masks', [])
        assert 'QK' not in masks, "QK should be skipped for RoPE models"
        assert 'OV' in masks
        assert 'MLP_in' in masks
        assert 'MLP_out' in masks

    def test_operand_range_produces_single_digit_answers(self, loaded_config):
        arith = loaded_config['arithmetic']
        start = arith['operand_range_start']
        end = arith['operand_range_end']
        max_answer = (end - 1) + (end - 1)
        assert max_answer <= 9, (
            f"Max answer {max_answer} > 9 — answers must be single-digit "
            f"for single-token label matching"
        )

    def test_batch_size_equals_dataset_size(self, loaded_config):
        arith = loaded_config['arithmetic']
        n_pairs = (arith['operand_range_end'] - arith['operand_range_start']) ** 2
        assert loaded_config['training']['batch_size'] == n_pairs, (
            "batch_size should equal full dataset for clean gradient signal"
        )

    def test_mask_init_value(self, loaded_config):
        init = loaded_config['masking']['mask_init_value']
        assert 0.0 < init < 1.0, f"mask_init_value={init} should be in (0, 1)"

    def test_l1_weight_is_positive(self, loaded_config):
        l1 = loaded_config['training']['l1_weight']
        assert l1 > 0, f"l1_weight={l1} must be positive for sparsity pressure"


# ======================================================================
# End-to-end: data → tokenize → label matching flow (mocked tokenizer)
# ======================================================================

class TestTokenizationFlow:
    """Test that arithmetic data flows correctly through the tokenization
    steps in train.py (using a mock tokenizer)."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Mock tokenizer that returns predictable token IDs."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0

        def mock_call(texts, return_tensors='pt', padding=True):
            # Return sequential token IDs: each text gets [1, 2, 3, ...]
            batch_size = len(texts) if isinstance(texts, list) else 1
            if isinstance(texts, str):
                texts = [texts]
            max_len = 6  # Short arithmetic prompts
            ids = torch.ones(batch_size, max_len, dtype=torch.long)
            for i in range(batch_size):
                ids[i, :5] = torch.arange(1, 6)  # Non-pad tokens
                ids[i, 5] = 0  # Pad
            return {'input_ids': ids}

        tokenizer.side_effect = mock_call
        tokenizer.__call__ = mock_call
        return tokenizer

    def test_label_preparation_for_batch(self, mock_tokenizer):
        """_prepare_label_strings should work on a real batch from the loader."""
        loader = load_arithmetic_dataset(
            batch_size=10, train=True, operand_range=range(0, 5)
        )
        batch = next(iter(loader))

        label_col, wrong_col = get_label_column_names('arithmetic')
        labels = _prepare_label_strings(batch[label_col], 'arithmetic')
        wrong_labels = _prepare_label_strings(batch[wrong_col], 'arithmetic')

        # All should be strings
        assert all(isinstance(s, str) for s in labels)
        assert all(isinstance(s, str) for s in wrong_labels)

        # Arithmetic labels already have space prefix — should be preserved
        for lbl in labels:
            assert lbl.startswith(' '), f"Label '{lbl}' should start with space"
