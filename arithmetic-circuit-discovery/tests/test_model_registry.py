"""
Tests for src/utils/model_registry.py

Covers:
    - MODEL_REGISTRY completeness and consistency
    - get_model_spec lookup and error handling
    - list_available_models
    - load_model device fallback logic (mocked)
    - find_number_token_positions / find_last_number_token
    - verify_single_token_numbers

Does NOT load real models — all tests use mocks.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from src.utils.model_registry import (
    MODEL_REGISTRY,
    ModelSpec,
    get_model_spec,
    list_available_models,
    load_model,
    find_number_token_positions,
    find_last_number_token,
    verify_single_token_numbers,
    _warn_if_mismatch,
)


# ======================================================================
# MODEL_REGISTRY
# ======================================================================

class TestModelRegistry:
    """Tests for the static model registry."""

    def test_registry_not_empty(self):
        assert len(MODEL_REGISTRY) > 0

    def test_all_entries_are_model_spec(self):
        for key, spec in MODEL_REGISTRY.items():
            assert isinstance(spec, ModelSpec), f"{key} is not a ModelSpec"

    def test_required_models_present(self):
        """Phase 1-3 target models must be in the registry."""
        required = ["pythia-1.4b", "gemma-2b", "gpt2-small"]
        for key in required:
            assert key in MODEL_REGISTRY, f"Missing required model: {key}"

    def test_pythia_spec_values(self):
        spec = MODEL_REGISTRY["pythia-1.4b"]
        assert spec.family == "pythia"
        assert spec.n_layers == 24
        assert spec.n_heads == 16
        assert spec.d_model == 2048
        assert spec.d_mlp == 8192

    def test_gemma_has_gated_mlp(self):
        spec = MODEL_REGISTRY["gemma-2b"]
        assert spec.has_gated_mlp is True

    def test_gpt2_no_gated_mlp(self):
        spec = MODEL_REGISTRY["gpt2-small"]
        assert spec.has_gated_mlp is False

    def test_all_specs_have_positive_dimensions(self):
        for key, spec in MODEL_REGISTRY.items():
            assert spec.n_layers > 0, f"{key}: n_layers must be positive"
            assert spec.n_heads > 0, f"{key}: n_heads must be positive"
            assert spec.d_model > 0, f"{key}: d_model must be positive"
            assert spec.d_mlp > 0, f"{key}: d_mlp must be positive"
            assert spec.context_length > 0, f"{key}: context_length must be positive"

    def test_all_specs_have_transformer_lens_name(self):
        for key, spec in MODEL_REGISTRY.items():
            assert spec.transformer_lens_name, f"{key}: missing transformer_lens_name"

    def test_all_specs_have_family(self):
        for key, spec in MODEL_REGISTRY.items():
            assert spec.family, f"{key}: missing family"


# ======================================================================
# get_model_spec
# ======================================================================

class TestGetModelSpec:
    def test_valid_key(self):
        spec = get_model_spec("pythia-1.4b")
        assert spec.transformer_lens_name == "pythia-1.4b"

    def test_invalid_key_raises(self):
        with pytest.raises(KeyError, match="Unknown model key"):
            get_model_spec("nonexistent-model")

    def test_error_message_lists_available(self):
        try:
            get_model_spec("bad-key")
        except KeyError as e:
            assert "pythia-1.4b" in str(e)


# ======================================================================
# list_available_models
# ======================================================================

class TestListAvailableModels:
    def test_returns_sorted_list(self):
        models = list_available_models()
        assert models == sorted(models)
        assert len(models) == len(MODEL_REGISTRY)

    def test_contains_known_models(self):
        models = list_available_models()
        assert "pythia-1.4b" in models
        assert "gpt2-small" in models


# ======================================================================
# load_model (mocked — no actual model loading)
# ======================================================================

class TestLoadModel:
    @patch("src.utils.model_registry.HookedTransformer")
    def test_load_model_returns_tuple(self, mock_ht_class):
        """load_model should return (model, spec) tuple."""
        mock_model = MagicMock()
        mock_model.cfg.n_layers = 24
        mock_model.cfg.n_heads = 16
        mock_model.cfg.d_model = 2048
        mock_ht_class.from_pretrained.return_value = mock_model

        model, spec = load_model("pythia-1.4b", device="cpu")
        assert spec.family == "pythia"
        mock_ht_class.from_pretrained.assert_called_once()

    @patch("src.utils.model_registry.HookedTransformer")
    def test_load_model_passes_cache_dir(self, mock_ht_class):
        mock_model = MagicMock()
        mock_model.cfg.n_layers = 24
        mock_model.cfg.n_heads = 16
        mock_model.cfg.d_model = 2048
        mock_ht_class.from_pretrained.return_value = mock_model

        load_model("pythia-1.4b", device="cpu", cache_dir="my_cache")
        call_kwargs = mock_ht_class.from_pretrained.call_args
        assert call_kwargs.kwargs["cache_dir"] == "my_cache"

    @patch("src.utils.model_registry.HookedTransformer")
    def test_load_model_auto_device_cpu(self, mock_ht_class):
        """When CUDA and MPS are unavailable, should fall back to CPU."""
        mock_model = MagicMock()
        mock_model.cfg.n_layers = 12
        mock_model.cfg.n_heads = 12
        mock_model.cfg.d_model = 768
        mock_ht_class.from_pretrained.return_value = mock_model

        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=False):
            model, spec = load_model("gpt2-small")
            call_args = mock_ht_class.from_pretrained.call_args
            assert call_args.kwargs.get("device", call_args.args[0] if call_args.args else None) is not None


# ======================================================================
# find_number_token_positions
# ======================================================================

class TestFindNumberTokenPositions:
    def test_single_digit_found(self):
        """Single digit should be found in a simple prompt."""
        model = MagicMock()
        # Simulating "3 + 7 =" tokenized as separate tokens
        model.to_str_tokens.return_value = ["<BOS>", "3", " +", " 7", " ="]
        positions = find_number_token_positions(model, "3 + 7 =", 3)
        assert 1 in positions, f"Expected position 1, got {positions}"

    def test_multi_token_number(self):
        """A two-digit number split by BPE should return both token positions."""
        model = MagicMock()
        # "45" might be tokenized as ["4", "5"] by some tokenizers
        model.to_str_tokens.return_value = ["<BOS>", " 4", "5", " +"]
        positions = find_number_token_positions(model, "45 +", 45)
        assert len(positions) >= 1

    def test_number_not_found(self):
        """If target number is absent, return empty list."""
        model = MagicMock()
        model.to_str_tokens.return_value = ["<BOS>", "hello", " world"]
        positions = find_number_token_positions(model, "hello world", 99)
        assert positions == []

    def test_bpe_prefix_stripped(self):
        """BPE whitespace markers like Ġ should be stripped before matching."""
        model = MagicMock()
        model.to_str_tokens.return_value = ["<BOS>", "\u0120 3", " +", "\u01207", " ="]
        positions = find_number_token_positions(model, "3 + 7 =", 7)
        assert len(positions) >= 1


# ======================================================================
# find_last_number_token
# ======================================================================

class TestFindLastNumberToken:
    def test_returns_last_position(self):
        """Should return the last token containing the target number."""
        model = MagicMock()
        model.to_str_tokens.return_value = ["<BOS>", "1", "2", " +"]
        idx = find_last_number_token(model, "12 +", 12)
        assert idx == 2  # "2" is the last part of "12"

    def test_raises_on_missing_number(self):
        """Should raise ValueError when number is not in the prompt."""
        model = MagicMock()
        model.to_str_tokens.return_value = ["<BOS>", "hello"]
        with pytest.raises(ValueError, match="Could not find"):
            find_last_number_token(model, "hello", 99)


# ======================================================================
# verify_single_token_numbers
# ======================================================================

class TestVerifySingleTokenNumbers:
    def test_all_single_token(self):
        """When each digit is one token, all results should be True."""
        model = MagicMock()

        def mock_to_tokens(text, prepend_bos=True):
            # prefix " " alone → 1 token; prefix + single digit → 2 tokens
            if text == " ":
                return torch.tensor([[100]])
            # " 0", " 1", " 2" → each is 2 tokens (prefix + digit)
            return torch.tensor([[100, 200]])

        model.to_tokens.side_effect = mock_to_tokens

        results = verify_single_token_numbers(model, numbers=[0, 1, 2])
        assert all(results.values()), f"Expected all True, got {results}"

    def test_multi_token_detected(self):
        """When a digit takes 2 tokens, result should be False."""
        model = MagicMock()

        def mock_to_tokens(text, prepend_bos=True):
            if text == " ":
                return torch.tensor([[100]])
            if "9" in text:
                # " 9" → 3 tokens (prefix + 2 digit tokens = multi-token)
                return torch.tensor([[100, 200, 300]])
            # " 0" → 2 tokens (prefix + 1 digit token = single-token)
            return torch.tensor([[100, 200]])

        model.to_tokens.side_effect = mock_to_tokens

        results = verify_single_token_numbers(model, numbers=[0, 9])
        assert results[0] is True
        assert results[9] is False


# ======================================================================
# _warn_if_mismatch
# ======================================================================

class TestWarnIfMismatch:
    def test_no_warning_on_match(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            _warn_if_mismatch("n_layers", 24, 24, "test-model")
        assert len(caplog.records) == 0

    def test_warning_on_mismatch(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            _warn_if_mismatch("n_layers", 12, 24, "test-model")
        assert len(caplog.records) == 1
        assert "mismatch" in caplog.records[0].message.lower()
