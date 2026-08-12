import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

# Adjust import based on your project structure
from src.models.arithmetic_pipeline import GeometricArithmeticPipeline

@pytest.fixture
def mock_model():
    """Creates a mock HookedTransformer to avoid loading real LLM weights during tests."""
    model = MagicMock()
    model.cfg.model_name = "mock-gpt2"

    # Mock parameters device
    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    model.parameters.return_value = iter([mock_param])

    return model

@pytest.fixture
def pipeline(mock_model):
    return GeometricArithmeticPipeline(mock_model)

def test_robust_tokenization(pipeline):
    """Validates that the pipeline can find fragmented numbers from BPE tokenizers."""
    # Simulate a fragmented BPE tokenization for " 45" -> [" ", " 4", "5"]
    pipeline.model.to_tokens.return_value = [[50256, 220, 15, 60]]
    pipeline.model.to_str_tokens.return_value = ["<|endoftext|>", " ", " 4", "5"]

    # It should find "5" (index 3) because it searches backwards for parts of "45"
    idx = pipeline.get_operand_token_index("The number 45", 45)
    assert idx == 3, "Failed to find the last token of a fragmented number"

def test_extract_ov_svd(pipeline):
    """Validates Phase 2: OV Weight Extraction and SVD math."""
    d_model, d_head = 64, 32
    layer, head = 0, 0

    # Create random weight matrices
    W_V = torch.randn(d_model, d_head)
    W_O = torch.randn(d_head, d_model)

    # Mock the TransformerLens weight dictionaries
    pipeline.model.W_V = torch.zeros(1, 1, d_model, d_head)
    pipeline.model.W_O = torch.zeros(1, 1, d_head, d_model)
    pipeline.model.W_V[layer, head] = W_V
    pipeline.model.W_O[layer, head] = W_O

    U, S, Vh = pipeline.extract_ov_svd(layer, head)

    assert U.shape == (d_model, d_model)
    assert Vh.shape == (d_model, d_model)
    assert S.shape == (d_model,)

    # Verify SVD reconstruction mathematically
    W_OV_reconstructed = U @ torch.diag(S) @ Vh
    W_OV_expected = W_V @ W_O
    assert torch.allclose(W_OV_reconstructed, W_OV_expected, atol=1e-4), "SVD reconstruction failed"

def test_input_plane_geometry(pipeline):
    """Validates Phase 3: CV and Linearity geometric math using perfect trigonometric data."""
    layer, head = 0, 0
    d_model = 64
    numbers = [1, 2, 3, 4, 5]

    # 1. Mock SVD extraction to return specific singular vectors
    U = torch.eye(d_model)
    Vh = torch.eye(d_model) # Vh[0] is [1,0,0...], Vh[1] is [0,1,0...]
    S = torch.ones(d_model)
    pipeline.extract_ov_svd = MagicMock(return_value=(U, S, Vh))

    # 2. Generate perfect circular activations (x = cos, y = sin)
    angles = [2 * np.pi * n / 10 for n in numbers]
    perfect_activations = torch.zeros(len(numbers), d_model)
    for i, theta in enumerate(angles):
        perfect_activations[i, 0] = np.cos(theta) # Projects perfectly onto Vh[0]
        perfect_activations[i, 1] = np.sin(theta) # Projects perfectly onto Vh[1]

    # 3. Mock the caching mechanism (STATEFUL MOCK)
    call_counter = [0]
    def mock_run_with_cache(tokens):
        idx = call_counter[0]
        # Pipeline expects shape [batch=1, seq_len=1, d_model]
        act = perfect_activations[idx].unsqueeze(0).unsqueeze(0)
        cache = {
            f"blocks.{layer}.hook_resid_pre": act
        }
        call_counter[0] += 1
        return None, cache

    pipeline.model.run_with_cache.side_effect = mock_run_with_cache
    pipeline.get_operand_token_index = MagicMock(return_value=0)

    # Run the test
    best_plane = pipeline.test_input_plane(layer, head, numbers)

    assert best_plane is not None, "Failed to identify the perfect circle"
    assert best_plane['k1'] == 0 and best_plane['k2'] == 1, "Failed to select the correct SVD directions"
    assert np.isclose(best_plane['cv'], 0.0, atol=1e-5), "Perfect circle should have 0.0 CV"
    assert np.isclose(best_plane['lin'], 1.0, atol=1e-5), "Perfect angular progression should have 1.0 linearity"

def test_mlp_subspace_alignment_gated(pipeline):
    """Validates Phase 5: Gated MLP (SwiGLU) alignment logic."""
    layer, head = 0, 0
    d_model, d_mlp = 64, 128

    # Mock Phase 2 output (Attention Plane)
    U_ov = torch.eye(d_model) # Plane is first two dimensions
    pipeline.extract_ov_svd = MagicMock(return_value=(U_ov, None, None))

    # Mock a Modern Gated MLP (LLaMA style)
    mock_mlp = MagicMock()
    mock_mlp.W_gate = torch.randn(d_model, d_mlp)
    mock_mlp.W_in = torch.randn(d_model, d_mlp)
    pipeline.model.blocks = [MagicMock(mlp=mock_mlp)]

    # Test
    result = pipeline.measure_mlp_alignment(layer, head, 0, 1)

    assert 'max_alignment' in result
    assert 'alignment_matrix' in result
    assert result['alignment_matrix'].shape == (2, 10) # 2 attention dims x 10 MLP reading dims

def test_causal_phase_shift_math(pipeline):
    """Validates Phase 6: Vector rotation math executes correctly without dimension errors."""
    layer, head = 0, 0
    d_model = 64

    v1 = torch.zeros(d_model); v1[0] = 1.0
    v2 = torch.zeros(d_model); v2[1] = 1.0

    pipeline.get_operand_token_index = MagicMock(return_value=0)

    # Mock clean residual stream
    clean_resid = torch.zeros(1, 1, d_model)
    clean_resid[0, 0, 0] = 1.0 # x = 1, y = 0

    pipeline.model.run_with_cache.return_value = (None, {f"blocks.{layer}.hook_resid_pre": clean_resid})

    # Mock patched run
    mock_logits = torch.zeros(1, 1, 50000)
    mock_logits[0, 0, 42] = 100.0 # Force token ID 42 to be predicted
    pipeline.model.run_with_hooks.return_value = mock_logits
    pipeline.model.to_string.return_value = "42"

    result = pipeline.causal_phase_shift(layer, head, v1, v2, a=10, b=20, shift_delta=2, period=10.0)

    assert result['original_sum'] == 30
    assert result['expected_shifted_sum'] == 32
    assert result['model_output'] == "42"
    assert pipeline.model.run_with_hooks.called, "Failed to apply causal hooks"