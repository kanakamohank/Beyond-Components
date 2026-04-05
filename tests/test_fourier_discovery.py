"""
Tests for src/analysis/fourier_discovery.py

Covers:
    - compute_fourier_power_spectrum correctness with known signals
    - identify_dominant_frequency logic
    - FourierResult and LayerFourierResult data classes
    - FourierDiscovery._resolve_positions
    - FourierDiscovery.analyze_layer
    - End-to-end run_all_layers with mock model

Does NOT require a real transformer model — all tests use mocks or synthetic data.
"""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch, PropertyMock

from src.analysis.fourier_discovery import (
    FourierDiscovery,
    FourierResult,
    LayerFourierResult,
    compute_fourier_power_spectrum,
    identify_dominant_frequency,
)


# ======================================================================
# compute_fourier_power_spectrum
# ======================================================================

class TestComputeFourierPowerSpectrum:
    """Tests for the core DFT computation."""

    def test_pure_cosine_signal(self):
        """A pure cosine at frequency k=1 should produce a peak at k=1."""
        N = 10
        d_model = 4
        activations = np.zeros((N, d_model))

        # Place a pure cosine in dimension 0: cos(2*pi*1*n/N)
        for n in range(N):
            activations[n, 0] = np.cos(2 * np.pi * 1 * n / N)

        power, dft = compute_fourier_power_spectrum(activations)

        # power should have N//2+1 = 6 entries
        assert power.shape == (N // 2 + 1,), f"Expected shape (6,), got {power.shape}"

        # k=1 should have the dominant power (excluding DC)
        assert np.argmax(power[1:]) + 1 == 1, (
            f"Expected dominant freq at k=1, got k={np.argmax(power[1:]) + 1}"
        )

    def test_pure_cosine_freq2(self):
        """A cosine at frequency k=2 should produce a peak at k=2."""
        N = 10
        d_model = 2
        activations = np.zeros((N, d_model))

        for n in range(N):
            activations[n, 0] = np.cos(2 * np.pi * 2 * n / N)

        power, _ = compute_fourier_power_spectrum(activations)
        dom_freq = np.argmax(power[1:]) + 1
        assert dom_freq == 2

    def test_dc_component(self):
        """A constant signal should have all power in DC (k=0)."""
        N = 10
        d_model = 3
        activations = np.ones((N, d_model)) * 5.0

        power, _ = compute_fourier_power_spectrum(activations)

        # DC should dominate
        assert power[0] > power[1:].sum(), (
            "Constant signal should have dominant DC component"
        )

    def test_white_noise_no_dominant_peak(self):
        """White noise should not have a strongly dominant frequency."""
        rng = np.random.RandomState(42)
        N = 100
        d_model = 16
        activations = rng.randn(N, d_model)

        power, _ = compute_fourier_power_spectrum(activations)

        # Check that no single non-DC freq has more than 5x the median
        non_dc = power[1:]
        ratio = non_dc.max() / np.median(non_dc)
        assert ratio < 10.0, (
            f"White noise should not have a dominant peak, got ratio {ratio:.1f}"
        )

    def test_two_frequencies(self):
        """Signal with two frequencies should show peaks at both."""
        N = 20
        d_model = 2
        activations = np.zeros((N, d_model))

        for n in range(N):
            activations[n, 0] = np.cos(2 * np.pi * 2 * n / N)  # freq 2
            activations[n, 1] = np.cos(2 * np.pi * 5 * n / N)  # freq 5

        power, _ = compute_fourier_power_spectrum(activations)

        # Both k=2 and k=5 should have significant power
        assert power[2] > np.median(power[1:]) * 2, "Missing peak at k=2"
        assert power[5] > np.median(power[1:]) * 2, "Missing peak at k=5"

    def test_minimum_input_size(self):
        """Should work with N=2 (minimum valid size)."""
        activations = np.array([[1.0, 0.0], [-1.0, 0.0]])
        power, dft = compute_fourier_power_spectrum(activations)
        assert power.shape == (2,)  # N//2+1 = 2

    def test_single_row_raises(self):
        """N=1 should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            compute_fourier_power_spectrum(np.array([[1.0, 2.0]]))

    def test_dft_shape(self):
        """Full DFT should have shape (N, d_model)."""
        N, d = 10, 8
        activations = np.random.randn(N, d)
        _, dft = compute_fourier_power_spectrum(activations)
        assert dft.shape == (N, d)


# ======================================================================
# identify_dominant_frequency
# ======================================================================

class TestIdentifyDominantFrequency:
    """Tests for dominant frequency identification."""

    def test_single_peak(self):
        """A spectrum with one clear peak should return that frequency."""
        power = np.array([100.0, 1.0, 50.0, 1.0, 1.0, 1.0])
        freq, ratio = identify_dominant_frequency(power)
        assert freq == 2, f"Expected freq=2, got {freq}"
        assert ratio > 1.0

    def test_dc_only_spectrum(self):
        """If only DC has power, dominant non-DC freq should have ratio ~0."""
        power = np.array([100.0, 0.0, 0.0])
        freq, ratio = identify_dominant_frequency(power)
        # ratio should be 0 or very small since median of [0,0] is 0
        assert ratio == 0.0

    def test_empty_spectrum(self):
        """Single-element spectrum should return (0, 0.0)."""
        power = np.array([5.0])
        freq, ratio = identify_dominant_frequency(power)
        assert freq == 0
        assert ratio == 0.0

    def test_ratio_calculation(self):
        """Verify ratio = power[dominant] / mean(other non-DC freqs)."""
        power = np.array([10.0, 2.0, 8.0, 2.0])
        freq, ratio = identify_dominant_frequency(power)
        assert freq == 2
        # Baseline = mean of non-DC excluding dominant (k=2):
        # non-DC = [2.0, 8.0, 2.0], dominant_idx=2 → other = [2.0, 2.0]
        expected_ratio = 8.0 / np.mean([2.0, 2.0])
        assert abs(ratio - expected_ratio) < 1e-6


# ======================================================================
# FourierResult and LayerFourierResult dataclasses
# ======================================================================

class TestResultDataclasses:
    """Tests for result data structures."""

    def test_fourier_result_creation(self):
        result = FourierResult(
            layer=5,
            hook_name="blocks.5.hook_resid_pre",
            power_spectrum=np.array([10.0, 5.0, 20.0]),
            dominant_frequency=2,
            dominant_frequency_power_ratio=4.0,
        )
        assert result.layer == 5
        assert result.dominant_frequency == 2
        assert result.activations is None
        assert result.dft_complex is None

    def test_layer_result_dominant_from_pre(self):
        """When only resid_pre exists, dominant should come from it."""
        pre = FourierResult(
            layer=3, hook_name="pre",
            power_spectrum=np.array([1.0, 5.0]),
            dominant_frequency=1,
            dominant_frequency_power_ratio=5.0,
        )
        lr = LayerFourierResult(layer=3, resid_pre=pre)
        assert lr.dominant_frequency == 1
        assert lr.dominant_frequency_power_ratio == 5.0

    def test_layer_result_picks_best_hook(self):
        """When both hooks exist, dominant should come from the stronger one."""
        pre = FourierResult(
            layer=3, hook_name="pre",
            power_spectrum=np.array([1.0, 5.0]),
            dominant_frequency=1,
            dominant_frequency_power_ratio=5.0,
        )
        post = FourierResult(
            layer=3, hook_name="post",
            power_spectrum=np.array([1.0, 20.0]),
            dominant_frequency=1,
            dominant_frequency_power_ratio=20.0,
        )
        lr = LayerFourierResult(layer=3, resid_pre=pre, resid_post=post)
        assert lr.dominant_frequency_power_ratio == 20.0


# ======================================================================
# FourierDiscovery.analyze_layer
# ======================================================================

class TestAnalyzeLayer:
    """Tests for the per-layer analysis method."""

    @pytest.fixture
    def mock_discovery(self):
        """Create a FourierDiscovery with a mock model."""
        model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        model.parameters.return_value = iter([mock_param])
        model.cfg.n_layers = 12
        model.cfg.n_heads = 8
        model.cfg.d_model = 64
        model.eval.return_value = None
        return FourierDiscovery(model, device=torch.device("cpu"))

    def test_analyze_layer_with_cosine(self, mock_discovery):
        """analyze_layer should detect a cosine signal."""
        N = 10
        d_model = 64
        activations = np.zeros((N, d_model))
        for n in range(N):
            activations[n, 0] = np.cos(2 * np.pi * 1 * n / N)

        result = mock_discovery.analyze_layer(
            activations, layer=3, hook_name="blocks.3.hook_resid_pre"
        )

        assert isinstance(result, FourierResult)
        assert result.layer == 3
        assert result.dominant_frequency == 1
        assert result.dominant_frequency_power_ratio > 1.0
        assert result.activations is None  # Not stored by default

    def test_analyze_layer_stores_activations(self, mock_discovery):
        """When store_activations=True, raw data should be saved."""
        activations = np.random.randn(10, 64)
        result = mock_discovery.analyze_layer(
            activations, layer=0, hook_name="test",
            store_activations=True,
        )
        assert result.activations is not None
        assert result.activations.shape == (10, 64)

    def test_analyze_layer_stores_dft(self, mock_discovery):
        """When store_dft=True, complex DFT should be saved."""
        activations = np.random.randn(10, 64)
        result = mock_discovery.analyze_layer(
            activations, layer=0, hook_name="test",
            store_dft=True,
        )
        assert result.dft_complex is not None
        assert result.dft_complex.shape == (10, 64)


# ======================================================================
# FourierDiscovery._resolve_positions
# ======================================================================

class TestResolvePositions:
    """Tests for token position resolution."""

    @pytest.fixture
    def discovery(self):
        model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        model.parameters.return_value = iter([mock_param])
        model.cfg.n_layers = 4
        model.cfg.n_heads = 4
        model.cfg.d_model = 32
        model.eval.return_value = None
        model.tokenizer.pad_token_id = 0
        return FourierDiscovery(model, device=torch.device("cpu"))

    def test_last_position_no_padding(self, discovery):
        """'last' with no padding should return seq_len - 1."""
        tokens = torch.tensor([[1, 2, 3, 4]])  # no pad tokens (pad=0)
        positions = discovery._resolve_positions(tokens, ["test"], "last")
        assert positions == [3]

    def test_last_position_with_padding(self, discovery):
        """'last' with padding should return last non-pad index."""
        tokens = torch.tensor([[1, 2, 3, 0, 0]])  # pad_id=0
        positions = discovery._resolve_positions(tokens, ["test"], "last")
        assert positions == [2]

    def test_integer_position(self, discovery):
        """Integer position should be returned as-is for all batch items."""
        tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
        positions = discovery._resolve_positions(tokens, ["a", "b"], "2")
        assert positions == [2, 2]

    def test_negative_integer_position(self, discovery):
        tokens = torch.tensor([[1, 2, 3]])
        positions = discovery._resolve_positions(tokens, ["test"], "-1")
        assert positions == [-1]

    def test_invalid_position_raises(self, discovery):
        tokens = torch.tensor([[1, 2, 3]])
        with pytest.raises(ValueError, match="Unknown position"):
            discovery._resolve_positions(tokens, ["test"], "invalid_pos")


# ======================================================================
# FourierDiscovery.summarize
# ======================================================================

class TestSummarize:
    """Tests for the human-readable summary generator."""

    def test_empty_results(self):
        summary = FourierDiscovery.summarize([])
        assert "No results" in summary

    def test_summary_contains_layer_info(self):
        pre = FourierResult(
            layer=5, hook_name="blocks.5.hook_resid_pre",
            power_spectrum=np.array([1.0, 5.0, 2.0, 1.0, 1.0, 1.0]),
            dominant_frequency=1,
            dominant_frequency_power_ratio=5.0,
        )
        lr = LayerFourierResult(layer=5, resid_pre=pre)
        summary = FourierDiscovery.summarize([lr])
        assert "5" in summary
        assert "5.0" in summary


# ======================================================================
# Integration: synthetic end-to-end with mock model
# ======================================================================

class TestAnalyzeLayerSynthetic:
    """Test analyze_layer with a known synthetic cosine signal."""

    def test_analyze_layer_synthetic_signal(self):
        """analyze_layer should detect a known cosine at freq=1."""
        N = 10
        d_model = 32
        model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        model.parameters.return_value = iter([mock_param])
        model.cfg.n_layers = 4
        model.cfg.n_heads = 4
        model.cfg.d_model = d_model
        model.eval.return_value = None

        discovery = FourierDiscovery(model, device=torch.device("cpu"))

        activations = np.zeros((N, d_model))
        for n in range(N):
            activations[n, 0] = np.cos(2 * np.pi * 1 * n / N)
            activations[n, 1] = np.sin(2 * np.pi * 1 * n / N)

        result = discovery.analyze_layer(
            activations, layer=2, hook_name="blocks.2.hook_resid_pre"
        )

        assert result.dominant_frequency == 1
        assert result.dominant_frequency_power_ratio > 5.0, (
            f"Expected strong signal, got ratio {result.dominant_frequency_power_ratio}"
        )


# ======================================================================
# Integration: collect_activations + run_all_layers with mock model
# ======================================================================

class _PromptCounter:
    """Track which prompt indices are in each batch for mock injection."""
    def __init__(self):
        self.call_count = 0

class TestCollectActivationsAndRunAllLayers:
    """Test collect_activations and run_all_layers through the full mock pipeline."""

    @pytest.fixture
    def mock_setup(self):
        """Build a mock model that returns synthetic activations with a
        known cosine at freq=1 in layer 1's resid_pre."""
        N = 10
        d_model = 16
        n_layers = 3
        n_heads = 2
        d_head = 8
        seq_len = 5

        model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        model.parameters.return_value = iter([mock_param])
        model.cfg.n_layers = n_layers
        model.cfg.n_heads = n_heads
        model.cfg.d_model = d_model
        model.eval.return_value = None
        model.tokenizer.pad_token_id = 0

        # Track batch offsets
        counter = _PromptCounter()

        def mock_to_tokens(texts):
            if isinstance(texts, str):
                texts = [texts]
            return torch.ones(len(texts), seq_len, dtype=torch.long)

        model.to_tokens.side_effect = mock_to_tokens

        def mock_run_with_cache(tokens, names_filter=None):
            batch_size = tokens.shape[0]
            cache = {}
            batch_offset = counter.call_count
            counter.call_count += batch_size

            for hook in (names_filter or []):
                if "hook_resid_pre" in hook or "hook_resid_post" in hook:
                    layer_idx = int(hook.split(".")[1])
                    tensor = torch.randn(batch_size, seq_len, d_model)
                    if layer_idx == 1 and "hook_resid_pre" in hook:
                        # Inject strong cosine at freq=1 at last token position
                        # Zero out noise at extraction position for clean signal
                        for b in range(batch_size):
                            n = batch_offset + b
                            tensor[b, -1, :] = 0.0
                            tensor[b, -1, 0] = 10.0 * np.cos(2 * np.pi * 1 * n / N)
                            tensor[b, -1, 1] = 10.0 * np.sin(2 * np.pi * 1 * n / N)
                    cache[hook] = tensor
                elif "attn.hook_z" in hook:
                    layer_idx = int(hook.split(".")[1])
                    tensor = torch.randn(batch_size, seq_len, n_heads, d_head)
                    if layer_idx == 1:
                        for b in range(batch_size):
                            n = batch_offset + b
                            tensor[b, -1, 0, :] = 0.0
                            tensor[b, -1, 0, 0] = 10.0 * np.cos(2 * np.pi * 1 * n / N)
                            tensor[b, -1, 0, 1] = 10.0 * np.sin(2 * np.pi * 1 * n / N)
                    cache[hook] = tensor

            return None, cache

        model.run_with_cache.side_effect = mock_run_with_cache

        prompts = [f"{n} + 0 =" for n in range(N)]

        return model, prompts, counter, N, d_model, n_layers

    def test_collect_activations(self, mock_setup):
        """collect_activations should return arrays of shape (N, d_model)."""
        model, prompts, counter, N, d_model, n_layers = mock_setup
        discovery = FourierDiscovery(model, device=torch.device("cpu"))

        acts = discovery.collect_activations(
            prompts=prompts, layers=[0, 1, 2], position="last", batch_size=5,
        )

        # Should have 3 hook entries (one per layer resid_pre)
        assert len(acts) == 3
        for hook, arr in acts.items():
            assert arr.shape == (N, d_model), f"{hook}: {arr.shape}"

    def test_run_all_layers(self, mock_setup):
        """run_all_layers should return LayerFourierResults with freq=1 strong in layer 1."""
        model, prompts, counter, N, d_model, n_layers = mock_setup
        discovery = FourierDiscovery(model, device=torch.device("cpu"))

        results = discovery.run_all_layers(
            prompts=prompts, layers=[0, 1, 2], position="last", batch_size=5,
        )

        assert len(results) == 3
        # Layer 1 should have the strongest signal
        layer1 = [r for r in results if r.layer == 1][0]
        assert layer1.dominant_frequency == 1
        assert layer1.dominant_frequency_power_ratio > 2.0

    def test_run_all_layers_with_resid_post(self, mock_setup):
        """include_resid_post should add post-layer results."""
        model, prompts, counter, N, d_model, n_layers = mock_setup
        discovery = FourierDiscovery(model, device=torch.device("cpu"))

        results = discovery.run_all_layers(
            prompts=prompts, layers=[1], position="last",
            include_resid_post=True, batch_size=10,
        )

        assert len(results) == 1
        assert results[0].resid_pre is not None
        assert results[0].resid_post is not None

    def test_analyze_attention_heads(self, mock_setup):
        """analyze_attention_heads should find significant heads in layer 1."""
        model, prompts, counter, N, d_model, n_layers = mock_setup
        discovery = FourierDiscovery(model, device=torch.device("cpu"))

        heads = discovery.analyze_attention_heads(
            prompts=prompts, layers=[0, 1, 2],
            position="last", batch_size=10, power_ratio_threshold=2.0,
        )

        # Layer 1, head 0 should be significant (we injected cosine there)
        assert (1, 0) in heads, f"Expected L1H0 in significant heads, got {list(heads.keys())}"
        assert heads[(1, 0)].dominant_frequency == 1
