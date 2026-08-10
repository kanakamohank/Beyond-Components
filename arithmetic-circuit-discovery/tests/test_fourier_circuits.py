"""
Phase 4: Unit tests for analyze_fourier_circuits.py core functions.

Tests cover:
  - fit_cosine_sine: pure cos, pure sin, mixed, DC-only, multi-frequency
  - fit_multi_frequency: combined multi-frequency fits
  - find_secondary_frequencies: peak detection after removing primary
  - analyze_neuron_trig_identity: separability, trig identity fit on synthetic data
"""

import math
import pytest
import numpy as np

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.analyze_fourier_circuits import (
    fit_cosine_sine,
    fit_multi_frequency,
    find_secondary_frequencies,
    analyze_neuron_trig_identity,
)


# ---------------------------------------------------------------------------
# fit_cosine_sine tests
# ---------------------------------------------------------------------------

class TestFitCosineSine:
    """Tests for single-frequency cos/sin fitting."""

    def test_pure_cosine(self):
        """A pure cosine signal should give A≈amplitude, B≈0, R²≈1."""
        N = 100
        f = 3
        n = np.arange(N, dtype=np.float64)
        signal = 2.0 * np.cos(2 * np.pi * f * n / N)

        A, B, C, amp, phase, r2, res_std = fit_cosine_sine(signal, f)

        assert abs(A - 2.0) < 1e-10
        assert abs(B) < 1e-10
        assert abs(C) < 1e-10
        assert abs(amp - 2.0) < 1e-10
        assert abs(phase) < 1e-10  # phase = atan2(0, 2) = 0
        assert r2 > 0.9999
        assert res_std < 1e-10

    def test_pure_sine(self):
        """A pure sine signal should give A≈0, B≈amplitude, R²≈1."""
        N = 100
        f = 5
        n = np.arange(N, dtype=np.float64)
        signal = 3.0 * np.sin(2 * np.pi * f * n / N)

        A, B, C, amp, phase, r2, res_std = fit_cosine_sine(signal, f)

        assert abs(A) < 1e-10
        assert abs(B - 3.0) < 1e-10
        assert abs(C) < 1e-10
        assert abs(amp - 3.0) < 1e-10
        assert abs(phase - np.pi / 2) < 1e-10  # phase = atan2(3, 0) = π/2
        assert r2 > 0.9999

    def test_cosine_with_dc_offset(self):
        """Cos + DC offset should be recovered exactly."""
        N = 64
        f = 2
        n = np.arange(N, dtype=np.float64)
        signal = 1.5 * np.cos(2 * np.pi * f * n / N) + 10.0

        A, B, C, amp, phase, r2, res_std = fit_cosine_sine(signal, f)

        assert abs(A - 1.5) < 1e-10
        assert abs(B) < 1e-10
        assert abs(C - 10.0) < 1e-10
        assert r2 > 0.9999

    def test_mixed_cos_sin(self):
        """A mixed cos+sin signal should recover both coefficients."""
        N = 100
        f = 4
        n = np.arange(N, dtype=np.float64)
        theta = 2 * np.pi * f * n / N
        signal = 3.0 * np.cos(theta) + 4.0 * np.sin(theta) + 1.0

        A, B, C, amp, phase, r2, res_std = fit_cosine_sine(signal, f)

        assert abs(A - 3.0) < 1e-10
        assert abs(B - 4.0) < 1e-10
        assert abs(C - 1.0) < 1e-10
        assert abs(amp - 5.0) < 1e-10  # sqrt(9+16)
        assert r2 > 0.9999

    def test_wrong_frequency_low_r2(self):
        """Fitting at the wrong frequency should give low R²."""
        N = 100
        f_true = 3
        f_fit = 7
        n = np.arange(N, dtype=np.float64)
        signal = 2.0 * np.cos(2 * np.pi * f_true * n / N)

        _, _, _, _, _, r2, _ = fit_cosine_sine(signal, f_fit)

        assert r2 < 0.05  # should be near 0

    def test_dc_only_signal(self):
        """A constant signal should give A=B=0, R²=0."""
        N = 50
        signal = np.full(N, 5.0)

        A, B, C, amp, phase, r2, res_std = fit_cosine_sine(signal, 3)

        assert abs(A) < 1e-10
        assert abs(B) < 1e-10
        assert abs(C - 5.0) < 1e-10
        assert r2 < 1e-10

    def test_noisy_cosine(self):
        """Cosine with noise should still have decent R²."""
        np.random.seed(42)
        N = 200
        f = 5
        n = np.arange(N, dtype=np.float64)
        signal = 4.0 * np.cos(2 * np.pi * f * n / N) + np.random.randn(N) * 0.5

        A, B, C, amp, phase, r2, res_std = fit_cosine_sine(signal, f)

        assert abs(A - 4.0) < 0.2
        assert r2 > 0.9  # SNR is high


# ---------------------------------------------------------------------------
# fit_multi_frequency tests
# ---------------------------------------------------------------------------

class TestFitMultiFrequency:
    """Tests for multi-frequency combined fitting."""

    def test_single_freq_matches_cosine_sine(self):
        """Single-frequency multi-fit should match fit_cosine_sine R²."""
        N = 100
        f = 3
        n = np.arange(N, dtype=np.float64)
        signal = 2.0 * np.cos(2 * np.pi * f * n / N) + 0.5

        _, _, _, _, _, r2_single, _ = fit_cosine_sine(signal, f)
        r2_multi = fit_multi_frequency(signal, [f])

        assert abs(r2_single - r2_multi) < 1e-10

    def test_two_frequencies_high_r2(self):
        """A signal with two frequencies should get high R² from multi-fit."""
        N = 100
        n = np.arange(N, dtype=np.float64)
        signal = (
            2.0 * np.cos(2 * np.pi * 3 * n / N)
            + 1.0 * np.sin(2 * np.pi * 7 * n / N)
        )

        r2 = fit_multi_frequency(signal, [3, 7])
        assert r2 > 0.9999

    def test_multi_freq_better_than_single(self):
        """Multi-freq fit should explain more variance than single-freq."""
        N = 100
        n = np.arange(N, dtype=np.float64)
        signal = (
            3.0 * np.cos(2 * np.pi * 2 * n / N)
            + 2.0 * np.sin(2 * np.pi * 5 * n / N)
            + 1.0 * np.cos(2 * np.pi * 10 * n / N)
        )

        r2_one = fit_multi_frequency(signal, [2])
        r2_all = fit_multi_frequency(signal, [2, 5, 10])

        assert r2_all > r2_one + 0.1
        assert r2_all > 0.999

    def test_empty_frequencies_returns_zero(self):
        """Empty frequency list should return R²=0."""
        signal = np.ones(50)
        assert fit_multi_frequency(signal, []) == 0.0

    def test_three_frequencies_with_noise(self):
        """Three frequencies + noise should give R² proportional to SNR."""
        np.random.seed(123)
        N = 200
        n = np.arange(N, dtype=np.float64)
        signal = (
            5.0 * np.cos(2 * np.pi * 1 * n / N)
            + 3.0 * np.sin(2 * np.pi * 4 * n / N)
            + 2.0 * np.cos(2 * np.pi * 20 * n / N)
            + np.random.randn(N) * 0.5
        )

        r2 = fit_multi_frequency(signal, [1, 4, 20])
        assert r2 > 0.95


# ---------------------------------------------------------------------------
# find_secondary_frequencies tests
# ---------------------------------------------------------------------------

class TestFindSecondaryFrequencies:
    """Tests for secondary frequency peak detection."""

    def test_single_secondary(self):
        """Signal with two frequencies: secondary should be found."""
        np.random.seed(99)
        N = 100
        n = np.arange(N, dtype=np.float64)
        signal = (
            5.0 * np.cos(2 * np.pi * 3 * n / N)
            + 3.0 * np.cos(2 * np.pi * 7 * n / N)
            + np.random.randn(N) * 0.1  # small noise for realistic baseline
        )

        secondaries = find_secondary_frequencies(signal, primary_freq=3, min_ratio=2.0)

        assert len(secondaries) >= 1
        freqs = [f for f, _ in secondaries]
        assert 7 in freqs

    def test_no_secondary_in_pure_tone(self):
        """A pure single-frequency signal should have no strong secondaries."""
        N = 100
        n = np.arange(N, dtype=np.float64)
        signal = 5.0 * np.cos(2 * np.pi * 3 * n / N)

        secondaries = find_secondary_frequencies(
            signal, primary_freq=3, min_ratio=2.0
        )

        # Should find no secondaries above threshold
        assert len(secondaries) == 0

    def test_max_secondary_limit(self):
        """Should not return more than max_secondary peaks."""
        N = 200
        n = np.arange(N, dtype=np.float64)
        signal = sum(
            (10 - k) * np.cos(2 * np.pi * (k + 1) * n / N)
            for k in range(6)
        )

        secondaries = find_secondary_frequencies(
            signal, primary_freq=1, min_ratio=1.5, max_secondary=2
        )

        assert len(secondaries) <= 2

    def test_returns_sorted_by_power(self):
        """Secondary peaks should be returned strongest first."""
        N = 200
        n = np.arange(N, dtype=np.float64)
        signal = (
            10.0 * np.cos(2 * np.pi * 2 * n / N)  # primary
            + 5.0 * np.cos(2 * np.pi * 5 * n / N)  # strongest secondary
            + 2.0 * np.cos(2 * np.pi * 8 * n / N)  # weaker secondary
        )

        secondaries = find_secondary_frequencies(
            signal, primary_freq=2, min_ratio=1.5, max_secondary=3
        )

        if len(secondaries) >= 2:
            # First secondary should be stronger
            assert secondaries[0][1] >= secondaries[1][1]


# ---------------------------------------------------------------------------
# analyze_neuron_trig_identity tests
# ---------------------------------------------------------------------------

class TestAnalyzeNeuronTrigIdentity:
    """Tests for MLP neuron trig identity analysis on synthetic data."""

    def test_perfect_trig_identity(self):
        """A perfect sin(f(a+b)/N) neuron should get high R²."""
        N = 30
        f = 3
        a = np.arange(N, dtype=np.float64)
        b = np.arange(N, dtype=np.float64)
        A, B = np.meshgrid(a, b, indexing="ij")
        activations = np.sin(2 * np.pi * f * (A + B) / N)

        result = analyze_neuron_trig_identity(activations, neuron_idx=0, layer=0, max_freq=9)

        assert result.trig_identity_r2 > 0.99
        assert result.sum_frequency == f

    def test_separable_function(self):
        """f(a)·g(b) should have high separability score."""
        N = 20
        a = np.arange(N, dtype=np.float64)
        b = np.arange(N, dtype=np.float64)
        A, B = np.meshgrid(a, b, indexing="ij")
        activations = np.sin(2 * np.pi * A / N) * np.cos(2 * np.pi * B / N)

        result = analyze_neuron_trig_identity(activations, neuron_idx=0, layer=0)

        assert result.separability_score > 0.9

    def test_random_noise_low_scores(self):
        """Random noise should have low trig R² and low separability."""
        np.random.seed(42)
        N = 20
        activations = np.random.randn(N, N)

        result = analyze_neuron_trig_identity(activations, neuron_idx=0, layer=0)

        assert result.trig_identity_r2 < 0.15
        assert result.separability_score < 0.3

    def test_constant_activation(self):
        """Constant activations should give ~0 for everything."""
        N = 20
        activations = np.full((N, N), 5.0)

        result = analyze_neuron_trig_identity(activations, neuron_idx=0, layer=0)

        assert result.activation_std < 1e-10
        assert result.trig_identity_r2 <= 0.0

    def test_single_operand_periodic(self):
        """sin(f·a/N) (depends only on a) should have high separability but low trig R²."""
        N = 20
        a = np.arange(N, dtype=np.float64)
        b = np.arange(N, dtype=np.float64)
        A, B = np.meshgrid(a, b, indexing="ij")
        activations = np.sin(2 * np.pi * 3 * A / N)  # only depends on a

        result = analyze_neuron_trig_identity(activations, neuron_idx=0, layer=0)

        # Separability should be very high (rank 1 after removing mean)
        assert result.separability_score > 0.95
        # Dominant 2-D freq should be (3, 0) — periodic in a, constant in b
        assert result.dominant_freq_2d[0] == 3 or result.dominant_freq_2d[1] == 0

    def test_result_fields_populated(self):
        """All result fields should be populated with sensible values."""
        np.random.seed(0)
        N = 15
        activations = np.random.randn(N, N) * 2.0

        result = analyze_neuron_trig_identity(
            activations, neuron_idx=42, layer=23, max_freq=5
        )

        assert result.layer == 23
        assert result.neuron_idx == 42
        assert isinstance(result.dominant_freq_2d, tuple)
        assert len(result.dominant_freq_2d) == 2
        assert result.power_ratio_2d >= 0
        assert 0 <= result.separability_score <= 1.0
        assert result.trig_identity_r2 >= -0.1  # can be slightly negative due to floating point
        assert result.sum_frequency >= 1
        assert result.activation_std > 0
