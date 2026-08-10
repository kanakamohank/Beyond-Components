"""Tests for experiments/causal_validation.py core functions.

Tests cover:
  - generate_corruption_pairs: pair generation, answer difference, corruption type
  - load_phase3_results: JSON parsing, mask filtering
  - aggregate_pair_results: mean/median/trimmed, degenerate filtering
"""

import json
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.causal_validation import (
    PairResult,
    aggregate_pair_results,
    generate_corruption_pairs,
    load_phase3_results,
)


# ---------------------------------------------------------------------------
# generate_corruption_pairs
# ---------------------------------------------------------------------------

class TestGenerateCorruptionPairs:
    """Tests for clean/corrupted pair generation."""

    def test_a_corruption_changes_a_only(self):
        """A-corruption should change operand a but keep b the same."""
        pairs = generate_corruption_pairs(range(0, 10), "A", n_pairs=20, seed=1)
        assert len(pairs) == 20
        for clean, corrupt in pairs:
            assert clean.operand_b == corrupt.operand_b, "b must stay the same"
            assert clean.operand_a != corrupt.operand_a, "a must change"
            assert clean.answer != corrupt.answer, "answers must differ"

    def test_b_corruption_changes_b_only(self):
        """B-corruption should change operand b but keep a the same."""
        pairs = generate_corruption_pairs(range(0, 10), "B", n_pairs=20, seed=2)
        assert len(pairs) == 20
        for clean, corrupt in pairs:
            assert clean.operand_a == corrupt.operand_a, "a must stay the same"
            assert clean.operand_b != corrupt.operand_b, "b must change"
            assert clean.answer != corrupt.answer, "answers must differ"

    def test_deterministic_with_seed(self):
        """Same seed should produce identical pairs."""
        p1 = generate_corruption_pairs(range(0, 10), "A", n_pairs=10, seed=42)
        p2 = generate_corruption_pairs(range(0, 10), "A", n_pairs=10, seed=42)
        for (c1, x1), (c2, x2) in zip(p1, p2):
            assert c1.prompt == c2.prompt
            assert x1.prompt == x2.prompt

    def test_different_seeds_differ(self):
        """Different seeds should (almost certainly) produce different pairs."""
        p1 = generate_corruption_pairs(range(0, 10), "A", n_pairs=10, seed=1)
        p2 = generate_corruption_pairs(range(0, 10), "A", n_pairs=10, seed=99)
        prompts1 = [c.prompt for c, _ in p1]
        prompts2 = [c.prompt for c, _ in p2]
        assert prompts1 != prompts2

    def test_small_range(self):
        """Should work with very small operand ranges."""
        pairs = generate_corruption_pairs(range(0, 3), "A", n_pairs=5, seed=7)
        assert len(pairs) > 0
        for clean, corrupt in pairs:
            assert clean.answer != corrupt.answer

    def test_n_pairs_limit(self):
        """Should respect the n_pairs limit."""
        pairs = generate_corruption_pairs(range(0, 10), "A", n_pairs=5, seed=1)
        assert len(pairs) == 5

    def test_prompts_are_valid(self):
        """Prompts should follow the arithmetic template."""
        pairs = generate_corruption_pairs(range(0, 10), "A", n_pairs=5, seed=1)
        for clean, corrupt in pairs:
            assert "+" in clean.prompt
            assert "=" in clean.prompt
            assert "+" in corrupt.prompt
            assert "=" in corrupt.prompt


# ---------------------------------------------------------------------------
# load_phase3_results
# ---------------------------------------------------------------------------

class TestLoadPhase3Results:
    """Tests for Phase 3 JSON loader."""

    def _make_json(self, results):
        """Write a temporary Phase 3 JSON file and return its path."""
        data = {"results": results, "timestamp": "test", "config": {}, "summary": {}}
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        return path

    def test_filters_by_mask_threshold(self):
        """Only directions with mask_value > 0.3 should be returned."""
        path = self._make_json([{
            "layer": 5, "head": 3, "component": "OV",
            "avg_mask": 0.5, "n_active_directions": 2, "n_periodic_directions": 2,
            "directions": [
                {"direction_idx": 0, "singular_value": 1.0, "mask_value": 0.8,
                 "effective_strength": 0.8, "dominant_frequency": 1, "power_ratio": 5.0},
                {"direction_idx": 1, "singular_value": 1.0, "mask_value": 0.1,
                 "effective_strength": 0.1, "dominant_frequency": 1, "power_ratio": 2.0},
            ],
        }])
        try:
            survivors = load_phase3_results(path)
            assert len(survivors) == 1
            assert survivors[0]["direction_idx"] == 0
            assert survivors[0]["layer"] == 5
            assert survivors[0]["head"] == 3
            assert survivors[0]["component"] == "OV"
        finally:
            os.unlink(path)

    def test_flattens_nested_structure(self):
        """Should flatten results[i].directions[j] into a flat list."""
        path = self._make_json([
            {
                "layer": 5, "head": 3, "component": "OV",
                "avg_mask": 0.5, "n_active_directions": 1, "n_periodic_directions": 1,
                "directions": [
                    {"direction_idx": 0, "mask_value": 0.9, "singular_value": 2.0,
                     "effective_strength": 1.8, "dominant_frequency": 1, "power_ratio": 10.0},
                ],
            },
            {
                "layer": 23, "head": None, "component": "MLP_out",
                "avg_mask": 0.5, "n_active_directions": 1, "n_periodic_directions": 1,
                "directions": [
                    {"direction_idx": 0, "mask_value": 0.95, "singular_value": 16.0,
                     "effective_strength": 15.2, "dominant_frequency": 1, "power_ratio": 20.0},
                ],
            },
        ])
        try:
            survivors = load_phase3_results(path)
            assert len(survivors) == 2
            layers = {s["layer"] for s in survivors}
            assert layers == {5, 23}
        finally:
            os.unlink(path)

    def test_empty_results(self):
        """Should return empty list for empty results."""
        path = self._make_json([])
        try:
            survivors = load_phase3_results(path)
            assert survivors == []
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# aggregate_pair_results
# ---------------------------------------------------------------------------

class TestAggregatePairResults:
    """Tests for aggregate statistics computation."""

    def _make_pair_result(self, logit_clean, logit_corrupt, logit_patched):
        """Create a PairResult with given logit values."""
        denom = logit_clean - logit_corrupt
        recovery = (logit_patched - logit_corrupt) / denom if abs(denom) > 1e-6 else 0.0
        return PairResult(
            clean_a=1, clean_b=2, corrupt_a=3, corrupt_b=2,
            clean_answer=3, corrupt_answer=5,
            logit_clean=logit_clean, logit_corrupt=logit_corrupt,
            logit_patched=logit_patched, logit_recovery=recovery,
            clean_rank=0, corrupt_rank=10, patched_rank=5,
        )

    def test_empty_input(self):
        agg = aggregate_pair_results([])
        assert agg["n_pairs"] == 0
        assert agg["mean_logit_recovery"] == 0.0

    def test_perfect_recovery(self):
        """All pairs fully recovered → mean/median = 1.0."""
        results = [self._make_pair_result(10.0, 2.0, 10.0) for _ in range(5)]
        agg = aggregate_pair_results(results)
        assert abs(agg["mean_logit_recovery"] - 1.0) < 1e-6
        assert abs(agg["median_logit_recovery"] - 1.0) < 1e-6

    def test_no_recovery(self):
        """No recovery → mean/median = 0.0."""
        results = [self._make_pair_result(10.0, 2.0, 2.0) for _ in range(5)]
        agg = aggregate_pair_results(results)
        assert abs(agg["mean_logit_recovery"]) < 1e-6
        assert abs(agg["median_logit_recovery"]) < 1e-6

    def test_partial_recovery(self):
        """50% recovery."""
        results = [self._make_pair_result(10.0, 2.0, 6.0) for _ in range(5)]
        agg = aggregate_pair_results(results)
        assert abs(agg["mean_logit_recovery"] - 0.5) < 1e-6

    def test_degenerate_filtering(self):
        """Pairs with tiny logit diff should be filtered."""
        good = [self._make_pair_result(10.0, 2.0, 6.0) for _ in range(5)]
        # This pair has logit_clean ≈ logit_corrupt → degenerate
        degen = self._make_pair_result(2.01, 2.0, 2.005)
        all_results = good + [degen]
        agg = aggregate_pair_results(all_results, min_logit_diff=0.1)
        # Should filter the degenerate pair
        assert agg["n_pairs_filtered"] == 5
        assert agg["n_pairs"] == 6
        assert abs(agg["mean_logit_recovery"] - 0.5) < 1e-6

    def test_rank_improvement(self):
        """Rank improvement = corrupt_rank - patched_rank."""
        results = [self._make_pair_result(10.0, 2.0, 6.0) for _ in range(3)]
        agg = aggregate_pair_results(results)
        # corrupt_rank=10, patched_rank=5 → improvement = 5
        assert abs(agg["mean_rank_improvement"] - 5.0) < 1e-6

    def test_trimmed_mean(self):
        """Trimmed mean should reduce outlier influence."""
        # 8 normal results + 2 extreme outliers
        results = [self._make_pair_result(10.0, 2.0, 6.0) for _ in range(8)]
        results.append(self._make_pair_result(10.0, 2.0, 100.0))  # extreme positive
        results.append(self._make_pair_result(10.0, 2.0, -80.0))  # extreme negative
        agg = aggregate_pair_results(results)
        # Trimmed mean should be closer to 0.5 than the raw mean
        assert abs(agg["trimmed_mean_recovery"] - 0.5) < abs(agg["mean_logit_recovery"] - 0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
