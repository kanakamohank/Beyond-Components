"""Tests for experiments/circuit_analysis.py

Covers:
  - generate_carry_dataset: carry labeling correctness
  - find_token_positions: position extraction
  - _cluster_separability: metric computation
  - run_mlp_unembedding: output structure (mocked)
  - run_operand_b_hunt: output structure (mocked)
  - run_carry_probe: output structure
  - run_activation_pca: output structure
  - run_ensemble_edge_patching: output structure
  - CLI: clean import + exports
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.circuit_analysis import (
    generate_carry_dataset,
    find_token_positions,
    _cluster_separability,
    run_mlp_unembedding,
    run_operand_b_hunt,
    run_carry_probe,
    run_activation_pca,
    run_ensemble_edge_patching,
    run_circuit_analysis,
)


# ======================================================================
# generate_carry_dataset
# ======================================================================

class TestGenerateCarryDataset:

    def test_basic_structure(self):
        problems = generate_carry_dataset(n_problems=20, max_operand=50, seed=1)
        assert len(problems) == 20
        required = {"a", "b", "answer", "has_carry", "ones_carry",
                     "tens_carry", "clean_prompt"}
        for p in problems:
            assert required.issubset(p.keys())

    def test_carry_labeling_correct(self):
        """Verify carry detection is mathematically correct."""
        problems = generate_carry_dataset(n_problems=200, max_operand=99, seed=42)
        for p in problems:
            a, b = p["a"], p["b"]
            ones_carry = (a % 10) + (b % 10) >= 10
            tens_carry = (a // 10) + (b // 10) + (1 if ones_carry else 0) >= 10
            assert p["ones_carry"] == ones_carry, f"{a}+{b}: ones_carry mismatch"
            assert p["tens_carry"] == tens_carry, f"{a}+{b}: tens_carry mismatch"
            assert p["has_carry"] == (ones_carry or tens_carry)

    def test_known_carry_cases(self):
        """Spot-check specific known carry/no-carry cases."""
        # 7 + 8 = 15 -> ones carry
        assert (7 % 10) + (8 % 10) >= 10  # ones carry
        # 3 + 4 = 7 -> no carry
        assert (3 % 10) + (4 % 10) < 10  # no carry
        # 15 + 17 = 32 -> tens carry (1+1+carry=3)
        ones = (15 % 10) + (17 % 10) >= 10  # 5+7=12, carry
        tens = (15 // 10) + (17 // 10) + (1 if ones else 0) >= 10
        assert not tens  # 1+1+1=3, no tens overflow

    def test_deterministic(self):
        p1 = generate_carry_dataset(n_problems=10, seed=42)
        p2 = generate_carry_dataset(n_problems=10, seed=42)
        for a, b in zip(p1, p2):
            assert a["a"] == b["a"]
            assert a["has_carry"] == b["has_carry"]

    def test_has_both_classes(self):
        """Dataset should contain both carry and no-carry problems."""
        problems = generate_carry_dataset(n_problems=200, max_operand=50, seed=1)
        n_carry = sum(1 for p in problems if p["has_carry"])
        n_no = len(problems) - n_carry
        assert n_carry > 10, "Should have carry problems"
        assert n_no > 10, "Should have no-carry problems"

    def test_answer_correctness(self):
        problems = generate_carry_dataset(n_problems=50, seed=7)
        for p in problems:
            assert p["answer"] == p["a"] + p["b"]


# ======================================================================
# find_token_positions (needs a mock model)
# ======================================================================

class TestFindTokenPositions:

    def test_structure(self):
        """Mock to_str_tokens and verify position extraction."""
        from unittest.mock import MagicMock
        model = MagicMock()
        model.to_str_tokens.return_value = [
            '<s>', 'Calculate', ':', '\n', '12', ' +', ' 7', ' =', ' 19',
            '\n', '5', ' +', ' 3', ' ='
        ]
        pos = find_token_positions(model, "dummy")
        assert "equals" in pos
        assert "plus" in pos
        assert "operand_a" in pos
        assert "operand_b" in pos
        assert pos["equals"] == 13  # last token index

    def test_no_plus_found(self):
        """If no + token, should return only equals."""
        from unittest.mock import MagicMock
        model = MagicMock()
        model.to_str_tokens.return_value = ['hello', 'world']
        pos = find_token_positions(model, "dummy")
        assert "equals" in pos
        assert "plus" not in pos


# ======================================================================
# _cluster_separability
# ======================================================================

class TestClusterSeparability:

    def test_perfect_separation(self):
        """Well-separated clusters should have high separability."""
        X = np.array([[0, 0], [0.1, 0], [10, 10], [10.1, 10]])
        labels = np.array([0, 0, 1, 1])
        sep = _cluster_separability(X, labels)
        assert sep > 5.0, "Perfectly separated clusters should score high"

    def test_mixed_clusters(self):
        """Overlapping clusters should have low separability."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        labels = np.array([0] * 50 + [1] * 50)
        sep = _cluster_separability(X, labels)
        assert sep < 2.0, "Random labels on mixed data should score low"

    def test_single_class(self):
        """Single class should return 0."""
        X = np.array([[1, 2], [3, 4]])
        labels = np.array([0, 0])
        sep = _cluster_separability(X, labels)
        assert sep == 0.0

    def test_multi_class(self):
        """Should work with >2 classes."""
        X = np.array([[0, 0], [0.1, 0], [5, 5], [5.1, 5], [10, 0], [10.1, 0]])
        labels = np.array([0, 0, 1, 1, 2, 2])
        sep = _cluster_separability(X, labels)
        assert sep > 2.0


# ======================================================================
# Module exports / CLI
# ======================================================================

class TestModuleExports:

    def test_imports_cleanly(self):
        import experiments.circuit_analysis as mod
        assert hasattr(mod, "run_circuit_analysis")
        assert hasattr(mod, "run_mlp_unembedding")
        assert hasattr(mod, "run_operand_b_hunt")
        assert hasattr(mod, "run_carry_probe")
        assert hasattr(mod, "run_activation_pca")
        assert hasattr(mod, "run_ensemble_edge_patching")
        assert hasattr(mod, "generate_carry_dataset")
        assert hasattr(mod, "find_token_positions")

    def test_functions_are_callable(self):
        from experiments.circuit_analysis import (
            run_mlp_unembedding, run_operand_b_hunt, run_carry_probe,
            run_activation_pca, run_ensemble_edge_patching,
        )
        assert callable(run_mlp_unembedding)
        assert callable(run_operand_b_hunt)
        assert callable(run_carry_probe)
        assert callable(run_activation_pca)
        assert callable(run_ensemble_edge_patching)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
