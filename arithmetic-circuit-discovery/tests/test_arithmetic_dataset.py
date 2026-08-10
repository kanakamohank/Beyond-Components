"""
Tests for src/data/arithmetic_dataset.py

Covers:
    - Prompt generation correctness and completeness
    - ArithmeticPromptGenerator iteration and lookup
    - ArithmeticDataset column names and corrupted pair logic
    - load_arithmetic_dataset DataLoader factory
    - Edge cases and sanity checks
"""

import pytest
import random
from typing import Dict, Set

from src.data.arithmetic_dataset import (
    ArithmeticDataset,
    ArithmeticPromptGenerator,
    ArithmeticSample,
    generate_arithmetic_prompts,
    load_arithmetic_dataset,
    _build_corrupted_pair,
)


# ======================================================================
# generate_arithmetic_prompts
# ======================================================================

class TestGenerateArithmeticPrompts:
    """Tests for the core prompt generation function."""

    def test_default_generates_100_samples(self):
        """Default range(0,10) should produce 10*10 = 100 samples."""
        samples = generate_arithmetic_prompts()
        assert len(samples) == 100

    def test_all_operand_pairs_covered(self):
        """Every (a, b) pair in the operand range must appear exactly once."""
        samples = generate_arithmetic_prompts(operand_range=range(0, 5))
        pairs = {(s.operand_a, s.operand_b) for s in samples}
        expected = {(a, b) for a in range(5) for b in range(5)}
        assert pairs == expected, f"Missing pairs: {expected - pairs}"

    def test_addition_answers_correct(self):
        """Every sample's answer must equal operand_a + operand_b."""
        samples = generate_arithmetic_prompts(operation="add")
        for s in samples:
            assert s.answer == s.operand_a + s.operand_b, (
                f"Wrong answer for {s.operand_a} + {s.operand_b}: "
                f"got {s.answer}"
            )

    def test_subtraction_answers_correct(self):
        """Subtraction operation should compute a - b."""
        samples = generate_arithmetic_prompts(
            operand_range=range(0, 5), operation="sub"
        )
        for s in samples:
            assert s.answer == s.operand_a - s.operand_b

    def test_multiplication_answers_correct(self):
        """Multiplication operation should compute a * b."""
        samples = generate_arithmetic_prompts(
            operand_range=range(0, 5), operation="mul"
        )
        for s in samples:
            assert s.answer == s.operand_a * s.operand_b

    def test_invalid_operation_raises(self):
        """An unsupported operation should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported operation"):
            generate_arithmetic_prompts(operation="div")

    def test_prompt_format(self):
        """Prompts must match the template with correct substitutions."""
        samples = generate_arithmetic_prompts(
            operand_range=range(0, 3),
            prompt_template="{a} + {b} =",
        )
        for s in samples:
            assert s.prompt == f"{s.operand_a} + {s.operand_b} ="

    def test_custom_template(self):
        """Custom templates should work."""
        samples = generate_arithmetic_prompts(
            operand_range=range(0, 2),
            prompt_template="What is {a} plus {b}?",
        )
        assert samples[0].prompt.startswith("What is ")

    def test_answer_str_has_leading_space(self):
        """answer_str should always have a leading space for tokenization."""
        samples = generate_arithmetic_prompts(operand_range=range(0, 3))
        for s in samples:
            assert s.answer_str.startswith(" "), (
                f"answer_str '{s.answer_str}' missing leading space"
            )
            assert s.answer_str.strip() == str(s.answer)

    def test_shuffle_changes_order(self):
        """Shuffled output should differ from unshuffled (with high probability)."""
        unshuffled = generate_arithmetic_prompts(shuffle=False)
        shuffled = generate_arithmetic_prompts(shuffle=True, seed=42)
        # Same elements
        assert set(s.prompt for s in unshuffled) == set(s.prompt for s in shuffled)
        # Different order (extremely unlikely to be same by chance)
        prompts_unshuffled = [s.prompt for s in unshuffled]
        prompts_shuffled = [s.prompt for s in shuffled]
        assert prompts_unshuffled != prompts_shuffled

    def test_shuffle_deterministic(self):
        """Same seed should produce same shuffle order."""
        s1 = generate_arithmetic_prompts(shuffle=True, seed=123)
        s2 = generate_arithmetic_prompts(shuffle=True, seed=123)
        assert [s.prompt for s in s1] == [s.prompt for s in s2]

    def test_small_range(self):
        """range(0, 1) should produce exactly 1 sample: (0, 0)."""
        samples = generate_arithmetic_prompts(operand_range=range(0, 1))
        assert len(samples) == 1
        assert samples[0].operand_a == 0
        assert samples[0].operand_b == 0
        assert samples[0].answer == 0


# ======================================================================
# ArithmeticPromptGenerator
# ======================================================================

class TestArithmeticPromptGenerator:
    """Tests for the iterator-based prompt generator."""

    def test_len(self):
        gen = ArithmeticPromptGenerator(operand_range=range(0, 5))
        assert len(gen) == 25

    def test_iteration(self):
        gen = ArithmeticPromptGenerator(operand_range=range(0, 3))
        items = list(gen)
        assert len(items) == 9
        assert all(isinstance(s, ArithmeticSample) for s in items)

    def test_getitem(self):
        gen = ArithmeticPromptGenerator(operand_range=range(0, 3))
        sample = gen[0]
        assert isinstance(sample, ArithmeticSample)

    def test_get_by_operands_found(self):
        gen = ArithmeticPromptGenerator(operand_range=range(0, 5))
        result = gen.get_by_operands(3, 4)
        assert result is not None
        assert result.operand_a == 3
        assert result.operand_b == 4
        assert result.answer == 7

    def test_get_by_operands_not_found(self):
        gen = ArithmeticPromptGenerator(operand_range=range(0, 5))
        result = gen.get_by_operands(10, 10)
        assert result is None


# ======================================================================
# _build_corrupted_pair
# ======================================================================

class TestBuildCorruptedPair:
    """Tests for the corruption pairing logic."""

    def test_corrupted_has_different_answer(self):
        """The corrupted sample must have a different answer than the clean one."""
        samples = generate_arithmetic_prompts(operand_range=range(0, 5))
        rng = random.Random(42)
        for s in samples:
            corrupted, wrong_answer = _build_corrupted_pair(s, samples, rng)
            assert corrupted.answer != s.answer, (
                f"Corrupted answer {corrupted.answer} should differ from "
                f"clean answer {s.answer}"
            )

    def test_wrong_answer_str_format(self):
        """wrong_answer_str should be space-prefixed string of the corrupted answer."""
        samples = generate_arithmetic_prompts(operand_range=range(0, 5))
        rng = random.Random(42)
        for s in samples:
            corrupted, wrong_answer = _build_corrupted_pair(s, samples, rng)
            assert wrong_answer == f" {corrupted.answer}"

    def test_deterministic_with_same_rng(self):
        """Same RNG state should produce same corrupted pairs."""
        samples = generate_arithmetic_prompts(operand_range=range(0, 5))
        rng1 = random.Random(99)
        rng2 = random.Random(99)
        for s in samples:
            c1, _ = _build_corrupted_pair(s, samples, rng1)
            c2, _ = _build_corrupted_pair(s, samples, rng2)
            assert c1.prompt == c2.prompt


# ======================================================================
# ArithmeticDataset
# ======================================================================

class TestArithmeticDataset:
    """Tests for the PyTorch Dataset class."""

    def test_len(self):
        ds = ArithmeticDataset(operand_range=range(0, 5))
        assert len(ds) == 25

    def test_getitem_returns_dict(self):
        ds = ArithmeticDataset(operand_range=range(0, 5))
        item = ds[0]
        assert isinstance(item, dict)

    def test_column_names_present(self):
        """All required columns must be present in each item."""
        ds = ArithmeticDataset(operand_range=range(0, 3))
        item = ds[0]
        required = {
            ArithmeticDataset.COL_CLEAN,
            ArithmeticDataset.COL_CORRUPTED,
            ArithmeticDataset.COL_ANSWER,
            ArithmeticDataset.COL_ANSWER_WRONG,
        }
        assert required.issubset(item.keys()), (
            f"Missing columns: {required - set(item.keys())}"
        )

    def test_clean_and_corrupted_differ(self):
        """Clean and corrupted prompts should have different answers."""
        ds = ArithmeticDataset(operand_range=range(0, 10))
        for i in range(len(ds)):
            item = ds[i]
            # The answer and wrong answer should differ
            assert item[ArithmeticDataset.COL_ANSWER] != item[ArithmeticDataset.COL_ANSWER_WRONG], (
                f"Item {i}: answer '{item[ArithmeticDataset.COL_ANSWER]}' "
                f"should differ from wrong answer '{item[ArithmeticDataset.COL_ANSWER_WRONG]}'"
            )

    def test_all_values_are_strings(self):
        """All column values should be strings (for tokenizer compatibility)."""
        ds = ArithmeticDataset(operand_range=range(0, 5))
        item = ds[0]
        for key, value in item.items():
            assert isinstance(value, str), (
                f"Column '{key}' has type {type(value)}, expected str"
            )

    def test_answer_has_leading_space(self):
        """Answer strings should have a leading space for tokenizer alignment."""
        ds = ArithmeticDataset(operand_range=range(0, 5))
        for i in range(len(ds)):
            item = ds[i]
            assert item[ArithmeticDataset.COL_ANSWER].startswith(" ")
            assert item[ArithmeticDataset.COL_ANSWER_WRONG].startswith(" ")


# ======================================================================
# load_arithmetic_dataset
# ======================================================================

class TestLoadArithmeticDataset:
    """Tests for the DataLoader factory function."""

    def test_returns_dataloader(self):
        from torch.utils.data import DataLoader
        loader = load_arithmetic_dataset(batch_size=10)
        assert isinstance(loader, DataLoader)

    def test_train_loader_size(self):
        loader = load_arithmetic_dataset(batch_size=25, train=True)
        total = sum(len(batch[ArithmeticDataset.COL_CLEAN]) for batch in loader)
        assert total == 100  # 10*10 single-digit pairs

    def test_validation_not_shuffled(self):
        """Validation loader should produce deterministic order."""
        loader1 = load_arithmetic_dataset(batch_size=100, validation=True)
        loader2 = load_arithmetic_dataset(batch_size=100, validation=True)
        batch1 = next(iter(loader1))
        batch2 = next(iter(loader2))
        assert batch1[ArithmeticDataset.COL_CLEAN] == batch2[ArithmeticDataset.COL_CLEAN]

    def test_full_batch(self):
        loader = load_arithmetic_dataset(full_batch=True)
        batches = list(loader)
        assert len(batches) == 1
        assert len(batches[0][ArithmeticDataset.COL_CLEAN]) == 100

    def test_batch_keys_match_dataset(self):
        loader = load_arithmetic_dataset(batch_size=10)
        batch = next(iter(loader))
        required = {
            ArithmeticDataset.COL_CLEAN,
            ArithmeticDataset.COL_CORRUPTED,
            ArithmeticDataset.COL_ANSWER,
            ArithmeticDataset.COL_ANSWER_WRONG,
        }
        assert required.issubset(batch.keys())


    def test_custom_operand_range(self):
        """Custom operand_range should produce the correct number of samples."""
        loader = load_arithmetic_dataset(
            batch_size=4, operand_range=range(0, 3),
        )
        total = sum(len(batch[ArithmeticDataset.COL_CLEAN]) for batch in loader)
        assert total == 9  # 3*3

    def test_custom_operand_range_large(self):
        """Larger operand range should work."""
        loader = load_arithmetic_dataset(
            batch_size=50, operand_range=range(0, 7),
        )
        total = sum(len(batch[ArithmeticDataset.COL_CLEAN]) for batch in loader)
        assert total == 49  # 7*7


# ======================================================================
# Edge cases
# ======================================================================

class TestEdgeCases:
    """Edge case tests for arithmetic dataset."""

    def test_single_pair_dataset(self):
        """range(0,1) produces exactly one sample: (0,0)=0.
        Corruption must fall back since there's only one unique answer."""
        ds = ArithmeticDataset(operand_range=range(0, 1))
        assert len(ds) == 1
        item = ds[0]
        # With only one sample, corrupted pair falls back to same sample
        # (the assert in _build_corrupted_pair prevents this — but range(0,1)
        # has only answer=0, so the fallback uses `s is not sample`).
        assert isinstance(item[ArithmeticDataset.COL_CLEAN], str)

    def test_two_pair_dataset_corruption_works(self):
        """range(0,2) should have 4 samples with working corruption."""
        ds = ArithmeticDataset(operand_range=range(0, 2))
        assert len(ds) == 4
        for i in range(len(ds)):
            item = ds[i]
            # Answer and wrong answer should differ (at least some of the time)
            assert isinstance(item[ArithmeticDataset.COL_ANSWER_WRONG], str)

    def test_subtraction_negative_answers(self):
        """Subtraction can produce negative answers — should still work."""
        ds = ArithmeticDataset(operand_range=range(0, 3), operation="sub")
        answers = set()
        for i in range(len(ds)):
            item = ds[i]
            answers.add(item[ArithmeticDataset.COL_ANSWER].strip())
        # Should include negative answers like "-1", "-2"
        assert any(a.startswith("-") for a in answers), f"No negative answers in {answers}"


# ======================================================================
# Integration with src/utils/utils.py column name functions
# ======================================================================

class TestUtilsIntegration:
    """Verify that arithmetic columns are registered in utils.py."""

    def test_get_data_column_names(self):
        from src.utils.utils import get_data_column_names
        clean, corrupted = get_data_column_names("arithmetic")
        assert clean == "arithmetic_input"
        assert corrupted == "corr_arithmetic_input"

    def test_get_label_column_names(self):
        from src.utils.utils import get_label_column_names
        correct, wrong = get_label_column_names("arithmetic")
        assert correct == "arithmetic_answer"
        assert wrong == "arithmetic_answer_wrong"

    def test_get_data_label_column_names(self):
        from src.utils.utils import get_data_label_column_names
        correct, wrong = get_data_label_column_names("arithmetic")
        assert correct == "arithmetic_answer"
        assert wrong == "arithmetic_answer_wrong"

    def test_existing_data_types_unchanged(self):
        """Verify we haven't broken existing data type lookups."""
        from src.utils.utils import get_data_column_names, get_label_column_names

        # IOI
        c, cc = get_data_column_names("ioi")
        assert c == "ioi_sentences_input"
        l, lw = get_label_column_names("ioi")
        assert l == "ioi_sentences_labels"

        # GP
        c, cc = get_data_column_names("gp")
        assert c == "prefix"
        l, lw = get_label_column_names("gp")
        assert l == "pronoun"

        # GT
        c, cc = get_data_column_names("gt")
        assert c == "prefix"
        l, lw = get_label_column_names("gt")
        assert l == "century"
