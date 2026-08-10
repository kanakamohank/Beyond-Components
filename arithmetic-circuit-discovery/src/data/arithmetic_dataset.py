"""
Arithmetic Dataset: Data generation for arithmetic circuit discovery.

Generates single-digit addition prompts with clean/corrupted pairs for use in
the Beyond Components SVD mask-learning pipeline. Designed to be model-agnostic:
all outputs are plain text — tokenization happens downstream.

Prompt format examples:
    Clean:     "3 + 7 ="   (answer: " 10")
    Corrupted: "5 + 2 ="   (answer: " 7")

Supports three usage modes:
    1. **Fourier discovery** (Phase 1): Only clean prompts are needed.
       Use ``generate_arithmetic_prompts`` or ``ArithmeticPromptGenerator``.
    2. **Mask learning** (Phase 3): Needs clean/corrupted pairs via
       ``ArithmeticDataset`` + ``load_arithmetic_dataset``.
    3. **Standalone iteration**: ``ArithmeticPromptGenerator`` yields
       ``(prompt, operand_a, operand_b, answer)`` tuples for analysis scripts.
"""

import itertools
import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core prompt generation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArithmeticSample:
    """A single arithmetic prompt with metadata.

    Attributes:
        prompt: The input text (e.g. ``"3 + 7 ="``).
        operand_a: First operand.
        operand_b: Second operand.
        answer: Correct answer as integer.
        answer_str: Answer as string with leading space (e.g. ``" 10"``).
    """
    prompt: str
    operand_a: int
    operand_b: int
    answer: int
    answer_str: str


def generate_arithmetic_prompts(
    operand_range: range = range(0, 10),
    operation: str = "add",
    prompt_template: str = "{a} + {b} =",
    shuffle: bool = False,
    seed: int = 42,
) -> List[ArithmeticSample]:
    """Generate all arithmetic prompts for the given operand range.

    Args:
        operand_range: Range of operands (default: 0-9 for single-digit).
        operation: One of ``"add"``, ``"sub"``, ``"mul"``.
        prompt_template: Format string with ``{a}`` and ``{b}`` placeholders.
        shuffle: Whether to shuffle the output list.
        seed: Random seed for reproducible shuffling.

    Returns:
        List of ``ArithmeticSample`` instances.

    Raises:
        ValueError: If *operation* is not supported.
    """
    _OPERATIONS = {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
    }
    if operation not in _OPERATIONS:
        raise ValueError(
            f"Unsupported operation '{operation}'. Choose from: {list(_OPERATIONS)}"
        )
    op_fn = _OPERATIONS[operation]

    samples: List[ArithmeticSample] = []
    for a, b in itertools.product(operand_range, repeat=2):
        answer = op_fn(a, b)
        prompt = prompt_template.format(a=a, b=b)
        samples.append(ArithmeticSample(
            prompt=prompt,
            operand_a=a,
            operand_b=b,
            answer=answer,
            answer_str=f" {answer}",
        ))

    # Sanity check: expected count
    expected = len(operand_range) ** 2
    assert len(samples) == expected, (
        f"Generated {len(samples)} samples, expected {expected}"
    )

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(samples)

    return samples


class ArithmeticPromptGenerator:
    """Iterator-based prompt generator for Fourier discovery analysis.

    Yields ``ArithmeticSample`` instances one at a time, which is
    useful for streaming analysis where we don't need a full DataLoader.

    Example::

        gen = ArithmeticPromptGenerator(operand_range=range(0, 10))
        for sample in gen:
            # sample.prompt = "3 + 7 ="
            # sample.answer = 10
            ...
    """

    def __init__(
        self,
        operand_range: range = range(0, 10),
        operation: str = "add",
        prompt_template: str = "{a} + {b} =",
        shuffle: bool = False,
        seed: int = 42,
    ):
        self.samples = generate_arithmetic_prompts(
            operand_range=operand_range,
            operation=operation,
            prompt_template=prompt_template,
            shuffle=shuffle,
            seed=seed,
        )
        logger.info(
            f"ArithmeticPromptGenerator: {len(self.samples)} prompts "
            f"({operation}, range {operand_range.start}-{operand_range.stop - 1})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def __getitem__(self, idx) -> ArithmeticSample:
        return self.samples[idx]

    def get_by_operands(self, a: int, b: int) -> Optional[ArithmeticSample]:
        """Look up a specific prompt by operand values.

        Args:
            a: First operand.
            b: Second operand.

        Returns:
            The matching ``ArithmeticSample``, or ``None`` if not found.
        """
        for s in self.samples:
            if s.operand_a == a and s.operand_b == b:
                return s
        return None


# ---------------------------------------------------------------------------
# PyTorch Dataset for mask-learning pipeline (Phase 3)
# ---------------------------------------------------------------------------

def _build_corrupted_pair(
    sample: ArithmeticSample,
    all_samples: List[ArithmeticSample],
    rng: random.Random,
) -> Tuple[ArithmeticSample, str]:
    """Pick a corrupted counterpart that has a *different* answer.

    The corrupted prompt keeps the same format but uses different operands
    whose sum differs from the clean answer. This mirrors the IOI corruption
    strategy: same structure, different content.

    .. note:: Phase 3 TODO — Add ``corruption_type`` parameter to distinguish:
       - **A-corruption**: change operand a only (a' + b =)
       - **B-corruption**: change operand b only (a + b' =)
       This is needed for cross-referencing A-vs-B corruption effects in the
       Phase 3 analysis table.  Current implementation picks any different-answer
       pair, which is sufficient for Phase 1-2 mask learning.

    Args:
        sample: The clean sample.
        all_samples: Pool of all samples to draw from.
        rng: Random number generator for reproducibility.

    Returns:
        Tuple of (corrupted_sample, wrong_answer_str).
    """
    candidates = [s for s in all_samples if s.answer != sample.answer]
    if not candidates:
        # Fallback: use any different sample (same answer but different operands)
        candidates = [s for s in all_samples if s is not sample]
    if not candidates:
        # Degenerate case: only 1 sample total (e.g. range(0,1) → only (0,0)=0).
        # Use itself as corruption — caller should be aware this is a no-op pair.
        candidates = [sample]
    corrupted = rng.choice(candidates)
    wrong_answer_str = f" {corrupted.answer}"
    return corrupted, wrong_answer_str


class ArithmeticDataset(Dataset):
    """PyTorch Dataset providing clean/corrupted arithmetic pairs.

    Compatible with the existing Beyond Components training pipeline
    (``experiments/train.py``). Each item is a dict with keys matching
    the column-name convention used by ``get_data_column_names("arithmetic")``.

    Column mapping:
        - ``arithmetic_input``: clean prompt text
        - ``corr_arithmetic_input``: corrupted prompt text
        - ``arithmetic_answer``: correct answer string (space-prefixed)
        - ``arithmetic_answer_wrong``: wrong answer string (space-prefixed)

    Args:
        operand_range: Range of operands.
        operation: Arithmetic operation.
        prompt_template: Prompt format string.
        seed: Seed for corruption randomness.
    """

    # Column names — must stay in sync with src/utils/utils.py additions
    COL_CLEAN = "arithmetic_input"
    COL_CORRUPTED = "corr_arithmetic_input"
    COL_ANSWER = "arithmetic_answer"
    COL_ANSWER_WRONG = "arithmetic_answer_wrong"

    def __init__(
        self,
        operand_range: range = range(0, 10),
        operation: str = "add",
        prompt_template: str = "{a} + {b} =",
        seed: int = 42,
    ):
        self.samples = generate_arithmetic_prompts(
            operand_range=operand_range,
            operation=operation,
            prompt_template=prompt_template,
            shuffle=True,
            seed=seed,
        )
        self._rng = random.Random(seed)

        # Pre-build corrupted pairs for determinism
        self._corrupted: List[Tuple[ArithmeticSample, str]] = []
        for s in self.samples:
            corr, wrong = _build_corrupted_pair(s, self.samples, self._rng)
            self._corrupted.append((corr, wrong))

        logger.info(
            f"ArithmeticDataset: {len(self)} samples "
            f"({operation}, range {operand_range.start}-{operand_range.stop - 1})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        clean = self.samples[idx]
        corrupted, wrong_answer = self._corrupted[idx]

        return {
            self.COL_CLEAN: clean.prompt,
            self.COL_CORRUPTED: corrupted.prompt,
            self.COL_ANSWER: clean.answer_str,
            self.COL_ANSWER_WRONG: wrong_answer,
        }


# ---------------------------------------------------------------------------
# DataLoader factory (mirrors load_ioi_dataset interface)
# ---------------------------------------------------------------------------

def load_arithmetic_dataset(
    batch_size: int = 64,
    full_batch: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
    validation: bool = False,
    train: bool = False,
    data_dir: Optional[str] = None,
    operand_range: Optional[range] = None,
    operation: str = "add",
    prompt_template: str = "{a} + {b} =",
    seed: int = 42,
) -> DataLoader:
    """Create a DataLoader for arithmetic data.

    Follows the same signature convention as ``load_ioi_dataset`` so it can
    be used as a drop-in replacement in ``train.py`` via
    ``getattr(data_loader, f"load_{data_type}_dataset")``.

    For arithmetic data the train/validation/test split is controlled by
    operand ranges:
        - **train**: 0-9 (all 100 single-digit pairs, shuffled)
        - **validation**: 0-9 (same range, different shuffle seed)
        - **test**: 0-9 (no shuffle, deterministic order)

    Args:
        batch_size: Batch size for the DataLoader.
        full_batch: If ``True``, sets batch_size = len(dataset).
        shuffle: Whether to shuffle the DataLoader.
        num_workers: DataLoader workers. Default 0 for MPS compatibility.
        validation: Load validation split.
        train: Load training split.
        data_dir: Unused (kept for API compatibility with other loaders).
        operand_range: Override the default range(0, 10).
        operation: Arithmetic operation type.
        prompt_template: Prompt format string.
        seed: Base random seed.

    Returns:
        A PyTorch ``DataLoader``.
    """
    if operand_range is None:
        operand_range = range(0, 10)

    # NOTE: All splits use the SAME 100 prompts with different shuffle seeds.
    # This is intentional for mask learning — we are discovering which existing
    # model directions matter, not learning new capabilities.  The "splits"
    # only differ in ordering to avoid overfitting to batch order.
    if train:
        split_seed = seed
    elif validation:
        split_seed = seed + 1000
        shuffle = False
    else:
        # Test split: deterministic
        split_seed = seed + 2000
        shuffle = False

    dataset = ArithmeticDataset(
        operand_range=operand_range,
        operation=operation,
        prompt_template=prompt_template,
        seed=split_seed,
    )

    if full_batch:
        batch_size = len(dataset)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    logger.info(
        f"Arithmetic DataLoader: {len(dataset)} samples, "
        f"batch_size={batch_size}, shuffle={shuffle}"
    )
    return loader
