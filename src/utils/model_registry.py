"""
Model Registry: Multi-model compatibility layer for arithmetic circuit discovery.

Provides a unified interface for working with different transformer architectures
(Pythia, Gemma, Phi-3, GPT-2) via TransformerLens. Handles model-specific
tokenization differences, especially for number tokens.

Supported model families:
    - Pythia (1.4B, 2.8B, 6.9B) — EleutherAI
    - Gemma (2B, 7B) — Google
    - Phi-3 (Mini) — Microsoft
    - GPT-2 (Small, Medium) — OpenAI
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from transformer_lens import HookedTransformer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    """Immutable specification for a supported model.

    Attributes:
        transformer_lens_name: Name accepted by ``HookedTransformer.from_pretrained``.
        family: Short family identifier (e.g. ``"pythia"``, ``"gemma"``).
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads per layer.
        d_model: Hidden dimension.
        d_mlp: MLP intermediate dimension.
        has_gated_mlp: Whether the MLP uses a gated architecture (SwiGLU / GeGLU).
        context_length: Maximum context length.
        single_digit_tokens: Whether digits 0-9 are each a single token.
        notes: Any useful caveats.
    """
    transformer_lens_name: str
    family: str
    n_layers: int
    n_heads: int
    d_model: int
    d_mlp: int
    has_gated_mlp: bool = False
    context_length: int = 2048
    single_digit_tokens: bool = True
    notes: str = ""


# Canonical registry — add new models here.
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    # ---- Pythia family (EleutherAI) ----
    "pythia-1.4b": ModelSpec(
        transformer_lens_name="pythia-1.4b",
        family="pythia",
        n_layers=24, n_heads=16, d_model=2048, d_mlp=8192,
        context_length=2048,
        notes="Good single-token arithmetic model on M1 Max.",
    ),
    "pythia-2.8b": ModelSpec(
        transformer_lens_name="pythia-2.8b",
        family="pythia",
        n_layers=32, n_heads=32, d_model=2560, d_mlp=10240,
        context_length=2048,
    ),
    "pythia-6.9b": ModelSpec(
        transformer_lens_name="pythia-6.9b",
        family="pythia",
        n_layers=32, n_heads=32, d_model=4096, d_mlp=16384,
        context_length=2048,
        notes="Cloud GPU recommended.",
    ),

    # ---- Gemma family (Google) ----
    "gemma-2b": ModelSpec(
        transformer_lens_name="gemma-2b",
        family="gemma",
        n_layers=18, n_heads=8, d_model=2048, d_mlp=16384,
        has_gated_mlp=True,
        context_length=8192,
        notes="Grouped-query attention, SentencePiece tokenizer.",
    ),

    # ---- Phi-3 family (Microsoft) ----
    "phi-3-mini": ModelSpec(
        transformer_lens_name="microsoft/Phi-3-mini-4k-instruct",
        family="phi3",
        n_layers=32, n_heads=32, d_model=3072, d_mlp=8192,
        has_gated_mlp=True,
        context_length=4096,
        notes="SuRoPE, grouped-query attention. TransformerLens support may vary.",
    ),

    # ---- GPT-2 family (OpenAI) — already validated in Phase 0 ----
    "gpt2-small": ModelSpec(
        transformer_lens_name="gpt2-small",
        family="gpt2",
        n_layers=12, n_heads=12, d_model=768, d_mlp=3072,
        context_length=1024,
    ),
    "gpt2-medium": ModelSpec(
        transformer_lens_name="gpt2-medium",
        family="gpt2",
        n_layers=24, n_heads=16, d_model=1024, d_mlp=4096,
        context_length=1024,
    ),
}


def get_model_spec(model_key: str) -> ModelSpec:
    """Look up a ``ModelSpec`` by its short key.

    Args:
        model_key: A key from ``MODEL_REGISTRY`` (e.g. ``"pythia-1.4b"``).

    Returns:
        The corresponding ``ModelSpec``.

    Raises:
        KeyError: If *model_key* is not in the registry.
    """
    if model_key not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(
            f"Unknown model key '{model_key}'. Available: {available}"
        )
    return MODEL_REGISTRY[model_key]


def list_available_models() -> List[str]:
    """Return a sorted list of all registered model keys."""
    return sorted(MODEL_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Model loading helper
# ---------------------------------------------------------------------------

def load_model(
    model_key: str,
    device: Optional[str] = None,
    cache_dir: str = "cache_dir",
    dtype: Optional[torch.dtype] = None,
) -> Tuple[HookedTransformer, ModelSpec]:
    """Load a ``HookedTransformer`` by its registry key.

    Automatically selects device using the fallback chain
    cuda → mps → cpu if *device* is ``None``.

    Args:
        model_key: Registry key (e.g. ``"pythia-1.4b"``).
        device: Target device string. ``None`` means auto-detect.
        cache_dir: HuggingFace cache directory.
        dtype: Optional torch dtype override (e.g. ``torch.float16``).

    Returns:
        Tuple of (model, spec).
    """
    spec = get_model_spec(model_key)

    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    logger.info(f"Loading {spec.transformer_lens_name} on {device}")

    kwargs = {"cache_dir": cache_dir}
    if dtype is not None:
        kwargs["dtype"] = dtype

    model = HookedTransformer.from_pretrained(
        spec.transformer_lens_name, device=device, **kwargs
    )

    # Sanity checks: verify loaded model matches registry spec
    cfg = model.cfg
    _warn_if_mismatch("n_layers", cfg.n_layers, spec.n_layers, model_key)
    _warn_if_mismatch("n_heads", cfg.n_heads, spec.n_heads, model_key)
    _warn_if_mismatch("d_model", cfg.d_model, spec.d_model, model_key)

    logger.info(
        f"Loaded {model_key}: {cfg.n_layers}L / {cfg.n_heads}H / d={cfg.d_model}"
    )
    return model, spec


def _warn_if_mismatch(field: str, actual, expected, model_key: str) -> None:
    """Emit a warning if a loaded model's config does not match the registry."""
    if actual != expected:
        logger.warning(
            f"Registry mismatch for {model_key}.{field}: "
            f"expected {expected}, got {actual}. "
            "Update MODEL_REGISTRY if this is intentional."
        )


# ---------------------------------------------------------------------------
# Tokenizer helpers — number-aware utilities
# ---------------------------------------------------------------------------

def find_number_token_positions(
    model: HookedTransformer,
    prompt: str,
    target_number: int,
) -> List[int]:
    """Find all token positions that encode (part of) *target_number*.

    Works across GPT-2 / Pythia (BPE), Gemma / Phi-3 (SentencePiece) by
    checking the decoded string of each token against the target.

    Args:
        model: A ``HookedTransformer`` with an attached tokenizer.
        prompt: The full text prompt.
        target_number: The integer to locate.

    Returns:
        List of 0-based token indices that contain digits of *target_number*.
        Empty list if not found.
    """
    str_tokens = model.to_str_tokens(prompt)
    target_str = str(target_number)
    positions: List[int] = []

    for i, tok in enumerate(str_tokens):
        # Strip common BPE / SentencePiece whitespace markers
        clean = tok.replace("Ġ", "").replace("▁", "").strip()
        if not clean:
            continue
        # For single-digit targets, exact match avoids false positives
        # (e.g. "1" matching inside "12"). For multi-digit targets,
        # check if the token contains the full target string OR if the
        # token is a contiguous sub-string of the target (multi-token number).
        if clean == target_str:
            positions.append(i)
        elif len(target_str) > 1 and (target_str in clean or clean in target_str):
            # Multi-digit: token could be a substring of the number
            # (e.g. target "12" split into tokens "1" and "2")
            positions.append(i)

    return positions


def find_last_number_token(
    model: HookedTransformer,
    prompt: str,
    target_number: int,
) -> int:
    """Return the *last* token position encoding *target_number*.

    This is the recommended default: for multi-token numbers the last
    sub-token is where the full number representation is complete.

    Args:
        model: A ``HookedTransformer``.
        prompt: The full text prompt.
        target_number: The integer to locate.

    Returns:
        Token index (0-based).

    Raises:
        ValueError: If *target_number* cannot be found in the tokenized prompt.
    """
    positions = find_number_token_positions(model, prompt, target_number)
    if not positions:
        raise ValueError(
            f"Could not find {target_number} in tokens of '{prompt}'. "
            f"Tokens: {model.to_str_tokens(prompt)}"
        )
    return positions[-1]


def verify_single_token_numbers(
    model: HookedTransformer,
    numbers: Optional[List[int]] = None,
    prefix: str = " ",
) -> Dict[int, bool]:
    """Check which numbers are single-token with the model's tokenizer.

    Args:
        model: A ``HookedTransformer``.
        numbers: Numbers to test (default: 0-9).
        prefix: Prefix before the number (e.g. ``" "`` for space-prefixed).

    Returns:
        Dict mapping each number to ``True`` if it is a single token.
    """
    if numbers is None:
        numbers = list(range(10))

    results: Dict[int, bool] = {}
    for n in numbers:
        text = f"{prefix}{n}"
        tokens = model.to_tokens(text, prepend_bos=False)
        # Subtract BOS if present, count remaining tokens
        n_tokens = tokens.shape[1]
        # The prefix itself may consume a token; we care about
        # whether the number part adds exactly one token.
        prefix_tokens = model.to_tokens(prefix, prepend_bos=False).shape[1]
        is_single = (n_tokens - prefix_tokens) == 1
        results[n] = is_single
        if not is_single:
            str_toks = model.to_str_tokens(text, prepend_bos=False)
            logger.warning(
                f"Number {n} is NOT single-token with prefix '{prefix}': "
                f"{n_tokens} tokens (prefix={prefix_tokens}), "
                f"tokens = {str_toks}"
            )

    return results
