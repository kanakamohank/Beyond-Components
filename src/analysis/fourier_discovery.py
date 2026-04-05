"""
Fourier Discovery: Bottom-up DFT analysis of residual stream activations.

Implements the Fourier-based approach from the helix paper (2502.00873) for
discovering periodic number representations in transformer models. This is a
**passive observation** tool — it does not modify the model.

Algorithm outline:
    1. For each number n in {0, …, N-1}, run a prompt through the model and
       cache the residual stream at every layer.
    2. Stack activations into a matrix A ∈ R^{N × d_model}.
    3. Compute the DFT along the "number" axis: F = DFT(A) ∈ C^{N × d_model}.
    4. For each frequency k, compute power P(k) = ‖F[k]‖².
    5. Peaks in P(k) reveal periodic structure (e.g. period-10 → k=1,
       period-5 → k=2 for N=10).

The module is model-agnostic — it only requires a ``HookedTransformer``.

Typical usage::

    from src.analysis.fourier_discovery import FourierDiscovery
    from src.data.arithmetic_dataset import ArithmeticPromptGenerator

    discovery = FourierDiscovery(model, device=device)
    prompts = ArithmeticPromptGenerator(operand_range=range(0, 10))
    results = discovery.run_all_layers(prompts)

    # Inspect which layers have strong periodic structure
    for lr in results:
        if lr.dominant_frequency_power_ratio > 5.0:
            print(f"Layer {lr.layer}: freq={lr.dominant_frequency}, "
                  f"power_ratio={lr.dominant_frequency_power_ratio:.1f}x")
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class FourierResult:
    """DFT result for a single layer and activation type.

    Attributes:
        layer: Layer index.
        hook_name: Full TransformerLens hook name.
        power_spectrum: 1-D array of shape ``(N//2+1,)`` — DFT power at each
            non-negative frequency.  Index *k* corresponds to a period of
            ``N / k`` (k=0 is the DC component / mean).
        dominant_frequency: Frequency index with the highest power
            (excluding DC, k=0).
        dominant_frequency_power_ratio: Ratio of dominant-frequency power to
            the median power of all non-DC frequencies.  A value ≫ 1 indicates
            a clear periodic signal.
        activations: Raw activation matrix ``(N, d_model)`` on CPU.
            Stored only if ``store_activations=True`` was passed.
        dft_complex: Full complex DFT matrix ``(N, d_model)`` on CPU.
            Stored only if ``store_dft=True`` was passed.
    """
    layer: int
    hook_name: str
    power_spectrum: np.ndarray
    dominant_frequency: int
    dominant_frequency_power_ratio: float
    activations: Optional[np.ndarray] = None
    dft_complex: Optional[np.ndarray] = None


@dataclass
class LayerFourierResult:
    """Aggregated Fourier results for one layer across hook types.

    Attributes:
        layer: Layer index.
        resid_pre: ``FourierResult`` for the residual stream *before* the layer.
        resid_post: ``FourierResult`` for the residual stream *after* the layer
            (``None`` if not requested).
        dominant_frequency: Best dominant frequency across hooks.
        dominant_frequency_power_ratio: Corresponding power ratio.
    """
    layer: int
    resid_pre: FourierResult
    resid_post: Optional[FourierResult] = None

    @property
    def dominant_frequency(self) -> int:
        """Best dominant frequency across available hooks."""
        candidates = [self.resid_pre]
        if self.resid_post is not None:
            candidates.append(self.resid_post)
        best = max(candidates, key=lambda r: r.dominant_frequency_power_ratio)
        return best.dominant_frequency

    @property
    def dominant_frequency_power_ratio(self) -> float:
        """Best power ratio across available hooks."""
        candidates = [self.resid_pre]
        if self.resid_post is not None:
            candidates.append(self.resid_post)
        return max(r.dominant_frequency_power_ratio for r in candidates)


# ---------------------------------------------------------------------------
# Core Fourier analysis
# ---------------------------------------------------------------------------

def compute_fourier_power_spectrum(
    activations: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the 1-D DFT power spectrum along the number axis.

    Args:
        activations: Array of shape ``(N, d_model)`` where row *i* is the
            activation for number *i*.

    Returns:
        Tuple of ``(power_spectrum, dft_complex)`` where:
            - ``power_spectrum``: shape ``(N//2+1,)`` — total power at each
              non-negative frequency, summed over the d_model dimension.
            - ``dft_complex``: shape ``(N, d_model)`` — full complex DFT.

    Raises:
        ValueError: If *activations* has fewer than 2 rows.
    """
    if activations.shape[0] < 2:
        raise ValueError(
            f"Need at least 2 number activations, got {activations.shape[0]}"
        )

    N, d = activations.shape

    # DFT along axis 0 (number axis)
    dft = np.fft.fft(activations, axis=0)  # (N, d)

    # Power = |F[k]|^2 summed over d_model dimensions
    # Only keep non-negative frequencies (real-signal symmetry)
    n_freqs = N // 2 + 1
    power = np.sum(np.abs(dft[:n_freqs, :]) ** 2, axis=1)  # (n_freqs,)

    return power, dft


def identify_dominant_frequency(
    power_spectrum: np.ndarray,
) -> Tuple[int, float]:
    """Find the dominant non-DC frequency and its power ratio.

    The power ratio is defined as::

        power[dominant_freq] / mean(power[1:] excluding dominant_freq)

    This gives a signal-to-noise ratio: how much stronger is the peak
    compared to the average of all *other* non-DC frequencies. A ratio ≫ 1
    means there is a clear periodic signal at that frequency.  Returns
    ``inf`` when the dominant frequency has power but all others are ~0
    (a perfectly clean periodic signal).

    Args:
        power_spectrum: 1-D array from ``compute_fourier_power_spectrum``.

    Returns:
        Tuple of ``(dominant_frequency_index, power_ratio)``.
        Returns ``(0, 0.0)`` if the spectrum has no non-DC components.
    """
    if len(power_spectrum) <= 1:
        return 0, 0.0

    # Exclude DC (k=0)
    non_dc = power_spectrum[1:]
    dominant_idx = int(np.argmax(non_dc)) + 1  # +1 to account for DC offset

    dominant_power = float(power_spectrum[dominant_idx])

    # Compute baseline: mean of all non-DC frequencies EXCLUDING the dominant
    other_powers = np.concatenate([non_dc[:dominant_idx - 1], non_dc[dominant_idx:]])
    if len(other_powers) > 0:
        baseline = float(np.mean(other_powers))
    else:
        baseline = 0.0

    if baseline < 1e-12:
        # If the dominant frequency has real power but baseline is ~0,
        # that IS a very strong signal.
        ratio = float("inf") if dominant_power > 1e-12 else 0.0
    else:
        ratio = dominant_power / baseline

    return dominant_idx, ratio


# ---------------------------------------------------------------------------
# Main discovery class
# ---------------------------------------------------------------------------

class FourierDiscovery:
    """Bottom-up Fourier analysis of number representations in a transformer.

    This class orchestrates the full pipeline: collect activations for each
    number, compute DFT power spectra, and identify layers with periodic
    structure.

    Args:
        model: A ``HookedTransformer`` to analyze.
        device: Device override. Defaults to the model's device.
    """

    def __init__(
        self,
        model: HookedTransformer,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = device if device is not None else next(model.parameters()).device
        self.cfg = model.cfg

        # Sanity check: model should be in eval mode
        self.model.eval()

    # ------------------------------------------------------------------
    # Activation collection
    # ------------------------------------------------------------------

    def collect_activations(
        self,
        prompts: List,
        layers: Optional[List[int]] = None,
        position: str = "last",
        include_resid_post: bool = False,
        batch_size: int = 16,
    ) -> Dict[str, np.ndarray]:
        """Collect residual stream activations for a list of prompts.

        Args:
            prompts: Iterable of objects with a ``.prompt`` attribute
                (e.g. ``ArithmeticSample``) or plain strings.
            layers: Layer indices to collect from. ``None`` → all layers.
            position: Token position to extract.
                - ``"last"``: last non-padding token (default, recommended).
                - ``"eq_sign"``: the ``=`` token position.
                - An integer: explicit 0-based index.
            include_resid_post: Also collect post-layer residual streams.
            batch_size: Number of prompts to process at once.

        Returns:
            Dict mapping hook names to numpy arrays of shape
            ``(n_prompts, d_model)``.
        """
        if layers is None:
            layers = list(range(self.cfg.n_layers))

        # Validate layers
        for l in layers:
            assert 0 <= l < self.cfg.n_layers, (
                f"Layer {l} out of range [0, {self.cfg.n_layers})"
            )

        # Build hook name list
        hook_names = []
        for l in layers:
            hook_names.append(f"blocks.{l}.hook_resid_pre")
            if include_resid_post:
                hook_names.append(f"blocks.{l}.hook_resid_post")

        # Extract prompt strings
        prompt_texts = []
        for p in prompts:
            text = p.prompt if hasattr(p, "prompt") else str(p)
            prompt_texts.append(text)

        n_prompts = len(prompt_texts)
        logger.info(
            f"Collecting activations: {n_prompts} prompts, "
            f"{len(layers)} layers, position='{position}'"
        )

        # Pre-allocate storage
        result: Dict[str, List[np.ndarray]] = {h: [] for h in hook_names}

        # Process in batches
        for batch_start in tqdm(
            range(0, n_prompts, batch_size),
            desc="Collecting activations",
            disable=n_prompts <= batch_size,
        ):
            batch_texts = prompt_texts[batch_start : batch_start + batch_size]

            # Tokenize
            tokens = self.model.to_tokens(batch_texts)
            tokens = tokens.to(self.device)

            # Forward pass with cache
            with torch.no_grad():
                _, cache = self.model.run_with_cache(
                    tokens, names_filter=hook_names
                )

            # Determine extraction positions for each item in the batch
            pos_indices = self._resolve_positions(
                tokens, batch_texts, position
            )

            # Extract activations at the resolved positions
            for hook in hook_names:
                if hook not in cache:
                    logger.warning(f"Hook '{hook}' not found in cache, skipping")
                    continue
                tensor = cache[hook]  # (batch, seq, d_model)
                for i, pos_idx in enumerate(pos_indices):
                    act = tensor[i, pos_idx, :].cpu().numpy()
                    result[hook].append(act)

            # Free cache memory
            del cache
            if self.device.type == "mps":
                torch.mps.empty_cache()

        # Stack into arrays
        stacked: Dict[str, np.ndarray] = {}
        for hook, arrays in result.items():
            if arrays:
                stacked[hook] = np.stack(arrays, axis=0)
            else:
                logger.warning(f"No activations collected for {hook}")

        return stacked

    def _resolve_positions(
        self,
        tokens: torch.Tensor,
        texts: List[str],
        position: str,
    ) -> List[int]:
        """Resolve token extraction positions for a batch.

        Args:
            tokens: Token tensor of shape ``(batch, seq_len)``.
            texts: Original text strings.
            position: Position strategy.

        Returns:
            List of integer indices, one per batch item.
        """
        batch_size, seq_len = tokens.shape

        if position == "last":
            # Last non-padding token
            if self.model.tokenizer.pad_token_id is not None:
                pad_id = self.model.tokenizer.pad_token_id
                lengths = (tokens != pad_id).sum(dim=1)
                return [(l.item() - 1) for l in lengths]
            else:
                return [seq_len - 1] * batch_size

        elif position == "eq_sign":
            # Find the '=' token
            positions = []
            for i, text in enumerate(texts):
                str_tokens = self.model.to_str_tokens(tokens[i])
                eq_pos = None
                for j, tok in enumerate(str_tokens):
                    if "=" in tok.strip():
                        eq_pos = j
                        break
                if eq_pos is None:
                    logger.warning(
                        f"'=' not found in '{text}', falling back to last token"
                    )
                    eq_pos = seq_len - 1
                positions.append(eq_pos)
            return positions

        elif isinstance(position, int) or (
            isinstance(position, str) and position.lstrip("-").isdigit()
        ):
            idx = int(position)
            return [idx] * batch_size

        else:
            raise ValueError(
                f"Unknown position strategy '{position}'. "
                "Use 'last', 'eq_sign', or an integer."
            )

    # ------------------------------------------------------------------
    # Fourier analysis
    # ------------------------------------------------------------------

    def analyze_layer(
        self,
        activations: np.ndarray,
        layer: int,
        hook_name: str,
        store_activations: bool = False,
        store_dft: bool = False,
    ) -> FourierResult:
        """Run Fourier analysis on activations for a single layer.

        Args:
            activations: Shape ``(N, d_model)``.
            layer: Layer index (for metadata).
            hook_name: Hook name (for metadata).
            store_activations: Whether to keep raw activations in the result.
            store_dft: Whether to keep the full complex DFT in the result.

        Returns:
            A ``FourierResult`` with power spectrum and dominant frequency.
        """
        power, dft = compute_fourier_power_spectrum(activations)
        dom_freq, power_ratio = identify_dominant_frequency(power)

        return FourierResult(
            layer=layer,
            hook_name=hook_name,
            power_spectrum=power,
            dominant_frequency=dom_freq,
            dominant_frequency_power_ratio=power_ratio,
            activations=activations if store_activations else None,
            dft_complex=dft if store_dft else None,
        )

    def run_all_layers(
        self,
        prompts: List,
        layers: Optional[List[int]] = None,
        position: str = "last",
        include_resid_post: bool = False,
        store_activations: bool = False,
        store_dft: bool = False,
        batch_size: int = 16,
    ) -> List[LayerFourierResult]:
        """Run Fourier discovery across all specified layers.

        This is the main entry point for Phase 1 analysis.

        Args:
            prompts: Iterable of ``ArithmeticSample`` or plain strings.
                Must be ordered by number value (i.e. prompt *i* corresponds
                to number *i*).
            layers: Layers to analyze. ``None`` → all layers.
            position: Token position strategy.
            include_resid_post: Also analyze post-layer residuals.
            store_activations: Keep raw activations in results.
            store_dft: Keep complex DFT matrices in results.
            batch_size: Batch size for activation collection.

        Returns:
            List of ``LayerFourierResult``, one per layer, sorted by layer.
        """
        if layers is None:
            layers = list(range(self.cfg.n_layers))

        # Collect all activations in one pass
        all_activations = self.collect_activations(
            prompts=prompts,
            layers=layers,
            position=position,
            include_resid_post=include_resid_post,
            batch_size=batch_size,
        )

        # Analyze each layer
        results: List[LayerFourierResult] = []
        for layer_idx in sorted(layers):
            pre_hook = f"blocks.{layer_idx}.hook_resid_pre"
            post_hook = f"blocks.{layer_idx}.hook_resid_post"

            # resid_pre
            if pre_hook not in all_activations:
                logger.warning(f"No activations for {pre_hook}, skipping layer {layer_idx}")
                continue
            pre_result = self.analyze_layer(
                all_activations[pre_hook],
                layer=layer_idx,
                hook_name=pre_hook,
                store_activations=store_activations,
                store_dft=store_dft,
            )

            # resid_post (optional)
            post_result = None
            if include_resid_post and post_hook in all_activations:
                post_result = self.analyze_layer(
                    all_activations[post_hook],
                    layer=layer_idx,
                    hook_name=post_hook,
                    store_activations=store_activations,
                    store_dft=store_dft,
                )

            results.append(LayerFourierResult(
                layer=layer_idx,
                resid_pre=pre_result,
                resid_post=post_result,
            ))

        # Log summary
        if results:
            best = max(results, key=lambda r: r.dominant_frequency_power_ratio)
            logger.info(
                f"Fourier discovery complete: {len(results)} layers analyzed. "
                f"Best: Layer {best.layer}, freq={best.dominant_frequency}, "
                f"power_ratio={best.dominant_frequency_power_ratio:.1f}x"
            )

        return results

    # ------------------------------------------------------------------
    # Per-head Fourier analysis
    # ------------------------------------------------------------------

    def analyze_attention_heads(
        self,
        prompts: List,
        layers: Optional[List[int]] = None,
        position: str = "last",
        batch_size: int = 16,
        power_ratio_threshold: float = 3.0,
    ) -> Dict[Tuple[int, int], FourierResult]:
        """Fourier analysis of individual attention head outputs.

        Collects ``hook_z`` (pre-OV-projection attention output) for each
        head and runs DFT analysis.  This identifies which specific heads
        carry periodic number information.

        Args:
            prompts: Ordered prompts (prompt *i* → number *i*).
            layers: Layers to scan. ``None`` → all.
            position: Token position strategy.
            batch_size: Batch size for forward passes.
            power_ratio_threshold: Minimum power ratio to report a head.

        Returns:
            Dict mapping ``(layer, head)`` to ``FourierResult`` for heads
            exceeding the threshold.
        """
        if layers is None:
            layers = list(range(self.cfg.n_layers))

        # Build hook names for attention outputs
        hook_names = [f"blocks.{l}.attn.hook_z" for l in layers]

        prompt_texts = [
            p.prompt if hasattr(p, "prompt") else str(p) for p in prompts
        ]
        n_prompts = len(prompt_texts)

        # Collect attention head outputs: (n_prompts, n_heads, d_head)
        head_acts: Dict[int, List[np.ndarray]] = {l: [] for l in layers}

        for batch_start in tqdm(
            range(0, n_prompts, batch_size),
            desc="Collecting head activations",
            disable=n_prompts <= batch_size,
        ):
            batch_texts = prompt_texts[batch_start : batch_start + batch_size]
            tokens = self.model.to_tokens(batch_texts).to(self.device)

            with torch.no_grad():
                _, cache = self.model.run_with_cache(
                    tokens, names_filter=hook_names
                )

            pos_indices = self._resolve_positions(tokens, batch_texts, position)

            for l_idx, layer in enumerate(layers):
                hook = hook_names[l_idx]
                if hook not in cache:
                    continue
                tensor = cache[hook]  # (batch, seq, n_heads, d_head)
                for i, pos_idx in enumerate(pos_indices):
                    # Extract all heads at once
                    act = tensor[i, pos_idx, :, :].cpu().numpy()  # (n_heads, d_head)
                    head_acts[layer].append(act)

            del cache
            if self.device.type == "mps":
                torch.mps.empty_cache()

        # Analyze each head
        significant_heads: Dict[Tuple[int, int], FourierResult] = {}

        for layer in sorted(layers):
            if not head_acts[layer]:
                continue
            stacked = np.stack(head_acts[layer], axis=0)  # (N, n_heads, d_head)
            n_heads = stacked.shape[1]

            for head in range(n_heads):
                head_data = stacked[:, head, :]  # (N, d_head)
                power, dft = compute_fourier_power_spectrum(head_data)
                dom_freq, ratio = identify_dominant_frequency(power)

                if ratio >= power_ratio_threshold:
                    result = FourierResult(
                        layer=layer,
                        hook_name=f"blocks.{layer}.attn.hook_z[head={head}]",
                        power_spectrum=power,
                        dominant_frequency=dom_freq,
                        dominant_frequency_power_ratio=ratio,
                    )
                    significant_heads[(layer, head)] = result
                    logger.info(
                        f"Significant head: L{layer}H{head}, "
                        f"freq={dom_freq}, ratio={ratio:.1f}x"
                    )

        logger.info(
            f"Head analysis complete: {len(significant_heads)} significant heads "
            f"out of {len(layers) * self.cfg.n_heads} total"
        )
        return significant_heads

    # ------------------------------------------------------------------
    # Utility: summarize results
    # ------------------------------------------------------------------

    @staticmethod
    def summarize(
        layer_results: List[LayerFourierResult],
        top_n: int = 10,
    ) -> str:
        """Generate a human-readable summary of Fourier discovery results.

        Args:
            layer_results: Output from ``run_all_layers``.
            top_n: Number of top layers to include.

        Returns:
            Formatted string summary.
        """
        if not layer_results:
            return "No results to summarize."

        sorted_results = sorted(
            layer_results,
            key=lambda r: r.dominant_frequency_power_ratio,
            reverse=True,
        )

        lines = ["=" * 60]
        lines.append("FOURIER DISCOVERY SUMMARY")
        lines.append("=" * 60)
        lines.append(f"Layers analyzed: {len(layer_results)}")
        lines.append(f"Top {min(top_n, len(sorted_results))} layers by power ratio:")
        lines.append("-" * 60)
        lines.append(f"{'Layer':>6} {'Freq':>5} {'Period':>8} {'Power Ratio':>12}")
        lines.append("-" * 60)

        n_prompts = (
            sorted_results[0].resid_pre.power_spectrum.shape[0] * 2 - 2
            if sorted_results else 0
        )

        for r in sorted_results[:top_n]:
            freq = r.dominant_frequency
            period = n_prompts / freq if freq > 0 else float("inf")
            ratio = r.dominant_frequency_power_ratio
            lines.append(f"{r.layer:>6d} {freq:>5d} {period:>8.1f} {ratio:>12.1f}x")

        lines.append("=" * 60)
        return "\n".join(lines)
