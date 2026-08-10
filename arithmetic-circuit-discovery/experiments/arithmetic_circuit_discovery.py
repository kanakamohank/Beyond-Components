#!/usr/bin/env python3
"""
Arithmetic Circuit Discovery — Unified Causal Pipeline

Finds the ACTUAL arithmetic circuit in a transformer model by combining:
  Stage 1: Logit Lens + Layer-Level Activation Patching
  Stage 2: Component-Level Patching (per attention head + per MLP)
  Stage 3: Mean Ablation + Attention Pattern Analysis + Position-Aware Patching

Unlike the SVD mask pipeline (Phases 2-5) which found output-formatting layers (L28-31)
via KL divergence, this script uses task-specific causal interventions that are
unbiased across layers.

Unlike the helix investigation which found representational (non-causal) geometry,
this script directly measures which components are NECESSARY for correct arithmetic.

Supported models (via model_registry or raw TransformerLens names):
  - pythia-1.4b, pythia-2.8b, pythia-6.9b
  - phi-3-mini (microsoft/Phi-3-mini-4k-instruct)
  - gpt2-small, gpt2-medium
  - gemma-2b
  - Any TransformerLens-compatible model name

Usage:
    python experiments/arithmetic_circuit_discovery.py --model-key pythia-1.4b
    python experiments/arithmetic_circuit_discovery.py --model microsoft/Phi-3-mini-4k-instruct
    python experiments/arithmetic_circuit_discovery.py --model-key phi-3-mini --stage 1
    python experiments/arithmetic_circuit_discovery.py --stage 2   # component patching only
    python experiments/arithmetic_circuit_discovery.py --operand-range 50  # max operand value
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from transformer_lens import HookedTransformer
from pathlib import Path
import json
import re
import gc
import random
import time
import argparse
import logging
import warnings
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════

DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"
OUTPUT_DIR = "arithmetic_circuit_results"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_model_name(model_key_or_name: str) -> str:
    """Resolve a model registry key to a TransformerLens name.

    Accepts either a registry key (e.g. 'pythia-1.4b') or a raw
    TransformerLens / HuggingFace name (e.g. 'microsoft/Phi-3-mini-4k-instruct').
    Registry keys take priority.
    """
    try:
        from src.utils.model_registry import get_model_spec
        spec = get_model_spec(model_key_or_name)
        logger.info(f"Resolved registry key '{model_key_or_name}' -> '{spec.transformer_lens_name}'")
        return spec.transformer_lens_name
    except (KeyError, ImportError):
        # Not a registry key — treat as raw model name
        return model_key_or_name


# Models known to produce incorrect results on MPS with bfloat16.
# These will be forced to CPU + float32 when MPS is the only accelerator.
_MPS_INCOMPATIBLE_MODELS = ("gemma",)


def auto_dtype(device: torch.device, model_name: str = "") -> torch.dtype:
    """Select dtype based on device capabilities and model family."""
    if device.type == "cuda":
        return torch.bfloat16
    elif device.type == "mps":
        return torch.bfloat16
    return torch.float32


def auto_device_dtype(
    model_name: str,
    device_override: Optional[str] = None,
    dtype_override: Optional[str] = None,
) -> Tuple[torch.device, torch.dtype]:
    """Choose device and dtype, with model-aware fallbacks.

    Gemma models produce silent garbage on MPS + bfloat16, so they are
    forced to CPU + float32 when no CUDA is available and no explicit
    override is given.
    """
    # Resolve device
    if device_override:
        device = torch.device(device_override)
    else:
        device = get_device()

    # Resolve dtype
    if dtype_override:
        dtype_map = {"float32": torch.float32, "float16": torch.float16,
                     "bfloat16": torch.bfloat16}
        dtype = dtype_map.get(dtype_override, torch.float32)
    else:
        dtype = auto_dtype(device, model_name)

    # Model-specific MPS workaround
    model_lower = model_name.lower()
    if device.type == "mps" and not device_override:
        if any(tag in model_lower for tag in _MPS_INCOMPATIBLE_MODELS):
            print(f"  ⚠ {model_name} is known to produce incorrect results on MPS.")
            print(f"    Falling back to CPU + float32. Use --device mps to force MPS.")
            device = torch.device("cpu")
            dtype = torch.float32

    return device, dtype


# ═════════════════════════════════════════════════════════════
# DATA GENERATION
# ═════════════════════════════════════════════════════════════

def generate_arithmetic_prompts(
    n_problems: int = 60,
    max_operand: int = 50,
    seed: int = 42,
    few_shot: bool = True,
) -> List[Dict[str, Any]]:
    """Generate addition problems with clean/corrupted pairs.

    For activation patching we need:
      - clean prompt:     "{a} + {b} ="  (correct answer = a+b)
      - corrupted prompt: "{a'} + {b} =" (different a', same format)

    The corruption changes the first operand so we can trace
    where the model processes operand information.
    """
    rng = random.Random(seed)
    problems = []

    few_shot_prefix = ""
    if few_shot:
        few_shot_prefix = "Calculate:\n12 + 7 = 19\n34 + 15 = 49\n"

    for _ in range(n_problems):
        a = rng.randint(1, max_operand)
        b = rng.randint(1, max_operand)
        answer = a + b

        # Corrupted: change a to a different value (same digit count to keep tokenization stable)
        a_corrupt = a
        while a_corrupt == a:
            a_corrupt = rng.randint(1, max_operand)
        answer_corrupt = a_corrupt + b

        clean_prompt = f"{few_shot_prefix}{a} + {b} ="
        corrupt_prompt = f"{few_shot_prefix}{a_corrupt} + {b} ="

        problems.append({
            "a": a, "b": b, "answer": answer,
            "a_corrupt": a_corrupt, "answer_corrupt": answer_corrupt,
            "clean_prompt": clean_prompt,
            "corrupt_prompt": corrupt_prompt,
        })

    return problems


def get_answer_token_id(model, answer: int) -> int:
    """Get the first token ID for the answer string (space-prefixed).

    For multi-digit answers like 57, the model needs to predict ' 5' as the
    first token. We return that token's ID.
    """
    answer_str = " " + str(answer)
    tokens = model.to_tokens(answer_str, prepend_bos=False)
    return tokens[0, 0].item()


def get_answer_digit_token_id(model, answer: int) -> int:
    """Get the first DIGIT token ID for the answer.

    For ' 45' tokenized as [space, '4', '5'], returns the '4' token.
    This is more discriminative than the space token for logit comparisons.
    """
    answer_str = " " + str(answer)
    tokens = model.to_tokens(answer_str, prepend_bos=False)
    # Return second token (first digit) if multi-token, else first
    if tokens.shape[1] > 1:
        return tokens[0, 1].item()
    return tokens[0, 0].item()


def get_answer_token_ids_all(model, answer: int) -> List[int]:
    """Get ALL token IDs for the answer string (for multi-token answers)."""
    answer_str = " " + str(answer)
    tokens = model.to_tokens(answer_str, prepend_bos=False)
    return tokens[0].tolist()


def check_answer_greedy(model, tokens: torch.Tensor, answer: int, max_new: int = 4) -> bool:
    """Check if greedy generation produces the correct answer.

    Generates up to max_new tokens and checks if the decoded output
    contains the correct answer string.
    """
    with torch.no_grad():
        generated = tokens.clone()
        for _ in range(max_new):
            logits = model(generated)
            next_tok = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            generated = torch.cat([generated, next_tok], dim=1)

    # Decode only the newly generated tokens
    new_tokens = generated[0, tokens.shape[1]:]
    decoded = model.tokenizer.decode(new_tokens.tolist()).strip()
    answer_str = str(answer)

    # Check if the answer starts with the correct number
    return decoded.startswith(answer_str)


# ═════════════════════════════════════════════════════════════
# STAGE 1: LOGIT LENS + LAYER-LEVEL ACTIVATION PATCHING
# ═════════════════════════════════════════════════════════════

def run_logit_lens(
    model: HookedTransformer,
    problems: List[Dict],
    device: torch.device,
) -> Dict[str, Any]:
    """Logit Lens: at each layer, project residual stream through W_U.

    Measures when the correct answer token first appears in top-k predictions.
    This tells us WHERE the model has computed the answer.
    """
    n_layers = model.cfg.n_layers
    print(f"\n{'=' * 70}")
    print(f"STAGE 1a: LOGIT LENS — When does the answer become decodable?")
    print(f"{'=' * 70}")
    print(f"  Model: {model.cfg.model_name}, {n_layers} layers")
    print(f"  Problems: {len(problems)}")

    # We need W_U for projecting residual stream to vocabulary
    W_U = model.W_U.detach().float()  # [d_model, vocab]
    # Some models have a final layer norm before unembedding
    has_ln_final = hasattr(model, 'ln_final') and model.ln_final is not None

    # Collect all hook names we need (resid_post at every layer)
    hook_names = [f"blocks.{L}.hook_resid_post" for L in range(n_layers)]

    # Per-layer metrics
    layer_rank_correct = defaultdict(list)      # rank of correct token
    layer_prob_correct = defaultdict(list)       # probability of correct token
    layer_top1_correct = defaultdict(list)       # is correct token top-1?
    layer_top5_correct = defaultdict(list)       # is correct token in top-5?
    layer_logit_correct = defaultdict(list)      # raw logit for correct token

    n_done = 0
    for prob in problems:
        tokens = model.to_tokens(prob["clean_prompt"])
        answer_tok = get_answer_token_id(model, prob["answer"])

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)

        last_pos = tokens.shape[1] - 1

        for L in range(n_layers):
            resid = cache[hook_names[L]][0, last_pos, :].float()

            # Apply final LayerNorm if model has one (most do)
            if has_ln_final:
                # Manually apply LN: normalize then scale/shift
                resid_normed = model.ln_final(resid.unsqueeze(0).to(device)).squeeze(0).float().cpu()
            else:
                resid_normed = resid

            logits = resid_normed @ W_U.cpu()  # [vocab]
            probs = F.softmax(logits, dim=-1)

            correct_logit = logits[answer_tok].item()
            correct_prob = probs[answer_tok].item()

            # Rank of correct token (0-indexed, lower is better)
            rank = (logits > correct_logit).sum().item()

            layer_rank_correct[L].append(rank)
            layer_prob_correct[L].append(correct_prob)
            layer_top1_correct[L].append(1.0 if rank == 0 else 0.0)
            layer_top5_correct[L].append(1.0 if rank < 5 else 0.0)
            layer_logit_correct[L].append(correct_logit)

        del cache
        n_done += 1
        if n_done % 20 == 0:
            print(f"  Processed {n_done}/{len(problems)} problems...")

    # Summarize
    results = {}
    print(f"\n  {'Layer':>6}  {'MeanRank':>9}  {'Top-1%':>7}  {'Top-5%':>7}  "
          f"{'MeanProb':>9}  {'MeanLogit':>10}")
    print(f"  {'─' * 62}")

    for L in range(n_layers):
        mean_rank = np.mean(layer_rank_correct[L])
        top1_pct = np.mean(layer_top1_correct[L]) * 100
        top5_pct = np.mean(layer_top5_correct[L]) * 100
        mean_prob = np.mean(layer_prob_correct[L])
        mean_logit = np.mean(layer_logit_correct[L])

        marker = ""
        if top1_pct > 30:
            marker = " ◀ ANSWER EMERGES"
        if top1_pct > 60:
            marker = " ◀◀ STRONG"
        if top1_pct > 80:
            marker = " ★ DOMINANT"

        print(f"  L{L:>4d}  {mean_rank:9.1f}  {top1_pct:6.1f}%  {top5_pct:6.1f}%  "
              f"{mean_prob:9.4f}  {mean_logit:10.2f}{marker}")

        results[L] = {
            "mean_rank": float(mean_rank),
            "top1_pct": float(top1_pct),
            "top5_pct": float(top5_pct),
            "mean_prob": float(mean_prob),
            "mean_logit": float(mean_logit),
        }

    return {"logit_lens": results}


def run_layer_activation_patching(
    model: HookedTransformer,
    problems: List[Dict],
    device: torch.device,
) -> Dict[str, Any]:
    """Layer-level activation patching: patch residual stream at each layer.

    For each layer L:
      1. Run clean prompt → get clean logit for correct answer
      2. Run corrupted prompt → get corrupted logit (lower, wrong answer)
      3. Run corrupted prompt WITH clean residual stream patched at layer L
         → measure how much logit recovers

    Recovery = (patched_logit - corrupt_logit) / (clean_logit - corrupt_logit)
    High recovery at layer L means that layer's residual stream contains
    critical information for computing the correct answer.
    """
    n_layers = model.cfg.n_layers
    print(f"\n{'=' * 70}")
    print(f"STAGE 1b: LAYER ACTIVATION PATCHING — Which layers are causal?")
    print(f"{'=' * 70}")

    layer_recovery = defaultdict(list)

    n_done = 0
    for prob in problems:
        clean_tokens = model.to_tokens(prob["clean_prompt"])
        corrupt_tokens = model.to_tokens(prob["corrupt_prompt"])
        answer_tok = get_answer_token_id(model, prob["answer"])
        last_pos = clean_tokens.shape[1] - 1

        # Ensure same sequence length (should be, since format is identical)
        if clean_tokens.shape[1] != corrupt_tokens.shape[1]:
            continue

        # Get clean and corrupt logits with full cache
        all_hooks = [f"blocks.{L}.hook_resid_post" for L in range(n_layers)]

        with torch.no_grad():
            clean_logits, clean_cache = model.run_with_cache(
                clean_tokens, names_filter=all_hooks
            )
            corrupt_logits, _ = model.run_with_cache(
                corrupt_tokens, names_filter=[]
            )

        clean_logit = clean_logits[0, last_pos, answer_tok].item()
        corrupt_logit = corrupt_logits[0, last_pos, answer_tok].item()
        logit_diff = clean_logit - corrupt_logit

        if abs(logit_diff) < 0.1:
            # Model doesn't distinguish clean from corrupt — skip
            del clean_cache
            continue

        # For each layer: run corrupt with clean residual patched at that layer
        for L in range(n_layers):
            hook_name = f"blocks.{L}.hook_resid_post"
            clean_act = clean_cache[hook_name].detach().clone()

            def make_patch_hook(clean_activation, pos):
                def hook_fn(value, hook):
                    value[0, pos, :] = clean_activation[0, pos, :]
                    return value
                return hook_fn

            patch_hook = make_patch_hook(clean_act, last_pos)

            with torch.no_grad():
                with model.hooks(fwd_hooks=[(hook_name, patch_hook)]):
                    patched_logits = model(corrupt_tokens)

            patched_logit = patched_logits[0, last_pos, answer_tok].item()
            recovery = (patched_logit - corrupt_logit) / logit_diff
            layer_recovery[L].append(recovery)

        del clean_cache
        n_done += 1
        if n_done % 10 == 0:
            print(f"  Processed {n_done}/{len(problems)} problems...")

    # Summarize
    results = {}
    print(f"\n  {'Layer':>6}  {'MeanRecov':>10}  {'StdRecov':>9}  {'Bar':}")
    print(f"  {'─' * 60}")

    for L in range(n_layers):
        if not layer_recovery[L]:
            continue
        mean_r = np.mean(layer_recovery[L])
        std_r = np.std(layer_recovery[L])
        bar_len = max(0, int(mean_r * 40))
        bar = '█' * bar_len

        marker = ""
        if mean_r > 0.5:
            marker = " ◀ CRITICAL"
        if mean_r > 0.8:
            marker = " ★ ESSENTIAL"

        print(f"  L{L:>4d}  {mean_r:10.3f}  {std_r:9.3f}  {bar}{marker}")

        results[L] = {
            "mean_recovery": float(mean_r),
            "std_recovery": float(std_r),
            "n_samples": len(layer_recovery[L]),
        }

    return {"layer_patching": results}


# ═════════════════════════════════════════════════════════════
# STAGE 2: COMPONENT-LEVEL PATCHING
# ═════════════════════════════════════════════════════════════

def run_component_patching(
    model: HookedTransformer,
    problems: List[Dict],
    device: torch.device,
    target_layers: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Component-level activation patching: patch individual attention heads and MLPs.

    For each component (attention head output or MLP output) at target layers:
      - Patch that component's output from clean into the corrupted run
      - Measure logit recovery for the correct answer

    This identifies the specific attention heads and MLP layers responsible
    for arithmetic computation.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    if target_layers is None:
        # Default: scan all layers
        target_layers = list(range(n_layers))

    print(f"\n{'=' * 70}")
    print(f"STAGE 2: COMPONENT PATCHING — Which heads and MLPs are causal?")
    print(f"{'=' * 70}")
    print(f"  Target layers: {target_layers[0]}–{target_layers[-1]} "
          f"({len(target_layers)} layers)")
    print(f"  Components per layer: {n_heads} attn heads + 1 MLP = {n_heads + 1}")
    print(f"  Total components: {len(target_layers) * (n_heads + 1)}")

    # Hook names we need to cache from clean run
    # Use hook_z for per-head output: shape [batch, seq, n_heads, d_head]
    # This is the per-head result BEFORE W_O projection and head summation
    attn_hooks = []
    mlp_hooks = []
    for L in target_layers:
        attn_hooks.append(f"blocks.{L}.attn.hook_z")
        mlp_hooks.append(f"blocks.{L}.hook_mlp_out")

    all_hooks = attn_hooks + mlp_hooks

    # Results storage
    head_recovery = defaultdict(list)   # (layer, head) -> list of recovery values
    mlp_recovery = defaultdict(list)    # layer -> list of recovery values

    n_done = 0
    for prob in problems:
        clean_tokens = model.to_tokens(prob["clean_prompt"])
        corrupt_tokens = model.to_tokens(prob["corrupt_prompt"])
        answer_tok = get_answer_token_id(model, prob["answer"])
        last_pos = clean_tokens.shape[1] - 1

        if clean_tokens.shape[1] != corrupt_tokens.shape[1]:
            continue

        # Get clean cache and baseline logits
        with torch.no_grad():
            clean_logits, clean_cache = model.run_with_cache(
                clean_tokens, names_filter=all_hooks
            )
            corrupt_logits, _ = model.run_with_cache(
                corrupt_tokens, names_filter=[]
            )

        clean_logit = clean_logits[0, last_pos, answer_tok].item()
        corrupt_logit = corrupt_logits[0, last_pos, answer_tok].item()
        logit_diff = clean_logit - corrupt_logit

        if abs(logit_diff) < 0.1:
            del clean_cache
            continue

        # ── Patch each attention head (at last_pos) ──
        for L in target_layers:
            hook_name = f"blocks.{L}.attn.hook_z"
            if hook_name not in clean_cache:
                continue
            clean_attn = clean_cache[hook_name].detach().clone()
            # hook_z shape: [batch, seq, n_heads, d_head]

            for H in range(n_heads):
                def make_head_hook(clean_act, pos, head_idx):
                    def hook_fn(value, hook):
                        value[0, pos, head_idx, :] = clean_act[0, pos, head_idx, :]
                        return value
                    return hook_fn

                patch_hook = make_head_hook(clean_attn, last_pos, H)

                with torch.no_grad():
                    with model.hooks(fwd_hooks=[(hook_name, patch_hook)]):
                        patched_logits = model(corrupt_tokens)

                patched_logit = patched_logits[0, last_pos, answer_tok].item()
                recovery = (patched_logit - corrupt_logit) / logit_diff
                head_recovery[(L, H)].append(recovery)

        # ── Patch each MLP ──
        for L in target_layers:
            hook_name = f"blocks.{L}.hook_mlp_out"
            if hook_name not in clean_cache:
                continue
            clean_mlp = clean_cache[hook_name].detach().clone()

            def make_mlp_hook(clean_act, pos):
                def hook_fn(value, hook):
                    value[0, pos, :] = clean_act[0, pos, :]
                    return value
                return hook_fn

            patch_hook = make_mlp_hook(clean_mlp, last_pos)

            with torch.no_grad():
                with model.hooks(fwd_hooks=[(hook_name, patch_hook)]):
                    patched_logits = model(corrupt_tokens)

            patched_logit = patched_logits[0, last_pos, answer_tok].item()
            recovery = (patched_logit - corrupt_logit) / logit_diff
            mlp_recovery[L].append(recovery)

        del clean_cache
        n_done += 1
        if n_done % 5 == 0:
            print(f"  Processed {n_done}/{len(problems)} problems...")

    # ── Summarize Attention Heads ──
    print(f"\n  {'=' * 70}")
    print(f"  ATTENTION HEAD RESULTS (sorted by mean recovery)")
    print(f"  {'=' * 70}")

    head_results = {}
    head_summary = []
    for (L, H), recoveries in head_recovery.items():
        mean_r = np.mean(recoveries)
        std_r = np.std(recoveries)
        head_results[f"L{L}_H{H}"] = {
            "layer": L, "head": H,
            "mean_recovery": float(mean_r),
            "std_recovery": float(std_r),
            "n_samples": len(recoveries),
        }
        head_summary.append((L, H, mean_r, std_r))

    # Sort by recovery (descending)
    head_summary.sort(key=lambda x: x[2], reverse=True)

    print(f"\n  Top 30 attention heads:")
    print(f"  {'Layer':>6}  {'Head':>5}  {'MeanRecov':>10}  {'StdRecov':>9}  {'Bar':}")
    print(f"  {'─' * 65}")
    for L, H, mean_r, std_r in head_summary[:30]:
        bar_len = max(0, int(mean_r * 40))
        bar = '█' * bar_len
        marker = ""
        if mean_r > 0.02:
            marker = " ◀"
        if mean_r > 0.05:
            marker = " ◀◀"
        if mean_r > 0.10:
            marker = " ★"
        print(f"  L{L:>4d}  H{H:>4d}  {mean_r:10.4f}  {std_r:9.4f}  {bar}{marker}")

    # ── Summarize MLPs ──
    print(f"\n  {'=' * 70}")
    print(f"  MLP RESULTS (sorted by mean recovery)")
    print(f"  {'=' * 70}")

    mlp_results = {}
    mlp_summary = []
    for L, recoveries in mlp_recovery.items():
        mean_r = np.mean(recoveries)
        std_r = np.std(recoveries)
        mlp_results[f"L{L}"] = {
            "layer": L,
            "mean_recovery": float(mean_r),
            "std_recovery": float(std_r),
            "n_samples": len(recoveries),
        }
        mlp_summary.append((L, mean_r, std_r))

    mlp_summary.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  {'Layer':>6}  {'MeanRecov':>10}  {'StdRecov':>9}  {'Bar':}")
    print(f"  {'─' * 55}")
    for L, mean_r, std_r in mlp_summary:
        bar_len = max(0, int(mean_r * 40))
        bar = '█' * bar_len
        marker = ""
        if mean_r > 0.02:
            marker = " ◀"
        if mean_r > 0.05:
            marker = " ◀◀"
        if mean_r > 0.10:
            marker = " ★"
        print(f"  L{L:>4d}  {mean_r:10.4f}  {std_r:9.4f}  {bar}{marker}")

    # ── Combined Circuit Summary ──
    print(f"\n  {'=' * 70}")
    print(f"  CIRCUIT SUMMARY: Components with recovery > 0.02")
    print(f"  {'=' * 70}")

    significant_heads = [(L, H, r, s) for L, H, r, s in head_summary if r > 0.02]
    significant_mlps = [(L, r, s) for L, r, s in mlp_summary if r > 0.02]

    if significant_heads:
        print(f"\n  Attention Heads ({len(significant_heads)}):")
        for L, H, r, s in significant_heads:
            print(f"    L{L} H{H}: recovery={r:.4f} ± {s:.4f}")
    else:
        print(f"\n  No attention heads with recovery > 0.02")

    if significant_mlps:
        print(f"\n  MLP Layers ({len(significant_mlps)}):")
        for L, r, s in significant_mlps:
            print(f"    L{L} MLP: recovery={r:.4f} ± {s:.4f}")
    else:
        print(f"\n  No MLP layers with recovery > 0.02")

    return {
        "head_patching": head_results,
        "mlp_patching": mlp_results,
        "top_heads": [(L, H, float(r)) for L, H, r, _ in head_summary[:20]],
        "top_mlps": [(L, float(r)) for L, r, _ in mlp_summary[:10]],
    }


# ═════════════════════════════════════════════════════════════
# STAGE 1+2 COMBINED: IDENTIFY CRITICAL LAYERS THEN DRILL DOWN
# ═════════════════════════════════════════════════════════════

def identify_critical_layers(layer_patching_results: Dict) -> List[int]:
    """From layer-level patching results, identify which layers to focus Stage 2 on."""
    patching = layer_patching_results.get("layer_patching", {})
    if not patching:
        return list(range(32))  # fallback: scan all

    # Find layers with substantial recovery
    layer_scores = []
    for L_str, data in patching.items():
        L = int(L_str)
        layer_scores.append((L, data["mean_recovery"]))

    layer_scores.sort(key=lambda x: x[1], reverse=True)

    # Take layers with recovery > 0.01, plus their neighbors
    critical = set()
    for L, r in layer_scores:
        if r > 0.01:
            critical.add(L)
            if L > 0:
                critical.add(L - 1)
            critical.add(min(L + 1, layer_scores[-1][0]))

    # Always include at least the top 10 layers by recovery
    for L, r in layer_scores[:10]:
        critical.add(L)

    result = sorted(critical)
    print(f"\n  Critical layers for Stage 2: {result}")
    return result


# ═════════════════════════════════════════════════════════════
# OPERAND POSITION PATCHING (bonus: patch at operand token)
# ═════════════════════════════════════════════════════════════

def run_layer_patching_at_all_positions(
    model: HookedTransformer,
    problems: List[Dict],
    device: torch.device,
) -> Dict[str, Any]:
    """Layer-level patching at ALL sequence positions to find information flow.

    This reveals whether critical information flows through:
    - The first operand position (a)
    - The operator position (+)
    - The second operand position (b)
    - The final position (=)
    """
    n_layers = model.cfg.n_layers
    print(f"\n{'=' * 70}")
    print(f"STAGE 1c: POSITION-AWARE PATCHING — Where does info flow?")
    print(f"{'=' * 70}")

    # For a subset of problems, patch at different positions
    subset = problems[:20]

    # Position categories
    position_recovery = defaultdict(lambda: defaultdict(list))

    for prob in subset:
        clean_tokens = model.to_tokens(prob["clean_prompt"])
        corrupt_tokens = model.to_tokens(prob["corrupt_prompt"])
        answer_tok = get_answer_token_id(model, prob["answer"])
        last_pos = clean_tokens.shape[1] - 1

        if clean_tokens.shape[1] != corrupt_tokens.shape[1]:
            continue

        # Identify token positions
        str_tokens = model.to_str_tokens(prob["clean_prompt"])
        positions = {}
        for idx, tok in enumerate(str_tokens):
            tok_str = tok.strip()
            if tok_str == '=':
                positions['equals'] = idx
            elif tok_str == '+':
                positions['plus'] = idx

        # The last operand a is just before the last +
        if 'plus' in positions:
            positions['operand_a'] = positions['plus'] - 1
            positions['operand_b'] = positions['plus'] + 1

        positions['final'] = last_pos

        all_hooks = [f"blocks.{L}.hook_resid_post" for L in range(n_layers)]

        with torch.no_grad():
            clean_logits, clean_cache = model.run_with_cache(
                clean_tokens, names_filter=all_hooks
            )
            corrupt_logits, _ = model.run_with_cache(
                corrupt_tokens, names_filter=[]
            )

        clean_logit = clean_logits[0, last_pos, answer_tok].item()
        corrupt_logit = corrupt_logits[0, last_pos, answer_tok].item()
        logit_diff = clean_logit - corrupt_logit

        if abs(logit_diff) < 0.1:
            del clean_cache
            continue

        for pos_name, pos_idx in positions.items():
            for L in range(n_layers):
                hook_name = f"blocks.{L}.hook_resid_post"
                clean_act = clean_cache[hook_name].detach().clone()

                def make_pos_hook(clean_activation, p):
                    def hook_fn(value, hook):
                        value[0, p, :] = clean_activation[0, p, :]
                        return value
                    return hook_fn

                patch_hook = make_pos_hook(clean_act, pos_idx)

                with torch.no_grad():
                    with model.hooks(fwd_hooks=[(hook_name, patch_hook)]):
                        patched_logits = model(corrupt_tokens)

                patched_logit = patched_logits[0, last_pos, answer_tok].item()
                recovery = (patched_logit - corrupt_logit) / logit_diff
                position_recovery[pos_name][L].append(recovery)

        del clean_cache

    # Summarize
    print(f"\n  Recovery by position and layer:")
    pos_names = sorted(position_recovery.keys())

    # Header
    header = f"  {'Layer':>6}"
    for pn in pos_names:
        header += f"  {pn:>12}"
    print(header)
    print(f"  {'─' * (8 + 14 * len(pos_names))}")

    results = {}
    for L in range(n_layers):
        row = f"  L{L:>4d}"
        layer_data = {}
        any_significant = False
        for pn in pos_names:
            if L in position_recovery[pn] and position_recovery[pn][L]:
                mean_r = np.mean(position_recovery[pn][L])
                row += f"  {mean_r:12.4f}"
                layer_data[pn] = float(mean_r)
                if mean_r > 0.02:
                    any_significant = True
            else:
                row += f"  {'—':>12}"
        if any_significant:
            row += " ◀"
        print(row)
        results[L] = layer_data

    return {"position_patching": results}


# ═════════════════════════════════════════════════════════════
# STAGE 3: MEAN ABLATION, ATTENTION PATTERNS, POSITION-AWARE HEADS
# ═════════════════════════════════════════════════════════════

def compute_mean_activations(
    model: HookedTransformer,
    n_prompts: int = 100,
    max_operand: int = 50,
    seed: int = 999,
) -> Dict[str, torch.Tensor]:
    """Compute mean activations over a dataset for mean ablation.

    Returns a dict mapping hook_name -> mean activation tensor.
    We collect means for hook_z (per-head) and hook_mlp_out.
    """
    rng = random.Random(seed)
    n_layers = model.cfg.n_layers

    hook_names = []
    for L in range(n_layers):
        hook_names.append(f"blocks.{L}.attn.hook_z")
        hook_names.append(f"blocks.{L}.hook_mlp_out")

    # Accumulate means online
    means = {}
    counts = 0

    for _ in range(n_prompts):
        a = rng.randint(1, max_operand)
        b = rng.randint(1, max_operand)
        prompt = f"Calculate:\n12 + 7 = 19\n34 + 15 = 49\n{a} + {b} ="
        tokens = model.to_tokens(prompt)
        last_pos = tokens.shape[1] - 1

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)

        for hname in hook_names:
            act = cache[hname][0, last_pos].float().cpu()
            if hname not in means:
                means[hname] = torch.zeros_like(act)
            means[hname] += act

        counts += 1
        del cache

    for hname in means:
        means[hname] /= counts

    return means


def run_mean_ablation(
    model: HookedTransformer,
    problems: List[Dict],
    device: torch.device,
    top_heads: List[Tuple[int, int, float]],
    top_mlps: List[Tuple[int, float]],
    mean_acts: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """Mean ablation (knockouts): set component to mean activation, measure accuracy drop.

    This tests NECESSITY: if ablating a component drops accuracy,
    that component is necessary for arithmetic.
    """
    print(f"\n{'=' * 70}")
    print(f"STAGE 3a: MEAN ABLATION — Which components are NECESSARY?")
    print(f"{'=' * 70}")

    # First, measure baseline accuracy using greedy generation
    n_correct_baseline = 0
    n_total = 0
    for prob in problems:
        tokens = model.to_tokens(prob["clean_prompt"])
        if check_answer_greedy(model, tokens, prob["answer"]):
            n_correct_baseline += 1
        n_total += 1

    baseline_acc = n_correct_baseline / max(n_total, 1)
    print(f"  Baseline accuracy (greedy gen): {baseline_acc:.1%} ({n_correct_baseline}/{n_total})")

    results = {"baseline_accuracy": float(baseline_acc)}

    # Ablate top attention heads
    print(f"\n  Ablating top {min(15, len(top_heads))} attention heads:")
    head_ablation_results = {}
    for L, H, orig_recovery in top_heads[:15]:
        hook_name = f"blocks.{L}.attn.hook_z"
        mean_act = mean_acts.get(hook_name)
        if mean_act is None:
            continue

        def make_ablation_hook(mean_activation, head_idx, pos):
            def hook_fn(value, hook):
                value[0, pos, head_idx, :] = mean_activation[head_idx, :].to(value.dtype).to(value.device)
                return value
            return hook_fn

        n_correct = 0
        for prob in problems:
            tokens = model.to_tokens(prob["clean_prompt"])
            last_pos = tokens.shape[1] - 1
            ablation_hook = make_ablation_hook(mean_act, H, last_pos)

            # Generate with the ablation hook active
            generated = tokens.clone()
            with torch.no_grad():
                for _ in range(4):
                    with model.hooks(fwd_hooks=[(hook_name, ablation_hook)]):
                        logits = model(generated)
                    next_tok = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                    generated = torch.cat([generated, next_tok], dim=1)
            new_tokens = generated[0, tokens.shape[1]:]
            decoded = model.tokenizer.decode(new_tokens.tolist()).strip()
            if decoded.startswith(str(prob["answer"])):
                n_correct += 1

        ablated_acc = n_correct / max(n_total, 1)
        drop = baseline_acc - ablated_acc
        marker = ""
        if drop > 0.05:
            marker = " ◀ NECESSARY"
        if drop > 0.15:
            marker = " ★ CRITICAL"
        print(f"    L{L:2d} H{H:2d}: acc={ablated_acc:.1%}, drop={drop:+.1%} "
              f"(patch recov={orig_recovery:.3f}){marker}")

        head_ablation_results[f"L{L}_H{H}"] = {
            "layer": L, "head": H,
            "ablated_accuracy": float(ablated_acc),
            "accuracy_drop": float(drop),
            "patch_recovery": float(orig_recovery),
        }

    results["head_ablation"] = head_ablation_results

    # Ablate top MLPs
    print(f"\n  Ablating top {min(10, len(top_mlps))} MLP layers:")
    mlp_ablation_results = {}
    for L, orig_recovery in top_mlps[:10]:
        hook_name = f"blocks.{L}.hook_mlp_out"
        mean_act = mean_acts.get(hook_name)
        if mean_act is None:
            continue

        def make_mlp_ablation_hook(mean_activation, pos):
            def hook_fn(value, hook):
                value[0, pos, :] = mean_activation.to(value.dtype).to(value.device)
                return value
            return hook_fn

        n_correct = 0
        for prob in problems:
            tokens = model.to_tokens(prob["clean_prompt"])
            last_pos = tokens.shape[1] - 1
            ablation_hook = make_mlp_ablation_hook(mean_act, last_pos)

            # Generate with the ablation hook active
            generated = tokens.clone()
            with torch.no_grad():
                for _ in range(4):
                    with model.hooks(fwd_hooks=[(hook_name, ablation_hook)]):
                        logits = model(generated)
                    next_tok = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                    generated = torch.cat([generated, next_tok], dim=1)
            new_tokens = generated[0, tokens.shape[1]:]
            decoded = model.tokenizer.decode(new_tokens.tolist()).strip()
            if decoded.startswith(str(prob["answer"])):
                n_correct += 1

        ablated_acc = n_correct / max(n_total, 1)
        drop = baseline_acc - ablated_acc
        marker = ""
        if drop > 0.05:
            marker = " ◀ NECESSARY"
        if drop > 0.15:
            marker = " ★ CRITICAL"
        print(f"    L{L:2d} MLP: acc={ablated_acc:.1%}, drop={drop:+.1%} "
              f"(patch recov={orig_recovery:.3f}){marker}")

        mlp_ablation_results[f"L{L}"] = {
            "layer": L,
            "ablated_accuracy": float(ablated_acc),
            "accuracy_drop": float(drop),
            "patch_recovery": float(orig_recovery),
        }

    results["mlp_ablation"] = mlp_ablation_results
    return {"mean_ablation": results}


def run_attention_pattern_analysis(
    model: HookedTransformer,
    problems: List[Dict],
    device: torch.device,
    routing_layers: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Analyze attention patterns to find routing heads.

    At layers L9-L15 (the routing zone from position patching), which heads
    attend from the '=' position to the operand positions?
    These are the "information routing" heads that copy operand values
    to the answer position.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    if routing_layers is None:
        routing_layers = list(range(8, 16))

    print(f"\n{'=' * 70}")
    print(f"STAGE 3b: ATTENTION PATTERN ANALYSIS — Which heads route operands?")
    print(f"{'=' * 70}")
    print(f"  Routing layers: {routing_layers}")

    # For each problem, extract attention patterns at routing layers
    hook_names = [f"blocks.{L}.attn.hook_pattern" for L in routing_layers]

    # Accumulate: for each head, how much does '=' attend to operand_a?
    head_attn_to_a = defaultdict(list)     # (L, H) -> list of attention weights
    head_attn_to_b = defaultdict(list)
    head_attn_to_plus = defaultdict(list)

    for prob in problems[:30]:
        tokens = model.to_tokens(prob["clean_prompt"])
        str_tokens = model.to_str_tokens(prob["clean_prompt"])
        last_pos = tokens.shape[1] - 1

        # Find positions
        plus_pos = None
        for idx, tok in enumerate(str_tokens):
            if tok.strip() == '+':
                plus_pos = idx
        if plus_pos is None:
            continue

        operand_a_pos = plus_pos - 1
        operand_b_pos = plus_pos + 1
        equals_pos = last_pos  # '=' is the last token

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)

        for L in routing_layers:
            hook_name = f"blocks.{L}.attn.hook_pattern"
            if hook_name not in cache:
                continue
            # pattern shape: [batch, n_heads, seq_q, seq_k]
            pattern = cache[hook_name][0].float().cpu()  # [n_heads, seq, seq]

            for H in range(n_heads):
                # Attention from equals_pos (query) to various key positions
                attn_to_a = pattern[H, equals_pos, operand_a_pos].item()
                attn_to_b = pattern[H, equals_pos, operand_b_pos].item()
                attn_to_plus = pattern[H, equals_pos, plus_pos].item()

                head_attn_to_a[(L, H)].append(attn_to_a)
                head_attn_to_b[(L, H)].append(attn_to_b)
                head_attn_to_plus[(L, H)].append(attn_to_plus)

        del cache

    # Find routing heads: those that consistently attend from '=' to operand_a
    print(f"\n  Heads attending from '=' to operand_a (sorted by mean attention):")
    print(f"  {'Layer':>6}  {'Head':>5}  {'→op_a':>8}  {'→op_b':>8}  {'→plus':>8}  {'Verdict'}")
    print(f"  {'─' * 60}")

    routing_summary = []
    for (L, H) in sorted(head_attn_to_a.keys()):
        mean_a = np.mean(head_attn_to_a[(L, H)])
        mean_b = np.mean(head_attn_to_b[(L, H)])
        mean_plus = np.mean(head_attn_to_plus[(L, H)])
        routing_summary.append((L, H, mean_a, mean_b, mean_plus))

    # Sort by attention to operand_a
    routing_summary.sort(key=lambda x: x[2], reverse=True)

    results = {}
    for L, H, mean_a, mean_b, mean_plus in routing_summary[:40]:
        verdict = ""
        if mean_a > 0.10:
            verdict = "ROUTES_A"
        if mean_b > 0.10:
            verdict += " ROUTES_B" if verdict else "ROUTES_B"
        if mean_a > 0.20:
            verdict = "★ STRONG_ROUTER_A"
        if not verdict:
            verdict = "—"

        print(f"  L{L:>4d}  H{H:>4d}  {mean_a:8.4f}  {mean_b:8.4f}  {mean_plus:8.4f}  {verdict}")

        results[f"L{L}_H{H}"] = {
            "layer": L, "head": H,
            "attn_to_operand_a": float(mean_a),
            "attn_to_operand_b": float(mean_b),
            "attn_to_plus": float(mean_plus),
        }

    return {"attention_patterns": results}


def run_position_aware_head_patching(
    model: HookedTransformer,
    problems: List[Dict],
    device: torch.device,
    target_layers: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Patch attention heads at the OPERAND position, not just the final position.

    This catches routing heads that copy operand information.
    At layers L0-L15, patch each head's output at the operand_a position
    from clean into corrupted run.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    if target_layers is None:
        target_layers = list(range(0, 16))

    print(f"\n{'=' * 70}")
    print(f"STAGE 3c: POSITION-AWARE HEAD PATCHING — Routing heads at operand_a")
    print(f"{'=' * 70}")
    print(f"  Patching at operand_a position for layers {target_layers[0]}-{target_layers[-1]}")

    hook_names = [f"blocks.{L}.attn.hook_z" for L in target_layers]

    head_recovery = defaultdict(list)

    n_done = 0
    for prob in problems:
        clean_tokens = model.to_tokens(prob["clean_prompt"])
        corrupt_tokens = model.to_tokens(prob["corrupt_prompt"])
        answer_tok = get_answer_token_id(model, prob["answer"])

        if clean_tokens.shape[1] != corrupt_tokens.shape[1]:
            continue

        last_pos = clean_tokens.shape[1] - 1

        # Find operand_a position
        str_tokens = model.to_str_tokens(prob["clean_prompt"])
        plus_pos = None
        for idx, tok in enumerate(str_tokens):
            if tok.strip() == '+':
                plus_pos = idx
        if plus_pos is None or plus_pos < 1:
            continue
        operand_a_pos = plus_pos - 1

        with torch.no_grad():
            clean_logits, clean_cache = model.run_with_cache(
                clean_tokens, names_filter=hook_names
            )
            corrupt_logits, _ = model.run_with_cache(
                corrupt_tokens, names_filter=[]
            )

        clean_logit = clean_logits[0, last_pos, answer_tok].item()
        corrupt_logit = corrupt_logits[0, last_pos, answer_tok].item()
        logit_diff = clean_logit - corrupt_logit

        if abs(logit_diff) < 0.1:
            del clean_cache
            continue

        for L in target_layers:
            hook_name = f"blocks.{L}.attn.hook_z"
            if hook_name not in clean_cache:
                continue
            clean_attn = clean_cache[hook_name].detach().clone()

            for H in range(n_heads):
                def make_head_hook(clean_act, pos, head_idx):
                    def hook_fn(value, hook):
                        value[0, pos, head_idx, :] = clean_act[0, pos, head_idx, :]
                        return value
                    return hook_fn

                patch_hook = make_head_hook(clean_attn, operand_a_pos, H)

                with torch.no_grad():
                    with model.hooks(fwd_hooks=[(hook_name, patch_hook)]):
                        patched_logits = model(corrupt_tokens)

                patched_logit = patched_logits[0, last_pos, answer_tok].item()
                recovery = (patched_logit - corrupt_logit) / logit_diff
                head_recovery[(L, H)].append(recovery)

        del clean_cache
        n_done += 1
        if n_done % 5 == 0:
            print(f"  Processed {n_done}/{len(problems)} problems...")

    # Summarize
    print(f"\n  Top heads by recovery when patched at operand_a position:")
    print(f"  {'Layer':>6}  {'Head':>5}  {'MeanRecov':>10}  {'StdRecov':>9}")
    print(f"  {'─' * 45}")

    head_summary = []
    results = {}
    for (L, H), recoveries in head_recovery.items():
        mean_r = np.mean(recoveries)
        std_r = np.std(recoveries)
        head_summary.append((L, H, mean_r, std_r))
        results[f"L{L}_H{H}"] = {
            "layer": L, "head": H,
            "mean_recovery": float(mean_r),
            "std_recovery": float(std_r),
            "n_samples": len(recoveries),
        }

    head_summary.sort(key=lambda x: x[2], reverse=True)

    for L, H, mean_r, std_r in head_summary[:30]:
        marker = ""
        if mean_r > 0.02:
            marker = " ◀ ROUTING"
        if mean_r > 0.05:
            marker = " ★ KEY ROUTER"
        print(f"  L{L:>4d}  H{H:>4d}  {mean_r:10.4f}  {std_r:9.4f}{marker}")

    return {"operand_head_patching": results}


# ═════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═════════════════════════════════════════════════════════════

def run_full_pipeline(
    model_name: str = DEFAULT_MODEL,
    stage: Optional[int] = None,
    n_problems_stage1: int = 40,
    n_problems_stage2: int = 20,
    max_operand: int = 50,
    output_dir: str = OUTPUT_DIR,
    device_override: Optional[str] = None,
    dtype_override: Optional[str] = None,
):
    """Run the full arithmetic circuit discovery pipeline."""
    # Resolve model name via registry if possible
    resolved_name = resolve_model_name(model_name)

    # Pick device + dtype with model-aware fallbacks
    device, dtype = auto_device_dtype(resolved_name, device_override, dtype_override)

    print(f"\n{'═' * 70}")
    print(f"ARITHMETIC CIRCUIT DISCOVERY")
    print(f"{'═' * 70}")
    print(f"  Model: {model_name}")
    if resolved_name != model_name:
        print(f"  Resolved: {resolved_name}")
    print(f"  Device: {device}  dtype: {dtype}")
    print(f"  Stage: {'all' if stage is None else stage}")
    print(f"  Operand range: 1–{max_operand}")

    # Load model
    print(f"\nLoading model...")
    t0 = time.time()
    model = HookedTransformer.from_pretrained(
        resolved_name, device=device, dtype=dtype
    )
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s: "
          f"{model.cfg.n_layers}L, {model.cfg.n_heads}H, "
          f"d_model={model.cfg.d_model}")

    # Generate problems
    problems_s1 = generate_arithmetic_prompts(
        n_problems=n_problems_stage1, max_operand=max_operand, seed=42
    )
    problems_s2 = generate_arithmetic_prompts(
        n_problems=n_problems_stage2, max_operand=max_operand, seed=123
    )

    all_results = {
        "model": model_name,
        "n_layers": model.cfg.n_layers,
        "n_heads": model.cfg.n_heads,
        "d_model": model.cfg.d_model,
        "max_operand": max_operand,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── STAGE 1 ──
    if stage is None or stage == 1:
        t1 = time.time()

        # 1a: Logit Lens
        logit_results = run_logit_lens(model, problems_s1, device)
        all_results.update(logit_results)

        # 1b: Layer-level activation patching
        layer_patch_results = run_layer_activation_patching(
            model, problems_s1, device
        )
        all_results.update(layer_patch_results)

        # 1c: Position-aware patching (smaller subset)
        position_results = run_layer_patching_at_all_positions(
            model, problems_s1[:20], device
        )
        all_results.update(position_results)

        print(f"\n  Stage 1 completed in {time.time() - t1:.1f}s")

        # Save intermediate results
        s1_path = Path(output_dir) / "stage1_results.json"
        with open(s1_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Stage 1 saved: {s1_path}")

    # ── STAGE 2 ──
    if stage is None or stage == 2:
        t2 = time.time()

        # If we have Stage 1 results, focus on critical layers
        if "layer_patching" in all_results:
            critical_layers = identify_critical_layers(all_results)
        else:
            # Try loading Stage 1 results
            s1_path = Path(output_dir) / "stage1_results.json"
            if s1_path.exists():
                with open(s1_path) as f:
                    s1_data = json.load(f)
                critical_layers = identify_critical_layers(s1_data)
            else:
                print("  No Stage 1 results — scanning all layers (slower)")
                critical_layers = list(range(model.cfg.n_layers))

        component_results = run_component_patching(
            model, problems_s2, device,
            target_layers=critical_layers,
        )
        all_results.update(component_results)

        print(f"\n  Stage 2 completed in {time.time() - t2:.1f}s")

        # Save Stage 2 results
        s2_path = Path(output_dir) / "stage2_results.json"
        with open(s2_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Stage 2 saved: {s2_path}")

    # ── STAGE 3 ──
    if stage is None or stage == 3:
        t3 = time.time()

        # Load Stage 2 results if not already in memory
        if "top_heads" not in all_results:
            s2_path = Path(output_dir) / "stage2_results.json"
            disc_path = Path(output_dir) / "discovery_results.json"
            for p in [s2_path, disc_path]:
                if p.exists():
                    with open(p) as f:
                        s2_data = json.load(f)
                    if "top_heads" in s2_data:
                        all_results.update(s2_data)
                        break

        top_heads = all_results.get("top_heads", [])
        top_mlps = all_results.get("top_mlps", [])

        if not top_heads and not top_mlps:
            print("  WARNING: No Stage 2 results found. Run --stage 2 first.")
        else:
            # Generate problems for Stage 3
            problems_s3 = generate_arithmetic_prompts(
                n_problems=40, max_operand=max_operand, seed=456
            )

            # 3a: Mean ablation (necessity testing)
            print("\n  Computing mean activations for ablation baseline...")
            mean_acts = compute_mean_activations(
                model, n_prompts=60, max_operand=max_operand
            )
            print(f"  Mean activations computed for {len(mean_acts)} hooks")

            ablation_results = run_mean_ablation(
                model, problems_s3, device,
                top_heads=[(L, H, r) for L, H, r in top_heads],
                top_mlps=[(L, r) for L, r in top_mlps],
                mean_acts=mean_acts,
            )
            all_results.update(ablation_results)

            del mean_acts
            gc.collect()

            # 3b: Attention pattern analysis (routing heads)
            attn_results = run_attention_pattern_analysis(
                model, problems_s3, device,
                routing_layers=list(range(8, 20)),
            )
            all_results.update(attn_results)

            # 3c: Position-aware head patching at operand_a
            operand_patch_results = run_position_aware_head_patching(
                model, problems_s3[:15], device,
                target_layers=list(range(0, 16)),
            )
            all_results.update(operand_patch_results)

        print(f"\n  Stage 3 completed in {time.time() - t3:.1f}s")

    # ── Save final results ──
    final_path = Path(output_dir) / "discovery_results.json"
    with open(final_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Final results saved: {final_path}")

    # ── Print final summary ──
    print(f"\n{'═' * 70}")
    print(f"DISCOVERY COMPLETE")
    print(f"{'═' * 70}")

    if "top_heads" in all_results:
        print(f"\n  Top 10 causal attention heads:")
        for L, H, r in all_results["top_heads"][:10]:
            print(f"    L{L} H{H}: {r:.4f}")

    if "top_mlps" in all_results:
        print(f"\n  Top 5 causal MLP layers:")
        for L, r in all_results["top_mlps"][:5]:
            print(f"    L{L} MLP: {r:.4f}")

    if "mean_ablation" in all_results:
        ma = all_results["mean_ablation"]
        print(f"\n  Baseline accuracy: {ma.get('baseline_accuracy', 0):.1%}")
        ha = ma.get("head_ablation", {})
        critical_heads = [(k, v) for k, v in ha.items() if v.get("accuracy_drop", 0) > 0.05]
        if critical_heads:
            print(f"  NECESSARY heads (drop > 5%):")
            for k, v in sorted(critical_heads, key=lambda x: x[1]["accuracy_drop"], reverse=True):
                print(f"    {k}: drop={v['accuracy_drop']:+.1%}")
        ma_mlp = ma.get("mlp_ablation", {})
        critical_mlps = [(k, v) for k, v in ma_mlp.items() if v.get("accuracy_drop", 0) > 0.05]
        if critical_mlps:
            print(f"  NECESSARY MLPs (drop > 5%):")
            for k, v in sorted(critical_mlps, key=lambda x: x[1]["accuracy_drop"], reverse=True):
                print(f"    {k}: drop={v['accuracy_drop']:+.1%}")

    return all_results


# ═════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arithmetic Circuit Discovery — Unified Causal Pipeline"
    )
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", type=str, default=None,
                             help="Raw TransformerLens / HuggingFace model name")
    model_group.add_argument("--model-key", type=str, default=None,
                             help="Model registry key (e.g. 'pythia-1.4b', 'phi-3-mini')")
    parser.add_argument("--stage", type=int, default=None, choices=[1, 2, 3],
                        help="Run only this stage (default: all)")
    parser.add_argument("--n-problems-s1", type=int, default=40,
                        help="Number of problems for Stage 1")
    parser.add_argument("--n-problems-s2", type=int, default=20,
                        help="Number of problems for Stage 2")
    parser.add_argument("--operand-range", type=int, default=50,
                        help="Maximum operand value")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Output directory")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cpu, cuda, mps)")
    parser.add_argument("--dtype", type=str, default=None,
                        choices=["float32", "float16", "bfloat16"],
                        help="Force dtype")
    args = parser.parse_args()

    # Determine model name: --model-key takes priority, then --model, then default
    model_name = args.model_key or args.model or DEFAULT_MODEL

    run_full_pipeline(
        model_name=model_name,
        stage=args.stage,
        n_problems_stage1=args.n_problems_s1,
        n_problems_stage2=args.n_problems_s2,
        max_operand=args.operand_range,
        output_dir=args.output_dir,
        device_override=args.device,
        dtype_override=args.dtype,
    )
