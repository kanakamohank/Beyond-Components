#!/usr/bin/env python3
"""
Mathematical Toolkit for Arithmetic Circuit Discovery

Five novel mathematical approaches to crack the addition circuit mystery:
  1. Fisher Information Geometry  — find the TRUE causal subspace
  2. Independent Component Analysis — resolve superposition without training
  3. CP Tensor Decomposition       — capture bilinear structure of f(a,b)=a+b
  4. Persistent Homology (TDA)     — basis-free topology detection + CRT test
  5. Wasserstein Geometry           — distributional structure across layers

Each approach addresses a specific failure mode of SVD/Fourier/PCA.
See MATHEMATICAL_TOOLKIT_PROPOSAL.md for full theoretical motivation.

Usage:
    python experiments/mathematical_toolkit.py --model phi-3-mini --analysis all
    python experiments/mathematical_toolkit.py --model gemma-2b --analysis fisher
    python experiments/mathematical_toolkit.py --analysis ica --operand-range 30
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import gc
import random
import time
import argparse
import logging
import warnings
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

# Core scientific computing
from scipy import stats
from scipy.linalg import svd as scipy_svd
from scipy.spatial.distance import pdist, squareform

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════
# DEVICE / MODEL UTILITIES (reused from arithmetic_circuit_discovery.py)
# ═════════════════════════════════════════════════════════════

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_model_name(key: str) -> str:
    MODEL_MAP = {
        "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
        "gemma-2b": "google/gemma-2-2b",
        "gemma-7b": "google/gemma-7b",
        "gpt2-small": "gpt2",
        "gpt2-medium": "gpt2-medium",
        "pythia-1.4b": "EleutherAI/pythia-1.4b",
        "pythia-6.9b": "EleutherAI/pythia-6.9b",
        "llama-3b": "meta-llama/Llama-3.2-3B",
    }
    return MODEL_MAP.get(key, key)


# ═════════════════════════════════════════════════════════════
# DATA GENERATION
# ═════════════════════════════════════════════════════════════

def generate_grid_prompts(
    max_operand: int = 30,
    few_shot: bool = True,
) -> List[Dict[str, Any]]:
    """Generate a FULL GRID of (a, b) pairs for tensor analysis.

    Unlike random sampling, the full grid is essential for:
    - Tensor decomposition (need complete T[a, b, d] tensor)
    - CRT analysis (need all residue classes represented)
    - Wasserstein (need enough samples per digit class)
    """
    few_shot_prefix = ""
    if few_shot:
        few_shot_prefix = "Calculate:\n12 + 7 = 19\n34 + 15 = 49\n"

    problems = []
    for a in range(max_operand):
        for b in range(max_operand):
            answer = a + b
            prompt = f"{few_shot_prefix}{a} + {b} ="
            problems.append({
                "a": a, "b": b, "answer": answer,
                "prompt": prompt,
                "ones_digit": answer % 10,
                "tens_digit": (answer // 10) % 10,
                "carry": 1 if (a % 10 + b % 10) >= 10 else 0,
                "a_mod2": a % 2, "b_mod2": b % 2,
                "a_mod5": a % 5, "b_mod5": b % 5,
                "ans_mod2": answer % 2, "ans_mod5": answer % 5,
            })
    return problems


def get_answer_token_id(model, answer: int) -> int:
    answer_str = " " + str(answer)
    tokens = model.to_tokens(answer_str, prepend_bos=False)
    return tokens[0, 0].item()


# ═════════════════════════════════════════════════════════════
# ACTIVATION COLLECTION (shared across all analyses)
# ═════════════════════════════════════════════════════════════

def collect_activations(
    model,
    problems: List[Dict],
    layers: List[int],
    position: str = "final",
    batch_size: int = 8,
    max_problems: Optional[int] = None,
) -> Dict[int, np.ndarray]:
    """Collect MLP output activations at specified layers.

    Args:
        model: HookedTransformer model
        problems: list of problem dicts with 'prompt' key
        layers: which layers to collect from
        position: 'final' (last token) or 'operand' (token before +)
        batch_size: batch size for inference
        max_problems: limit number of problems (None = all)

    Returns:
        Dict mapping layer -> np.ndarray of shape (n_problems, d_model)
    """
    if max_problems:
        problems = problems[:max_problems]

    hook_names = [f"blocks.{l}.hook_mlp_out" for l in layers]
    activations = {l: [] for l in layers}

    n_batches = (len(problems) + batch_size - 1) // batch_size
    logger.info(f"Collecting activations from {len(problems)} problems, {len(layers)} layers, position='{position}'")

    for batch_idx in range(n_batches):
        batch = problems[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        prompts = [p["prompt"] for p in batch]

        tokens = model.to_tokens(prompts, prepend_bos=True)

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)

        # Pre-compute plus positions if needed
        plus_positions = None
        if position == "operand":
            plus_positions = []
            for i, prompt in enumerate(prompts):
                toks = model.to_tokens(prompt, prepend_bos=True)
                tok_strs = [model.tokenizer.decode([t]) for t in toks[0]]
                plus_pos = None
                for j, s in enumerate(tok_strs):
                    if '+' in s:
                        plus_pos = j
                        break
                if plus_pos is None or plus_pos == 0:
                    plus_pos = len(tok_strs) - 3
                plus_positions.append(max(plus_pos - 1, 0))

        for l in layers:
            hook = f"blocks.{l}.hook_mlp_out"
            acts = cache[hook]  # (batch, seq_len, d_model)

            if position == "operand" and plus_positions is not None:
                for i in range(len(prompts)):
                    activations[l].append(acts[i, plus_positions[i], :].cpu().float().numpy())
            else:
                # "final" or fallback: last token
                for i in range(acts.shape[0]):
                    activations[l].append(acts[i, -1, :].cpu().float().numpy())

        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (batch_idx + 1) % 10 == 0:
            logger.info(f"  Batch {batch_idx + 1}/{n_batches}")

    result = {}
    for l in layers:
        result[l] = np.array(activations[l], dtype=np.float32)
        logger.info(f"  Layer {l}: shape {result[l].shape}")

    return result


def collect_residual_activations(
    model,
    problems: List[Dict],
    layers: List[int],
    position: str = "final",
    batch_size: int = 8,
    max_problems: Optional[int] = None,
) -> Dict[int, np.ndarray]:
    """Collect residual stream activations (hook_resid_post) at specified layers."""
    if max_problems:
        problems = problems[:max_problems]

    hook_names = [f"blocks.{l}.hook_resid_post" for l in layers]
    activations = {l: [] for l in layers}

    n_batches = (len(problems) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        batch = problems[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        prompts = [p["prompt"] for p in batch]
        tokens = model.to_tokens(prompts, prepend_bos=True)

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)

        for l in layers:
            hook = f"blocks.{l}.hook_resid_post"
            acts = cache[hook]
            pos_acts = acts[:, -1, :]  # last token by default

            for i in range(pos_acts.shape[0]):
                activations[l].append(pos_acts[i].cpu().float().numpy())

        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {l: np.array(activations[l], dtype=np.float32) for l in layers}


# ═════════════════════════════════════════════════════════════
# ANALYSIS 1: FISHER INFORMATION GEOMETRY
# ═════════════════════════════════════════════════════════════

def fisher_information_analysis(
    model,
    problems: List[Dict],
    layers: List[int],
    n_problems: int = 200,
    top_k: int = 30,
) -> Dict[str, Any]:
    """Compute Fisher Information Matrix to find the true causal subspace.

    The Fisher IM eigenvectors are the directions that maximally affect
    the probability of the correct answer token. Unlike SVD directions
    (which maximize variance) or PCA (which finds orthogonal components),
    Fisher directions are guaranteed to be on the causal path.

    Returns:
        Dict with eigenvalues, eigenvectors, and analysis per layer.
    """
    logger.info("=" * 60)
    logger.info("ANALYSIS 1: FISHER INFORMATION GEOMETRY")
    logger.info("=" * 60)

    device = next(model.parameters()).device
    subset = problems[:n_problems]
    results = {}

    for layer in layers:
        logger.info(f"\n--- Layer {layer} ---")
        hook_name = f"blocks.{layer}.hook_resid_post"

        # Accumulate outer products of gradients
        d_model = model.cfg.d_model
        fisher_matrix = np.zeros((d_model, d_model), dtype=np.float64)
        grad_norms = []
        n_valid = 0

        for i, prob in enumerate(subset):
            tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
            answer_tok = get_answer_token_id(model, prob["answer"])

            # We need gradients w.r.t. the residual stream at this layer
            # Use a hook to capture and enable grad on the activation
            activation_holder = {}

            def capture_hook(act, hook, holder=activation_holder):
                holder['act'] = act
                act.requires_grad_(True)
                act.retain_grad()
                return act

            try:
                with model.hooks(fwd_hooks=[(hook_name, capture_hook)]):
                    logits = model(tokens)

                # Log probability of correct answer at last position
                log_probs = F.log_softmax(logits[0, -1], dim=-1)
                log_p = log_probs[answer_tok]

                # Gradient of log p(correct) w.r.t. hidden state
                log_p.backward(retain_graph=False)

                if 'act' in activation_holder and activation_holder['act'].grad is not None:
                    # Gradient at the last token position
                    g = activation_holder['act'].grad[0, -1].detach().cpu().float().numpy()
                    grad_norms.append(np.linalg.norm(g))
                    # Outer product accumulation (rank-1 update)
                    fisher_matrix += np.outer(g, g)
                    n_valid += 1

            except Exception as e:
                logger.debug(f"  Problem {i} failed: {e}")
                continue
            finally:
                model.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if (i + 1) % 50 == 0:
                logger.info(f"  Processed {i + 1}/{len(subset)} problems ({n_valid} valid)")

        if n_valid < 10:
            logger.warning(f"  Only {n_valid} valid gradients — skipping layer {layer}")
            continue

        fisher_matrix /= n_valid

        # Eigendecomposition (symmetric positive semidefinite)
        eigenvalues, eigenvectors = np.linalg.eigh(fisher_matrix)
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Analysis
        total_info = eigenvalues.sum()
        cumulative = np.cumsum(eigenvalues) / total_info if total_info > 0 else np.zeros_like(eigenvalues)
        dims_90 = int(np.searchsorted(cumulative, 0.90)) + 1
        dims_95 = int(np.searchsorted(cumulative, 0.95)) + 1
        dims_99 = int(np.searchsorted(cumulative, 0.99)) + 1

        # Effective dimension (participation ratio)
        if total_info > 0:
            p = eigenvalues / total_info
            p = p[p > 1e-10]
            eff_dim = np.exp(-np.sum(p * np.log(p)))
        else:
            eff_dim = 0

        logger.info(f"  Valid gradients: {n_valid}")
        logger.info(f"  Mean gradient norm: {np.mean(grad_norms):.4f}")
        logger.info(f"  Top 5 eigenvalues: {eigenvalues[:5]}")
        logger.info(f"  Dims for 90% info: {dims_90}")
        logger.info(f"  Dims for 95% info: {dims_95}")
        logger.info(f"  Dims for 99% info: {dims_99}")
        logger.info(f"  Effective dimension: {eff_dim:.1f}")

        results[layer] = {
            "eigenvalues": eigenvalues[:top_k].tolist(),
            "eigenvectors_topk": eigenvectors[:, :top_k],  # (d_model, top_k) — not serialized
            "dims_90": dims_90,
            "dims_95": dims_95,
            "dims_99": dims_99,
            "effective_dim": float(eff_dim),
            "n_valid": n_valid,
            "mean_grad_norm": float(np.mean(grad_norms)),
            "cumulative_variance_topk": cumulative[:top_k].tolist(),
        }

    return {"fisher": results}


# ═════════════════════════════════════════════════════════════
# ANALYSIS 2: INDEPENDENT COMPONENT ANALYSIS
# ═════════════════════════════════════════════════════════════

def ica_analysis(
    activations: Dict[int, np.ndarray],
    problems: List[Dict],
    n_components: int = 30,
) -> Dict[str, Any]:
    """Apply ICA to MLP activations to find statistically independent features.

    Unlike PCA (orthogonal, variance-maximizing), ICA finds directions that
    are statistically independent — exactly what superposition creates.

    Tests each IC against arithmetic properties:
    - ones digit, tens digit, carry bit
    - CRT components (mod 2, mod 5)
    - individual operands a, b
    """
    from sklearn.decomposition import FastICA, PCA

    logger.info("=" * 60)
    logger.info("ANALYSIS 2: INDEPENDENT COMPONENT ANALYSIS")
    logger.info("=" * 60)

    # Build label arrays from problems
    n = len(problems)
    labels = {
        "ones_digit": np.array([p["ones_digit"] for p in problems[:n]]),
        "tens_digit": np.array([p["tens_digit"] for p in problems[:n]]),
        "carry": np.array([p["carry"] for p in problems[:n]]),
        "ans_mod2": np.array([p["ans_mod2"] for p in problems[:n]]),
        "ans_mod5": np.array([p["ans_mod5"] for p in problems[:n]]),
        "a_mod2": np.array([p["a_mod2"] for p in problems[:n]]),
        "b_mod2": np.array([p["b_mod2"] for p in problems[:n]]),
        "a_mod5": np.array([p["a_mod5"] for p in problems[:n]]),
        "b_mod5": np.array([p["b_mod5"] for p in problems[:n]]),
        "a": np.array([p["a"] for p in problems[:n]]),
        "b": np.array([p["b"] for p in problems[:n]]),
        "answer": np.array([p["answer"] for p in problems[:n]]),
    }

    results = {}

    for layer, acts in activations.items():
        logger.info(f"\n--- Layer {layer} ---")
        acts_layer = acts[:n]

        # Step 1: PCA pre-reduction (ICA is cubic in dimensionality)
        n_pca = min(100, acts_layer.shape[1], acts_layer.shape[0] - 1)
        pca = PCA(n_components=n_pca, random_state=42)
        acts_pca = pca.fit_transform(acts_layer)
        pca_var_explained = pca.explained_variance_ratio_.sum()
        logger.info(f"  PCA pre-reduction: {acts_layer.shape[1]} -> {n_pca} dims ({pca_var_explained:.1%} var)")

        # Step 2: FastICA
        n_ic = min(n_components, n_pca)
        try:
            ica = FastICA(n_components=n_ic, random_state=42, max_iter=500, tol=1e-4)
            sources = ica.fit_transform(acts_pca)  # (n_problems, n_ic)
            logger.info(f"  ICA: extracted {n_ic} independent components")
        except Exception as e:
            logger.warning(f"  ICA failed: {e}")
            continue

        # Step 3: Correlate each IC with arithmetic labels
        ic_correlations = {}
        for ic_idx in range(n_ic):
            ic_signal = sources[:, ic_idx]
            corrs = {}
            for label_name, label_vals in labels.items():
                if label_vals.dtype in [np.int64, np.int32, np.float64, np.float32]:
                    r, p_val = stats.spearmanr(ic_signal, label_vals[:len(ic_signal)])
                    corrs[label_name] = {"r": float(r), "p": float(p_val)}
            ic_correlations[ic_idx] = corrs

        # Step 4: Find best IC for each arithmetic property
        best_ics = {}
        for label_name in labels:
            best_r = 0
            best_ic = -1
            for ic_idx, corrs in ic_correlations.items():
                if label_name in corrs and abs(corrs[label_name]["r"]) > abs(best_r):
                    best_r = corrs[label_name]["r"]
                    best_ic = ic_idx
            best_ics[label_name] = {"ic": best_ic, "r": float(best_r)}
            if abs(best_r) > 0.3:
                logger.info(f"  {label_name:12s} -> IC {best_ic:2d} (r = {best_r:+.3f})")

        # Step 5: Mutual information between ICs (are they truly independent?)
        # Use discretized mutual info as a sanity check
        from sklearn.metrics import mutual_info_score
        mi_matrix = np.zeros((n_ic, n_ic))
        for i in range(n_ic):
            for j in range(i + 1, n_ic):
                # Discretize into 20 bins for MI estimation
                si = np.digitize(sources[:, i], np.linspace(sources[:, i].min(), sources[:, i].max(), 20))
                sj = np.digitize(sources[:, j], np.linspace(sources[:, j].min(), sources[:, j].max(), 20))
                mi = mutual_info_score(si, sj)
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi

        mean_mi = mi_matrix[np.triu_indices(n_ic, k=1)].mean()
        logger.info(f"  Mean pairwise MI: {mean_mi:.4f} (lower = more independent)")

        # Step 6: Compare PCA vs ICA separability for key features
        # For each label, measure how well the best PCA component vs best IC separates it
        pca_best = {}
        for label_name, label_vals in labels.items():
            best_r_pca = 0
            for pc_idx in range(min(n_ic, acts_pca.shape[1])):
                r, _ = stats.spearmanr(acts_pca[:, pc_idx], label_vals[:len(acts_pca)])
                if abs(r) > abs(best_r_pca):
                    best_r_pca = r
            pca_best[label_name] = float(best_r_pca)

        # Report ICA advantage
        logger.info(f"\n  ICA vs PCA comparison (|Spearman r|):")
        logger.info(f"  {'Property':12s} | {'ICA':>8s} | {'PCA':>8s} | {'Δ':>8s}")
        logger.info(f"  {'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
        for label_name in ["carry", "ones_digit", "tens_digit", "ans_mod2", "ans_mod5"]:
            r_ica = abs(best_ics[label_name]["r"])
            r_pca = abs(pca_best[label_name])
            delta = r_ica - r_pca
            marker = " ★" if delta > 0.05 else ""
            logger.info(f"  {label_name:12s} | {r_ica:8.3f} | {r_pca:8.3f} | {delta:+8.3f}{marker}")

        results[layer] = {
            "n_components": n_ic,
            "pca_var_explained": float(pca_var_explained),
            "best_ics": best_ics,
            "pca_best": pca_best,
            "mean_mutual_info": float(mean_mi),
            "ic_correlations_top5": {
                k: {lbl: v for lbl, v in sorted(corrs.items(), key=lambda x: -abs(x[1]["r"]))[:5]}
                for k, corrs in ic_correlations.items()
            },
        }

    return {"ica": results}


# ═════════════════════════════════════════════════════════════
# ANALYSIS 3: CP TENSOR DECOMPOSITION
# ═════════════════════════════════════════════════════════════

def tensor_decomposition_analysis(
    activations: Dict[int, np.ndarray],
    problems: List[Dict],
    max_operand: int = 30,
    max_rank: int = 15,
    n_pca_dims: int = 50,
) -> Dict[str, Any]:
    """CP tensor decomposition to capture the bilinear structure of f(a,b)=a+b.

    Forms tensor T[a, b, d] from MLP activations and decomposes it into
    rank-1 components. Each component (u_r, v_r, w_r) represents an
    independent computational channel where:
    - u_r encodes information about operand a
    - v_r encodes information about operand b
    - w_r identifies which activation dimensions this channel uses
    """
    try:
        import tensorly as tl
        from tensorly.decomposition import parafac
        tl.set_backend('numpy')
    except ImportError:
        logger.error("tensorly not installed. Run: pip install tensorly")
        return {"tensor": {"error": "tensorly not installed"}}

    logger.info("=" * 60)
    logger.info("ANALYSIS 3: CP TENSOR DECOMPOSITION")
    logger.info("=" * 60)

    results = {}

    for layer, acts in activations.items():
        logger.info(f"\n--- Layer {layer} ---")

        # Build the 3-way tensor T[a, b, d]
        # First, PCA reduce dimensionality for tractability
        from sklearn.decomposition import PCA
        n_dims = min(n_pca_dims, acts.shape[1], acts.shape[0] - 1)
        pca = PCA(n_components=n_dims, random_state=42)
        acts_reduced = pca.fit_transform(acts)

        # Map (a, b) -> index in problems list
        ab_to_idx = {}
        for idx, p in enumerate(problems):
            if idx < len(acts):
                ab_to_idx[(p["a"], p["b"])] = idx

        # Build tensor
        A_range = max_operand
        B_range = max_operand
        tensor = np.zeros((A_range, B_range, n_dims), dtype=np.float32)
        mask = np.zeros((A_range, B_range), dtype=bool)

        for a in range(A_range):
            for b in range(B_range):
                if (a, b) in ab_to_idx:
                    idx = ab_to_idx[(a, b)]
                    tensor[a, b, :] = acts_reduced[idx]
                    mask[a, b] = True

        coverage = mask.sum() / mask.size
        logger.info(f"  Tensor shape: {tensor.shape}, coverage: {coverage:.1%}")

        if coverage < 0.5:
            logger.warning(f"  Low coverage ({coverage:.1%}), skipping")
            continue

        # CP decomposition at multiple ranks
        rank_results = []
        for rank in [2, 4, 6, 8, 10, max_rank]:
            try:
                weights, factors = parafac(
                    tl.tensor(tensor), rank=rank,
                    init='random', random_state=42,
                    n_iter_max=200, tol=1e-6,
                )
                # factors = [u (A_range, rank), v (B_range, rank), w (n_dims, rank)]
                u, v, w = factors

                # Reconstruct and compute fit
                recon = tl.cp_to_tensor((weights, factors))
                residual = np.linalg.norm(tensor - recon) / np.linalg.norm(tensor)
                fit = 1 - residual

                # Analyze each rank-1 component
                component_analysis = []
                for r in range(rank):
                    ur = u[:, r]  # function of a
                    vr = v[:, r]  # function of b
                    wr = w[:, r]  # activation dimensions

                    # FFT of u_r (function of a) to find periodicity
                    u_fft = np.abs(np.fft.rfft(ur - ur.mean()))
                    u_freqs = np.fft.rfftfreq(len(ur), d=1.0)
                    u_dominant_freq = u_freqs[1:][np.argmax(u_fft[1:])]
                    u_dominant_period = 1.0 / u_dominant_freq if u_dominant_freq > 0 else np.inf
                    u_fft_strength = u_fft[1:].max() / (u_fft[1:].mean() + 1e-10)

                    # FFT of v_r (function of b)
                    v_fft = np.abs(np.fft.rfft(vr - vr.mean()))
                    v_freqs = np.fft.rfftfreq(len(vr), d=1.0)
                    v_dominant_freq = v_freqs[1:][np.argmax(v_fft[1:])]
                    v_dominant_period = 1.0 / v_dominant_freq if v_dominant_freq > 0 else np.inf
                    v_fft_strength = v_fft[1:].max() / (v_fft[1:].mean() + 1e-10)

                    # Check CRT frequencies: T=2 (mod 2) and T=5 (mod 5)
                    u_t2_power = 0
                    u_t5_power = 0
                    u_t10_power = 0
                    if len(ur) >= 10:
                        # Power at frequency 1/2 (every other number)
                        f2_idx = np.argmin(np.abs(u_freqs - 0.5))
                        f5_idx = np.argmin(np.abs(u_freqs - 0.2))
                        f10_idx = np.argmin(np.abs(u_freqs - 0.1))
                        u_t2_power = float(u_fft[f2_idx] / (u_fft.sum() + 1e-10))
                        u_t5_power = float(u_fft[f5_idx] / (u_fft.sum() + 1e-10))
                        u_t10_power = float(u_fft[f10_idx] / (u_fft.sum() + 1e-10))

                    # Cosine similarity between u_r and v_r (trig identity signature:
                    # if cos(a)*cos(b), then u_r ≈ v_r; if sin(a)*sin(b), also u_r ≈ v_r)
                    uv_cos = float(np.dot(ur, vr[:len(ur)]) / (np.linalg.norm(ur) * np.linalg.norm(vr[:len(ur)]) + 1e-10))

                    component_analysis.append({
                        "rank_component": r,
                        "weight": float(weights[r]) if hasattr(weights, '__getitem__') else float(weights),
                        "u_dominant_period": float(u_dominant_period),
                        "v_dominant_period": float(v_dominant_period),
                        "u_fft_strength": float(u_fft_strength),
                        "v_fft_strength": float(v_fft_strength),
                        "u_t2_power": u_t2_power,
                        "u_t5_power": u_t5_power,
                        "u_t10_power": u_t10_power,
                        "uv_cosine": uv_cos,
                    })

                rank_results.append({
                    "rank": rank,
                    "fit": float(fit),
                    "components": component_analysis,
                })
                logger.info(f"  Rank {rank:2d}: fit = {fit:.4f}")

                # Log interesting components
                for ca in component_analysis:
                    if ca["u_fft_strength"] > 3 or ca["v_fft_strength"] > 3:
                        logger.info(
                            f"    Component {ca['rank_component']}: "
                            f"u_period={ca['u_dominant_period']:.1f} (SNR={ca['u_fft_strength']:.1f}), "
                            f"v_period={ca['v_dominant_period']:.1f} (SNR={ca['v_fft_strength']:.1f}), "
                            f"cos(u,v)={ca['uv_cosine']:.3f}"
                        )
                        if ca["u_t2_power"] > 0.1 or ca["u_t5_power"] > 0.1:
                            logger.info(
                                f"      CRT: T2={ca['u_t2_power']:.3f}, "
                                f"T5={ca['u_t5_power']:.3f}, T10={ca['u_t10_power']:.3f}"
                            )

            except Exception as e:
                logger.warning(f"  Rank {rank} decomposition failed: {e}")
                continue

        results[layer] = {"rank_decompositions": rank_results}

    return {"tensor": results}


# ═════════════════════════════════════════════════════════════
# ANALYSIS 4: PERSISTENT HOMOLOGY (TDA) + CRT
# ═════════════════════════════════════════════════════════════

def tda_analysis(
    activations: Dict[int, np.ndarray],
    problems: List[Dict],
    max_operand: int = 30,
    n_pca_dims: int = 20,
) -> Dict[str, Any]:
    """Persistent homology to detect topological structure (circles, tori).

    If the model uses mod-10 arithmetic: expect 1 persistent 1-cycle
    If the model uses CRT (mod-2 × mod-5): expect 2 persistent 1-cycles + 1 persistent 2-cycle

    This is a topological fingerprint that distinguishes the two hypotheses
    without assuming any basis or period.
    """
    try:
        from ripser import ripser
    except ImportError:
        logger.error("ripser not installed. Run: pip install ripser")
        return {"tda": {"error": "ripser not installed"}}

    logger.info("=" * 60)
    logger.info("ANALYSIS 4: PERSISTENT HOMOLOGY (TDA) + CRT")
    logger.info("=" * 60)

    from sklearn.decomposition import PCA

    results = {}

    for layer, acts in activations.items():
        logger.info(f"\n--- Layer {layer} ---")

        # Group activations by ones digit of ANSWER
        digit_groups = defaultdict(list)
        for idx, p in enumerate(problems):
            if idx < len(acts):
                digit_groups[p["ones_digit"]].append(acts[idx])

        # Compute mean activation per digit (centroid-based topology)
        digit_centroids = np.zeros((10, acts.shape[1]), dtype=np.float32)
        for d in range(10):
            if digit_groups[d]:
                digit_centroids[d] = np.mean(digit_groups[d], axis=0)

        # PCA reduce for computational tractability
        n_dims = min(n_pca_dims, acts.shape[1])
        pca = PCA(n_components=n_dims, random_state=42)
        pca.fit(acts)
        centroids_pca = pca.transform(digit_centroids)

        # Test 1: Persistent homology of digit centroids
        logger.info(f"  Test 1: H_1 of digit centroids (10 points in {n_dims}D)")
        rips = ripser(centroids_pca, maxdim=2)
        dgms = rips['dgms']

        h0_bars = len(dgms[0]) if len(dgms) > 0 else 0
        h1_bars = len(dgms[1]) if len(dgms) > 1 else 0
        h2_bars = len(dgms[2]) if len(dgms) > 2 else 0

        # Persistence = death - birth. Long bars = robust features.
        h1_persistence = []
        if len(dgms) > 1:
            for birth, death in dgms[1]:
                if not np.isinf(death):
                    h1_persistence.append(float(death - birth))
        h1_persistence.sort(reverse=True)

        h2_persistence = []
        if len(dgms) > 2:
            for birth, death in dgms[2]:
                if not np.isinf(death):
                    h2_persistence.append(float(death - birth))
        h2_persistence.sort(reverse=True)

        logger.info(f"  H0 bars: {h0_bars} (connected components)")
        logger.info(f"  H1 bars: {h1_bars} (loops/circles)")
        logger.info(f"  H2 bars: {h2_bars} (voids/tori)")
        if h1_persistence:
            logger.info(f"  H1 persistence (top 5): {h1_persistence[:5]}")
        if h2_persistence:
            logger.info(f"  H2 persistence (top 3): {h2_persistence[:3]}")

        # CRT test: separate mod-2 and mod-5 groupings
        # Group by mod-2 residue of answer
        mod2_centroids = np.zeros((2, acts.shape[1]))
        for d in range(10):
            if digit_groups[d]:
                mod2_centroids[d % 2] += np.mean(digit_groups[d], axis=0)
        mod2_centroids /= 5  # 5 digits per mod-2 class

        # Group by mod-5 residue of answer
        mod5_centroids = np.zeros((5, acts.shape[1]))
        for d in range(10):
            if digit_groups[d]:
                mod5_centroids[d % 5] += np.mean(digit_groups[d], axis=0)
        mod5_centroids /= 2  # 2 digits per mod-5 class

        mod5_pca = pca.transform(mod5_centroids)

        # Test 2: Is the mod-5 centroid arrangement circular?
        logger.info(f"\n  Test 2: Mod-5 centroid circularity")
        rips_mod5 = ripser(mod5_pca, maxdim=1)
        h1_mod5 = len(rips_mod5['dgms'][1]) if len(rips_mod5['dgms']) > 1 else 0
        h1_mod5_persist = []
        if len(rips_mod5['dgms']) > 1:
            for b, d in rips_mod5['dgms'][1]:
                if not np.isinf(d):
                    h1_mod5_persist.append(float(d - b))
        h1_mod5_persist.sort(reverse=True)
        logger.info(f"  Mod-5 H1 bars: {h1_mod5}, persistence: {h1_mod5_persist[:3]}")

        # Test 3: Distance matrix analysis
        # Compute pairwise distances between digit centroids
        dist_matrix = squareform(pdist(centroids_pca, metric='euclidean'))

        # Expected circular distance: d(i,j) = min(|i-j|, 10-|i-j|)
        circular_expected = np.zeros((10, 10))
        for i in range(10):
            for j in range(10):
                circular_expected[i, j] = min(abs(i - j), 10 - abs(i - j))

        # Expected CRT distance: sqrt(d2(i,j)^2 + d5(i,j)^2)
        crt_expected = np.zeros((10, 10))
        for i in range(10):
            for j in range(10):
                d2 = abs((i % 2) - (j % 2))  # 0 or 1
                d5 = min(abs((i % 5) - (j % 5)), 5 - abs((i % 5) - (j % 5)))  # 0-2
                crt_expected[i, j] = np.sqrt(d2**2 + d5**2)

        # Correlation of actual distances with circular vs CRT expectations
        triu = np.triu_indices(10, k=1)
        r_circular, p_circular = stats.spearmanr(dist_matrix[triu], circular_expected[triu])
        r_crt, p_crt = stats.spearmanr(dist_matrix[triu], crt_expected[triu])

        logger.info(f"\n  Test 3: Distance matrix structure")
        logger.info(f"  Correlation with circular (mod-10): r = {r_circular:.3f} (p = {p_circular:.4f})")
        logger.info(f"  Correlation with CRT (mod-2×mod-5): r = {r_crt:.3f} (p = {p_crt:.4f})")
        if r_crt > r_circular + 0.05:
            logger.info(f"  → CRT hypothesis favored (Δr = {r_crt - r_circular:.3f})")
        elif r_circular > r_crt + 0.05:
            logger.info(f"  → Circular hypothesis favored (Δr = {r_circular - r_crt:.3f})")
        else:
            logger.info(f"  → Inconclusive (Δr = {abs(r_circular - r_crt):.3f})")

        # Test 4: Full point cloud topology (subsample for speed)
        logger.info(f"\n  Test 4: Full point cloud topology (subsampled)")
        n_subsample = min(200, len(acts))
        subsample_idx = np.random.RandomState(42).choice(len(acts), n_subsample, replace=False)
        acts_sub = pca.transform(acts[subsample_idx])

        try:
            rips_full = ripser(acts_sub, maxdim=1)
            h1_full = len(rips_full['dgms'][1]) if len(rips_full['dgms']) > 1 else 0
            h1_full_persist = []
            if len(rips_full['dgms']) > 1:
                for b, d in rips_full['dgms'][1]:
                    if not np.isinf(d):
                        h1_full_persist.append(float(d - b))
            h1_full_persist.sort(reverse=True)
            logger.info(f"  Full cloud H1 bars: {h1_full}")
            if h1_full_persist:
                logger.info(f"  Top 5 H1 persistence: {h1_full_persist[:5]}")
        except Exception as e:
            logger.warning(f"  Full cloud TDA failed: {e}")
            h1_full = 0
            h1_full_persist = []

        results[layer] = {
            "centroid_h0": h0_bars,
            "centroid_h1": h1_bars,
            "centroid_h2": h2_bars,
            "h1_persistence": h1_persistence[:10],
            "h2_persistence": h2_persistence[:5],
            "mod5_h1_bars": h1_mod5,
            "mod5_h1_persistence": h1_mod5_persist[:5],
            "r_circular": float(r_circular),
            "p_circular": float(p_circular),
            "r_crt": float(r_crt),
            "p_crt": float(p_crt),
            "full_cloud_h1": h1_full,
            "full_cloud_h1_persistence": h1_full_persist[:10],
        }

    return {"tda": results}


# ═════════════════════════════════════════════════════════════
# ANALYSIS 5: WASSERSTEIN (OPTIMAL TRANSPORT) GEOMETRY
# ═════════════════════════════════════════════════════════════

def wasserstein_analysis(
    activations: Dict[int, np.ndarray],
    problems: List[Dict],
    n_pca_dims: int = 30,
) -> Dict[str, Any]:
    """Wasserstein distance between activation distributions grouped by digit.

    For each ones-digit d ∈ {0,...,9}, collects the distribution of activations.
    Computes pairwise Wasserstein-2 distances to build a 10×10 distance matrix.
    Tests whether this matrix has circular (mod-10) or CRT (mod-2 × mod-5) structure.
    """
    try:
        import ot  # Python Optimal Transport
    except ImportError:
        # Fallback to scipy for simpler 1D Wasserstein
        logger.warning("POT not installed. Using scipy 1D Wasserstein fallback.")
        ot = None

    logger.info("=" * 60)
    logger.info("ANALYSIS 5: WASSERSTEIN GEOMETRY")
    logger.info("=" * 60)

    from sklearn.decomposition import PCA
    from scipy.stats import wasserstein_distance as w1d

    results = {}

    for layer, acts in activations.items():
        logger.info(f"\n--- Layer {layer} ---")

        # PCA reduce
        n_dims = min(n_pca_dims, acts.shape[1], acts.shape[0] - 1)
        pca = PCA(n_components=n_dims, random_state=42)
        acts_pca = pca.fit_transform(acts)

        # Group by ones digit of answer
        digit_acts = defaultdict(list)
        for idx, p in enumerate(problems):
            if idx < len(acts_pca):
                digit_acts[p["ones_digit"]].append(acts_pca[idx])

        for d in range(10):
            digit_acts[d] = np.array(digit_acts[d])
            logger.info(f"  Digit {d}: {len(digit_acts[d])} samples")

        # Compute pairwise Wasserstein distances
        W = np.zeros((10, 10))

        if ot is not None:
            # Use Sinkhorn (entropic regularization) for approximate W2
            for i in range(10):
                for j in range(i + 1, 10):
                    xi = digit_acts[i]
                    xj = digit_acts[j]
                    if len(xi) == 0 or len(xj) == 0:
                        continue
                    # Cost matrix
                    M = ot.dist(xi, xj, metric='sqeuclidean')
                    # Normalize cost matrix to avoid numerical issues
                    M_max = M.max()
                    if M_max > 0:
                        M_norm = M / M_max
                    else:
                        continue
                    # Uniform weights
                    a_weights = np.ones(len(xi)) / len(xi)
                    b_weights = np.ones(len(xj)) / len(xj)
                    # Sinkhorn with adaptive regularization
                    try:
                        reg = max(1e-2, M_norm.mean() * 0.05)
                        w2_sq = ot.sinkhorn2(a_weights, b_weights, M_norm, reg=reg)
                        W[i, j] = np.sqrt(max(0, float(w2_sq) * M_max))
                        W[j, i] = W[i, j]
                    except Exception:
                        # Fallback: use centroid distance
                        W[i, j] = np.linalg.norm(xi.mean(axis=0) - xj.mean(axis=0))
                        W[j, i] = W[i, j]
        else:
            # Fallback: average 1D Wasserstein across PCA dimensions
            for i in range(10):
                for j in range(i + 1, 10):
                    xi = digit_acts[i]
                    xj = digit_acts[j]
                    if len(xi) == 0 or len(xj) == 0:
                        continue
                    w_sum = 0
                    for d_idx in range(n_dims):
                        w_sum += w1d(xi[:, d_idx], xj[:, d_idx]) ** 2
                    W[i, j] = np.sqrt(w_sum)
                    W[j, i] = W[i, j]

        # Analyze the distance matrix
        # 1. Circular fit
        circular_expected = np.zeros((10, 10))
        for i in range(10):
            for j in range(10):
                circular_expected[i, j] = min(abs(i - j), 10 - abs(i - j))

        # 2. CRT fit
        crt_expected = np.zeros((10, 10))
        for i in range(10):
            for j in range(10):
                d2 = abs((i % 2) - (j % 2))
                d5 = min(abs((i % 5) - (j % 5)), 5 - abs((i % 5) - (j % 5)))
                crt_expected[i, j] = np.sqrt(d2**2 + d5**2)

        triu = np.triu_indices(10, k=1)
        r_circ, p_circ = stats.spearmanr(W[triu], circular_expected[triu])
        r_crt, p_crt = stats.spearmanr(W[triu], crt_expected[triu])

        # 3. MDS embedding to 2D for visualization
        from sklearn.manifold import MDS
        try:
            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
            embedding = mds.fit_transform(W)
            # Check if MDS embedding is circular: compute angles and test ordering
            angles = np.arctan2(embedding[:, 1] - embedding[:, 1].mean(),
                               embedding[:, 0] - embedding[:, 0].mean())
            # Correlation of angle order with digit order
            r_angle, _ = stats.spearmanr(angles, np.arange(10))
        except Exception:
            embedding = None
            r_angle = 0.0

        # 4. Group by carry vs no-carry
        carry_acts = defaultdict(list)
        nocarry_acts = defaultdict(list)
        for idx, p in enumerate(problems):
            if idx < len(acts_pca):
                if p["carry"]:
                    carry_acts[p["ones_digit"]].append(acts_pca[idx])
                else:
                    nocarry_acts[p["ones_digit"]].append(acts_pca[idx])

        # Mean separation between carry and no-carry for each digit
        carry_separations = {}
        for d in range(10):
            if carry_acts[d] and nocarry_acts[d]:
                c_mean = np.mean(carry_acts[d], axis=0)
                nc_mean = np.mean(nocarry_acts[d], axis=0)
                carry_separations[d] = float(np.linalg.norm(c_mean - nc_mean))

        mean_carry_sep = np.mean(list(carry_separations.values())) if carry_separations else 0
        # Compare with mean inter-digit separation
        mean_digit_sep = W[triu].mean()

        logger.info(f"\n  Wasserstein Distance Matrix (10×10):")
        logger.info(f"  {'':3s}" + "".join(f"{d:7d}" for d in range(10)))
        for i in range(10):
            row = f"  {i:2d} " + "".join(f"{W[i,j]:7.2f}" for j in range(10))
            logger.info(row)

        logger.info(f"\n  Circular correlation: r = {r_circ:.3f} (p = {p_circ:.4f})")
        logger.info(f"  CRT correlation:     r = {r_crt:.3f} (p = {p_crt:.4f})")
        logger.info(f"  MDS angle ordering:  r = {r_angle:.3f}")
        logger.info(f"  Mean inter-digit Wasserstein: {mean_digit_sep:.4f}")
        logger.info(f"  Mean carry separation:        {mean_carry_sep:.4f}")
        if mean_digit_sep > 0:
            logger.info(f"  Carry/digit ratio:            {mean_carry_sep/mean_digit_sep:.3f}")

        results[layer] = {
            "distance_matrix": W.tolist(),
            "r_circular": float(r_circ),
            "p_circular": float(p_circ),
            "r_crt": float(r_crt),
            "p_crt": float(p_crt),
            "mds_angle_correlation": float(r_angle),
            "mean_digit_separation": float(mean_digit_sep),
            "mean_carry_separation": float(mean_carry_sep),
            "carry_separations": carry_separations,
        }

    return {"wasserstein": results}


# ═════════════════════════════════════════════════════════════
# CROSS-VALIDATION: FISHER SUBSPACE + ICA/FOURIER
# ═════════════════════════════════════════════════════════════

def cross_validate_fisher_ica(
    fisher_results: Dict,
    activations: Dict[int, np.ndarray],
    problems: List[Dict],
) -> Dict[str, Any]:
    """Project activations into the Fisher subspace, then run ICA and Fourier.

    This tests the key prediction: ICA and Fourier analysis should work
    BETTER in the Fisher subspace because it filters out causally irrelevant
    dimensions (where the noise lives).
    """
    from sklearn.decomposition import FastICA

    logger.info("=" * 60)
    logger.info("CROSS-VALIDATION: FISHER SUBSPACE + ICA/FOURIER")
    logger.info("=" * 60)

    n = len(problems)
    labels = {
        "carry": np.array([p["carry"] for p in problems[:n]]),
        "ones_digit": np.array([p["ones_digit"] for p in problems[:n]]),
        "tens_digit": np.array([p["tens_digit"] for p in problems[:n]]),
        "ans_mod2": np.array([p["ans_mod2"] for p in problems[:n]]),
        "ans_mod5": np.array([p["ans_mod5"] for p in problems[:n]]),
    }

    results = {}

    for layer in fisher_results:
        if "eigenvectors_topk" not in fisher_results[layer]:
            continue

        logger.info(f"\n--- Layer {layer} ---")
        V = fisher_results[layer]["eigenvectors_topk"]  # (d_model, top_k)
        eff_dim = fisher_results[layer]["effective_dim"]

        if layer not in activations:
            continue

        acts = activations[layer][:n]

        # Project activations into Fisher subspace
        n_fisher = min(int(eff_dim) + 5, V.shape[1])  # effective dim + margin
        V_sub = V[:, :n_fisher]
        acts_fisher = acts @ V_sub  # (n_problems, n_fisher)

        logger.info(f"  Fisher subspace: {n_fisher} dims (effective dim = {eff_dim:.1f})")

        # ICA in Fisher subspace
        n_ic = min(n_fisher, 20)
        try:
            ica = FastICA(n_components=n_ic, random_state=42, max_iter=500)
            sources = ica.fit_transform(acts_fisher)

            # Correlate with labels
            fisher_ica_corrs = {}
            for label_name, label_vals in labels.items():
                best_r = 0
                for ic_idx in range(n_ic):
                    r, _ = stats.spearmanr(sources[:, ic_idx], label_vals[:len(sources)])
                    if abs(r) > abs(best_r):
                        best_r = r
                fisher_ica_corrs[label_name] = float(best_r)

            logger.info(f"  ICA in Fisher subspace:")
            for lbl, r in fisher_ica_corrs.items():
                marker = " ★★" if abs(r) > 0.5 else " ★" if abs(r) > 0.3 else ""
                logger.info(f"    {lbl:12s}: r = {r:+.3f}{marker}")

        except Exception as e:
            logger.warning(f"  Fisher-ICA failed: {e}")
            fisher_ica_corrs = {}

        # Fourier analysis in Fisher subspace
        # Group by ones digit, compute mean Fisher-projected activation
        digit_means = {}
        for d in range(10):
            mask = labels["ones_digit"][:len(acts_fisher)] == d
            if mask.sum() > 0:
                digit_means[d] = acts_fisher[mask].mean(axis=0)

        if len(digit_means) == 10:
            digit_array = np.array([digit_means[d] for d in range(10)])  # (10, n_fisher)
            # FFT of each Fisher dimension as function of digit
            fisher_fourier = {}
            for dim in range(n_fisher):
                signal = digit_array[:, dim]
                fft_vals = np.abs(np.fft.rfft(signal - signal.mean()))
                # Check for T=5 (freq = 2/10 = 0.2) and T=2 (freq = 5/10 = 0.5)
                freqs = np.fft.rfftfreq(10, d=1.0)
                t5_power = float(fft_vals[2]) if len(fft_vals) > 2 else 0  # freq = 2/10
                t2_power = float(fft_vals[5]) if len(fft_vals) > 5 else 0  # freq = 5/10
                t10_power = float(fft_vals[1]) if len(fft_vals) > 1 else 0  # freq = 1/10
                fisher_fourier[dim] = {
                    "t2_power": t2_power,
                    "t5_power": t5_power,
                    "t10_power": t10_power,
                }

            # Aggregate: what fraction of Fisher dims show CRT vs mod-10 dominance?
            crt_dominant = 0
            mod10_dominant = 0
            for dim, ff in fisher_fourier.items():
                crt_total = ff["t2_power"] + ff["t5_power"]
                if crt_total > ff["t10_power"]:
                    crt_dominant += 1
                elif ff["t10_power"] > crt_total:
                    mod10_dominant += 1

            logger.info(f"\n  Fourier in Fisher subspace:")
            logger.info(f"  CRT-dominant dims (T2+T5 > T10): {crt_dominant}/{n_fisher}")
            logger.info(f"  Mod-10-dominant dims (T10 > T2+T5): {mod10_dominant}/{n_fisher}")

        results[layer] = {
            "n_fisher_dims": n_fisher,
            "fisher_ica_correlations": fisher_ica_corrs,
            "crt_dominant_dims": crt_dominant if 'crt_dominant' in dir() else 0,
            "mod10_dominant_dims": mod10_dominant if 'mod10_dominant' in dir() else 0,
        }

    return {"cross_validation": results}


# ═════════════════════════════════════════════════════════════
# MAIN: UNIFIED EXPERIMENT RUNNER
# ═════════════════════════════════════════════════════════════

def run_all_analyses(
    model_key: str = "phi-3-mini",
    max_operand: int = 20,
    analyses: List[str] = None,
    device_override: Optional[str] = None,
    batch_size: int = 4,
):
    """Run all five mathematical analyses + cross-validation.

    Args:
        model_key: model registry key or HuggingFace name
        max_operand: max value for a and b (grid will be max_operand^2 problems)
        analyses: list of analyses to run ('fisher', 'ica', 'tensor', 'tda', 'wasserstein', 'cross')
                  or None for all
        device_override: force specific device
        batch_size: batch size for model inference
    """
    if analyses is None:
        analyses = ["fisher", "ica", "tensor", "tda", "wasserstein", "cross"]

    model_name = resolve_model_name(model_key)
    logger.info(f"Model: {model_name}")
    logger.info(f"Max operand: {max_operand} ({max_operand**2} total problems)")

    # Determine device
    if device_override:
        device = torch.device(device_override)
    else:
        device = get_device()

    dtype = torch.bfloat16 if device.type in ("cuda", "mps") else torch.float32
    logger.info(f"Device: {device}, dtype: {dtype}")

    # Load model
    logger.info("Loading model...")
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=dtype)
    model.eval()
    logger.info(f"Model loaded: {model.cfg.n_layers}L, {model.cfg.n_heads}H, d={model.cfg.d_model}")

    # Determine compute-zone layers based on model
    n_layers = model.cfg.n_layers
    # Compute zone is roughly 60-85% depth (from your circuit analysis findings)
    compute_start = int(n_layers * 0.55)
    compute_end = int(n_layers * 0.85)
    output_layer = n_layers - 1
    layers = list(range(compute_start, compute_end + 1))
    # Add a couple of early layers for comparison
    early_layer = int(n_layers * 0.2)
    layers = [early_layer] + layers + [output_layer]
    logger.info(f"Target layers: {layers}")

    # Generate problems (full grid for tensor decomposition)
    problems = generate_grid_prompts(max_operand=max_operand)
    logger.info(f"Generated {len(problems)} problems")

    # Collect activations (shared across analyses)
    logger.info("\n" + "=" * 60)
    logger.info("COLLECTING ACTIVATIONS")
    logger.info("=" * 60)
    mlp_activations = collect_activations(
        model, problems, layers,
        position="final", batch_size=batch_size,
        max_problems=len(problems),
    )

    all_results = {
        "model": model_name,
        "max_operand": max_operand,
        "n_problems": len(problems),
        "layers": layers,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }

    # Run analyses
    fisher_data = None

    if "fisher" in analyses:
        fisher_out = fisher_information_analysis(
            model, problems, layers, n_problems=min(200, len(problems)),
        )
        all_results.update(fisher_out)
        fisher_data = fisher_out.get("fisher", {})

    if "ica" in analyses:
        ica_out = ica_analysis(mlp_activations, problems)
        all_results.update(ica_out)

    if "tensor" in analyses:
        tensor_out = tensor_decomposition_analysis(
            mlp_activations, problems, max_operand=max_operand,
        )
        all_results.update(tensor_out)

    if "tda" in analyses:
        tda_out = tda_analysis(mlp_activations, problems, max_operand=max_operand)
        all_results.update(tda_out)

    if "wasserstein" in analyses:
        wass_out = wasserstein_analysis(mlp_activations, problems)
        all_results.update(wass_out)

    if ("cross" in analyses or ("fisher" in analyses and "ica" in analyses)) and fisher_data:
        cross_out = cross_validate_fisher_ica(fisher_data, mlp_activations, problems)
        all_results.update(cross_out)

    # Save results
    output_dir = Path("mathematical_toolkit_results")
    output_dir.mkdir(exist_ok=True)

    # Remove non-serializable numpy arrays before saving
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()
                    if not isinstance(v, np.ndarray)}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    safe_results = make_serializable(all_results)
    timestamp = all_results["timestamp"]
    model_tag = model_key.replace("/", "_")
    fname = output_dir / f"toolkit_{model_tag}_{timestamp}.json"
    with open(fname, 'w') as f:
        json.dump(safe_results, f, indent=2)
    logger.info(f"\nResults saved to {fname}")

    # Print summary
    print_summary(all_results)

    return all_results


def print_summary(results: Dict):
    """Print a concise summary of all analyses."""
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: MATHEMATICAL TOOLKIT RESULTS")
    logger.info("=" * 60)

    # Fisher summary
    if "fisher" in results:
        logger.info("\n📐 FISHER INFORMATION:")
        for layer, data in results["fisher"].items():
            logger.info(f"  L{layer}: effective dim = {data['effective_dim']:.1f}, "
                       f"90% info in {data['dims_90']} dims")

    # ICA summary
    if "ica" in results:
        logger.info("\n🔬 ICA (Superposition Resolution):")
        for layer, data in results["ica"].items():
            best = data.get("best_ics", {})
            carry_r = abs(best.get("carry", {}).get("r", 0))
            ones_r = abs(best.get("ones_digit", {}).get("r", 0))
            logger.info(f"  L{layer}: carry r={carry_r:.3f}, ones_digit r={ones_r:.3f}")

    # Tensor summary
    if "tensor" in results:
        logger.info("\n🧊 TENSOR DECOMPOSITION:")
        for layer, data in results["tensor"].items():
            for rd in data.get("rank_decompositions", []):
                if rd["rank"] == 10:
                    logger.info(f"  L{layer}: rank-10 fit = {rd['fit']:.4f}")

    # TDA summary
    if "tda" in results:
        logger.info("\n🔄 PERSISTENT HOMOLOGY:")
        for layer, data in results["tda"].items():
            logger.info(f"  L{layer}: H1={data['centroid_h1']} loops, "
                       f"r_circular={data['r_circular']:.3f}, r_crt={data['r_crt']:.3f}")

    # Wasserstein summary
    if "wasserstein" in results:
        logger.info("\n📊 WASSERSTEIN GEOMETRY:")
        for layer, data in results["wasserstein"].items():
            logger.info(f"  L{layer}: r_circular={data['r_circular']:.3f}, "
                       f"r_crt={data['r_crt']:.3f}, carry_sep={data['mean_carry_separation']:.4f}")


# ═════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mathematical Toolkit for Arithmetic Circuit Discovery")
    parser.add_argument("--model", default="phi-3-mini", help="Model key or HuggingFace name")
    parser.add_argument("--operand-range", type=int, default=20, help="Max operand value (grid = range^2)")
    parser.add_argument("--analysis", default="all",
                       help="Which analyses: all, fisher, ica, tensor, tda, wasserstein, cross (comma-separated)")
    parser.add_argument("--device", default=None, help="Force device (cuda/mps/cpu)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for inference")
    args = parser.parse_args()

    if args.analysis == "all":
        analyses = None
    else:
        analyses = [a.strip() for a in args.analysis.split(",")]

    run_all_analyses(
        model_key=args.model,
        max_operand=args.operand_range,
        analyses=analyses,
        device_override=args.device,
        batch_size=args.batch_size,
    )
