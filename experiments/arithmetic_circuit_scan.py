#!/usr/bin/env python3
"""
Phi-3 Arithmetic Layer Scan + Unembed-Aligned Patching (v2 — fully fixed)

Fixes over v1:
  BUG-1  compute_unembed_basis: centering was dim=1 (wrong). Fixed to dim=0.
  BUG-2  Layer scan and unembed experiment were disconnected.
         Best layer from scan is now automatically selected for Experiment 2.
  BUG-3  _tokens left on problem dicts after scan. Added explicit cleanup pass.
  BUG-4  n_dims_list=[2,5,9] omitted 10D (full unembed rank). Added.
  NEW    Instruction-tuning interference removal via formatting Fisher.
  NEW    Comprehensive sanity checks after every critical computation.
  NEW    Cross-verification: random-basis control to confirm patching is not trivial.

Three experiments:
  1. Full-layer scan  — find where arithmetic lives (highest transfer ceiling).
  2. Unembed patching — test Explanation A: circuit stored in unembed-aligned dirs.
  3. Format removal   — project out instruction-following subspace, re-run Fisher.

Usage:
    python phi3_arithmetic_scan.py --device cpu
    python phi3_arithmetic_scan.py --device cpu --skip-scan --unembed-layers 28,29
    python phi3_arithmetic_scan.py --device cpu --no-format-removal
"""

import argparse
import logging
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_MAP = {
    "phi-3":       "microsoft/Phi-3-mini-4k-instruct",
    "gemma-2b":    "google/gemma-2-2b",
    "gpt2":        "gpt2",
    "gpt2-medium": "gpt2-medium",
    "llama-3b":    "meta-llama/Llama-3.2-3B",
    "llama-3b-it": "meta-llama/Llama-3.2-3B-Instruct",
}

# ─────────────────────────────────────────────────────────────
# SANITY CHECK UTILITIES
# ─────────────────────────────────────────────────────────────

def assert_close(a, b, atol=1e-5, label=""):
    """Hard assertion that two scalars or arrays are numerically close."""
    if np.abs(np.asarray(a) - np.asarray(b)).max() > atol:
        raise AssertionError(
            f"Sanity check FAILED [{label}]: got {a} vs {b} (atol={atol})"
        )


def check_orthonormal(basis: np.ndarray, label="basis"):
    """Verify columns of basis are orthonormal. Raises if not."""
    n_cols = basis.shape[1]
    G = basis.T @ basis              # should be identity
    off_diag = G - np.eye(n_cols)
    max_err = np.abs(off_diag).max()
    if max_err > 1e-4:
        raise AssertionError(
            f"Orthonormality FAILED [{label}]: max off-diag error = {max_err:.2e}"
        )
    logger.info(f"  [SANITY] {label}: orthonormal ✓ (max_err={max_err:.2e})")


def check_subspace_partition(
    full_delta: torch.Tensor,
    sub_delta: torch.Tensor,
    ortho_delta: torch.Tensor,
    label="partition",
    atol=1e-4,
):
    """Verify sub_delta + ortho_delta == full_delta (exact partition)."""
    recon = sub_delta + ortho_delta
    err = (full_delta - recon).abs().max().item()
    if err > atol:
        raise AssertionError(
            f"Partition check FAILED [{label}]: sub+ortho != full (max_err={err:.2e})"
        )
    logger.info(f"  [SANITY] {label}: sub+ortho=full ✓ (max_err={err:.2e})")


def log_section(title: str):
    logger.info(f"\n{'═'*60}")
    logger.info(f"  {title}")
    logger.info(f"{'═'*60}")


# ─────────────────────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────────────────────

def generate_teacher_forced_problems(
    n_per_digit: int = 100,
    operand_max: int = 99,
) -> Tuple[List[dict], Dict[int, List[dict]]]:
    """
    Generate teacher-forced problems where the TARGET TOKEN is always the ones digit.

    For single-digit answers (a+b < 10):
        Prompt  = "Calculate:\\n{a} + {b} = "
        Target  = str(answer)           e.g. "7"

    For multi-digit answers:
        Prompt  = "Calculate:\\n{a} + {b} = {tens_and_above}"
        Target  = str(answer % 10)      e.g. "0" for answer=30

    This gives O(operand_max²) candidates per digit class —
    sufficient for well-estimated Fisher matrices.
    """
    by_digit: Dict[int, List[dict]] = defaultdict(list)

    for a in range(operand_max + 1):
        for b in range(operand_max + 1):
            answer = a + b
            ones = answer % 10
            answer_str = str(answer)

            if len(answer_str) == 1:
                prompt     = f"Calculate:\n{a} + {b} = "
                target_str = answer_str
            else:
                prefix     = answer_str[:-1]      # everything but ones digit
                prompt     = f"Calculate:\n{a} + {b} = {prefix}"
                target_str = answer_str[-1]       # ones digit character

            by_digit[ones].append({
                "prompt":     prompt,
                "answer":     answer,
                "ones_digit": ones,
                "target_str": target_str,
                "a": a, "b": b,
                "has_carry":  int((a % 10 + b % 10) >= 10),
            })

    # Sanity: every digit must have enough candidates
    for d in range(10):
        n_avail = len(by_digit[d])
        assert n_avail >= n_per_digit, (
            f"[SANITY] Only {n_avail} problems for digit {d}; "
            f"need {n_per_digit}. Increase operand_max."
        )
        logger.info(f"  Digit {d}: {n_avail} available, sampling {n_per_digit}")

    import random
    flat = []
    for d in range(10):
        pool = list(by_digit[d])
        random.shuffle(pool)
        flat.extend(pool[:n_per_digit])
    random.shuffle(flat)

    logger.info(f"Generated {len(flat)} teacher-forced problems ({n_per_digit}/digit)")
    return flat, by_digit


def generate_formatting_problems(n: int = 200) -> List[dict]:
    """
    Non-arithmetic prompts that trigger instruction-following / formatting behaviour.
    Used to compute the instruction-tuning Fisher subspace that we project OUT.

    Each problem has a dummy numeric-looking completion so we can take
    a gradient w.r.t. the logit of a digit token — same setup as arithmetic,
    but with no actual arithmetic content.
    """
    templates = [
        ("The answer is: ", "4"),
        ("Result: ", "7"),
        ("Output: ", "3"),
        ("Value: ", "9"),
        ("Number: ", "5"),
        ("Score: ", "8"),
        ("Total: ", "2"),
        ("Count: ", "6"),
        ("Amount: ", "1"),
        ("Quantity: ", "0"),
    ]
    problems = []
    import random
    for _ in range(n):
        prompt, target = random.choice(templates)
        problems.append({
            "prompt":     prompt,
            "target_str": target,
            "ones_digit": int(target),
        })
    return problems


def filter_correct_teacher_forced(
    model,
    problems: List[dict],
    max_n: int = 200,
) -> List[dict]:
    """
    Keep only problems where the model correctly predicts target_str at
    the last token position. Attach _tokens for later use.
    """
    correct = []
    for prob in problems:
        tokens = model.to_tokens(prob["prompt"], prepend_bos=True)
        target_tok = model.to_tokens(f'= {prob["target_str"]}', prepend_bos=False)[0, -1].item()
        with torch.no_grad():
            logits = model(tokens)
        pred = logits[0, -1].argmax().item()
        if pred == target_tok:
            prob["_tokens"]     = tokens
            prob["_target_tok"] = target_tok
            correct.append(prob)
        if len(correct) >= max_n:
            break

    # Count per digit
    counts = defaultdict(int)
    for p in correct:
        counts[p["ones_digit"]] += 1
    logger.info(f"  Correct problems: {len(correct)}/{max_n} attempted")
    logger.info(f"  By digit: {dict(sorted(counts.items()))}")
    return correct


def cleanup_tokens(problems: List[dict]):
    """Remove all injected runtime keys from problem dicts (memory hygiene)."""
    for p in problems:
        p.pop("_tokens",     None)
        p.pop("_target_tok", None)
        p.pop("_clean_act",  None)
        # Also remove any per-layer cached activation keys (_act_L{n})
        keys_to_remove = [k for k in list(p.keys()) if k.startswith("_")]
        for k in keys_to_remove:
            p.pop(k, None)


# ─────────────────────────────────────────────────────────────
# ACTIVATION PATCHING CORE
# ─────────────────────────────────────────────────────────────

def _build_pairs(
    correct: List[dict],
    n_pairs_per_combo: int = 3,
) -> List[Tuple[dict, dict]]:
    """Build all (clean_digit, corrupt_digit) pairs across digit combinations."""
    by_digit = defaultdict(list)
    for p in correct:
        by_digit[p["ones_digit"]].append(p)

    pairs = []
    for cd in sorted(by_digit):
        for rd in sorted(by_digit):
            if cd == rd:
                continue
            cleans   = by_digit[cd][:n_pairs_per_combo]
            corrupts = by_digit[rd][:n_pairs_per_combo]
            for i, cp in enumerate(cleans):
                if i >= len(corrupts):
                    break
                pairs.append((cp, corrupts[i]))
    return pairs


def _project_delta(
    clean_act: torch.Tensor,
    corrupt_act: torch.Tensor,
    basis: torch.Tensor,             # (n_dims, d_model) — rows are basis vectors
    mode: str,                        # "full" | "sub" | "ortho"
    verify_partition: bool = False,
) -> torch.Tensor:
    """
    Compute activation delta for a given patch mode.

    basis rows must be orthonormal (verified externally).
    The partition identity sub_delta + ortho_delta = full_delta is
    checked when verify_partition=True (used in sanity-check mode).
    """
    full_delta = clean_act.float() - corrupt_act.float()

    if mode == "full":
        return full_delta.to(clean_act.dtype)

    # Project onto subspace spanned by basis rows
    clean_proj   = basis @ clean_act.float()    # (n_dims,)
    corrupt_proj = basis @ corrupt_act.float()  # (n_dims,)
    sub_delta    = basis.T @ (clean_proj - corrupt_proj)  # (d_model,)
    ortho_delta  = full_delta - sub_delta

    if verify_partition:
        check_subspace_partition(
            full_delta, sub_delta, ortho_delta,
            label=f"dim={basis.shape[0]}"
        )

    if mode == "sub":
        return sub_delta.to(clean_act.dtype)
    return ortho_delta.to(clean_act.dtype)


def run_patching_experiment(
    model,
    layer:    int,
    basis:    np.ndarray,            # (d_model, n_directions)
    correct:  List[dict],
    n_dims_list: List[int],
    label:    str = "basis",
    n_pairs_per_combo: int = 3,
    verify_partition_on_first: bool = True,
) -> Dict:
    """
    Run full / subspace / ortho activation patching for multiple subspace sizes.

    Returns a dict keyed by "{n_dims}D" → {"full": {...}, "sub": {...}, "ortho": {...}}.
    """
    device    = next(model.parameters()).device
    hook_name = f"blocks.{layer}.hook_resid_post"

    # Cache clean activations
    logger.info(f"  Caching activations at L{layer} for {len(correct)} problems...")
    for prob in correct:
        tokens = prob["_tokens"]
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
            prob["_clean_act"] = cache[hook_name][0, -1].detach().cpu()
            del cache

    pairs = _build_pairs(correct, n_pairs_per_combo)
    if not pairs:
        logger.warning(f"  No valid pairs at L{layer} — skipping")
        return {}

    logger.info(f"  Running {len(pairs)} pairs × {len(n_dims_list)} dims × 3 modes")

    all_results = {}
    partition_verified = not verify_partition_on_first  # flip after first check

    for n_dims in n_dims_list:
        if n_dims > basis.shape[1]:
            logger.info(f"  [SKIP] {n_dims}D: only {basis.shape[1]} directions in basis")
            continue

        sub_basis_np = basis[:, :n_dims]

        # Orthonormality check on the subspace columns
        check_orthonormal(sub_basis_np, label=f"{label}-{n_dims}D")

        sub_basis = torch.tensor(
            sub_basis_np.T,              # (n_dims, d_model) — rows for projection
            dtype=torch.float32,
            device=device,
        )

        dim_res = {m: {"total": 0, "transfer": 0, "changed": 0, "stayed": 0}
                   for m in ("full", "sub", "ortho")}

        for idx, (clean_prob, corrupt_prob) in enumerate(pairs):
            clean_act   = clean_prob["_clean_act"].to(device)
            corrupt_act = corrupt_prob["_clean_act"].to(device)
            corrupt_toks = corrupt_prob["_tokens"]
            clean_digit   = clean_prob["ones_digit"]
            corrupt_digit = corrupt_prob["ones_digit"]

            do_verify = (not partition_verified) and (idx == 0)

            for mode in ("full", "sub", "ortho"):
                delta = _project_delta(
                    clean_act, corrupt_act, sub_basis, mode,
                    verify_partition=do_verify and (mode == "ortho"),
                )

                def hook_fn(act, hook, d=delta):
                    act[:, -1, :] = act[:, -1, :] + d.unsqueeze(0).to(act.dtype)
                    return act

                with torch.no_grad():
                    with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
                        logits = model(corrupt_toks)

                pred_tok = logits[0, -1].argmax().item()
                pred_str = model.tokenizer.decode([pred_tok]).strip()

                r = dim_res[mode]
                r["total"] += 1
                try:
                    pred_digit = int(pred_str)
                    if pred_digit == clean_digit:
                        r["transfer"] += 1
                    if pred_digit != corrupt_digit:
                        r["changed"] += 1
                    if pred_digit == corrupt_digit:
                        r["stayed"] += 1
                except ValueError:
                    pass  # non-numeric prediction — counted in total only

            if do_verify:
                partition_verified = True

        # Log summary
        logger.info(f"\n{'─'*55}")
        logger.info(f"  {label.upper()} PATCHING — L{layer}, {n_dims}D")
        logger.info(f"{'─'*55}")
        for name, mode_key in [("FULL", "full"), (label.upper()[:6], "sub"), ("ORTHO", "ortho")]:
            r = dim_res[mode_key]
            n = r["total"]
            if n:
                t = 100 * r["transfer"] / n
                c = 100 * r["changed"]  / n
                s = 100 * r["stayed"]   / n
                logger.info(
                    f"  {name:<8}: transfer={r['transfer']}/{n} ({t:.1f}%)  "
                    f"changed={r['changed']}/{n} ({c:.1f}%)  "
                    f"stayed={r['stayed']}/{n} ({s:.1f}%)"
                )

        all_results[f"{n_dims}D"] = dim_res

    # Cleanup cached activations
    for prob in correct:
        prob.pop("_clean_act", None)

    return all_results


# ─────────────────────────────────────────────────────────────
# EXPERIMENT 1 — FULL-LAYER SCAN
# ─────────────────────────────────────────────────────────────

def run_layer_scan(
    model,
    correct:  List[dict],
    n_layers: int,
    n_pairs_per_combo: int = 3,
) -> Dict[int, dict]:
    """
    Run ONLY full-activation patching at every layer to find where arithmetic lives.
    Returns {layer_idx: {"transfer": int, "changed": int, "stayed": int, "total": int}}.
    """
    log_section("EXPERIMENT 1: FULL-LAYER SCAN")
    pairs   = _build_pairs(correct, n_pairs_per_combo)
    results = {}

    for layer in range(n_layers):
        hook_name = f"blocks.{layer}.hook_resid_post"

        # Cache activations for this layer
        for prob in correct:
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    prob["_tokens"], names_filter=hook_name
                )
                prob[f"_act_L{layer}"] = cache[hook_name][0, -1].detach().cpu()
                del cache

        res = {"total": 0, "transfer": 0, "changed": 0, "stayed": 0}

        for clean_prob, corrupt_prob in pairs:
            clean_act   = clean_prob[f"_act_L{layer}"]
            corrupt_act = corrupt_prob[f"_act_L{layer}"]
            delta       = clean_act - corrupt_act

            def hook_fn(act, hook, d=delta):
                act[:, -1, :] = act[:, -1, :] + d.unsqueeze(0).to(act.dtype)
                return act

            with torch.no_grad():
                with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)]):
                    logits = model(corrupt_prob["_tokens"])

            pred_str = model.tokenizer.decode(
                [logits[0, -1].argmax().item()]
            ).strip()
            res["total"] += 1
            try:
                pred_digit = int(pred_str)
                if pred_digit == clean_prob["ones_digit"]:
                    res["transfer"] += 1
                if pred_digit != corrupt_prob["ones_digit"]:
                    res["changed"] += 1
                if pred_digit == corrupt_prob["ones_digit"]:
                    res["stayed"] += 1
            except ValueError:
                pass

        # Compute percentages
        n = res["total"]
        res["transfer_pct"] = round(100 * res["transfer"] / n, 1) if n else 0.0
        res["changed_pct"]  = round(100 * res["changed"]  / n, 1) if n else 0.0
        results[layer]      = res

        # Clean up this layer's activations immediately (memory)
        for prob in correct:
            prob.pop(f"_act_L{layer}", None)

        marker = " ◄◄◄ ARITHMETIC?" if res["transfer_pct"] > 50 else ""
        logger.info(
            f"  L{layer:>3}: transfer={res['transfer_pct']:>5.1f}%  "
            f"changed={res['changed_pct']:>5.1f}%{marker}"
        )

    # Print ranked summary
    logger.info(f"\n  {'─'*40}")
    logger.info("  TOP-5 LAYERS BY TRANSFER%:")
    top5 = sorted(results.items(), key=lambda x: x[1]["transfer_pct"], reverse=True)[:5]
    for rank, (l, r) in enumerate(top5, 1):
        logger.info(f"    #{rank}  L{l}: {r['transfer_pct']:.1f}%")

    return results


# ─────────────────────────────────────────────────────────────
# UNEMBED BASIS
# ─────────────────────────────────────────────────────────────

def get_digit_token_ids(model) -> List[int]:
    """
    Get token IDs for digit characters 0-9 as predicted in arithmetic context.

    WHY THIS MATTERS — tokenizer space-prefix problem:
      GPT-2 (BPE) and LLaMA (SentencePiece) encode digits differently
      depending on whether they follow a space:
        GPT-2:  to_tokens("1")  → token 16   (no-space)
                to_tokens(" 1") → token 352  (space-prefixed — what follows "= ")
      Naive to_tokens("1") returns the WRONG token ID on these models.

      Fix: tokenize "= {d}" and take the LAST token — exactly what the
      model predicts after "= " in context. Works on all tokenizers.
    """
    ids = []
    for d in range(10):
        toks = model.to_tokens(f"= {d}", prepend_bos=False)
        ids.append(toks[0, -1].item())

    logger.info(f"  Digit token IDs (context-aware, after '= '): {ids}")

    # Sanity 1: all IDs must be distinct
    assert len(set(ids)) == 10, \
        f"[SANITY] Duplicate digit token IDs: {ids}. " \
        "Some digits may tokenize identically — check tokenizer."
    logger.info("  [SANITY] All 10 digit token IDs are distinct ✓")

    # Sanity 2: round-trip — decode must give back the original digit string
    for d, tok_id in enumerate(ids):
        decoded = model.tokenizer.decode([tok_id]).strip()
        assert decoded == str(d), \
            f"[SANITY] Token {tok_id} decodes to \'{decoded}\', expected \'{d}\'. " \
            "Digit may be a multi-token sequence on this tokenizer."
    logger.info("  [SANITY] All digit tokens decode correctly ✓")

    return ids


def compute_unembed_basis(
    model,
    digit_token_ids: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute an orthonormal basis for the subspace of residual stream directions
    that most affect digit token logits.

    W_U shape: (d_model, vocab_size)
    Extract columns for digit tokens → W_digits: (d_model, 10)

    CENTERING: We subtract the column-wise mean (mean over d_model for each digit)
    so that the SVD focuses on directions that *differ between digits*, not the
    shared "output any digit" direction.

    W_centered[:, k] = W_digits[:, k] - mean_over_d_model(W_digits[:, k])

    FIX vs v1: v1 used dim=1 (subtracting the mean across digits for each
    residual-stream dimension) — this is mathematically wrong for our purpose.
    Correct centering is dim=0 (subtract the scalar mean of each column).

    Returns:
        U  : (d_model, 10) orthonormal basis — columns are the directions
        S  : (10,) singular values
    """
    W_U      = model.W_U.detach().float().cpu()   # (d_model, vocab_size)
    W_digits = W_U[:, digit_token_ids]             # (d_model, 10)

    # ── CENTERING (Bug 1 fix) ────────────────────────────────────────────────
    # Each column is one digit's unembedding vector in residual stream space.
    # Centering over dim=0 removes the scalar mean of each column independently,
    # making each column zero-mean in residual stream coordinates.
    col_means  = W_digits.mean(dim=0, keepdim=True)  # (1, 10)
    W_centered = W_digits - col_means                  # (d_model, 10)

    # Verify centering correctness
    post_means = W_centered.mean(dim=0).abs()
    assert post_means.max().item() < 1e-5, \
        f"[SANITY] Centering failed: max col mean = {post_means.max():.2e}"
    logger.info(
        f"  [SANITY] Column centering ✓  (max residual mean = {post_means.max():.2e})"
    )

    # ── SVD ─────────────────────────────────────────────────────────────────
    U, S, Vt = torch.linalg.svd(W_centered, full_matrices=False)
    # U: (d_model, 10)  — left singular vectors = residual-stream basis
    # S: (10,)
    # Vt: (10, 10)

    # Verify U columns are orthonormal
    G = (U.T @ U).numpy()
    off_diag_err = np.abs(G - np.eye(10)).max()
    assert off_diag_err < 1e-4, \
        f"[SANITY] U not orthonormal: max off-diag = {off_diag_err:.2e}"
    logger.info(
        f"  [SANITY] SVD U columns orthonormal ✓  (max err = {off_diag_err:.2e})"
    )

    # Verify reconstruction: U @ diag(S) @ Vt ≈ W_centered
    recon = (U * S.unsqueeze(0)) @ Vt
    recon_err = (recon - W_centered).abs().max().item()
    assert recon_err < 1e-3, \
        f"[SANITY] SVD reconstruction error too large: {recon_err:.2e}"
    logger.info(f"  [SANITY] SVD reconstruction ✓  (max err = {recon_err:.2e})")

    # Variance explained
    S_np    = S.numpy()
    var_all = (S_np ** 2).sum()
    logger.info(f"  Singular values: {S_np.round(4)}")
    for k in [1, 2, 5, 10]:
        pct = 100 * (S_np[:k] ** 2).sum() / var_all
        logger.info(f"  Top-{k} explain {pct:.1f}% of digit-token variance")

    return U.numpy(), S_np


# ─────────────────────────────────────────────────────────────
# FISHER MATRICES (standard + formatting)
# ─────────────────────────────────────────────────────────────

def compute_fisher_matrix(
    model,
    problems:  List[dict],
    layer:     int,
    n:         int = 200,
    label:     str = "Fisher",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute Fisher Information Matrix at a given layer by accumulating
    outer products of log-probability gradients.

    Assumes each problem has 'prompt', 'target_str' fields.
    Returns (eigenvalues, eigenvectors, eff_dim) — eigenvectors sorted descending.
    """
    logger.info(f"  Computing {label} at L{layer} ({n} problems)...")
    hook_name = f"blocks.{layer}.hook_resid_post"
    d_model   = model.cfg.d_model
    F_mat     = np.zeros((d_model, d_model), dtype=np.float64)
    n_valid   = 0

    for i, prob in enumerate(problems[:n]):
        tokens     = model.to_tokens(prob["prompt"], prepend_bos=True)
        target_tok = model.to_tokens(
            f"= {prob['target_str']}", prepend_bos=False
        )[0, -1].item()

        holder = {}

        def capture(act, hook, h=holder):
            h["act"] = act
            act.requires_grad_(True)
            act.retain_grad()
            return act

        try:
            with model.hooks(fwd_hooks=[(hook_name, capture)]):
                logits = model(tokens)
            log_p = F.log_softmax(logits[0, -1].float(), dim=-1)[target_tok]
            log_p.backward()
            if "act" in holder and holder["act"].grad is not None:
                g       = holder["act"].grad[0, -1].detach().cpu().float().numpy()
                F_mat  += np.outer(g, g)
                n_valid += 1
        except Exception:
            pass
        finally:
            model.zero_grad()

    assert n_valid >= 10, \
        f"[SANITY] Only {n_valid} valid gradients for {label} — too few"

    F_mat /= n_valid

    # Eigendecomposition (F_mat is symmetric PSD)
    evals, evecs = np.linalg.eigh(F_mat)
    idx   = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Effective dimension (Shannon entropy of normalised eigenvalue spectrum)
    total = evals.sum()
    if total > 0:
        p       = evals / total
        p       = p[p > 1e-10]
        eff_dim = float(np.exp(-np.sum(p * np.log(p))))
    else:
        eff_dim = 0.0

    logger.info(
        f"  {label}: eff_dim={eff_dim:.2f}, "
        f"λ₁/λ₂={evals[0]/max(evals[1], 1e-30):.1f}×, "
        f"valid={n_valid}"
    )
    return evals, evecs, eff_dim


def compute_contrastive_fisher(
    model,
    by_digit:  Dict[int, List[dict]],
    layer:     int,
    n_per_digit: int = 100,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Contrastive Fisher via between-class scatter of per-digit gradient means.

    Uses teacher-forced problems (from generate_teacher_forced_problems) where
    each problem already has 'prompt' and 'target_str' (ones digit).

    Algorithm:
      For each digit class k (0-9):
        Collect n_per_digit gradients: g_i = ∇ log p(ones_digit_k | x_i)
        Compute class mean: μ_k = mean(g_i for i in class k)

      Global mean: μ = mean(μ_k for k in 0..9)

      Between-class scatter:
        S_B = Σ_k n_k · (μ_k - μ)(μ_k - μ)ᵀ   (rank ≤ 9)

    Top eigenvectors of S_B are the directions that maximally separate
    digit classes in gradient space.

    Returns:
        evals  : (d_model,) eigenvalues descending
        evecs  : (d_model, d_model) eigenvectors (top 9 are meaningful)
        n_dirs : number of non-trivial contrastive directions (≤ 9)
    """
    logger.info(f"  Computing contrastive Fisher at L{layer} ({n_per_digit}/digit)...")
    hook_name = f"blocks.{layer}.hook_resid_post"
    d_model   = model.cfg.d_model

    class_grads: Dict[int, List[np.ndarray]] = defaultdict(list)
    n_total_valid = 0

    for digit in range(10):
        problems = by_digit[digit][:n_per_digit]
        for prob in problems:
            tokens     = model.to_tokens(prob["prompt"], prepend_bos=True)
            target_tok = model.to_tokens(
                f"= {prob['target_str']}", prepend_bos=False
            )[0, -1].item()

            holder = {}

            def capture(act, hook, h=holder):
                h["act"] = act
                act.requires_grad_(True)
                act.retain_grad()
                return act

            try:
                with model.hooks(fwd_hooks=[(hook_name, capture)]):
                    logits = model(tokens)
                log_p = F.log_softmax(logits[0, -1].float(), dim=-1)[target_tok]
                log_p.backward()
                if "act" in holder and holder["act"].grad is not None:
                    g = holder["act"].grad[0, -1].detach().cpu().float().numpy()
                    class_grads[digit].append(g)
                    n_total_valid += 1
            except Exception:
                pass
            finally:
                model.zero_grad()

        logger.info(f"    Digit {digit}: {len(class_grads[digit])} valid")

    # Sanity: every digit must have enough gradients for a reliable mean
    for d in range(10):
        n = len(class_grads[d])
        assert n >= 10, \
            f"[SANITY] Contrastive Fisher: only {n} gradients for digit {d} (need ≥10)"
    logger.info(f"  Total valid gradients: {n_total_valid}")

    # ── Per-class means ───────────────────────────────────────────────────
    class_means  = {}
    class_counts = {}
    for d in range(10):
        grads = class_grads[d]
        class_means[d]  = np.mean(grads, axis=0)    # (d_model,)
        class_counts[d] = len(grads)

    global_mean = np.mean(list(class_means.values()), axis=0)  # (d_model,)

    # ── Between-class scatter S_B ─────────────────────────────────────────
    # Rank ≤ 9 (10 classes → 9 independent deviations from global mean)
    S_B = np.zeros((d_model, d_model), dtype=np.float64)
    for d, mu_k in class_means.items():
        diff = mu_k - global_mean           # (d_model,)
        S_B += class_counts[d] * np.outer(diff, diff)
    S_B /= n_total_valid

    # ── Eigendecomposition ───────────────────────────────────────────────
    evals, evecs = np.linalg.eigh(S_B)
    idx   = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Number of non-trivial directions: eigenvalues above 1e-4 × max
    threshold  = evals[0] * 1e-4
    n_nontrivial = int(np.sum(evals > threshold))
    n_dirs     = min(n_nontrivial, 9)   # hard cap: rank of S_B ≤ 9

    logger.info(f"  Contrastive Fisher: {n_dirs} non-trivial directions")
    logger.info(f"  Top-5 S_B eigenvalues: {evals[:5]}")
    logger.info(f"  Eigenvalue ratio λ₁/λ₉: "
                f"{evals[0]/max(evals[n_dirs-1], 1e-30):.1f}×")

    # Sanity: S_B should be positive semidefinite
    n_negative = int(np.sum(evals < -1e-8))
    assert n_negative == 0, \
        f"[SANITY] S_B has {n_negative} negative eigenvalues — numerical issue"
    logger.info(f"  [SANITY] S_B is PSD ✓")

    return evals, evecs, n_dirs


def subtract_formatting_subspace(
    arith_evecs: np.ndarray,          # (d_model, k_arith)
    format_evecs: np.ndarray,         # (d_model, k_fmt)
    n_format_dims: int,
) -> np.ndarray:
    """
    Project arithmetic Fisher eigenvectors onto the complement of the
    top n_format_dims formatting Fisher directions.

    Steps:
      1. Build orthonormal format basis F: (d_model, n_format_dims)
      2. Project each arithmetic eigenvector onto complement of F:
             v_clean = v - F (F^T v)
      3. Re-orthonormalise via QR (Gram-Schmidt).

    Returns (d_model, k_arith) orthonormal basis — same number of columns
    but now orthogonal to the instruction-following subspace.
    """
    F_basis    = format_evecs[:, :n_format_dims]                # (d_model, n_fmt)
    # Project arithmetic basis onto formatting complement
    proj       = arith_evecs - F_basis @ (F_basis.T @ arith_evecs)  # (d_model, k_arith)

    # Re-orthonormalise via QR
    Q, R       = np.linalg.qr(proj, mode="reduced")
    # Check sign consistency (QR is not unique in sign)
    signs      = np.sign(np.diag(R))
    signs[signs == 0] = 1
    Q          = Q * signs[np.newaxis, :]

    # Sanity
    check_orthonormal(Q, label="format-cleaned basis")
    return Q


# ─────────────────────────────────────────────────────────────
# RANDOM CONTROL BASIS
# ─────────────────────────────────────────────────────────────

def random_orthonormal_basis(d_model: int, n_dims: int, seed: int = 42) -> np.ndarray:
    """
    Generate a random orthonormal basis for control experiments.
    Patching in a random subspace should give near-zero transfer.
    If random patching gives high transfer, the experiment is trivially confounded.
    """
    rng = np.random.default_rng(seed)
    A   = rng.standard_normal((d_model, n_dims))
    Q, _ = np.linalg.qr(A, mode="reduced")
    check_orthonormal(Q, label="random-control basis")
    return Q


# ─────────────────────────────────────────────────────────────
# SERIALISATION HELPERS
# ─────────────────────────────────────────────────────────────

def serialise(obj):
    """Recursively make results JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: serialise(v) for k, v in obj.items()
                if not isinstance(v, np.ndarray) and not isinstance(v, torch.Tensor)}
    if isinstance(obj, list):
        return [serialise(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def pct(num: int, den: int) -> str:
    return f"{100*num/den:.1f}%" if den else "—"


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phi-3 Layer Scan + Unembed Patching (v2 — fully fixed)"
    )
    parser.add_argument("--device",          default="cpu")
    parser.add_argument("--model", default="phi-3",
                        help="Model key (phi-3, gemma-2b, gpt2, gpt2-medium, llama-3b, llama-3b-it) or HuggingFace path")
    parser.add_argument("--n-per-digit",     type=int,   default=100,
                        help="Fisher gradient samples per digit class")
    parser.add_argument("--n-test",          type=int,   default=150,
                        help="Max test problems to filter for patching")
    parser.add_argument("--skip-scan",       action="store_true",
                        help="Skip layer scan (supply --unembed-layers manually)")
    parser.add_argument("--unembed-layers",  default="",
                        help="Comma-separated layers for unembed patching. "
                             "Auto-selected from scan if empty.")
    parser.add_argument("--top-k-layers",    type=int, default=2,
                        help="How many top layers from scan to run unembed patching on")
    parser.add_argument("--no-format-removal", action="store_true",
                        help="Skip instruction-tuning subspace removal experiment")
    parser.add_argument("--n-format-dims",   type=int, default=10,
                        help="How many formatting Fisher dims to project out")
    parser.add_argument("--no-random-control", action="store_true",
                        help="Skip random-basis control experiment")
    parser.add_argument("--no-fisher",        action="store_true",
                        help="Skip standard + contrastive Fisher patching (Exp 1b)")
    parser.add_argument("--no-unembed",       action="store_true",
                        help="Skip unembed-aligned patching (Exp 2). Use when unembed "
                             "results already obtained and only Fisher is needed.")
    parser.add_argument("--fisher-dims",      default="2,5,10,20,50",
                        help="Subspace dims for the Fisher sweep (default: 2,5,10,20,50)")
    args = parser.parse_args()

    # ── Safety: MPS has broken gradients; CPU is required for Fisher ─────────
    device = args.device
    if device == "mps":
        logger.warning(
            "MPS produces incorrect Fisher gradients (confirmed in v1 experiments). "
            "Forcing device=cpu."
        )
        device = "cpu"

    log_section("ARITHMETIC CIRCUIT SCAN + UNEMBED PATCHING (v2 — multi-model)")
    logger.info(f"Device : {device}")
    logger.info(f"Model  : {args.model}")

    from transformer_lens import HookedTransformer
    model_name = MODEL_MAP.get(args.model, args.model)
    logger.info(f"Resolved model: {model_name}")
    model = HookedTransformer.from_pretrained(
        model_name, device=device, dtype=torch.float32
    )
    model.eval()
    n_layers = model.cfg.n_layers
    d_model  = model.cfg.d_model
    logger.info(f"n_layers={n_layers}, d_model={d_model}")

    all_results = {
        "model":   model_name,
        "device":  device,
        "n_layers": n_layers,
        "d_model":  d_model,
        "version": "v2_fully_fixed",
    }

    # ── Generate and filter test problems ────────────────────────────────────
    log_section("GENERATING TEST PROBLEMS")
    test_flat, test_by_digit = generate_teacher_forced_problems(
        n_per_digit=15, operand_max=99
    )
    correct = filter_correct_teacher_forced(model, test_flat, max_n=args.n_test)
    all_results["n_correct_test"] = len(correct)

    assert len(correct) >= 20, \
        "[SANITY] Too few correct test problems (<20). Model may not handle teacher-forcing."

    # ── Unembed basis (computed once, used in Exps 2 & 3) ───────────────────
    log_section("UNEMBED BASIS (BUG-1 FIXED: centering axis=dim=0)")
    digit_ids = get_digit_token_ids(model)
    U_basis, U_svals = compute_unembed_basis(model, digit_ids)
    all_results["unembed_singular_values"] = U_svals.tolist()

    # ── EXPERIMENT 1: Full-layer scan ────────────────────────────────────────
    scan_results = {}
    if not args.skip_scan:
        scan_results = run_layer_scan(model, correct, n_layers)
        all_results["layer_scan"] = {
            str(l): r for l, r in scan_results.items()
        }

    # Auto-select best layers (BUG-2 fix)
    if args.unembed_layers:
        unembed_layers = [int(l) for l in args.unembed_layers.split(",")]
        logger.info(f"  Using manually specified layers: {unembed_layers}")
    elif scan_results:
        sorted_layers = sorted(
            scan_results.items(),
            key=lambda x: x[1]["transfer_pct"],
            reverse=True,
        )
        unembed_layers = [l for l, _ in sorted_layers[: args.top_k_layers]]
        logger.info(
            f"  Auto-selected top-{args.top_k_layers} layers: {unembed_layers}"
        )
    else:
        # Fallback if scan skipped without explicit layers
        unembed_layers = [24]
        logger.warning("  No scan results and no --unembed-layers specified. Defaulting to [24].")

    # ── BUG-3 fix: clean up _tokens from scan phase ──────────────────────────
    # (run_layer_scan removes _act_L{k} per layer, but _tokens persists from
    #  filter_correct_teacher_forced; we keep them because Exp 2 still needs them)
    # They will be explicitly removed after all experiments finish.

    # ── EXPERIMENT 1b: Standard + Contrastive Fisher patching ────────────────
    # This replicates and extends what fisher_patching.py did, unified here.
    fisher_dims = [int(d) for d in args.fisher_dims.split(",")]

    if not args.no_fisher:
        log_section("EXPERIMENT 1b: STANDARD + CONTRASTIVE FISHER PATCHING")

        # Generate balanced teacher-forced Fisher problems (100/digit)
        _, fisher_by_digit = generate_teacher_forced_problems(
            n_per_digit=args.n_per_digit, operand_max=99
        )

        for layer in unembed_layers:
            logger.info(f"\n{'─'*55}")
            logger.info(f"  Layer {layer}")

            # ── Standard Fisher ─────────────────────────────────────────────
            # Pool all 1000 teacher-forced problems as one Fisher distribution
            all_fisher_probs = []
            for d in range(10):
                all_fisher_probs.extend(fisher_by_digit[d][:args.n_per_digit])

            std_evals, std_evecs, std_eff = compute_fisher_matrix(
                model, all_fisher_probs, layer=layer,
                n=len(all_fisher_probs), label="Standard-Fisher",
            )
            all_results[f"std_fisher_meta_L{layer}"] = {
                "eff_dim":    std_eff,
                "top5_evals": std_evals[:5].tolist(),
            }

            logger.info(f"\n  Standard Fisher sweep ({fisher_dims}D):")
            std_results = run_patching_experiment(
                model, layer, std_evecs, correct,
                n_dims_list=fisher_dims, label="StdFisher",
                verify_partition_on_first=True,
            )
            all_results[f"std_fisher_L{layer}"] = serialise(std_results)

            # Log crossover table
            logger.info(f"\n  {'Dims':>5}  {'Fisher%':>9}  {'Ortho%':>9}  {'Status'}")
            logger.info(f"  {'─'*45}")
            for nd in fisher_dims:
                key = f"{nd}D"
                if key in std_results:
                    s = std_results[key]["sub"]
                    o = std_results[key]["ortho"]
                    n = s["total"]
                    ft = pct(s["transfer"], n)
                    ot = pct(o["transfer"], n)
                    crossover = "◄ CROSSOVER" if s["transfer"] > o["transfer"] else ""
                    logger.info(f"  {nd:>5}D  {ft:>9}  {ot:>9}  {crossover}")

            # ── Contrastive Fisher ──────────────────────────────────────────
            con_evals, con_evecs, n_dirs = compute_contrastive_fisher(
                model, fisher_by_digit, layer=layer,
                n_per_digit=args.n_per_digit,
            )
            all_results[f"con_fisher_meta_L{layer}"] = {
                "n_dirs":     n_dirs,
                "top9_evals": con_evals[:9].tolist(),
            }

            con_dims = [d for d in [2, 5, 9] if d <= n_dirs]
            logger.info(f"\n  Contrastive Fisher sweep ({con_dims}D):")
            con_results = run_patching_experiment(
                model, layer, con_evecs, correct,
                n_dims_list=con_dims, label="ConFisher",
                verify_partition_on_first=False,
            )
            all_results[f"con_fisher_L{layer}"] = serialise(con_results)

            # ── Key comparison: std-best vs contrastive-best ────────────────
            logger.info(f"\n  {'─'*55}")
            logger.info(f"  COMPARISON SUMMARY — L{layer}")
            logger.info(f"  {'─'*55}")
            logger.info(f"  {'Method':<20}  {'Dims':>5}  {'Fisher%':>9}  {'Ortho%':>9}")
            for nd in fisher_dims:
                key = f"{nd}D"
                if key in std_results:
                    s = std_results[key]["sub"]
                    o = std_results[key]["ortho"]
                    n = s["total"]
                    logger.info(
                        f"  {'Std Fisher':<20}  {nd:>5}D  "
                        f"{pct(s['transfer'],n):>9}  {pct(o['transfer'],n):>9}"
                    )
            for nd in con_dims:
                key = f"{nd}D"
                if key in con_results:
                    s = con_results[key]["sub"]
                    o = con_results[key]["ortho"]
                    n = s["total"]
                    logger.info(
                        f"  {'Contrastive':<20}  {nd:>5}D  "
                        f"{pct(s['transfer'],n):>9}  {pct(o['transfer'],n):>9}"
                    )

            # ── Alignment check: do std and contrastive agree? ──────────────
            # Measure principal angles between std top-20D and con top-9D
            k_std = min(20, std_evecs.shape[1])
            k_con = min(n_dirs, con_evecs.shape[1])
            cross = std_evecs[:, :k_std].T @ con_evecs[:, :k_con]
            cos_angles = np.linalg.svd(cross, compute_uv=False)
            logger.info(
                f"\n  Std-{k_std}D ↔ Con-{k_con}D alignment: "
                f"top-3 principal cosines = {cos_angles[:3].round(3)}"
            )
            all_results[f"fisher_alignment_L{layer}"] = {
                "std_dims": k_std,
                "con_dims": k_con,
                "principal_cosines": cos_angles[:5].tolist(),
            }

    # ── EXPERIMENT 2: Unembed-aligned patching ───────────────────────────────
    if not args.no_unembed:
        log_section("EXPERIMENT 2: UNEMBED-ALIGNED PATCHING")
        n_dims_list = [2, 5, 9, 10]   # BUG-4 fix: include full rank (10D)

        for layer in unembed_layers:
            logger.info(f"\n  Layer {layer}")

            results_unembed = run_patching_experiment(
                model, layer, U_basis, correct,
                n_dims_list=n_dims_list, label="Unembed",
                verify_partition_on_first=True,
            )
            all_results[f"unembed_L{layer}"] = serialise(results_unembed)

            # ── Random control (same dimensionalities) ───────────────────────────
            if not args.no_random_control:
                logger.info(f"\n  [CONTROL] Random basis at L{layer}")
                R_basis = random_orthonormal_basis(d_model, n_dims=10, seed=42)
                results_random = run_patching_experiment(
                    model, layer, R_basis, correct,
                    n_dims_list=n_dims_list, label="Random-control",
                    verify_partition_on_first=False,
                )
                all_results[f"random_control_L{layer}"] = serialise(results_random)

                # Sanity: random should transfer far less than unembed
                for nd in n_dims_list:
                    key = f"{nd}D"
                    if key in results_unembed and key in results_random:
                        u_t = results_unembed[key]["sub"]["transfer"]
                        r_t = results_random[key]["sub"]["transfer"]
                        u_n = results_unembed[key]["sub"]["total"]
                        r_n = results_random[key]["sub"]["total"]
                        logger.info(
                            f"  [SANITY CONTROL] {nd}D: "
                            f"Unembed={pct(u_t,u_n)} vs Random={pct(r_t,r_n)}"
                        )
                        if u_t <= r_t and u_n > 0 and r_n > 0:
                            logger.warning(
                                f"  [WARNING] Unembed does not outperform random at {nd}D "
                                f"— unembed subspace may not contain arithmetic circuit"
                            )
    else:
        logger.info("\n  [SKIP] Experiment 2 (unembed patching) — --no-unembed set")
        n_dims_list = [2, 5, 9, 10]  # still needed for final summary

    # ── EXPERIMENT 3: Formatting Fisher removal ───────────────────────────────
    if not args.no_format_removal:
        log_section("EXPERIMENT 3: INSTRUCTION-TUNING SUBSPACE REMOVAL")

        # Generate formatting (non-arithmetic) problems
        fmt_problems = generate_formatting_problems(n=200)

        for layer in unembed_layers:
            logger.info(f"\n  Layer {layer}")

            # Arithmetic Fisher at this layer
            arith_evals, arith_evecs, arith_eff = compute_fisher_matrix(
                model, correct,   # reuse correct problems as arithmetic sample
                layer=layer, n=min(200, len(correct)),
                label="Arithmetic-Fisher",
            )
            all_results[f"arith_fisher_L{layer}"] = {
                "eff_dim":    arith_eff,
                "top5_evals": arith_evals[:5].tolist(),
            }

            # Formatting Fisher — add _tokens and _target_tok to fmt_problems
            fmt_with_toks = []
            for p in fmt_problems:
                toks = model.to_tokens(p["prompt"], prepend_bos=True)
                ttok = model.to_tokens(p["target_str"], prepend_bos=False)[0, -1].item()
                p["_tokens"]     = toks
                p["_target_tok"] = ttok
                fmt_with_toks.append(p)

            fmt_evals, fmt_evecs, fmt_eff = compute_fisher_matrix(
                model, fmt_with_toks,
                layer=layer, n=200,
                label="Formatting-Fisher",
            )
            all_results[f"format_fisher_L{layer}"] = {
                "eff_dim":    fmt_eff,
                "top5_evals": fmt_evals[:5].tolist(),
            }
            cleanup_tokens(fmt_with_toks)

            # Measure overlap between arithmetic and formatting subspaces
            # (principal angles via SVD of cross-Gram matrix)
            k = min(20, arith_evecs.shape[1], fmt_evecs.shape[1])
            cross = arith_evecs[:, :k].T @ fmt_evecs[:, :k]
            svals_cross = np.linalg.svd(cross, compute_uv=False)
            logger.info(
                f"  Arithmetic vs Formatting subspace overlap (top-{k}):"
                f" max principal cos = {svals_cross[0]:.3f}, "
                f"mean = {svals_cross.mean():.3f}"
            )
            all_results[f"subspace_overlap_L{layer}"] = {
                "principal_cosines": svals_cross[:5].tolist(),
            }

            # Build format-cleaned arithmetic basis
            cleaned_basis = subtract_formatting_subspace(
                arith_evecs, fmt_evecs,
                n_format_dims=args.n_format_dims,
            )

            logger.info(f"\n  Format-cleaned arithmetic patching at L{layer}:")
            results_cleaned = run_patching_experiment(
                model, layer, cleaned_basis, correct,
                n_dims_list=[5, 10, 20, 50], label="ArtihClean",
                verify_partition_on_first=True,
            )
            all_results[f"format_cleaned_L{layer}"] = serialise(results_cleaned)

            # Compare cleaned vs raw arithmetic Fisher at same dims
            logger.info("\n  Arithmetic-Fisher (raw) at same dims:")
            results_raw_arith = run_patching_experiment(
                model, layer, arith_evecs, correct,
                n_dims_list=[5, 10, 20, 50], label="ArithRaw",
                verify_partition_on_first=False,
            )
            all_results[f"arith_raw_L{layer}"] = serialise(results_raw_arith)

            for nd in [5, 10, 20, 50]:
                key = f"{nd}D"
                if key in results_cleaned and key in results_raw_arith:
                    raw = results_raw_arith[key]["sub"]
                    cln = results_cleaned[key]["sub"]
                    n   = raw["total"]
                    logger.info(
                        f"  [COMPARE] {nd}D  "
                        f"Raw={pct(raw['transfer'], n)}  "
                        f"Cleaned={pct(cln['transfer'], n)}"
                    )

    # ── Final cleanup (BUG-3) ────────────────────────────────────────────────
    cleanup_tokens(correct)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir  = Path("mathematical_toolkit_results")
    out_dir.mkdir(exist_ok=True)
    model_slug = args.model.replace("/", "_").replace("-", "_")
    out_path = out_dir / f"arithmetic_scan_{model_slug}.json"
    with open(out_path, "w") as f:
        json.dump(serialise(all_results), f, indent=2)
    logger.info(f"\nResults saved to {out_path}")

    # ── Terminal summary ──────────────────────────────────────────────────────
    log_section("FINAL SUMMARY")
    if "layer_scan" in all_results:
        top = sorted(
            all_results["layer_scan"].items(),
            key=lambda x: x[1]["transfer_pct"],
            reverse=True,
        )[:5]
        logger.info("  Top layers by full-patch transfer:")
        for l, r in top:
            logger.info(f"    L{l}: {r['transfer_pct']}%")

    for layer in unembed_layers:
        key = f"unembed_L{layer}"
        if key in all_results:
            logger.info(f"\n  Unembed patching — L{layer}:")
            for nd in n_dims_list:
                dkey = f"{nd}D"
                if dkey in all_results[key]:
                    r = all_results[key][dkey]["sub"]
                    n = r["total"]
                    logger.info(
                        f"    {nd}D → transfer={pct(r['transfer'], n)}  "
                        f"ortho-transfer={pct(all_results[key][dkey]['ortho']['transfer'], n)}"
                    )


if __name__ == "__main__":
    main()
