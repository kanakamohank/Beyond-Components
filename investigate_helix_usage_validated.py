#!/usr/bin/env python3
"""
VALIDATED Helix Usage Investigation

Fixed version that aligns with the actual findings from online_svd_scanner.py:
  1. Uses OV-matrix SVD (W_V @ W_O) — not raw activation SVD
  2. Phase-corrected metric with offset sweep (not buggy raw correlation)
  3. Fourier isolation to extract T=N component from superposition
  4. Causal phase-shift test (rotate in helix plane, check output shift)
  5. Cross-task helix persistence (does the same helix appear for
     counting, dates, ordering, arithmetic?)
  6. Subspace Vocabulary Projection — sweep theta around the OV circle
     and project through W_U to decode what tokens the helix maps to.
     Uses Vt rows (output/writing directions) per TransformerLens
     row-vector convention: y = x @ W_OV = x @ U @ S @ Vt
  7. Proper operand-position extraction and names_filter for memory safety

References:
  - online_svd_scanner.py: collect_activations, phase_match_optimized,
    fourier_isolate_TN, causal_phase_shift_test, analyze_helix_corrected
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
import re
import gc
import random
import itertools
import time
from typing import Dict, List, Tuple, Any, Optional
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONSTANTS: Corrected helix coordinates from experimental logs
# ─────────────────────────────────────────────────────────────
KNOWN_HELIX_CONFIG = {
    "gpt2-small": {
        "helix_layer": 10, "helix_head": 2,
        "svd_dims": (2, 7),
        "best_resid_layer": 10,
        "period": 99.0,
        "layer_range": range(6, 12),
        "notes": "T~99 monotone helix, no modular structure",
    },
    "google/gemma-2-2b": {
        "helix_layer": 9, "helix_head": 1,
        "svd_dims": (1, 2),
        "best_resid_layer": 9,
        "period": 9.9,
        "layer_range": range(6, 15),
        "notes": "T=3.3/9.9 multi-frequency, CV=0.293, Lin=0.999",
    },
    "google/gemma-7b": {
        "helix_layer": 14, "helix_head": 2,
        "svd_dims": (2, 5),
        "best_resid_layer": 21,
        "period": 10.0,
        "layer_range": range(14, 26),
        "notes": "Phase_corr=0.885, CV=0.392 at L21 operand",
    },
    "microsoft/Phi-3-mini-4k-instruct": {
        "helix_layer": 24, "helix_head": 28,
        "svd_dims": (3, 7),
        "best_resid_layer": 25,
        "period": 11.74,
        "layer_range": range(20, 32),
        "notes": "Clean clock after Fourier isolation CV=0.102",
    },
    "EleutherAI/gpt-neo-125M": {
        "helix_layer": 2, "helix_head": 2,
        "svd_dims": (0, 1),
        "best_resid_layer": 2,
        "period": 99.0,
        "layer_range": range(0, 12),
        "notes": "Head scan: L2 H2 35.6x MLP selectivity",
    },
    "EleutherAI/gpt-j-6B": {
        "helix_layer": 3, "helix_head": 13,
        "svd_dims": (2, 3),
        "best_resid_layer": 3,
        "period": 99.0,
        "layer_range": range(0, 28),
        "notes": "σ=1.014 best ratio, CV=0.273, Lin=0.928, T=99",
    },
    "gpt2-medium": {
        "helix_layer": 19, "helix_head": 12,
        "svd_dims": (0, 1),
        "best_resid_layer": 19,
        "period": 99.0,
        "layer_range": range(12, 24),
        "notes": "Cross-scale GPT-2 family check, dims TBD by sweep",
    },
    "meta-llama/Llama-3.2-3B": {
        "helix_layer": 1, "helix_head": 17,
        "svd_dims": (2, 6),
        "best_resid_layer": 13,
        "period": 99.0,
        "layer_range": range(1, 20),
        "notes": "T=99 + T=2 parity, no modular structure",
    },
    "meta-llama/Llama-3.2-3B-Instruct": {
        "helix_layer": 1, "helix_head": 17,
        "svd_dims": (4, 6),
        "best_resid_layer": 13,
        "period": 2.0,
        "layer_range": range(1, 20),
        "notes": "CV=0.255, Lin=0.917, T=2.0 parity, instruct-tuned",
    },
}


class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass


# ═════════════════════════════════════════════════════════════
# CORE METRICS (ported from online_svd_scanner.py)
# ═════════════════════════════════════════════════════════════

def phase_match_optimized(angles_rad: np.ndarray,
                          ones_digits: np.ndarray,
                          period: float = 10.0,
                          n_offsets: int = 720) -> Tuple[float, float]:
    """Phase-corrected circular alignment score.

    Sweeps over phase offsets to find the best alignment between
    measured angles and ideal modular angles.  Returns (score, phi).
    """
    ideal_base = (ones_digits / period) * 2 * np.pi
    best_alignment = -1.0
    best_phi = 0.0
    for phi in np.linspace(0, 2 * np.pi, n_offsets, endpoint=False):
        alignment = float(np.mean(np.cos(angles_rad - ideal_base - phi)))
        if alignment > best_alignment:
            best_alignment = alignment
            best_phi = phi
    return best_alignment, best_phi


def fourier_isolate_TN(acts_tensor: torch.Tensor,
                       valid_ns: list,
                       period: float) -> Tuple[torch.Tensor, float]:
    """Isolates the T=N Fourier component from activations."""
    ns = np.array(valid_ns, dtype=np.float32)
    cosT = np.cos(2 * np.pi * ns / period)
    sinT = np.sin(2 * np.pi * ns / period)
    basis = np.stack([cosT, sinT, np.ones(len(ns))], axis=1)

    acts_np = acts_tensor.numpy()
    A, _, _, _ = np.linalg.lstsq(basis, acts_np, rcond=None)
    acts_T = basis[:, :2] @ A[:2, :]

    ss_total = np.sum((acts_np - acts_np.mean(0)) ** 2)
    r2 = float(np.sum(acts_T ** 2) / (ss_total + 1e-8))
    return torch.tensor(acts_T, dtype=torch.float32), r2


# ═════════════════════════════════════════════════════════════
# INVESTIGATOR CLASS
# ═════════════════════════════════════════════════════════════

class HelixUsageInvestigator:
    """Validates helix usage with OV-SVD, phase-correction, Fourier
    isolation, causal intervention, and cross-task persistence tests."""

    def __init__(self, model_name: str = "gpt2-small"):
        self.model_name = model_name
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        # Look up corrected config
        cfg = KNOWN_HELIX_CONFIG.get(model_name)
        if cfg:
            self.helix_layer = cfg["helix_layer"]
            self.helix_head = cfg["helix_head"]
            self.svd_dims = cfg["svd_dims"]
            self.best_resid_layer = cfg["best_resid_layer"]
            self.expected_period = cfg["period"]
            self.layer_range = cfg["layer_range"]
        else:
            warnings.warn(f"Unknown model {model_name} — using heuristic defaults")
            self.helix_layer = None
            self.helix_head = None
            self.svd_dims = (0, 1)
            self.best_resid_layer = None
            self.expected_period = 10.0
            self.layer_range = None

        self._load_and_validate_model()
        self._set_and_validate_target_head()
        self._run_initial_sanity_checks()

    # ── Model loading ──────────────────────────────────────────

    def _load_and_validate_model(self):
        print(f"Loading {self.model_name}...")
        try:
            self.model = HookedTransformer.from_pretrained(
                self.model_name, device=self.device, dtype=torch.bfloat16
            )
            self.model.eval()
            print(f"  Model loaded: {self.model.cfg.n_layers}L, "
                  f"{self.model.cfg.n_heads}H, d={self.model.cfg.d_model}")
        except Exception as e:
            raise ValidationError(f"Failed to load model: {e}")

    def _set_and_validate_target_head(self):
        if self.helix_layer is None:
            self.helix_layer = self.model.cfg.n_layers // 2
            self.helix_head = 0
            self.best_resid_layer = self.helix_layer
            self.layer_range = range(
                max(0, self.helix_layer - 6),
                min(self.model.cfg.n_layers, self.helix_layer + 6)
            )

        assert self.helix_layer < self.model.cfg.n_layers, \
            f"Layer {self.helix_layer} >= {self.model.cfg.n_layers}"
        assert self.helix_head < self.model.cfg.n_heads, \
            f"Head {self.helix_head} >= {self.model.cfg.n_heads}"
        print(f"  Target: L{self.helix_layer} H{self.helix_head}, "
              f"SVD dims {self.svd_dims}, "
              f"resid L{self.best_resid_layer}, T={self.expected_period}")

    def _run_initial_sanity_checks(self):
        print("  Sanity checks...")
        hook = f"blocks.{self.best_resid_layer}.hook_resid_pre"
        tokens = self.model.to_tokens("The number 42")
        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens, names_filter=hook)
        assert hook in cache, f"Hook {hook} missing from cache"
        shape = cache[hook].shape
        assert shape[0] == 1 and shape[2] == self.model.cfg.d_model
        print(f"  Sanity OK  (activation shape {tuple(shape)})")

    # ── Activation collection (with operand-position logic) ───

    def collect_activations(self, layer: int, max_n: int = 100,
                            fixed_b: int = 5,
                            position: str = "operand") -> Tuple[torch.Tensor, List[int]]:
        """Collect residual-stream activations at the operand or final position."""
        hook_name = f"blocks.{layer}.hook_resid_pre"
        acts, valid_ns = [], []

        for n in range(max_n):
            prompt = f"What is {n} + {fixed_b}?"
            tokens = self.model.to_tokens(prompt)
            str_tokens = self.model.to_str_tokens(prompt)

            if position == "operand":
                target_pos = -1
                for idx, tok in enumerate(str_tokens):
                    if '+' in tok:
                        target_pos = idx - 1
                        break
                if target_pos <= 0:
                    continue
            else:
                target_pos = len(str_tokens) - 1

            with torch.no_grad():
                _, cache = self.model.run_with_cache(
                    tokens, names_filter=hook_name
                )
            resid = cache[hook_name][0, target_pos, :].cpu().float()
            acts.append(resid)
            valid_ns.append(n)

        if not acts:
            raise ValidationError(f"No valid activations at L{layer}")
        return torch.stack(acts), valid_ns

    # ── OV-matrix SVD helix analysis ──────────────────────────

    def analyze_ov_helix(self, layer: int, head: int,
                         acts_tensor: torch.Tensor,
                         valid_ns: List[int],
                         label: str = "") -> Dict[str, Any]:
        """Analyze helix in the OV reading subspace (correct method)."""
        W_V = self.model.W_V[layer, head].detach().float().cpu()
        W_O = self.model.W_O[layer, head].detach().float().cpu()
        W_OV = W_V @ W_O
        U, S, Vt = torch.linalg.svd(W_OV, full_matrices=False)

        top_U = U[:, :10]
        acts_in_U = (acts_tensor @ top_U).numpy()

        n_samples, n_features = acts_in_U.shape
        safe_n = min(10, n_samples, n_features)
        pca = PCA(n_components=safe_n)
        acts_pca = pca.fit_transform(acts_in_U)

        ones_arr = np.array([n % 10 for n in valid_ns])
        best_score, best_result = -np.inf, None

        safe_pairs = min(8, safe_n)
        for k1, k2 in itertools.combinations(range(safe_pairs), 2):
            coords = torch.tensor(acts_pca[:, [k1, k2]]).float()
            coords = coords - coords.mean(0)
            radii = coords.norm(dim=1)
            angles = torch.atan2(coords[:, 1], coords[:, 0]).numpy()

            rm = radii.mean().item()
            if rm < 1e-8:
                continue
            cv = (radii.std() / rm).item()

            if np.abs(np.diff(angles)).mean() >= np.pi * 0.75:
                continue

            unwrapped = np.unwrap(angles)
            lin_raw = abs(float(np.corrcoef(valid_ns, unwrapped)[0, 1]))

            slope, _ = np.polyfit(valid_ns, unwrapped, 1)
            period = abs(2 * np.pi / slope) if abs(slope) > 1e-8 else np.inf

            phase_corr, best_phi = phase_match_optimized(
                angles, ones_arr, period=self.expected_period
            )
            period_err = abs(period - self.expected_period) / self.expected_period
            score = phase_corr - cv - 0.3 * period_err

            if score > best_score:
                best_score = score
                best_result = dict(
                    k1=k1, k2=k2, cv=cv, lin_raw=lin_raw,
                    phase_corr=phase_corr,
                    phase_offset_deg=float(np.degrees(best_phi)),
                    period=period, score=score,
                )

        if best_result is None:
            print(f"  {label}: No valid PC pairs found.")
            return {"error": "no_valid_pairs"}

        r = best_result
        print(f"\n  {label}")
        print(f"    PC pair       : ({r['k1']}, {r['k2']})")
        print(f"    Radius CV     : {r['cv']:.4f}   (target < 0.20)")
        print(f"    Phase_corr    : {r['phase_corr']:.4f}   (target > 0.85)")
        print(f"    Phase offset  : {r['phase_offset_deg']:.1f} deg")
        print(f"    Period T      : {r['period']:.2f}   (target ~{self.expected_period})")
        print(f"    Lin (raw n)   : {r['lin_raw']:.4f}")

        t_ok = abs(r['period'] - self.expected_period) / self.expected_period < 0.15
        if r['cv'] < 0.20 and r['phase_corr'] > 0.85 and t_ok:
            r['verdict'] = "CLEAN_CLOCK"
            print(f"    --> CLEAN T={self.expected_period} CLOCK FACE CONFIRMED")
        elif r['phase_corr'] > 0.85 and t_ok:
            r['verdict'] = "IMPURE_CLOCK"
            print(f"    --> T={self.expected_period} CLOCK present but impure CV")
        elif r['lin_raw'] > 0.90:
            r['verdict'] = "MONOTONE_HELIX"
            print(f"    --> MONOTONE HELIX (Vector Translation, not modular)")
        else:
            r['verdict'] = "NO_HELIX"
            print(f"    --> NO CLEAN HELIX")

        return r

    # ── Direct residual-stream PCA helix analysis ─────────────

    def analyze_residual_helix(self, acts_tensor: torch.Tensor,
                               valid_ns: List[int],
                               label: str = "") -> Dict[str, Any]:
        """Analyze helix directly in residual stream via PCA."""
        ones_arr = np.array([n % 10 for n in valid_ns])
        acts_np = acts_tensor.numpy()
        acts_c = acts_np - acts_np.mean(0, keepdims=True)

        n_samples, n_features = acts_c.shape
        safe_n = min(15, n_samples, n_features)
        pca = PCA(n_components=safe_n)
        acts_pca = pca.fit_transform(acts_c)

        best_score, best_result = -np.inf, None
        safe_pairs = min(12, safe_n)

        for k1, k2 in itertools.combinations(range(safe_pairs), 2):
            coords = torch.tensor(acts_pca[:, [k1, k2]]).float()
            coords = coords - coords.mean(0)
            radii = coords.norm(dim=1)
            angles = torch.atan2(coords[:, 1], coords[:, 0]).numpy()

            rm = radii.mean().item()
            if rm < 1e-8:
                continue
            cv = (radii.std() / rm).item()

            if np.abs(np.diff(angles)).mean() >= np.pi * 0.75:
                continue

            unwrapped = np.unwrap(angles)
            lin_raw = abs(float(np.corrcoef(valid_ns, unwrapped)[0, 1]))
            if np.isnan(lin_raw):
                lin_raw = 0.0

            slope, _ = np.polyfit(valid_ns, unwrapped, 1)
            period = abs(2 * np.pi / slope) if abs(slope) > 1e-8 else np.inf

            phase_corr, best_phi = phase_match_optimized(
                angles, ones_arr, period=self.expected_period
            )
            period_err = abs(period - self.expected_period) / self.expected_period
            score = phase_corr - cv - 0.3 * period_err

            if score > best_score:
                best_score = score
                best_result = dict(
                    k1=k1, k2=k2, cv=cv, lin_raw=lin_raw,
                    phase_corr=phase_corr,
                    phase_offset_deg=float(np.degrees(best_phi)),
                    period=period, score=score,
                )

        if best_result is None:
            print(f"  {label}: No valid PC pairs.")
            return {"error": "no_valid_pairs"}

        r = best_result
        print(f"\n  {label}")
        print(f"    CV={r['cv']:.3f}  Phase_corr={r['phase_corr']:.3f}  "
              f"T={r['period']:.2f}  Lin={r['lin_raw']:.3f}")
        return r

    # ── Fourier isolation + circle check ──────────────────────

    def test_fourier_isolation(self, acts_tensor: torch.Tensor,
                               valid_ns: List[int],
                               period: float,
                               label: str = "") -> Dict[str, Any]:
        """Isolate the T=period Fourier component and check for circle."""
        acts_iso, r2 = fourier_isolate_TN(acts_tensor, valid_ns, period)
        print(f"\n  {label} — Fourier Isolation T={period:.1f}")
        print(f"    Variance explained: {r2:.1%}")

        ones = np.array([n % int(round(period)) for n in valid_ns])
        acts_np = acts_iso.numpy()
        acts_c = acts_np - acts_np.mean(0, keepdims=True)

        safe_n = min(5, acts_c.shape[0], acts_c.shape[1])
        if safe_n < 2:
            return {"error": "insufficient_rank", "r2": r2}

        pca = PCA(n_components=safe_n)
        pca_out = pca.fit_transform(acts_c)

        best_score, best = -np.inf, None
        n_pairs = min(4, pca_out.shape[1])

        for k1, k2 in itertools.combinations(range(n_pairs), 2):
            coords = torch.tensor(pca_out[:, [k1, k2]]).float()
            coords = coords - coords.mean(0)
            radii = coords.norm(dim=1)
            angles = torch.atan2(coords[:, 1], coords[:, 0]).numpy()

            rm = radii.mean().item()
            if rm < 1e-8:
                continue
            cv = (radii.std() / rm).item()

            if np.abs(np.diff(angles)).mean() >= np.pi * 0.75:
                continue

            best_phase, best_phi = -1.0, 0.0
            for phi in np.linspace(0, 2 * np.pi, 360, endpoint=False):
                p = float(np.mean(np.cos(angles - (ones / period) * 2 * np.pi - phi)))
                if p > best_phase:
                    best_phase, best_phi = p, phi

            score = best_phase - cv
            if score > best_score:
                best_score = score
                best = dict(k1=k1, k2=k2, cv=cv, phase_corr=best_phase,
                            phase_offset_deg=float(np.degrees(best_phi)),
                            r2=r2)

        if best is None:
            print(f"    No valid pairs after isolation")
            return {"error": "no_valid_pairs", "r2": r2}

        r = best
        cv_ok = r['cv'] < 0.20
        pc_ok = r['phase_corr'] > 0.85
        print(f"    CV    : {r['cv']:.4f}  {'< 0.20' if cv_ok else '> 0.20 (high)'}")
        print(f"    Phase : {r['phase_corr']:.4f}  {'> 0.85' if pc_ok else '< 0.85'}")

        if cv_ok and pc_ok:
            r['verdict'] = "CLEAN_CLOCK_AFTER_ISOLATION"
            print(f"    --> CLEAN CLOCK FACE after isolation")
        else:
            r['verdict'] = "NO_CLEAN_CLOCK_AFTER_ISOLATION"
            print(f"    --> No clean clock even after isolation")
        return r

    # ── Causal phase-shift test ───────────────────────────────

    def causal_phase_shift_test(self, layer: int, head: int,
                                valid_ns: List[int],
                                period: float = 10.0,
                                n_tests: int = 20,
                                seed: int = 42) -> Dict[str, Any]:
        """Rotate representation in helix plane, check if output shifts.

        Uses U columns (reading directions) for the rotation because we
        are modifying hook_resid_pre — the INPUT to the attention layer.
        The head reads from U-column space, so rotating in that plane
        changes what the head perceives.
        """
        random.seed(seed)
        torch.manual_seed(seed)

        d1, d2 = self.svd_dims
        W_V = self.model.W_V[layer, head].detach().float().cpu()
        W_O = self.model.W_O[layer, head].detach().float().cpu()
        W_OV = W_V @ W_O
        U, S, _ = torch.linalg.svd(W_OV, full_matrices=False)
        v1, v2 = U[:, d1], U[:, d2]

        ratio = (S[d1] / S[d2]).item()
        print(f"  SVD dims ({d1}, {d2}), sigma ratio = {ratio:.3f}")

        hook_name = f"blocks.{layer}.hook_resid_pre"
        print(f"\n  Causal Phase-Shift: L{layer} H{head}, T={period}")

        no_carry, carry = [], []
        for _ in range(n_tests * 6):
            if len(no_carry) >= n_tests and len(carry) >= n_tests:
                break
            a, b = random.randint(5, 45), random.randint(5, 45)
            delta = random.choice([1, 2])
            if (a % 10) + delta > 9:
                continue
            ones_sum = (a % 10) + (b % 10)
            case = (a, b, delta)
            if ones_sum >= 10 and len(carry) < n_tests:
                carry.append(case)
            elif ones_sum < 10 and len(no_carry) < n_tests:
                no_carry.append(case)

        results = {}
        for case_type, cases in [("no_carry", no_carry), ("carry", carry)]:
            successes, details = 0, []
            for a, b, delta in cases:
                prompt = f"Math:\n10 + 10 = 20\n21 + 13 = 34\n{a} + {b} ="
                str_tokens = self.model.to_str_tokens(prompt)
                tokens = self.model.to_tokens(prompt)

                target_pos = -1
                for idx, tok in enumerate(str_tokens):
                    if '+' in tok:
                        target_pos = idx - 1
                if target_pos <= 0:
                    continue

                with torch.no_grad():
                    _, cache = self.model.run_with_cache(
                        tokens, names_filter=hook_name
                    )
                h_orig = cache[hook_name][0, target_pos, :].cpu().float()

                theta = 2.0 * np.pi * delta / period
                c1 = (h_orig @ v1).item()
                c2 = (h_orig @ v2).item()
                ct, st = np.cos(theta), np.sin(theta)
                h_rot = (h_orig
                         - c1 * v1 - c2 * v2
                         + (ct * c1 - st * c2) * v1
                         + (st * c1 + ct * c2) * v2)

                def hook(value, hook):
                    if value.shape[1] > target_pos:
                        value[0, target_pos, :] = h_rot.to(value.dtype)
                    return value

                with torch.no_grad():
                    with self.model.hooks(fwd_hooks=[(hook_name, hook)]):
                        out_tokens = self.model.generate(
                            tokens, max_new_tokens=4,
                            prepend_bos=False, verbose=False
                        )

                out_str = self.model.tokenizer.decode(
                    out_tokens[0, tokens.shape[1]:]
                )
                nums = re.findall(r'\d+', out_str)
                pred_o = int(nums[0]) % 10 if nums else -1
                expected = (a + b + delta) % 10
                ok = pred_o == expected
                successes += int(ok)
                details.append(
                    f"    {a}+{b} rot+{delta}: pred={pred_o} exp={expected} "
                    f"{'OK' if ok else 'FAIL'}  (raw: '{out_str.strip()}')"
                )

            rate = successes / max(len(cases), 1)
            results[case_type] = rate
            print(f"\n  [{case_type.upper()}]")
            for d in details[:6]:
                print(d)
            if len(details) > 6:
                print(f"    ... ({len(details) - 6} more)")
            print(f"  Success: {successes}/{len(cases)} = {rate:.1%}")

        return results

    # ── Subspace Vocabulary Projection ────────────────────────

    def subspace_vocab_projection(self, layer: int = None, head: int = None,
                                  dim1: int = None, dim2: int = None,
                                  n_angles: int = 36,
                                  output_dir: str = "helix_usage_validated"
                                  ) -> Dict[str, Any]:
        """Sweep theta around the OV helix circle and project through W_U.

        This answers: 'What does the circle MEAN in vocabulary space?'

        TransformerLens convention (row-vector): y = x @ W_OV = x @ U @ S @ Vt
          - U columns = INPUT/READING directions (what the head reads from
            the residual stream).  Projecting U columns through W_U shows
            which tokens' embeddings align with each angle.
          - Vt rows = OUTPUT/WRITING directions (what the head writes back
            to the residual stream).  Projecting Vt rows through W_U shows
            which tokens the head *promotes* at each angle.  This is the
            primary "decode the circle" result (the ML researcher's
            "winning move").

        For tied-embedding models (GPT-2, GPT-J, Gemma): W_U = W_E^T, so
        the reading lens via W_U is equivalent to using W_E.
        For untied-embedding models (Llama): W_E is used for reading lens
        if available, otherwise W_U is used as an approximation.
        """
        layer = layer if layer is not None else self.helix_layer
        head = head if head is not None else self.helix_head
        dim1 = dim1 if dim1 is not None else self.svd_dims[0]
        dim2 = dim2 if dim2 is not None else self.svd_dims[1]

        print(f"\n{'=' * 65}")
        print(f"SUBSPACE VOCABULARY PROJECTION")
        print(f"Layer {layer}, Head {head}, SVD dims ({dim1}, {dim2})")
        print(f"{'=' * 65}")

        # Extract OV-SVD
        W_V = self.model.W_V[layer, head].detach().float().cpu()
        W_O = self.model.W_O[layer, head].detach().float().cpu()
        W_OV = W_V @ W_O  # [d_model, d_model], rank <= d_head
        U, S, Vt = torch.linalg.svd(W_OV, full_matrices=False)

        # ── Sanity check 1: dims within range ─────────────────
        n_sv = S.shape[0]
        assert dim1 < n_sv and dim2 < n_sv, \
            f"SVD dims ({dim1}, {dim2}) out of range for {n_sv} singular values"
        assert dim1 != dim2, f"SVD dims must be different, got ({dim1}, {dim2})"

        # ── Sanity check 2: sigma ratio ───────────────────────
        ratio = (S[dim1] / S[dim2]).item()
        print(f"\n  SANITY CHECKS:")
        print(f"    Sigma[{dim1}] = {S[dim1].item():.4f}")
        print(f"    Sigma[{dim2}] = {S[dim2].item():.4f}")
        print(f"    Ratio = {ratio:.4f}  (ideal = 1.0)")
        if abs(ratio - 1.0) > 0.3:
            print(f"    WARNING: Sigma ratio far from 1.0 — circle is an "
                  f"ellipse, projection quality may degrade")
        else:
            print(f"    OK: Balanced singular values for clean rotation")

        # ── Sanity check 3: W_U shape ─────────────────────────
        W_U = self.model.W_U.detach().float().cpu()  # [d_model, vocab_size]
        d_model = self.model.cfg.d_model
        assert W_U.shape[0] == d_model, \
            f"W_U shape mismatch: {W_U.shape[0]} != d_model={d_model}"
        print(f"    W_U shape: {tuple(W_U.shape)} (d_model={d_model}, "
              f"vocab={W_U.shape[1]})")

        # Reading directions (U columns — input space)
        u1, u2 = U[:, dim1], U[:, dim2]
        # Writing directions (Vt rows — output space, the "winning move")
        # Scaled by singular values to preserve relative importance
        w1 = S[dim1] * Vt[dim1, :]
        w2 = S[dim2] * Vt[dim2, :]

        # Use W_E for reading lens if available and different from W_U^T
        W_read = W_U  # default: use W_U for both
        if hasattr(self.model, 'W_E'):
            W_E = self.model.W_E.detach().float().cpu()  # [vocab, d_model]
            # Check if embeddings are tied (W_U ≈ W_E^T)
            sample_cos = F.cosine_similarity(
                W_U[:, 0].unsqueeze(0), W_E[0, :].unsqueeze(0)
            ).item()
            if abs(sample_cos) < 0.99:
                print(f"    Untied embeddings detected (cos={sample_cos:.3f})")
                print(f"    Using W_E for reading lens, W_U for writing lens")
                W_read = W_E.T  # [d_model, vocab] to match W_U shape
            else:
                print(f"    Tied embeddings confirmed (cos={sample_cos:.3f})")

        # ── Sanity check 4: self-consistency with number tokens ─
        self._verify_number_token_ordering(
            u1, u2, w1, w2, W_read, W_U)

        # ── Begin theta sweep ─────────────────────────────────
        thetas = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        angle_step = 360.0 / n_angles

        reading_results = []
        writing_results = []

        print(f"\n  {'Angle':>7}  {'READING (input) top tokens':<45}  "
              f"{'WRITING (output) top tokens'}")
        print(f"  {'─' * 100}")

        for i, theta in enumerate(thetas):
            ct, st = np.cos(theta), np.sin(theta)

            # Reading direction at this angle (uses W_E for untied models)
            read_dir = u1 * ct + u2 * st
            read_logits = read_dir @ W_read
            read_top = read_logits.topk(5)

            # Writing direction at this angle
            write_dir = w1 * ct + w2 * st
            write_logits = write_dir @ W_U
            write_top = write_logits.topk(5)

            # Decode tokens
            read_toks = []
            for tok_id, logit in zip(read_top.indices.tolist(),
                                     read_top.values.tolist()):
                tok_str = self.model.tokenizer.decode([tok_id]).replace('\n', '\\n')
                read_toks.append(f"'{tok_str}'({logit:+.1f})")

            write_toks = []
            for tok_id, logit in zip(write_top.indices.tolist(),
                                     write_top.values.tolist()):
                tok_str = self.model.tokenizer.decode([tok_id]).replace('\n', '\\n')
                write_toks.append(f"'{tok_str}'({logit:+.1f})")

            deg = i * angle_step
            read_str = "  ".join(read_toks[:3])
            write_str = "  ".join(write_toks[:3])
            print(f"  {deg:6.1f}°  {read_str:<45}  {write_str}")

            reading_results.append({
                "angle_deg": deg,
                "tokens": [(self.model.tokenizer.decode([t]), l)
                           for t, l in zip(read_top.indices.tolist(),
                                           read_top.values.tolist())]
            })
            writing_results.append({
                "angle_deg": deg,
                "tokens": [(self.model.tokenizer.decode([t]), l)
                           for t, l in zip(write_top.indices.tolist(),
                                           write_top.values.tolist())]
            })

        # ── Analysis: Check for number progression ────────────
        print(f"\n  {'=' * 65}")
        print(f"  ANALYSIS: Checking for structured token progression")
        print(f"  {'=' * 65}")

        for label, results_list in [("READING", reading_results),
                                    ("WRITING", writing_results)]:
            top1_tokens = [r["tokens"][0][0].strip() for r in results_list]
            number_count = 0
            number_sequence = []
            for tok in top1_tokens:
                try:
                    val = int(tok)
                    number_count += 1
                    number_sequence.append(val)
                except ValueError:
                    number_sequence.append(None)

            pct_numbers = number_count / len(top1_tokens) * 100
            print(f"\n  {label} lens:")
            print(f"    Top-1 tokens that are numbers: "
                  f"{number_count}/{len(top1_tokens)} ({pct_numbers:.0f}%)")

            if number_count >= len(top1_tokens) * 0.5:
                nums_only = [n for n in number_sequence if n is not None]
                if len(nums_only) >= 5:
                    diffs = np.diff(nums_only)
                    monotone = np.all(diffs > 0) or np.all(diffs < 0)
                    if monotone:
                        print(f"    --> MONOTONE NUMBER PROGRESSION: "
                              f"{nums_only[:8]}...")
                        print(f"        This is a NUMBER LINE / "
                              f"MAGNITUDE ENCODER")
                    else:
                        mod10 = [n % 10 for n in nums_only]
                        if len(set(mod10)) >= 8:
                            print(f"    --> MODULAR CYCLING: "
                                  f"mod10 = {mod10[:12]}...")
                            print(f"        This is a BASE-10 MODULAR WHEEL")
                        else:
                            print(f"    --> PARTIAL number structure: "
                                  f"{nums_only[:8]}...")
            else:
                print(f"    --> No clear number structure "
                      f"(mostly non-numeric tokens)")
                unique_top1 = list(dict.fromkeys(top1_tokens))[:15]
                print(f"    Top-1 unique: {unique_top1}")

            # Check for calendar/day patterns
            day_names = {"monday", "tuesday", "wednesday", "thursday",
                         "friday", "saturday", "sunday"}
            month_names = {"january", "february", "march", "april", "may",
                           "june", "july", "august", "september", "october",
                           "november", "december"}
            top1_lower = [t.lower().strip() for t in top1_tokens]
            day_hits = sum(1 for t in top1_lower if t in day_names)
            month_hits = sum(1 for t in top1_lower if t in month_names)
            if day_hits >= 3:
                print(f"    --> DAY-OF-WEEK CYCLE detected ({day_hits} hits)")
            if month_hits >= 3:
                print(f"    --> MONTH CYCLE detected ({month_hits} hits)")

        # ── Save plot ─────────────────────────────────────────
        Path(output_dir).mkdir(exist_ok=True)
        self._plot_vocab_projection(reading_results, writing_results,
                                    layer, head, dim1, dim2, output_dir)

        return {"reading": reading_results, "writing": writing_results}

    def _verify_number_token_ordering(self, u1, u2, w1, w2, W_read, W_U):
        """Sanity check: do number token embeddings map to ordered angles?

        If the circle encodes numbers, then projecting the embeddings of
        '0', '1', '2', ... '9' onto the (u1, u2) and (w1, w2) planes
        should yield monotonically increasing or cycling angles.
        """
        print(f"\n  SELF-CONSISTENCY CHECK: Number token ordering")
        tokenizer = self.model.tokenizer
        number_strs = [str(i) for i in range(10)]

        for label, v1, v2, W_proj in [
            ("READING (U cols)", u1, u2, W_read),
            ("WRITING (Vt rows)", w1, w2, W_U),
        ]:
            angles = []
            valid_nums = []
            for n_str in number_strs:
                # add_special_tokens=False avoids BOS being prepended
                tok_ids = tokenizer.encode(n_str, add_special_tokens=False)
                if len(tok_ids) == 1:
                    tok_id = tok_ids[0]
                elif len(tok_ids) >= 2:
                    # Some tokenizers split single digits — use last token
                    tok_id = tok_ids[-1]
                else:
                    continue

                # Get the embedding/unembedding vector for this token
                # W_proj is [d_model, vocab], so W_proj[:, tok_id] is the direction
                tok_vec = W_proj[:, tok_id]

                # Project onto the 2D SVD plane
                c1 = (tok_vec @ v1).item()
                c2 = (tok_vec @ v2).item()
                angle = np.arctan2(c2, c1)
                angles.append(angle)
                valid_nums.append(int(n_str))

            if len(angles) < 5:
                print(f"    {label}: Only {len(angles)} number tokens found — skip")
                continue

            # Check if angles are monotonically ordered (unwrapped)
            unwrapped = np.unwrap(angles)
            diffs = np.diff(unwrapped)
            monotone_up = np.all(diffs > 0)
            monotone_down = np.all(diffs < 0)

            angle_degs = [f"{np.degrees(a):+6.1f}" for a in angles]
            print(f"    {label}:")
            print(f"      Tokens 0-9 angles: {angle_degs}")

            if monotone_up or monotone_down:
                direction = "increasing" if monotone_up else "decreasing"
                print(f"      --> PASS: Angles are monotonically {direction}")
            else:
                # Check correlation instead
                corr = abs(float(np.corrcoef(valid_nums, unwrapped)[0, 1]))
                print(f"      --> Angle-number correlation: {corr:.3f}")
                if corr > 0.85:
                    print(f"      --> PASS: Strong linear ordering (r={corr:.3f})")
                elif corr > 0.50:
                    print(f"      --> PARTIAL: Moderate ordering (r={corr:.3f})")
                else:
                    print(f"      --> FAIL: No clear ordering (r={corr:.3f})")
                    print(f"      This may indicate the helix encodes something "
                          f"other than digit magnitude")

    def _plot_vocab_projection(self, reading_results, writing_results,
                               layer, head, dim1, dim2, output_dir):
        """Generate a polar plot showing token labels at each angle."""
        try:
            fig, (ax1, ax2) = plt.subplots(
                1, 2, figsize=(16, 7),
                subplot_kw=dict(projection='polar'))

            for ax, results, title in [
                (ax1, reading_results, "READING (Input)"),
                (ax2, writing_results, "WRITING (Output)")
            ]:
                angles = [np.radians(r["angle_deg"]) for r in results]
                labels = [r["tokens"][0][0].strip()[:8] for r in results]
                logits = [r["tokens"][0][1] for r in results]

                logit_arr = np.array(logits)
                norm = (logit_arr - logit_arr.min()) / (
                    logit_arr.max() - logit_arr.min() + 1e-8)

                for angle, label, nl in zip(angles, labels, norm):
                    ax.annotate(label, xy=(angle, 1.0),
                                fontsize=7, ha='center', va='center',
                                color=plt.cm.viridis(nl),
                                fontweight='bold')

                ax.set_title(
                    f"{title}\nL{layer}H{head} dims({dim1},{dim2})",
                    fontsize=10, pad=20)
                ax.set_rticks([])
                ax.set_yticklabels([])

            plt.tight_layout()
            fname = (f"vocab_projection_L{layer}H{head}"
                     f"_d{dim1}d{dim2}.png")
            fpath = Path(output_dir) / fname
            plt.savefig(fpath, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\n  Plot saved: {fpath}")
        except Exception as e:
            print(f"\n  Plot failed: {e}")

    # ── Automatic SVD Dim-Pair Sweep ────────────────────────

    def sweep_dim_pairs_vocab(self, layer: int = None, head: int = None,
                              top_k: int = 10, n_angles: int = 36,
                              output_dir: str = "helix_usage_validated"
                              ) -> Dict[str, Any]:
        """Sweep ALL pairs of top-k SVD dims to find the 2D plane
        that maximally aligns with number tokens in vocabulary space.

        This removes the dependency on PCA-derived dims which were
        found in activation space, not static weight space.
        """
        layer = layer if layer is not None else self.helix_layer
        head = head if head is not None else self.helix_head

        print(f"\n{'=' * 65}")
        print(f"AUTOMATIC SVD DIM-PAIR SWEEP")
        print(f"Layer {layer}, Head {head}, top-{top_k} dims, "
              f"{n_angles} angles")
        print(f"{'=' * 65}")

        # Extract OV-SVD
        W_V = self.model.W_V[layer, head].detach().float().cpu()
        W_O = self.model.W_O[layer, head].detach().float().cpu()
        W_OV = W_V @ W_O
        U, S, Vt = torch.linalg.svd(W_OV, full_matrices=False)
        W_U = self.model.W_U.detach().float().cpu()

        n_sv = min(top_k, S.shape[0])
        tokenizer = self.model.tokenizer
        thetas = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

        # Pre-encode digit tokens
        digit_tok_ids = {}
        for d in range(10):
            ids = tokenizer.encode(str(d), add_special_tokens=False)
            digit_tok_ids[d] = ids[-1] if ids else None

        results = []
        print(f"\n  {'Pair':>8}  {'σ ratio':>8}  {'%Nums':>6}  "
              f"{'r(digits)':>10}  {'MaxLogit':>9}  {'Score':>7}")
        print(f"  {'─' * 60}")

        for d1 in range(n_sv):
            for d2 in range(d1 + 1, n_sv):
                # Writing directions (Vt rows scaled by sigma)
                w1 = S[d1] * Vt[d1, :]
                w2 = S[d2] * Vt[d2, :]
                ratio = (S[d1] / S[d2]).item()

                # Sweep theta and decode top tokens
                num_count = 0
                max_logit = -float('inf')
                top1_tokens = []

                for theta in thetas:
                    ct, st = np.cos(theta), np.sin(theta)
                    write_dir = w1 * ct + w2 * st
                    logits = write_dir @ W_U
                    top_val, top_idx = logits.topk(1)
                    max_logit = max(max_logit, top_val.item())
                    tok_str = tokenizer.decode(
                        [top_idx.item()]).strip()
                    top1_tokens.append(tok_str)
                    try:
                        int(tok_str)
                        num_count += 1
                    except ValueError:
                        pass

                pct_nums = num_count / n_angles * 100

                # Digit ordering correlation
                digit_corr = 0.0
                angles_d = []
                nums_d = []
                for digit, tid in digit_tok_ids.items():
                    if tid is None:
                        continue
                    tok_vec = W_U[:, tid]
                    c1 = (tok_vec @ (S[d1] * Vt[d1, :])).item()
                    c2 = (tok_vec @ (S[d2] * Vt[d2, :])).item()
                    angles_d.append(np.arctan2(c2, c1))
                    nums_d.append(digit)
                if len(angles_d) >= 5:
                    unwrapped = np.unwrap(angles_d)
                    digit_corr = abs(float(
                        np.corrcoef(nums_d, unwrapped)[0, 1]))

                # Composite score: prioritize number tokens, then
                # digit ordering, then logit magnitude
                score = (pct_nums / 100.0 * 3.0
                         + digit_corr * 2.0
                         + min(max_logit / 10.0, 1.0))

                results.append({
                    "dims": (d1, d2),
                    "ratio": ratio,
                    "pct_nums": pct_nums,
                    "digit_corr": digit_corr,
                    "max_logit": max_logit,
                    "score": score,
                    "sample_tokens": top1_tokens[:6],
                })

                print(f"  ({d1:2d},{d2:2d})  "
                      f"{ratio:8.3f}  "
                      f"{pct_nums:5.1f}%  "
                      f"{digit_corr:10.3f}  "
                      f"{max_logit:9.2f}  "
                      f"{score:7.3f}")

        # Sort by score
        results.sort(key=lambda r: r["score"], reverse=True)

        print(f"\n  {'=' * 65}")
        print(f"  TOP 5 DIM PAIRS BY COMPOSITE SCORE")
        print(f"  {'=' * 65}")
        for i, r in enumerate(results[:5]):
            d1, d2 = r["dims"]
            print(f"  #{i+1}  dims=({d1},{d2})  "
                  f"score={r['score']:.3f}  "
                  f"%nums={r['pct_nums']:.0f}%  "
                  f"r={r['digit_corr']:.3f}  "
                  f"logit={r['max_logit']:.2f}  "
                  f"σ={r['ratio']:.3f}")
            print(f"       sample: {r['sample_tokens']}")

        # Run full projection on the best pair
        best = results[0]
        bd1, bd2 = best["dims"]
        print(f"\n  Running full projection on best pair ({bd1}, {bd2})...")
        full_result = self.subspace_vocab_projection(
            layer=layer, head=head, dim1=bd1, dim2=bd2,
            n_angles=n_angles, output_dir=output_dir)

        return {
            "sweep_results": results[:10],
            "best_dims": best["dims"],
            "best_score": best["score"],
            "best_projection": full_result,
        }

    # ── MLP Translation Lens ─────────────────────────────────

    def mlp_translation_lens(self, layer: int = None, head: int = None,
                             dim1: int = None, dim2: int = None,
                             mlp_layer: int = None,
                             n_angles: int = 36,
                             top_n_neurons: int = 20,
                             output_dir: str = "helix_usage_validated"
                             ) -> Dict[str, Any]:
        """Project the OV circle through MLP input weights instead of W_U.

        Instead of asking 'what vocabulary tokens does this angle promote?',
        this asks 'which MLP neurons does this angle EXCITE?'

        If the circle is a geometric dial that the MLP reads to trigger
        specific computation gates, this will reveal the mapping:
          θ → neuron activation pattern

        For gated MLPs (Gemma, Phi-3, Llama): uses both W_gate and W_in.
        For standard MLPs (GPT-2): uses W_in only.
        """
        layer = layer if layer is not None else self.helix_layer
        head = head if head is not None else self.helix_head
        dim1 = dim1 if dim1 is not None else self.svd_dims[0]
        dim2 = dim2 if dim2 is not None else self.svd_dims[1]
        mlp_layer = mlp_layer if mlp_layer is not None else layer

        print(f"\n{'=' * 65}")
        print(f"MLP TRANSLATION LENS")
        print(f"OV Head: L{layer} H{head}, SVD dims ({dim1}, {dim2})")
        print(f"MLP Target: Layer {mlp_layer}")
        print(f"{'=' * 65}")

        # Extract OV-SVD writing directions
        W_V = self.model.W_V[layer, head].detach().float().cpu()
        W_O = self.model.W_O[layer, head].detach().float().cpu()
        W_OV = W_V @ W_O
        U, S, Vt = torch.linalg.svd(W_OV, full_matrices=False)

        w1 = S[dim1] * Vt[dim1, :]  # [d_model]
        w2 = S[dim2] * Vt[dim2, :]  # [d_model]

        print(f"  Sigma[{dim1}] = {S[dim1].item():.4f}")
        print(f"  Sigma[{dim2}] = {S[dim2].item():.4f}")
        print(f"  Ratio = {(S[dim1] / S[dim2]).item():.4f}")

        # Extract MLP input weights
        mlp_module = self.model.blocks[mlp_layer].mlp
        if hasattr(mlp_module, 'W_gate') and hasattr(mlp_module, 'W_in'):
            W_gate = mlp_module.W_gate.detach().float().cpu()  # [d_model, d_mlp]
            W_in = mlp_module.W_in.detach().float().cpu()      # [d_model, d_mlp]
            W_mlp = torch.cat([W_gate, W_in], dim=1)
            d_mlp_gate = W_gate.shape[1]
            d_mlp_in = W_in.shape[1]
            mlp_type = "Gated"
            print(f"  Gated MLP: W_gate [{W_gate.shape[0]}×{d_mlp_gate}] "
                  f"+ W_in [{W_in.shape[0]}×{d_mlp_in}]")
            print(f"  Total neurons: {W_mlp.shape[1]} "
                  f"(gate: 0–{d_mlp_gate-1}, "
                  f"value: {d_mlp_gate}–{W_mlp.shape[1]-1})")
        elif hasattr(mlp_module, 'W_in'):
            W_mlp = mlp_module.W_in.detach().float().cpu()  # [d_model, d_mlp]
            d_mlp_gate = 0
            mlp_type = "Standard"
            print(f"  Standard MLP: W_in [{W_mlp.shape[0]}×{W_mlp.shape[1]}]")
        else:
            print(f"  ERROR: Unrecognized MLP architecture")
            return {"error": "unrecognized_mlp"}

        # Project writing directions onto MLP neurons
        # w @ W_mlp gives [d_mlp] — how much each neuron is excited
        thetas = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        angle_step = 360.0 / n_angles

        # Track per-neuron activation across all angles
        n_neurons = W_mlp.shape[1]
        neuron_activations = np.zeros((n_angles, n_neurons))

        angle_results = []

        print(f"\n  {'Angle':>7}  {'Top Neurons (idx: activation)':}")
        print(f"  {'─' * 80}")

        for i, theta in enumerate(thetas):
            ct, st = np.cos(theta), np.sin(theta)
            write_dir = w1 * ct + w2 * st
            # Project through MLP weights: [d_model] @ [d_model, d_mlp] = [d_mlp]
            neuron_act = (write_dir @ W_mlp).numpy()
            neuron_activations[i] = neuron_act

            # Top activated neurons at this angle
            top_idx = np.argsort(neuron_act)[-top_n_neurons:][::-1]
            top_vals = neuron_act[top_idx]

            deg = i * angle_step
            top_str = "  ".join(
                [f"N{idx}({val:+.2f})"
                 for idx, val in zip(top_idx[:5], top_vals[:5])])

            # Label gate vs value neurons
            labels = []
            for idx in top_idx[:5]:
                if mlp_type == "Gated" and idx < d_mlp_gate:
                    labels.append(f"gate:{idx}")
                elif mlp_type == "Gated":
                    labels.append(f"val:{idx - d_mlp_gate}")
                else:
                    labels.append(f"N{idx}")

            label_str = "  ".join(
                [f"{lbl}({val:+.2f})"
                 for lbl, val in zip(labels, top_vals[:5])])
            print(f"  {deg:6.1f}°  {label_str}")

            angle_results.append({
                "angle_deg": deg,
                "top_neurons": [(int(idx), float(val))
                                for idx, val in zip(
                                    top_idx[:top_n_neurons],
                                    top_vals[:top_n_neurons])],
            })

        # ── Analysis: Find neurons with strongest angular tuning ──
        print(f"\n  {'=' * 65}")
        print(f"  ANALYSIS: Neurons with strongest angular selectivity")
        print(f"  {'=' * 65}")

        # For each neuron, compute the range (max - min) of activation
        # across angles. High range = angle-selective neuron.
        neuron_range = neuron_activations.max(axis=0) - \
            neuron_activations.min(axis=0)
        top_selective = np.argsort(neuron_range)[-20:][::-1]

        print(f"\n  Top 20 angle-selective neurons (by activation range):")
        print(f"  {'Neuron':>8}  {'Type':>6}  {'Range':>7}  "
              f"{'Peak°':>6}  {'Min°':>6}  {'Max Act':>8}  {'Min Act':>8}")
        print(f"  {'─' * 65}")

        selective_neurons = []
        for nidx in top_selective:
            act_range = neuron_range[nidx]
            peak_angle = thetas[np.argmax(neuron_activations[:, nidx])]
            trough_angle = thetas[np.argmin(neuron_activations[:, nidx])]
            max_act = neuron_activations[:, nidx].max()
            min_act = neuron_activations[:, nidx].min()

            if mlp_type == "Gated" and nidx < d_mlp_gate:
                ntype = "gate"
                local_idx = nidx
            elif mlp_type == "Gated":
                ntype = "value"
                local_idx = nidx - d_mlp_gate
            else:
                ntype = "std"
                local_idx = nidx

            print(f"  {nidx:>8d}  {ntype:>6}  {act_range:7.3f}  "
                  f"{np.degrees(peak_angle):6.1f}  "
                  f"{np.degrees(trough_angle):6.1f}  "
                  f"{max_act:8.3f}  {min_act:8.3f}")

            selective_neurons.append({
                "global_idx": int(nidx),
                "local_idx": int(local_idx),
                "type": ntype,
                "range": float(act_range),
                "peak_deg": float(np.degrees(peak_angle)),
                "trough_deg": float(np.degrees(trough_angle)),
                "max_act": float(max_act),
                "min_act": float(min_act),
            })

        # Check if angle-selective neurons show structured progression
        peak_angles = [n["peak_deg"] for n in selective_neurons[:10]]
        if len(set([int(p) % 360 for p in peak_angles])) >= 5:
            print(f"\n  --> DIVERSE PEAK ANGLES detected among top neurons")
            print(f"     Peaks at: {[f'{p:.0f}°' for p in peak_angles]}")
            print(f"     This suggests the circle IS a geometric dial")
            print(f"     with different neurons tuned to different angles")
        else:
            print(f"\n  --> CLUSTERED peak angles — neurons may respond "
                  f"to a single direction, not a dial")

        # ── Activation range statistics ───────────────────────
        top10_range = np.mean([n["range"] for n in selective_neurons[:10]])
        median_range = np.median(neuron_range)
        print(f"\n  Activation range stats:")
        print(f"    Top-10 mean range: {top10_range:.4f}")
        print(f"    Median range:      {median_range:.4f}")
        print(f"    Selectivity ratio: {top10_range / (median_range + 1e-8):.1f}x")

        # ── Save plot ─────────────────────────────────────────
        Path(output_dir).mkdir(exist_ok=True)
        self._plot_mlp_lens(neuron_activations, selective_neurons,
                            layer, head, dim1, dim2, mlp_layer,
                            mlp_type, d_mlp_gate, output_dir)

        return {
            "mlp_layer": mlp_layer,
            "mlp_type": mlp_type,
            "n_neurons": n_neurons,
            "angle_results": angle_results,
            "selective_neurons": selective_neurons[:20],
            "top10_mean_range": float(top10_range),
            "median_range": float(median_range),
        }

    def _plot_mlp_lens(self, neuron_activations, selective_neurons,
                       layer, head, dim1, dim2, mlp_layer,
                       mlp_type, d_mlp_gate, output_dir):
        """Plot angular activation profiles of top selective neurons."""
        try:
            n_angles = neuron_activations.shape[0]
            thetas_deg = np.linspace(0, 360, n_angles, endpoint=False)

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Plot 1: Top 10 selective neurons' activation curves
            ax = axes[0, 0]
            for i, n in enumerate(selective_neurons[:10]):
                nidx = n["global_idx"]
                label = f"{n['type']}:{n['local_idx']}"
                ax.plot(thetas_deg, neuron_activations[:, nidx],
                        label=label, alpha=0.8, linewidth=1.5)
            ax.set_xlabel("Angle (degrees)")
            ax.set_ylabel("Neuron Activation")
            ax.set_title("Top 10 Angle-Selective Neurons")
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)

            # Plot 2: Polar plot of peak angles
            ax2 = fig.add_subplot(2, 2, 2, projection='polar')
            for i, n in enumerate(selective_neurons[:15]):
                angle_rad = np.radians(n["peak_deg"])
                ax2.annotate(
                    f"{n['type']}:{n['local_idx']}",
                    xy=(angle_rad, 1.0),
                    fontsize=6, ha='center', va='center',
                    color=plt.cm.tab20(i / 15),
                    fontweight='bold')
            ax2.set_title("Peak Activation Angles", fontsize=10, pad=20)
            ax2.set_rticks([])

            # Plot 3: Heatmap of top neurons vs angle
            ax3 = axes[1, 0]
            top_idxs = [n["global_idx"] for n in selective_neurons[:15]]
            heatmap_data = neuron_activations[:, top_idxs].T
            im = ax3.imshow(heatmap_data, aspect='auto',
                            cmap='RdBu_r',
                            extent=[0, 360, len(top_idxs), 0])
            ax3.set_xlabel("Angle (degrees)")
            ax3.set_ylabel("Neuron (rank by selectivity)")
            ax3.set_title("Neuron Activation Heatmap")
            plt.colorbar(im, ax=ax3, shrink=0.8)

            # Plot 4: Activation range distribution
            ax4 = axes[1, 1]
            all_ranges = neuron_activations.max(axis=0) - \
                neuron_activations.min(axis=0)
            ax4.hist(all_ranges, bins=50, alpha=0.7, color='steelblue')
            for n in selective_neurons[:5]:
                ax4.axvline(n["range"], color='red', alpha=0.5,
                            linestyle='--', linewidth=1)
            ax4.set_xlabel("Activation Range")
            ax4.set_ylabel("Count")
            ax4.set_title("Neuron Angular Selectivity Distribution")
            ax4.grid(True, alpha=0.3)

            fig.suptitle(
                f"MLP Translation Lens — L{layer}H{head} "
                f"dims({dim1},{dim2}) → MLP L{mlp_layer} "
                f"({mlp_type})",
                fontsize=12, fontweight='bold')
            plt.tight_layout()

            fname = (f"mlp_lens_L{layer}H{head}_d{dim1}d{dim2}"
                     f"_mlp{mlp_layer}.png")
            fpath = Path(output_dir) / fname
            plt.savefig(fpath, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\n  MLP lens plot saved: {fpath}")
        except Exception as e:
            print(f"\n  MLP lens plot failed: {e}")

    # ── MLP Neuron Forward Trace ─────────────────────────────

    def trace_neurons_to_vocab(self, layer: int = None, head: int = None,
                               dim1: int = None, dim2: int = None,
                               mlp_layer: int = None,
                               n_angles: int = 36,
                               top_n: int = 20,
                               output_dir: str = "helix_usage_validated"
                               ) -> Dict[str, Any]:
        """Trace angle-selective MLP neurons forward through W_out → W_U.

        For each top angle-selective neuron found by the MLP Translation
        Lens, extract its W_out row (what it writes to the residual stream)
        and project through W_U (unembedding) to decode the vocabulary
        tokens that neuron promotes when active.

        This closes the full circuit:
          OV circle → MLP gate neuron → W_out → W_U → vocabulary token
        """
        layer = layer if layer is not None else self.helix_layer
        head = head if head is not None else self.helix_head
        dim1 = dim1 if dim1 is not None else self.svd_dims[0]
        dim2 = dim2 if dim2 is not None else self.svd_dims[1]
        mlp_layer = mlp_layer if mlp_layer is not None else layer

        print(f"\n{'=' * 70}")
        print(f"MLP NEURON FORWARD TRACE: W_out → W_U → Vocabulary")
        print(f"OV Head: L{layer} H{head}, SVD dims ({dim1}, {dim2})")
        print(f"MLP Target: Layer {mlp_layer}")
        print(f"{'=' * 70}")

        # 1. Extract the SVD Writing Directions (Vt rows)
        W_V = self.model.W_V[layer, head].detach().float().cpu()
        W_O = self.model.W_O[layer, head].detach().float().cpu()
        W_OV = W_V @ W_O
        U, S, Vt = torch.linalg.svd(W_OV, full_matrices=False)

        w1 = S[dim1] * Vt[dim1, :]
        w2 = S[dim2] * Vt[dim2, :]

        # 2. Extract MLP Input Weights
        mlp_module = self.model.blocks[mlp_layer].mlp
        is_gated = hasattr(mlp_module, 'W_gate') and hasattr(mlp_module, 'W_in')

        if is_gated:
            W_gate = mlp_module.W_gate.detach().float().cpu()
            W_in_mlp = mlp_module.W_in.detach().float().cpu()
            W_mlp = torch.cat([W_gate, W_in_mlp], dim=1)
            d_mlp = W_gate.shape[1]
            print(f"  Gated MLP: W_gate [{W_gate.shape[0]}×{d_mlp}] "
                  f"+ W_in [{W_in_mlp.shape[0]}×{d_mlp}]")
        else:
            W_mlp = mlp_module.W_in.detach().float().cpu()
            d_mlp = W_mlp.shape[1]
            print(f"  Standard MLP: W_in [{W_mlp.shape[0]}×{d_mlp}]")

        # 3. Find Top Angle-Selective Neurons
        thetas = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        neuron_acts = np.zeros((n_angles, W_mlp.shape[1]))

        for i, theta in enumerate(thetas):
            ct, st = np.cos(theta), np.sin(theta)
            write_dir = w1 * ct + w2 * st
            neuron_acts[i] = (write_dir @ W_mlp).numpy()

        act_ranges = neuron_acts.max(axis=0) - neuron_acts.min(axis=0)
        top_global_idxs = np.argsort(act_ranges)[-top_n:][::-1]

        # 4. Extract W_out and W_U, sanity check shapes
        W_U = self.model.W_U.detach().float().cpu()   # [d_model, d_vocab]
        W_out = mlp_module.W_out.detach().float().cpu()  # [d_mlp, d_model]

        print(f"\n  [SANITY CHECKS]")
        print(f"    W_out shape: {tuple(W_out.shape)} "
              f"(Expected: [d_mlp, d_model])")
        print(f"    W_U shape:   {tuple(W_U.shape)} "
              f"(Expected: [d_model, d_vocab])")
        assert W_out.shape[1] == W_U.shape[0], \
            "CRITICAL: W_out output dim does not match W_U input dim!"
        print(f"    -> Dimension match confirmed. "
              f"Matrix multiplication is safe.\n")

        # 5. Project each top neuron's W_out row through W_U
        print(f"  {'Neuron':<12} | {'Peak°':>6} | "
              f"{'Top Output Tokens (W_out @ W_U)'}")
        print("  " + "-" * 80)

        trace_results = []
        for global_idx in top_global_idxs:
            peak_angle = np.degrees(
                thetas[np.argmax(neuron_acts[:, global_idx])])
            act_range = act_ranges[global_idx]

            # Translate global index to local W_out index
            if is_gated:
                ntype = "gate" if global_idx < d_mlp else "val"
                local_idx = global_idx % d_mlp
            else:
                ntype = "std"
                local_idx = global_idx

            # Extract the exact row vector this neuron writes
            neuron_out_vec = W_out[local_idx, :]  # [d_model]

            # Project through vocabulary
            vocab_logits = neuron_out_vec @ W_U  # [d_vocab]
            top_vals, top_tok_ids = vocab_logits.topk(6)

            # Decode tokens
            decoded_toks = []
            for tok_id, logit in zip(top_tok_ids.tolist(),
                                     top_vals.tolist()):
                tok_str = self.model.tokenizer.decode(
                    [tok_id]).replace('\n', '\\n')
                decoded_toks.append(f"'{tok_str}'({logit:+.1f})")

            label = f"{ntype}:{local_idx}"
            print(f"  {label:<12} | {peak_angle:>5.0f}° | "
                  f"{', '.join(decoded_toks)}")

            trace_results.append({
                "global_idx": int(global_idx),
                "local_idx": int(local_idx),
                "type": ntype,
                "peak_deg": float(peak_angle),
                "act_range": float(act_range),
                "top_tokens": [
                    {"token": self.model.tokenizer.decode([tid]),
                     "logit": float(lv)}
                    for tid, lv in zip(top_tok_ids.tolist(),
                                       top_vals.tolist())
                ],
            })

        # 6. Summary: check if any neurons project to number tokens
        number_pattern = re.compile(r'^\s*\d+\s*$')
        neurons_with_numbers = 0
        for tr in trace_results:
            has_num = any(number_pattern.match(t["token"])
                         for t in tr["top_tokens"])
            if has_num:
                neurons_with_numbers += 1

        print(f"\n  {'=' * 65}")
        print(f"  SUMMARY")
        print(f"  {'=' * 65}")
        print(f"  Neurons traced: {top_n}")
        print(f"  Neurons with number tokens in top-6: "
              f"{neurons_with_numbers}/{top_n} "
              f"({100*neurons_with_numbers/top_n:.0f}%)")

        if neurons_with_numbers > top_n * 0.3:
            print(f"  --> STRONG: Angle-selective MLP neurons "
                  f"project to number tokens")
            print(f"      Full circuit confirmed: "
                  f"OV circle → MLP gate → W_out → number tokens")
        elif neurons_with_numbers > 0:
            print(f"  --> PARTIAL: Some angle-selective neurons "
                  f"project to numbers")
        else:
            print(f"  --> NO number tokens found in neuron outputs")
            print(f"      The MLP neurons may encode intermediate "
                  f"representations,")
            print(f"      not direct vocabulary predictions")

        Path(output_dir).mkdir(exist_ok=True)
        return {
            "layer": layer, "head": head,
            "dims": (dim1, dim2), "mlp_layer": mlp_layer,
            "trace_results": trace_results,
            "neurons_with_numbers": neurons_with_numbers,
            "total_traced": top_n,
        }

    # ── Causal MLP Neuron Ablation ──────────────────────────

    def causal_mlp_ablation(self, layer: int = None, head: int = None,
                            dim1: int = None, dim2: int = None,
                            mlp_layer: int = None,
                            top_n: int = 20, n_prompts: int = 40,
                            digits: int = None,
                            output_dir: str = "helix_usage_validated"
                            ) -> Dict[str, Any]:
        """Zero out top angle-selective MLP neurons and measure
        arithmetic accuracy drop.

        Three conditions are tested:
          1. Baseline (no ablation)
          2. Targeted ablation (top angle-selective neurons zeroed)
          3. Random control (same number of random neurons zeroed)

        If targeted ablation hurts accuracy significantly more than
        the random control, the circuit is both real and sparse.
        """
        layer = layer if layer is not None else self.helix_layer
        head = head if head is not None else self.helix_head
        dim1 = dim1 if dim1 is not None else self.svd_dims[0]
        dim2 = dim2 if dim2 is not None else self.svd_dims[1]
        mlp_layer = mlp_layer if mlp_layer is not None else layer

        # Auto-detect digit count based on model
        if digits is None:
            if "gpt2" in self.model_name.lower():
                digits = 1
            else:
                digits = 2

        print(f"\n{'=' * 70}")
        print(f"CAUSAL MLP ABLATION")
        print(f"OV Head: L{layer} H{head}, SVD dims ({dim1}, {dim2})")
        print(f"MLP Target: Layer {mlp_layer} | Digits: {digits}")
        print(f"{'=' * 70}")

        # 1. Find target neurons (same logic as trace)
        W_V = self.model.W_V[layer, head].detach().float().cpu()
        W_O = self.model.W_O[layer, head].detach().float().cpu()
        W_OV = W_V @ W_O
        U, S, Vt = torch.linalg.svd(W_OV, full_matrices=False)

        w1 = S[dim1] * Vt[dim1, :]
        w2 = S[dim2] * Vt[dim2, :]

        mlp_module = self.model.blocks[mlp_layer].mlp
        is_gated = (hasattr(mlp_module, 'W_gate')
                    and hasattr(mlp_module, 'W_in'))

        if is_gated:
            W_gate = mlp_module.W_gate.detach().float().cpu()
            W_in_mlp = mlp_module.W_in.detach().float().cpu()
            W_mlp = torch.cat([W_gate, W_in_mlp], dim=1)
            d_mlp = W_gate.shape[1]
        else:
            W_mlp = mlp_module.W_in.detach().float().cpu()
            d_mlp = W_mlp.shape[1]

        thetas = np.linspace(0, 2 * np.pi, 36, endpoint=False)
        neuron_acts = np.zeros((36, W_mlp.shape[1]))
        for i, theta in enumerate(thetas):
            ct, st = np.cos(theta), np.sin(theta)
            write_dir = w1 * ct + w2 * st
            neuron_acts[i] = (write_dir @ W_mlp).numpy()

        act_ranges = neuron_acts.max(axis=0) - neuron_acts.min(axis=0)
        top_global_idxs = np.argsort(act_ranges)[-top_n:][::-1]

        # Map to physical neuron indices (dedup gate/value)
        target_neurons = sorted(set(
            [int(idx % d_mlp) for idx in top_global_idxs]))
        print(f"  Mapped {top_n} SVD targets to "
              f"{len(target_neurons)} physical MLP neurons "
              f"(out of {d_mlp} total)")

        # Generate random control neurons (same count, different set)
        random.seed(99)
        all_neurons = list(range(d_mlp))
        remaining = [n for n in all_neurons if n not in target_neurons]
        random_neurons = sorted(random.sample(
            remaining, min(len(target_neurons), len(remaining))))
        print(f"  Random control: {len(random_neurons)} neurons")

        # 2. Generate math prompts
        random.seed(42)
        prompts, expected = [], []
        min_val = 10 ** (digits - 1) if digits > 1 else 2
        max_val = (10 ** digits) - 1 if digits > 1 else 9

        for _ in range(n_prompts):
            a = random.randint(min_val, max_val)
            b = random.randint(min_val, max_val)
            ans = a + b
            if digits == 1:
                prompt = (f"Calculate:\n2 + 3 = 5\n1 + 4 = 5\n"
                          f"3 + 3 = 6\n1 + 1 = 2\n4 + 5 = 9\n"
                          f"{a} + {b} =")
            else:
                prompt = (f"Calculate:\n12 + 15 = 27\n"
                          f"41 + 23 = 64\n11 + 18 = 29\n"
                          f"33 + 14 = 47\n22 + 35 = 57\n"
                          f"{a} + {b} =")
            prompts.append(prompt)
            expected.append(str(ans))

        print(f"  Generated {n_prompts} few-shot "
              f"{digits}-digit addition prompts.")

        # 3. Helper: evaluate accuracy with optional ablation hook
        hook_name = f"blocks.{mlp_layer}.mlp.hook_post"

        def _eval(neurons_to_zero=None, label=""):
            correct = 0
            for prompt_str, exp_str in zip(prompts, expected):
                tokens = self.model.to_tokens(prompt_str)
                max_new = len(exp_str) + 1

                # Use add_hook / reset_hooks for robust generation
                if neurons_to_zero is not None:
                    def ablation_hook(acts, hook,
                                      _nz=neurons_to_zero):
                        acts[:, -1, _nz] = 0.0
                        return acts
                    self.model.add_hook(hook_name, ablation_hook)

                try:
                    with torch.no_grad():
                        out = self.model.generate(
                            tokens, max_new_tokens=max_new,
                            temperature=0.0, verbose=False)
                finally:
                    if neurons_to_zero is not None:
                        self.model.reset_hooks()

                gen = self.model.tokenizer.decode(
                    out[0][tokens.shape[1]:])
                if exp_str in gen.replace(" ", ""):
                    correct += 1

            acc = correct / len(prompts)
            print(f"      {label}: {acc:.1%} "
                  f"({correct}/{len(prompts)})")
            return acc

        # 4. Run three conditions
        print(f"\n  [1] Baseline (no ablation)...")
        baseline_acc = _eval(None, "Baseline")

        if baseline_acc < 0.15:
            print(f"\n  ⚠️  Baseline too low ({baseline_acc:.0%}). "
                  f"Model struggles with {digits}-digit addition.")
            print(f"      Ablation test is not meaningful.")
            Path(output_dir).mkdir(exist_ok=True)
            return {
                "baseline_acc": float(baseline_acc),
                "ablated_acc": None,
                "random_acc": None,
                "status": "baseline_too_low",
            }

        print(f"  [2] Targeted ablation "
              f"({len(target_neurons)} neurons)...")
        ablated_acc = _eval(target_neurons, "Ablated")

        print(f"  [3] Random control "
              f"({len(random_neurons)} neurons)...")
        random_acc = _eval(random_neurons, "Random")

        # 5. Summary
        targeted_drop = baseline_acc - ablated_acc
        random_drop = baseline_acc - random_acc
        specificity = targeted_drop - random_drop

        print(f"\n  {'=' * 60}")
        print(f"  RESULTS")
        print(f"  {'=' * 60}")
        print(f"  Baseline:         {baseline_acc:.1%}")
        print(f"  Targeted ablated: {ablated_acc:.1%} "
              f"(drop: {targeted_drop:+.1%})")
        print(f"  Random control:   {random_acc:.1%} "
              f"(drop: {random_drop:+.1%})")
        print(f"  Specificity:      {specificity:+.1%} "
              f"(targeted - random)")

        if targeted_drop > 0.15 and specificity > 0.10:
            print(f"\n  🔥 CIRCUIT IS NECESSARY AND SPARSE!")
            print(f"     Ablating {len(target_neurons)}/{d_mlp} "
                  f"neurons causes {targeted_drop:.0%} drop")
            print(f"     Random ablation only causes "
                  f"{random_drop:.0%} drop")
            verdict = "circuit_confirmed"
        elif targeted_drop > 0.10:
            print(f"\n  ⚠️  Moderate drop — circuit contributes but "
                  f"may not be sole pathway")
            verdict = "partial"
        else:
            print(f"\n  ○ Circuit remained resilient or routed "
                  f"around the ablation")
            verdict = "resilient"

        Path(output_dir).mkdir(exist_ok=True)
        return {
            "layer": layer, "head": head,
            "dims": (dim1, dim2), "mlp_layer": mlp_layer,
            "n_target_neurons": len(target_neurons),
            "n_random_neurons": len(random_neurons),
            "d_mlp": d_mlp,
            "baseline_acc": float(baseline_acc),
            "ablated_acc": float(ablated_acc),
            "random_acc": float(random_acc),
            "targeted_drop": float(targeted_drop),
            "random_drop": float(random_drop),
            "specificity": float(specificity),
            "verdict": verdict,
        }

    # ── Concept Compass Runtime Validation ──────────────────

    def validate_concept_compass(self, layer: int = None,
                                  head: int = None,
                                  dim1: int = None, dim2: int = None,
                                  prompts: Dict[str, List[str]] = None,
                                  output_dir: str = "helix_usage_validated"
                                  ) -> Dict[str, Any]:
        """Feed targeted semantic prompts and check if runtime
        activation angles match predicted category positions.

        Projects the residual stream (before the target layer's
        attention) onto the 2D reading plane defined by U columns
        of the OV SVD.  If different semantic categories cluster
        at different angles, the Concept Compass is used at
        runtime — not just a weight-space artifact.
        """
        layer = layer if layer is not None else self.helix_layer
        head = head if head is not None else self.helix_head
        dim1 = dim1 if dim1 is not None else self.svd_dims[0]
        dim2 = dim2 if dim2 is not None else self.svd_dims[1]

        if prompts is None:
            prompts = {
                "Operations / Programs (~0°-40° or ~330°-350°)": [
                    "The software company launched a new program",
                    "The scientists conducted several successful experiments",
                    "The military unit commenced their tactical operations",
                ],
                "Teams / Groups (~50°-120°)": [
                    "The championship was won by the rival teams",
                    "The presentation was delivered by the management team",
                    "The community organized into small neighborhood groups",
                ],
                "Tools / Components (~130°-150°)": [
                    "The carpenter organized his workbench and tools",
                    "The chemistry teacher explained the structure of atoms",
                    "The digital parser broke the text into individual tokens",
                ],
                "Areas / Geography (~220°-280°)": [
                    "The explorers mapped out the uncharted territory",
                    "The wildlife reserve covers several large areas",
                    "The climate varies significantly across these regions",
                ],
            }

        print(f"\n{'=' * 75}")
        print(f"RUNTIME CONCEPT COMPASS VALIDATION")
        print(f"Target: L{layer} H{head}, SVD Dims ({dim1}, {dim2})")
        print(f"{'=' * 75}")

        # 1. Extract reading directions (U columns) of the OV matrix
        print(f"  [1] Extracting SVD Reading Subspace...")
        W_V = self.model.W_V[layer, head].detach().float().cpu()
        W_O = self.model.W_O[layer, head].detach().float().cpu()
        W_OV = W_V @ W_O
        U, S, Vt = torch.linalg.svd(W_OV, full_matrices=False)

        u1 = U[:, dim1]  # Reading direction 1 (x-axis)
        u2 = U[:, dim2]  # Reading direction 2 (y-axis)

        hook_name = f"blocks.{layer}.hook_resid_pre"

        print(f"  [2] Running Prompts and Calculating "
              f"Runtime Angles...\n")

        all_results = {}
        for category, prompt_list in prompts.items():
            print(f"  [ Category: {category} ]")
            print(f"  {'-' * 70}")

            cat_angles = []
            for prompt_str in prompt_list:
                tokens = self.model.to_tokens(prompt_str)

                # Sanity: show which token we're analyzing
                last_tok_id = tokens[0, -1].item()
                last_tok_str = self.model.tokenizer.decode(
                    [last_tok_id])

                with torch.no_grad():
                    _, cache = self.model.run_with_cache(
                        tokens, names_filter=hook_name)

                resid = cache[hook_name][0, -1, :].cpu().float()

                c1 = (resid @ u1).item()
                c2 = (resid @ u2).item()

                angle_rad = np.arctan2(c2, c1)
                angle_deg = float(np.degrees(angle_rad) % 360)

                cat_angles.append(angle_deg)
                print(f"    Token: '{last_tok_str:<12}' | "
                      f"Angle: {angle_deg:>5.1f}°")

            mean_angle = float(np.mean(cat_angles))
            std_angle = float(np.std(cat_angles))
            print(f"    >>> Mean: {mean_angle:.1f}° ± "
                  f"{std_angle:.1f}°\n")

            all_results[category] = {
                "angles": cat_angles,
                "mean": mean_angle,
                "std": std_angle,
            }

        # 3. Summary: check if categories separate
        means = [v["mean"] for v in all_results.values()]
        overall_spread = float(np.std(means))

        print(f"  {'=' * 65}")
        print(f"  SUMMARY")
        print(f"  {'=' * 65}")
        for cat, res in all_results.items():
            print(f"  {cat[:40]:<42} "
                  f"mean={res['mean']:>5.1f}° ± "
                  f"{res['std']:>4.1f}°")

        print(f"\n  Category mean spread (std): "
              f"{overall_spread:.1f}°")

        if overall_spread > 30:
            print(f"  --> STRONG angular separation between "
                  f"semantic categories")
            print(f"      The Concept Compass is active at "
                  f"runtime!")
        elif overall_spread > 15:
            print(f"  --> MODERATE separation — some angular "
                  f"clustering visible")
        else:
            print(f"  --> WEAK separation — categories overlap "
                  f"in angle space")

        Path(output_dir).mkdir(exist_ok=True)
        return {
            "layer": layer, "head": head,
            "dims": (dim1, dim2),
            "categories": all_results,
            "overall_spread": overall_spread,
        }

    # ── Cross-task helix persistence ──────────────────────────

    def test_cross_task_persistence(self) -> Dict[str, Any]:
        """Does the helical structure appear across different task types?

        If the helix is a *representational scaffold* (our hypothesis),
        it should appear whenever numbers are processed — regardless of
        task.  If it only appears in arithmetic, it's task-specific.
        """
        print("\n" + "=" * 65)
        print("CROSS-TASK HELIX PERSISTENCE TEST")
        print("=" * 65)

        task_prompts = {
            "arithmetic": {
                "template": "What is {n} + 5?",
                "position": "operand",
            },
            "counting": {
                "template": "Count to {n}:",
                "position": "last",
            },
            "ordinal": {
                "template": "The {n}th item in the list is",
                "position": "last",
            },
            "comparison": {
                "template": "Is {n} greater than 50?",
                "position": "last",
            },
            "date": {
                "template": "January {n} is a",
                "position": "last",
            },
            "plain_number": {
                "template": "The number {n}",
                "position": "last",
            },
        }

        layer = self.best_resid_layer
        hook_name = f"blocks.{layer}.hook_resid_pre"
        results = {}

        for task_name, cfg in task_prompts.items():
            print(f"\n  Task: {task_name}")
            acts_list, ns_list = [], []

            for n in range(0, 60):
                prompt = cfg["template"].format(n=n)
                tokens = self.model.to_tokens(prompt)
                str_tokens = self.model.to_str_tokens(prompt)

                if cfg["position"] == "operand":
                    target_pos = -1
                    for idx, tok in enumerate(str_tokens):
                        if '+' in tok:
                            target_pos = idx - 1
                            break
                    if target_pos <= 0:
                        continue
                else:
                    target_pos = len(str_tokens) - 1

                with torch.no_grad():
                    _, cache = self.model.run_with_cache(
                        tokens, names_filter=hook_name
                    )
                resid = cache[hook_name][0, target_pos, :].cpu().float()

                if torch.isnan(resid).any() or resid.norm() < 1e-6:
                    continue
                acts_list.append(resid)
                ns_list.append(n)

            if len(acts_list) < 15:
                print(f"    Only {len(acts_list)} valid — skipping")
                results[task_name] = {"error": "insufficient_data"}
                continue

            acts_t = torch.stack(acts_list)
            r = self.analyze_residual_helix(
                acts_t, ns_list, label=f"  {task_name} L{layer}"
            )
            results[task_name] = r

        # Summary
        print(f"\n  {'Task':<18} {'Phase_corr':>10} {'CV':>8} {'Period':>8} {'Lin':>6}")
        print("  " + "-" * 56)
        for task_name, r in results.items():
            if "error" in r:
                print(f"  {task_name:<18} {'ERROR':>10}")
            else:
                print(f"  {task_name:<18} {r['phase_corr']:>10.3f} "
                      f"{r['cv']:>8.3f} {r['period']:>8.2f} "
                      f"{r['lin_raw']:>6.3f}")

        return results

    # ── Negative control ──────────────────────────────────────

    def run_negative_control(self) -> Dict[str, Any]:
        """Scrambled-number control: helix should vanish."""
        print("\n" + "=" * 65)
        print("NEGATIVE CONTROL: Scrambled Number Order")
        print("=" * 65)

        layer = self.best_resid_layer
        hook_name = f"blocks.{layer}.hook_resid_pre"

        # Collect activations for numbers 0-59
        acts_list, ns_list = [], []
        for n in range(60):
            prompt = f"What is {n} + 5?"
            tokens = self.model.to_tokens(prompt)
            str_tokens = self.model.to_str_tokens(prompt)
            target_pos = -1
            for idx, tok in enumerate(str_tokens):
                if '+' in tok:
                    target_pos = idx - 1
                    break
            if target_pos <= 0:
                continue

            with torch.no_grad():
                _, cache = self.model.run_with_cache(
                    tokens, names_filter=hook_name
                )
            resid = cache[hook_name][0, target_pos, :].cpu().float()
            acts_list.append(resid)
            ns_list.append(n)

        if len(acts_list) < 15:
            return {"error": "insufficient_data"}

        acts_t = torch.stack(acts_list)

        # Real order
        print("  Real number order:")
        real = self.analyze_residual_helix(
            acts_t, ns_list, label="  REAL"
        )

        # Scrambled order — permute the number labels
        rng = np.random.RandomState(42)
        scrambled_ns = list(ns_list)
        rng.shuffle(scrambled_ns)
        print("  Scrambled number order:")
        scrambled = self.analyze_residual_helix(
            acts_t, scrambled_ns, label="  SCRAMBLED"
        )

        # The scrambled version should have much worse phase_corr
        real_pc = real.get('phase_corr', 0)
        scram_pc = scrambled.get('phase_corr', 0)
        delta = real_pc - scram_pc
        print(f"\n  Phase_corr delta (real - scrambled): {delta:+.3f}")
        if delta > 0.15:
            print(f"  --> PASS: Real order has stronger circular structure")
        else:
            print(f"  --> WARNING: Scrambled order is comparable — "
                  f"circular structure may be spurious")

        return {"real": real, "scrambled": scrambled, "delta": delta}

    # ── Main pipeline ─────────────────────────────────────────

    def run_validated_investigation(self, output_dir: str = "helix_usage_validated"):
        Path(output_dir).mkdir(exist_ok=True)

        print(f"\n{'=' * 65}")
        print(f"VALIDATED HELIX INVESTIGATION: {self.model_name}")
        print(f"Helix Head: L{self.helix_layer} H{self.helix_head}")
        print(f"SVD Dims: {self.svd_dims}")
        print(f"Resid Layer: L{self.best_resid_layer}")
        print(f"Expected Period: T={self.expected_period}")
        print(f"Device: {self.device}")
        print(f"{'=' * 65}")

        all_results = {}

        # ── Test 1: OV-SVD helix at known head ───────────────
        print(f"\n{'─' * 65}")
        print("TEST 1: OV-Matrix SVD Helix (correct method)")
        print(f"{'─' * 65}")
        acts_op, ns_op = self.collect_activations(
            self.best_resid_layer, max_n=100, position="operand"
        )
        ov_result = self.analyze_ov_helix(
            self.helix_layer, self.helix_head,
            acts_op, ns_op,
            label=f"OV-SVD L{self.helix_layer}H{self.helix_head}"
        )
        all_results["ov_helix"] = ov_result

        # ── Test 2: Fourier isolation ─────────────────────────
        print(f"\n{'─' * 65}")
        print("TEST 2: Fourier Isolation")
        print(f"{'─' * 65}")
        fourier_result = self.test_fourier_isolation(
            acts_op, ns_op, period=self.expected_period,
            label=f"L{self.best_resid_layer}"
        )
        all_results["fourier_isolation"] = fourier_result

        # ── Test 3: Negative control ──────────────────────────
        print(f"\n{'─' * 65}")
        print("TEST 3: Negative Control (scrambled labels)")
        print(f"{'─' * 65}")
        control_result = self.run_negative_control()
        all_results["negative_control"] = control_result

        # ── Test 4: Cross-task persistence ────────────────────
        print(f"\n{'─' * 65}")
        print("TEST 4: Cross-Task Helix Persistence")
        print(f"{'─' * 65}")
        cross_task = self.test_cross_task_persistence()
        all_results["cross_task"] = cross_task

        # ── Test 5: Causal phase-shift ────────────────────────
        print(f"\n{'─' * 65}")
        print("TEST 5: Causal Phase-Shift (does rotation change output?)")
        print(f"{'─' * 65}")
        causal_result = self.causal_phase_shift_test(
            self.helix_layer, self.helix_head, ns_op,
            period=self.expected_period, n_tests=20
        )
        all_results["causal_phase_shift"] = causal_result

        # ── Test 6: Subspace Vocabulary Projection ────────────
        print(f"\n{'─' * 65}")
        print("TEST 6: Subspace Vocabulary Projection (decode the circle)")
        print(f"{'─' * 65}")
        vocab_result = self.subspace_vocab_projection(output_dir=output_dir)
        all_results["vocab_projection"] = vocab_result

        # ── Final verdict ─────────────────────────────────────
        self._print_verdict(all_results)
        self._save_results(all_results, output_dir)

        return all_results

    def _print_verdict(self, results: Dict[str, Any]):
        print(f"\n{'=' * 65}")
        print(f"FINAL VERDICT: {self.model_name}")
        print(f"{'=' * 65}")

        # 1. Does the helix exist representationally?
        ov = results.get("ov_helix", {})
        ov_pc = ov.get("phase_corr", 0)
        ov_cv = ov.get("cv", 1)
        print(f"\n  1. Helix exists in OV subspace?")
        print(f"     Phase_corr={ov_pc:.3f}, CV={ov_cv:.3f}")
        if ov_pc > 0.75:
            print(f"     --> YES: Helical structure present")
        elif ov.get("lin_raw", 0) > 0.90:
            print(f"     --> MONOTONE only (not modular)")
        else:
            print(f"     --> NO clear helix")

        # 2. Does Fourier isolation help?
        fi = results.get("fourier_isolation", {})
        fi_cv = fi.get("cv", 1)
        fi_pc = fi.get("phase_corr", 0)
        fi_r2 = fi.get("r2", 0)
        print(f"\n  2. Fourier isolation reveals circle?")
        print(f"     R2={fi_r2:.1%}, CV={fi_cv:.3f}, Phase={fi_pc:.3f}")
        if fi_cv < 0.20 and fi_pc > 0.85:
            print(f"     --> YES: Clean clock after denoising")
        else:
            print(f"     --> NO clean clock even after isolation")

        # 3. Negative control
        nc = results.get("negative_control", {})
        delta = nc.get("delta", 0)
        print(f"\n  3. Negative control (scrambled)?")
        print(f"     Phase delta (real - scrambled): {delta:+.3f}")
        if delta > 0.15:
            print(f"     --> PASS: Structure is number-order-dependent")
        else:
            print(f"     --> FAIL: Structure may be spurious")

        # 4. Cross-task persistence
        ct = results.get("cross_task", {})
        tasks_with_helix = []
        for task, r in ct.items():
            if isinstance(r, dict) and r.get("phase_corr", 0) > 0.60:
                tasks_with_helix.append(task)
        print(f"\n  4. Cross-task persistence?")
        print(f"     Tasks showing helix (phase_corr > 0.60): {tasks_with_helix or 'none'}")
        if len(tasks_with_helix) >= 3:
            print(f"     --> SCAFFOLD: Helix appears across tasks (representational)")
        elif len(tasks_with_helix) == 1:
            print(f"     --> TASK-SPECIFIC: Only in {tasks_with_helix[0]}")
        else:
            print(f"     --> WEAK/ABSENT across tasks")

        # 5. Causal test
        causal = results.get("causal_phase_shift", {})
        nc_rate = causal.get("no_carry", 0)
        c_rate = causal.get("carry", 0)
        print(f"\n  5. Causal phase-shift changes output?")
        print(f"     No-carry: {nc_rate:.1%}, Carry: {c_rate:.1%}")
        if nc_rate > 0.30:
            print(f"     --> YES: Helix causally controls arithmetic")
        else:
            print(f"     --> NO: Helix is NOT causally used for arithmetic")

        # 6. Vocabulary projection
        vp = results.get("vocab_projection", {})
        if vp:
            print(f"\n  6. Subspace vocabulary projection?")
            print(f"     (See detailed output above and saved plot)")

        # Overall
        print(f"\n  {'─' * 55}")
        is_representational = (ov_pc > 0.60 and delta > 0.10
                               and len(tasks_with_helix) >= 2)
        is_causal = nc_rate > 0.30
        if is_representational and not is_causal:
            print(f"  CONCLUSION: Helix is a REPRESENTATIONAL SCAFFOLD")
            print(f"  It organises numbers geometrically but does NOT")
            print(f"  causally drive arithmetic computation.")
        elif is_representational and is_causal:
            print(f"  CONCLUSION: Helix is a CAUSAL MECHANISM")
            print(f"  It both represents AND computes via rotation.")
        elif not is_representational:
            print(f"  CONCLUSION: NO CLEAR HELIX in this model/layer")
        print(f"  {'─' * 55}")

    def _save_results(self, results: Dict[str, Any], output_dir: str):
        """Save JSON-serializable results."""
        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_clean(v) for v in obj]
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, range):
                return list(obj)
            return obj

        clean_results = _clean(results)
        clean_results["model"] = self.model_name
        clean_results["config"] = {
            "helix_layer": self.helix_layer,
            "helix_head": self.helix_head,
            "svd_dims": list(self.svd_dims),
            "best_resid_layer": self.best_resid_layer,
            "expected_period": self.expected_period,
        }

        out_path = Path(output_dir) / "validated_results.json"
        with open(out_path, 'w') as f:
            json.dump(clean_results, f, indent=2, default=str)
        print(f"\n  Results saved to {out_path}")


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("VALIDATED HELIX USAGE INVESTIGATION")
    print("=" * 65)

    # Default to gpt2-small for fast iteration.
    # Change to "google/gemma-7b" or "microsoft/Phi-3-mini-4k-instruct"
    # for production runs on models with known strong helix signals.
    model_name = "gpt2-small"
    projection_only = "--projection-only" in sys.argv
    sweep_mode = "--sweep" in sys.argv
    mlp_lens_mode = "--mlp-lens" in sys.argv
    trace_mode = "--trace" in sys.argv
    ablate_mode = "--ablate" in sys.argv
    compass_mode = "--compass" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        model_name = args[0]

    # Parse optional --mlp-layer N for MLP lens
    mlp_target_layer = None
    for i, a in enumerate(sys.argv):
        if a == "--mlp-layer" and i + 1 < len(sys.argv):
            mlp_target_layer = int(sys.argv[i + 1])

    try:
        investigator = HelixUsageInvestigator(model_name)

        if sweep_mode:
            print("\n  Running SVD Dim-Pair Sweep + MLP Lens...")
            sweep_result = investigator.sweep_dim_pairs_vocab()
            # Also run MLP lens with the best dims found
            best_d1, best_d2 = sweep_result["best_dims"]
            print(f"\n  Running MLP lens with best dims ({best_d1}, {best_d2})...")
            investigator.mlp_translation_lens(
                dim1=best_d1, dim2=best_d2,
                mlp_layer=mlp_target_layer)
            # Also run MLP lens on next layer
            next_layer = investigator.helix_layer + 1
            if next_layer < investigator.model.cfg.n_layers:
                print(f"\n  Running MLP lens on next layer ({next_layer})...")
                investigator.mlp_translation_lens(
                    dim1=best_d1, dim2=best_d2,
                    mlp_layer=next_layer)
            # If --trace also requested, run it with sweep dims
            if trace_mode:
                print(f"\n  Running forward trace with sweep dims ({best_d1}, {best_d2})...")
                investigator.trace_neurons_to_vocab(
                    dim1=best_d1, dim2=best_d2,
                    mlp_layer=mlp_target_layer)
        elif compass_mode:
            print("\n  Running Concept Compass Validation...")
            investigator.validate_concept_compass()
        elif ablate_mode:
            print("\n  Running Causal MLP Neuron Ablation...")
            investigator.causal_mlp_ablation(
                mlp_layer=mlp_target_layer)
        elif trace_mode:
            print("\n  Running MLP Neuron Forward Trace...")
            investigator.trace_neurons_to_vocab(
                mlp_layer=mlp_target_layer)
        elif mlp_lens_mode:
            print("\n  Running MLP Translation Lens...")
            investigator.mlp_translation_lens(
                mlp_layer=mlp_target_layer)
            # Also try next layer
            next_layer = investigator.helix_layer + 1
            if next_layer < investigator.model.cfg.n_layers:
                investigator.mlp_translation_lens(
                    mlp_layer=next_layer)
        elif projection_only:
            print("\n  Running ONLY Subspace Vocabulary Projection...")
            investigator.subspace_vocab_projection()
        else:
            results = investigator.run_validated_investigation()

        print("\n  Investigation completed successfully.")
    except ValidationError as e:
        print(f"\n  VALIDATION ERROR: {e}")
        return False
    except Exception as e:
        print(f"\n  UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)