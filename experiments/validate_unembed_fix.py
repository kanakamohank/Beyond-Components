#!/usr/bin/env python3
"""Quick validation of compute_unembed_basis_direct_answer."""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from arithmetic_circuit_scan_updated import (
    compute_unembed_basis,
    compute_unembed_basis_direct_answer,
    get_digit_token_ids,
    get_context_target_tok,
    MODEL_MAP,
)
from transformer_lens import HookedTransformer

print("=" * 60)
print("VALIDATION: compute_unembed_basis_direct_answer")
print("=" * 60)

# Load model
model_name = MODEL_MAP["llama-3b"]
model = HookedTransformer.from_pretrained(model_name, device="cpu", dtype=torch.float32)
model.eval()

# ── Test 1: Standard basis (single-digit tokens) ──────────────────
print("\n── Test 1: Standard unembed basis (single-digit tokens) ──")
digit_ids = get_digit_token_ids(model)
print(f"  Single-digit token IDs: {digit_ids}")
U_std, S_std = compute_unembed_basis(model, digit_ids)
print(f"  Shape: U={U_std.shape}, S={S_std.shape}")
print(f"  Singular values: {S_std.round(4)}")

# ── Test 2: Direct-answer basis (group-mean of answer tokens) ─────
print("\n── Test 2: Direct-answer unembed basis (group-mean) ──")
U_da, S_da = compute_unembed_basis_direct_answer(model, operand_max=99)
print(f"  Shape: U={U_da.shape}, S={S_da.shape}")
print(f"  Singular values: {S_da.round(4)}")

# ── Test 3: Verify the bases are DIFFERENT ────────────────────────
print("\n── Test 3: Compare bases ──")
# Principal angles between the two 10D subspaces
cross = U_std.T @ U_da
cos_angles = np.linalg.svd(cross, compute_uv=False)
print(f"  Principal cosines (std vs direct-answer): {cos_angles.round(4)}")
print(f"  Mean cosine: {cos_angles.mean():.4f}")
if cos_angles.mean() > 0.95:
    print("  ⚠️  Bases are very similar — fix may not help much")
elif cos_angles.mean() < 0.5:
    print("  ★ Bases are VERY different — confirms the bug!")
else:
    print("  Bases are moderately different")

# ── Test 4: Check that answer tokens cover all ones digits ────────
print("\n── Test 4: Answer token coverage ──")
sample_prompt = "0 + 0 = "
from collections import defaultdict
groups = defaultdict(list)
for ans in range(199):
    tok_id = get_context_target_tok(model, sample_prompt, str(ans))
    groups[ans % 10].append(tok_id)
    
for d in range(10):
    unique_toks = len(set(groups[d]))
    print(f"  Digit {d}: {len(groups[d])} tokens, {unique_toks} unique")

# ── Test 5: Check single-digit tokens are a SUBSET of answer tokens ──
print("\n── Test 5: Single-digit token overlap ──")
all_answer_toks = set()
for ans in range(199):
    tok_id = get_context_target_tok(model, sample_prompt, str(ans))
    all_answer_toks.add(tok_id)
    
overlap = sum(1 for t in digit_ids if t in all_answer_toks)
print(f"  {overlap}/10 single-digit tokens appear among answer tokens")
print(f"  Total unique answer tokens: {len(all_answer_toks)}")

# ── Test 6: Sanity — W_U column norms for answer vs digit tokens ──
print("\n── Test 6: W_U column norm comparison ──")
W_U = model.W_U.detach().float().cpu()
digit_norms = torch.norm(W_U[:, digit_ids], dim=0).numpy()
print(f"  Single-digit W_U norms: {digit_norms.round(4)}")

# Sample a few multi-digit answer tokens
for ans in [10, 21, 50, 99, 150, 198]:
    tok_id = get_context_target_tok(model, sample_prompt, str(ans))
    norm = torch.norm(W_U[:, tok_id]).item()
    print(f"  Answer {ans:>3} (tok={tok_id}): W_U norm = {norm:.4f}")

print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)
