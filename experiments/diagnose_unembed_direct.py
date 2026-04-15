#!/usr/bin/env python3
"""
Diagnose why direct-answer unembed patching yields ~0% even at L27.

Hypotheses:
  H1: Within-group W_U variance is huge — group means wash out to near-zero.
  H2: The group-mean 9D basis doesn't capture individual answer-token W_U columns.
  H3: The single-digit W_U basis (teacher-forced) actually works better for direct-answer too.
  H4: Code bug in how the basis is applied during patching.
"""
import sys, torch, numpy as np
from collections import defaultdict

sys.path.insert(0, ".")
from experiments.arithmetic_circuit_scan_updated import (
    get_context_target_tok,
    get_digit_token_ids,
    compute_unembed_basis,
    compute_unembed_basis_direct_answer,
    MODEL_MAP,
)
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)

print("=" * 70)
print("DIAGNOSIS: Why is direct-answer unembed patching near-zero?")
print("=" * 70)

# ── Load model ────────────────────────────────────────────────────────
model_name = MODEL_MAP["llama-3b"]
model = HookedTransformer.from_pretrained(model_name, device="cpu", dtype=torch.float32)
model.eval()

W_U = model.W_U.detach().float().cpu()  # (d_model, vocab_size)
d_model = W_U.shape[0]

# ── Collect answer token IDs and their W_U columns ───────────────────
sample_prompt = "0 + 0 = "
answer_tok_ids = {}
for ans in range(199):
    tid = get_context_target_tok(model, sample_prompt, str(ans))
    answer_tok_ids[ans] = tid

# Group by ones digit
groups = defaultdict(list)
for ans, tid in answer_tok_ids.items():
    groups[ans % 10].append((ans, tid))

# ── TEST 1: Within-group W_U cosine similarity ──────────────────────
print("\n" + "─" * 70)
print("TEST 1: Within-group W_U cosine similarity")
print("  If low → group-mean approach destroys signal")
print("─" * 70)

for d in range(10):
    tids = [tid for _, tid in groups[d]]
    cols = W_U[:, tids]  # (d_model, n)
    # Normalize columns
    norms = cols.norm(dim=0, keepdim=True)
    cols_norm = cols / norms
    # Pairwise cosine similarity
    cos_mat = (cols_norm.T @ cols_norm).numpy()
    # Off-diagonal mean
    n = len(tids)
    mask = ~np.eye(n, dtype=bool)
    mean_cos = cos_mat[mask].mean()
    std_cos = cos_mat[mask].std()
    print(f"  Digit {d}: n={n:2d} tokens, mean_cos={mean_cos:.4f} ± {std_cos:.4f}")

# ── TEST 2: Group-mean norm vs individual norms ─────────────────────
print("\n" + "─" * 70)
print("TEST 2: Group-mean W_U norm vs individual token norms")
print("  If group-mean norm << individual norms → cancellation")
print("─" * 70)

for d in range(10):
    tids = [tid for _, tid in groups[d]]
    cols = W_U[:, tids]
    individual_norms = cols.norm(dim=0).numpy()
    group_mean = cols.mean(dim=1)
    group_mean_norm = group_mean.norm().item()
    mean_individual = individual_norms.mean()
    ratio = group_mean_norm / mean_individual
    print(f"  Digit {d}: group_mean_norm={group_mean_norm:.4f}, "
          f"mean_individual_norm={mean_individual:.4f}, ratio={ratio:.4f}")

# ── TEST 3: How much of individual W_U columns does the group-mean basis capture? ──
print("\n" + "─" * 70)
print("TEST 3: Variance captured by group-mean 9D basis")
print("  For each answer token, what fraction of its W_U column lies in the 9D basis?")
print("─" * 70)

U_da, S_da = compute_unembed_basis_direct_answer(model, operand_max=99)
U_da_t = torch.from_numpy(U_da[:, :9]).float()  # (d_model, 9)

# Also get teacher-forced single-digit basis
digit_ids = get_digit_token_ids(model)
U_tf, S_tf = compute_unembed_basis(model, digit_ids)
U_tf_t = torch.from_numpy(U_tf[:, :9]).float()  # (d_model, 9)

print(f"\n  Direct-answer basis singular values: {S_da[:9].round(4)}")
print(f"  Teacher-forced basis singular values: {S_tf[:9].round(4)}")

# For each answer token, project W_U column onto both bases
da_captured = []
tf_captured = []
for ans in range(199):
    col = W_U[:, answer_tok_ids[ans]]  # (d_model,)
    col_norm_sq = col.norm() ** 2

    # Project onto direct-answer basis
    proj_da = U_da_t @ (U_da_t.T @ col)
    da_frac = (proj_da.norm() ** 2 / col_norm_sq).item()
    da_captured.append(da_frac)

    # Project onto teacher-forced basis
    proj_tf = U_tf_t @ (U_tf_t.T @ col)
    tf_frac = (proj_tf.norm() ** 2 / col_norm_sq).item()
    tf_captured.append(tf_frac)

da_captured = np.array(da_captured)
tf_captured = np.array(tf_captured)

print(f"\n  Direct-answer basis captures {da_captured.mean()*100:.1f}% ± {da_captured.std()*100:.1f}% of individual W_U columns")
print(f"  Teacher-forced basis captures {tf_captured.mean()*100:.1f}% ± {tf_captured.std()*100:.1f}% of individual W_U columns")

# By ones digit
for d in range(10):
    mask = [ans % 10 == d for ans in range(199)]
    da_d = da_captured[mask].mean() * 100
    tf_d = tf_captured[mask].mean() * 100
    print(f"    Digit {d}: DA basis={da_d:.1f}%  TF basis={tf_d:.1f}%")

# ── TEST 4: Subspace overlap between DA and TF bases ────────────────
print("\n" + "─" * 70)
print("TEST 4: Principal angles between DA and TF 9D bases")
print("  If orthogonal → completely different subspaces")
print("─" * 70)

# Principal cosines via SVD of U_da^T @ U_tf
cross = U_da_t.T @ U_tf_t  # (9, 9)
_, sigmas, _ = torch.linalg.svd(cross)
print(f"  Principal cosines: {sigmas.numpy().round(4)}")
print(f"  Mean principal angle: {np.degrees(np.arccos(sigmas.numpy().clip(0,1))).mean():.1f}°")

# ── TEST 5: What if we use ALL 199 answer token W_U columns directly (no grouping)?
print("\n" + "─" * 70)
print("TEST 5: SVD of ALL 199 answer token W_U columns (no grouping)")
print("  This tests if the full answer-token W_U subspace is low-dimensional")
print("─" * 70)

all_tids = [answer_tok_ids[a] for a in range(199)]
W_answers = W_U[:, all_tids]  # (d_model, 199)
# Center
W_centered = W_answers - W_answers.mean(dim=1, keepdim=True)
U_all, S_all, _ = torch.linalg.svd(W_centered, full_matrices=False)
S_all_np = S_all.numpy()

total_var = (S_all_np ** 2).sum()
for k in [1, 2, 5, 9, 10, 20, 50]:
    pct = 100 * (S_all_np[:k] ** 2).sum() / total_var
    print(f"  Top-{k:2d} explain {pct:.1f}% of answer-token W_U variance")

print(f"\n  Top-20 singular values: {S_all_np[:20].round(3)}")

# ── TEST 6: Direct comparison — how much does answer-token SVD basis capture?
print("\n" + "─" * 70)
print("TEST 6: Patching-relevant test — project answer W_U cols onto 9D/50D bases")
print("─" * 70)

for k in [9, 20, 50]:
    U_k = U_all[:, :k].float()
    captured = []
    for ans in range(199):
        col = W_U[:, answer_tok_ids[ans]]
        proj = U_k @ (U_k.T @ col)
        frac = (proj.norm() ** 2 / col.norm() ** 2).item()
        captured.append(frac)
    captured = np.array(captured)
    print(f"  All-answer SVD {k:2d}D basis: captures {captured.mean()*100:.1f}% ± {captured.std()*100:.1f}%")

# ── TEST 7: Compare single-digit token IDs to answer token IDs
print("\n" + "─" * 70)
print("TEST 7: Token ID comparison — single-digit vs answer tokens")
print("─" * 70)
print(f"  Single-digit token IDs: {digit_ids}")
print(f"  Answer token IDs for 0-9: {[answer_tok_ids[a] for a in range(10)]}")
print(f"  Match: {all(digit_ids[a] == answer_tok_ids[a] for a in range(10))}")

# ── TEST 8: Effective dimensionality of the answer-token W_U subspace
print("\n" + "─" * 70)
print("TEST 8: Effective dimensionality of answer-token W_U subspace")
print("  (participation ratio of singular values)")
print("─" * 70)

S2 = S_all_np ** 2
S2_norm = S2 / S2.sum()
eff_dim = 1.0 / (S2_norm ** 2).sum()
print(f"  Effective dimensionality: {eff_dim:.1f}")
print(f"  Total non-zero dimensions: {(S_all_np > 1e-6).sum()}")

print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETE")
print("=" * 70)
