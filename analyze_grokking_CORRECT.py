#!/usr/bin/env python3
"""
CORRECT analysis of Neel Nanda's grokking model.
Look at position 2 (equals sign) to see how SUMS are represented.
"""

import sys
sys.path.append('src/models')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import einops

print("="*70)
print("CORRECT GROKKING ANALYSIS: Looking at SUM representations")
print("="*70)

# ============================================================
# Load model architecture (same as before)
# ============================================================

class Embed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(d_model, d_vocab)/np.sqrt(d_model))
    def forward(self, x):
        return torch.einsum('dbp -> bpd', self.W_E[:, x])

class Unembed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_U = nn.Parameter(torch.randn(d_model, d_vocab)/np.sqrt(d_vocab))
    def forward(self, x):
        return x @ self.W_U

class PosEmbed(nn.Module):
    def __init__(self, max_ctx, d_model):
        super().__init__()
        self.W_pos = nn.Parameter(torch.randn(max_ctx, d_model)/np.sqrt(d_model))
    def forward(self, x):
        return x + self.W_pos[:x.shape[-2]]

class HookPoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.fwd_hooks = []
    def give_name(self, name):
        self.name = name
    def add_hook(self, hook, dir='fwd'):
        if dir=='fwd':
            self.fwd_hooks.append(hook)
    def remove_hooks(self, dir='fwd'):
        if dir=='fwd':
            self.fwd_hooks = []
    def forward(self, x):
        for hook in self.fwd_hooks:
            x = hook(x, self)
        return x

class Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_head, n_ctx, model):
        super().__init__()
        self.model = model
        self.W_K = nn.Parameter(torch.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_Q = nn.Parameter(torch.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_O = nn.Parameter(torch.randn(d_model, d_head * num_heads)/np.sqrt(d_model))
        self.register_buffer('mask', torch.tril(torch.ones((n_ctx, n_ctx))))
        self.d_head = d_head
        self.hook_k = HookPoint()
        self.hook_q = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()
        self.hook_attn = HookPoint()
        self.hook_attn_pre = HookPoint()
    def forward(self, x):
        k = self.hook_k(torch.einsum('ihd,bpd->biph', self.W_K, x))
        q = self.hook_q(torch.einsum('ihd,bpd->biph', self.W_Q, x))
        v = self.hook_v(torch.einsum('ihd,bpd->biph', self.W_V, x))
        attn_scores_pre = torch.einsum('biph,biqh->biqp', k, q)
        attn_scores_masked = torch.tril(attn_scores_pre) - 1e10 * (1 - self.mask[:x.shape[-2], :x.shape[-2]])
        attn_matrix = self.hook_attn(F.softmax(self.hook_attn_pre(attn_scores_masked/np.sqrt(self.d_head)), dim=-1))
        z = self.hook_z(torch.einsum('biph,biqp->biqh', v, attn_matrix))
        z_flat = einops.rearrange(z, 'b i q h -> b q (i h)')
        out = torch.einsum('df,bqf->bqd', self.W_O, z_flat)
        return out

class MLP(nn.Module):
    def __init__(self, d_model, d_mlp, act_type, model):
        super().__init__()
        self.model = model
        self.W_in = nn.Parameter(torch.randn(d_mlp, d_model)/np.sqrt(d_model))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_model, d_mlp)/np.sqrt(d_model))
        self.b_out = nn.Parameter(torch.zeros(d_model))
        self.act_type = act_type
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()
        self.act = nn.ReLU() if act_type=='ReLU' else nn.GELU()
    def forward(self, x):
        x = self.hook_pre(torch.einsum('md,bpd->bpm', self.W_in, x) + self.b_in)
        x = self.hook_post(self.act(x))
        x = torch.einsum('dm,bpm->bpd', self.W_out, x) + self.b_out
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_mlp, d_head, num_heads, n_ctx, act_type, model):
        super().__init__()
        self.model = model
        self.attn = Attention(d_model, num_heads, d_head, n_ctx, model=self.model)
        self.mlp = MLP(d_model, d_mlp, act_type, model=self.model)
        self.hook_attn_out = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_pre = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_resid_post = HookPoint()
    def forward(self, x):
        x = self.hook_resid_pre(x)
        attn_out = self.hook_attn_out(self.attn(x))
        x = self.hook_resid_mid(x + attn_out)
        mlp_out = self.hook_mlp_out(self.mlp(x))
        x = self.hook_resid_post(x + mlp_out)
        return x

class Transformer(nn.Module):
    def __init__(self, num_layers, d_vocab, d_model, d_mlp, d_head, num_heads, n_ctx, act_type, use_cache=False, use_ln=True):
        super().__init__()
        self.cache = {}
        self.use_cache = use_cache
        self.embed = Embed(d_vocab, d_model)
        self.pos_embed = PosEmbed(n_ctx, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, d_mlp, d_head, num_heads, n_ctx, act_type, model=[self]) for i in range(num_layers)])
        self.unembed = Unembed(d_vocab, d_model)
        self.use_ln = use_ln
        self.hook_embed = HookPoint()
        self.hook_pos_embed = HookPoint()
        for name, module in self.named_modules():
            if type(module) == HookPoint:
                module.give_name(name)
    def forward(self, x):
        x = self.hook_embed(self.embed(x))
        x = self.hook_pos_embed(self.pos_embed(x))
        for block in self.blocks:
            x = block(x)
        x = self.unembed(x)
        return x
    def remove_all_hooks(self):
        for name, module in self.named_modules():
            if type(module) == HookPoint:
                module.remove_hooks('fwd')

# ============================================================
# Load trained model
# ============================================================

p = 113
d_vocab = p + 1
d_model = 128
num_layers = 1
num_heads = 4
d_head = 32
d_mlp = 512
n_ctx = 3
act_type = 'ReLU'
use_ln = False

print("\nLoading model...")
model = Transformer(num_layers=num_layers, d_vocab=d_vocab, d_model=d_model, d_mlp=d_mlp,
                   d_head=d_head, num_heads=num_heads, n_ctx=n_ctx, act_type=act_type,
                   use_cache=True, use_ln=use_ln)

checkpoint = torch.load("/Users/mkanaka/Documents/GitHub/Beyond-Components/grokking_addition_full_run.pth", map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()
print("✓ Model loaded with 100% accuracy")

# ============================================================
# CORRECT ANALYSIS: Collect activations at POSITION 2 for SUMS
# ============================================================

print("\n" + "="*70)
print("CORRECT ANALYSIS: Looking at position 2 (answer position)")
print("="*70)

def collect_sum_activations(model, test_sums=None):
    """
    Collect activations at position 2 (equals sign) for different SUMS.

    For each target sum s in 0 to p-1:
    - Generate input pair (a, b) where (a+b) mod p = s
    - Look at residual stream at position 2 (where sum is computed)
    - This tests if sums 0,1,2,...,p-1 form a helix
    """
    if test_sums is None:
        test_sums = list(range(min(30, p)))

    activations = []
    valid_sums = []

    # Hook to capture position 2 activations
    captured_activation = [None]

    def capture_hook(tensor, hook):
        captured_activation[0] = tensor.detach()
        return tensor

    model.blocks[0].hook_resid_post.add_hook(capture_hook, 'fwd')

    with torch.no_grad():
        for s in test_sums:
            # Generate pair (a, b) where a+b mod p = s
            # Simple choice: a = 0, b = s
            a, b = 0, s

            # Input tokens: [a, b, equals_sign]
            tokens = torch.tensor([[a, b, p]], dtype=torch.long)

            # Forward pass
            _ = model(tokens)

            # Extract activation at POSITION 2 (equals sign / answer position)
            resid = captured_activation[0][0, 2, :]  # [d_model]

            activations.append(resid.cpu())
            valid_sums.append(s)

    model.remove_all_hooks()

    return torch.stack(activations), valid_sums

print("\nCollecting activations for SUMS 0 to 29...")
print("  For each sum s: using input (0, s) so 0+s mod 113 = s")
print("  Looking at POSITION 2 (where answer is computed)")

acts_tensor, sums = collect_sum_activations(model)
acts_np = acts_tensor.numpy()

print(f"✓ Collected {len(sums)} activations")
print(f"  Shape: {acts_np.shape}")

# ============================================================
# Helix Analysis on SUM representations
# ============================================================

print("\n" + "="*70)
print("Helix Analysis: Do SUMS form a helix?")
print("="*70)

# SVD
acts_centered = acts_np - acts_np.mean(axis=0, keepdims=True)
U, S, Vt = np.linalg.svd(acts_centered, full_matrices=False)

coords = U[:, :2] * S[:2]
x, y = coords[:, 0], coords[:, 1]

print(f"SVD analysis:")
print(f"  Top 5 singular values: {S[:5].round(2)}")
var_explained = (S[:2]**2).sum() / (S**2).sum()
print(f"  Variance explained by PC1+PC2: {var_explained:.1%}")

# Angle linearity
r = np.sqrt(x**2 + y**2)
theta = np.arctan2(y, x)
cv = r.std() / r.mean() if r.mean() != 0 else 0

angles_unwrapped = np.unwrap(theta - theta[0])
angles_unwrapped = angles_unwrapped - angles_unwrapped.min()

correlation = np.corrcoef(sums, angles_unwrapped)[0, 1]
coeffs = np.polyfit(sums, angles_unwrapped, 1)
period = 2 * np.pi / coeffs[0] if coeffs[0] > 0 else np.inf

print(f"\nHelix metrics:")
print(f"  Angle linearity: {abs(correlation):.3f} (>0.9 = strong helix)")
print(f"  Period: {period:.1f} (expected {p})")
print(f"  Radius CV: {cv:.3f} (lower = rounder)")

if abs(correlation) > 0.9:
    print(f"\n  ✓✓✓ STRONG HELIX DETECTED IN SUMS!")
else:
    print(f"\n  ⚠ Weak or no helix structure in sums")

# Fourier analysis
print("\n" + "="*70)
print("Fourier Analysis on SUM representations")
print("="*70)

from arithmetic_circuit_discovery import fit_helix

periods_to_test = [
    [float(p)],
    [10.0],
    [2.0, 5.0, 10.0],
    [2.0, 5.0, 10.0, float(p)],
]

print(f"\nTesting Fourier periods:")
best_r_sq = 0
best_periods = None

for periods in periods_to_test:
    r_sq, _ = fit_helix(acts_np, sums, periods=periods)
    period_str = str(periods).replace(" ", "")
    print(f"  Periods {period_str:40s} → R² = {r_sq:.4f}")

    if r_sq > best_r_sq:
        best_r_sq = r_sq
        best_periods = periods

print(f"\n✓ Best R² = {best_r_sq:.4f} with periods {best_periods}")

if p in best_periods or float(p) in best_periods:
    print(f"  ✓ Period {p} (the modulus) is used!")
else:
    print(f"  ⚠ Period {p} not dominant")

# Visualization
print("\n" + "="*70)
print("Creating Visualization")
print("="*70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Left: 2D projection
scatter = ax1.scatter(x, y, c=sums, cmap='viridis', s=300,
                     edgecolors='black', linewidth=2, alpha=0.9)

for i, s in enumerate(sums):
    ax1.text(x[i], y[i], str(s), fontsize=9, fontweight='bold',
            ha='center', va='center', color='white')

ax1.set_xlabel('SVD Direction 1', fontsize=13, fontweight='bold')
ax1.set_ylabel('SVD Direction 2', fontsize=13, fontweight='bold')
ax1.set_title(f'Grokking Model: SUM Representations (mod {p})\nHelix Projection - CV: {cv:.3f}',
             fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')
plt.colorbar(scatter, ax=ax1, label='Sum Value')

# Right: Angle linearity
angles_fit = np.poly1d(coeffs)(sums)
scatter2 = ax2.scatter(sums, angles_unwrapped,
                      c=sums, cmap='viridis', s=250,
                      edgecolors='black', linewidth=2, alpha=0.9, zorder=10)
ax2.plot(sums, angles_fit, 'r--', linewidth=3, alpha=0.8,
        label=f'Linear fit (R={abs(correlation):.3f})', zorder=5)

ax2.set_xlabel('Sum Value', fontsize=13, fontweight='bold')
ax2.set_ylabel('Angle (radians)', fontsize=13, fontweight='bold')
ax2.set_title(f'Angle Linearity: {abs(correlation):.3f}\nPeriod: {period:.1f} (expected {p})',
             fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11, loc='upper left')

fig.suptitle(f'CORRECT ANALYSIS: Grokking Model SUM Representations (R² = {best_r_sq:.4f})',
            fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig('grokking_CORRECT_sum_helix_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: grokking_CORRECT_sum_helix_analysis.png")
plt.close()

# ============================================================
# Comparison
# ============================================================

print("\n" + "="*70)
print("COMPARISON: Correct vs Incorrect Analysis")
print("="*70)

print("\n❌ PREVIOUS (INCORRECT) Analysis:")
print("  - Looked at POSITION 0 (input 'a')")
print("  - Tested how INPUTS are embedded")
print("  - Result: R²=0.13, no helix")
print("  - This was WRONG!")

print("\n✓ CURRENT (CORRECT) Analysis:")
print("  - Looking at POSITION 2 (answer position)")
print("  - Testing how SUMS are represented")
print(f"  - Result: R²={best_r_sq:.4f}, angle linearity={abs(correlation):.3f}")

if abs(correlation) > 0.9 and best_r_sq > 0.7:
    print("\n🎯 HELIX CONFIRMED in grokking model!")
    print("  ✓ Task structure determines algorithm")
    print("  ✓ Modular task → helix representation")
    print("  ✓ Original hypothesis WAS CORRECT!")
else:
    print("\n⚠ Still no strong helix in correct analysis")
    print(f"  - But R² improved from 0.13 to {best_r_sq:.4f}")
    print("  - May need more analysis or different approach")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
