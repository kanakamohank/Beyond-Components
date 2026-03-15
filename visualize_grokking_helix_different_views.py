#!/usr/bin/env python3
"""
Visualize grokking model's helix from different viewing angles:
- Top view: Components 1 vs 2 (circular)
- Side view: Component 1 vs 3 (vertical spiral)
- Front view: Component 2 vs 3 (vertical spiral)
- 3D view: All three components together
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import einops

# ============================================================
# Model architecture (same as before)
# ============================================================

class Embed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(d_model, d_vocab) / np.sqrt(d_model))

    def forward(self, x):
        return torch.einsum('dbp -> bpd', self.W_E[:, x])


class Unembed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_U = nn.Parameter(torch.randn(d_model, d_vocab) / np.sqrt(d_vocab))

    def forward(self, x):
        return (x @ self.W_U)


class PosEmbed(nn.Module):
    def __init__(self, max_ctx, d_model):
        super().__init__()
        self.W_pos = nn.Parameter(torch.randn(max_ctx, d_model) / np.sqrt(d_model))

    def forward(self, x):
        return x + self.W_pos[:x.shape[-2]]


class Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_head, n_ctx):
        super().__init__()
        self.W_K = nn.Parameter(torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_Q = nn.Parameter(torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_O = nn.Parameter(torch.randn(d_model, d_head * num_heads) / np.sqrt(d_model))
        self.register_buffer('mask', torch.tril(torch.ones((n_ctx, n_ctx))))
        self.d_head = d_head

    def forward(self, x):
        k = torch.einsum('ihd,bpd->biph', self.W_K, x)
        q = torch.einsum('ihd,bpd->biph', self.W_Q, x)
        v = torch.einsum('ihd,bpd->biph', self.W_V, x)
        attn_scores_pre = torch.einsum('biph,biqh->biqp', k, q)
        attn_scores_masked = torch.tril(attn_scores_pre) - 1e10 * (1 - self.mask[:x.shape[-2], :x.shape[-2]])
        attn_matrix = F.softmax(attn_scores_masked / np.sqrt(self.d_head), dim=-1)
        z = torch.einsum('biph,biqp->biqh', v, attn_matrix)
        z_flat = einops.rearrange(z, 'b i q h -> b q (i h)')
        out = torch.einsum('df,bqf->bqd', self.W_O, z_flat)
        return out


class MLP(nn.Module):
    def __init__(self, d_model, d_mlp):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(d_mlp, d_model) / np.sqrt(d_model))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_model, d_mlp) / np.sqrt(d_model))
        self.b_out = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        x = torch.einsum('md,bpd->bpm', self.W_in, x) + self.b_in
        x = torch.relu(x)
        x = torch.einsum('dm,bpm->bpd', self.W_out, x) + self.b_out
        return x


class Transformer(nn.Module):
    def __init__(self, num_layers, d_vocab, d_model, d_mlp, d_head, num_heads, n_ctx):
        super().__init__()
        self.embed = Embed(d_vocab, d_model)
        self.pos_embed = PosEmbed(n_ctx, d_model)
        self.unembed = Unembed(d_vocab, d_model)
        self.blocks = nn.ModuleList([nn.ModuleDict({
            'attn': Attention(d_model, num_heads, d_head, n_ctx),
            'mlp': MLP(d_model, d_mlp)
        }) for _ in range(num_layers)])

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_embed(x)
        for block in self.blocks:
            x = x + block['attn'](x)
            x = x + block['mlp'](x)
        return self.unembed(x)


print("="*70)
print("GROKKING MODEL: Different Views of the Helix")
print("="*70)

# Load checkpoint
checkpoint = torch.load(
    "/Users/mkanaka/Documents/GitHub/Beyond-Components/grokking_addition_full_run.pth",
    map_location='cpu'
)

config = checkpoint['config']
p = config['p']
d_model = config['d_model']

print(f"\nConfig: p={p}, d_model={d_model}")

# Create and load model
model = Transformer(
    num_layers=1,
    d_vocab=p + 1,
    d_model=d_model,
    d_mlp=4 * d_model,
    d_head=d_model // 4,
    num_heads=4,
    n_ctx=3
)
model.load_state_dict(checkpoint['model'])
model.eval()

# ============================================================
# Collect activations
# ============================================================

print("\nCollecting activations at position 2...")

captured_activations = []

def hook_fn(module, input, output):
    captured_activations.append(output[0, 2, :].detach().clone())
    return output

handle = model.blocks[-1]['mlp'].register_forward_hook(hook_fn)

test_sums = list(range(p))

with torch.no_grad():
    for s in test_sums:
        a, b = 0, s
        tokens = torch.tensor([[a, b, p]], dtype=torch.long)
        _ = model(tokens)

handle.remove()

activations = torch.stack(captured_activations)  # [113, 128]
activations_centered = activations - activations.mean(dim=0, keepdim=True)

print(f"Activations shape: {activations.shape}")

# ============================================================
# Perform SVD
# ============================================================

U_torch, S_torch, Vt_torch = torch.linalg.svd(activations_centered, full_matrices=False)
U = U_torch.numpy()
S = S_torch.numpy()

print(f"\nSVD Results:")
print(f"  Top 10 singular values: {S[:10].round(2)}")

cumulative_var = np.cumsum(S**2) / (S**2).sum()
print(f"\nVariance explained:")
for i in range(10):
    print(f"  Component {i+1}: {cumulative_var[i]:.1%}")

# ============================================================
# Create different views
# ============================================================

fig = plt.figure(figsize=(20, 15))

# Colors for points
colors = test_sums
cmap = plt.cm.viridis

# ============================================================
# 1. TOP VIEW (Component 1 vs 2) - Should show circle
# ============================================================

ax1 = plt.subplot(3, 3, 1)
coords_top = U[:, :2] @ np.diag(S[:2])
x_top, y_top = coords_top[:, 0], coords_top[:, 1]

scatter1 = ax1.scatter(x_top, y_top, c=colors, cmap=cmap, s=60,
                       edgecolors='black', linewidth=0.5, alpha=0.8)
ax1.plot(x_top, y_top, 'gray', alpha=0.3, linewidth=0.5)

# Annotate some points
for i in [0, p//4, p//2, 3*p//4, p-1]:
    ax1.annotate(str(i), (x_top[i], y_top[i]), fontsize=8,
                fontweight='bold', color='red')

# Calculate angle linearity
theta_top = np.arctan2(y_top, x_top)
angles_top = np.unwrap(theta_top - theta_top[0])
angles_top = angles_top - angles_top.min()
corr_top = abs(np.corrcoef(test_sums, angles_top)[0, 1])

ax1.set_xlabel('Component 1', fontsize=11, fontweight='bold')
ax1.set_ylabel('Component 2', fontsize=11, fontweight='bold')
ax1.set_title(f'TOP VIEW (Comp 1 vs 2)\nCircle! Angle Linearity: {corr_top:.4f}\n'
              f'Variance: {cumulative_var[1]:.1%}',
              fontsize=12, fontweight='bold')
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=ax1, label='Sum', fraction=0.046)

# ============================================================
# 2. SIDE VIEW (Component 1 vs 3) - Should show vertical spiral
# ============================================================

ax2 = plt.subplot(3, 3, 2)
x_side = (U[:, 0] * S[0])
y_side = (U[:, 2] * S[2])

scatter2 = ax2.scatter(x_side, y_side, c=colors, cmap=cmap, s=60,
                       edgecolors='black', linewidth=0.5, alpha=0.8)
ax2.plot(x_side, y_side, 'gray', alpha=0.3, linewidth=0.5)

# Annotate
for i in [0, p//4, p//2, 3*p//4, p-1]:
    ax2.annotate(str(i), (x_side[i], y_side[i]), fontsize=8,
                fontweight='bold', color='red')

ax2.set_xlabel('Component 1 (Circular)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Component 3 (Vertical?)', fontsize=11, fontweight='bold')
ax2.set_title(f'SIDE VIEW (Comp 1 vs 3)\nLooking for vertical spiral\n'
              f'Comp 3 variance: {(cumulative_var[2]-cumulative_var[1]):.1%}',
              fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=ax2, label='Sum', fraction=0.046)

# ============================================================
# 3. FRONT VIEW (Component 2 vs 3)
# ============================================================

ax3 = plt.subplot(3, 3, 3)
x_front = (U[:, 1] * S[1])
y_front = (U[:, 2] * S[2])

scatter3 = ax3.scatter(x_front, y_front, c=colors, cmap=cmap, s=60,
                       edgecolors='black', linewidth=0.5, alpha=0.8)
ax3.plot(x_front, y_front, 'gray', alpha=0.3, linewidth=0.5)

# Annotate
for i in [0, p//4, p//2, 3*p//4, p-1]:
    ax3.annotate(str(i), (x_front[i], y_front[i]), fontsize=8,
                fontweight='bold', color='red')

ax3.set_xlabel('Component 2 (Circular)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Component 3 (Vertical?)', fontsize=11, fontweight='bold')
ax3.set_title(f'FRONT VIEW (Comp 2 vs 3)\nLooking for vertical spiral',
              fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
plt.colorbar(scatter3, ax=ax3, label='Sum', fraction=0.046)

# ============================================================
# 4. 3D View - All three components
# ============================================================

ax4 = plt.subplot(3, 3, 4, projection='3d')
x_3d = (U[:, 0] * S[0])
y_3d = (U[:, 1] * S[1])
z_3d = (U[:, 2] * S[2])

scatter4 = ax4.scatter(x_3d, y_3d, z_3d, c=colors, cmap=cmap, s=60,
                       edgecolors='black', linewidth=0.5, alpha=0.8)

# Draw connecting lines
for i in range(len(test_sums)-1):
    ax4.plot([x_3d[i], x_3d[i+1]], [y_3d[i], y_3d[i+1]], [z_3d[i], z_3d[i+1]],
            'gray', alpha=0.2, linewidth=0.5)

ax4.set_xlabel('Component 1', fontsize=10, fontweight='bold')
ax4.set_ylabel('Component 2', fontsize=10, fontweight='bold')
ax4.set_zlabel('Component 3', fontsize=10, fontweight='bold')
ax4.set_title(f'3D VIEW (Comp 1, 2, 3)\n{cumulative_var[2]:.1%} variance',
              fontsize=12, fontweight='bold')

# ============================================================
# 5-7. Try different component combinations to find helix
# ============================================================

view_configs = [
    (0, 3, "Comp 1 vs 4"),
    (0, 4, "Comp 1 vs 5"),
    (1, 4, "Comp 2 vs 5"),
]

for idx, (comp1, comp2, title) in enumerate(view_configs):
    ax = plt.subplot(3, 3, 5 + idx)

    x = (U[:, comp1] * S[comp1])
    y = (U[:, comp2] * S[comp2])

    scatter = ax.scatter(x, y, c=colors, cmap=cmap, s=60,
                        edgecolors='black', linewidth=0.5, alpha=0.8)
    ax.plot(x, y, 'gray', alpha=0.3, linewidth=0.5)

    # Annotate
    for i in [0, p//4, p//2, 3*p//4, p-1]:
        ax.annotate(str(i), (x[i], y[i]), fontsize=7,
                   fontweight='bold', color='red')

    ax.set_xlabel(f'Component {comp1+1}', fontsize=10, fontweight='bold')
    ax.set_ylabel(f'Component {comp2+1}', fontsize=10, fontweight='bold')
    ax.set_title(f'{title}\nSearching for spiral pattern',
                fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Sum', fraction=0.046)

# ============================================================
# 8. Variance contribution of each component
# ============================================================

ax8 = plt.subplot(3, 3, 8)
components = np.arange(1, 11)
variances = np.diff(np.concatenate([[0], cumulative_var[:10]])) * 100

bars = ax8.bar(components, variances, alpha=0.8, edgecolor='black')
bars[0].set_color('red')
bars[1].set_color('orange')
bars[2].set_color('yellow')

ax8.set_xlabel('Component', fontsize=11, fontweight='bold')
ax8.set_ylabel('Variance Contribution (%)', fontsize=11, fontweight='bold')
ax8.set_title('Individual Component Contributions\n(Red/Orange/Yellow = Comp 1/2/3)',
              fontsize=11, fontweight='bold')
ax8.grid(True, alpha=0.3, axis='y')

# ============================================================
# 9. Vertical progression analysis
# ============================================================

ax9 = plt.subplot(3, 3, 9)

# Check if any component shows monotonic increase (vertical progression)
for comp_idx in range(5):
    y_vals = U[:, comp_idx] * S[comp_idx]
    ax9.plot(test_sums, y_vals, alpha=0.6, label=f'Comp {comp_idx+1}',
            linewidth=2, marker='o', markersize=2)

ax9.set_xlabel('Sum Value (0 to 112)', fontsize=11, fontweight='bold')
ax9.set_ylabel('Component Value', fontsize=11, fontweight='bold')
ax9.set_title('Components vs Sum Value\n(Looking for monotonic progression)',
              fontsize=11, fontweight='bold')
ax9.legend(fontsize=8)
ax9.grid(True, alpha=0.3)

plt.suptitle('Grokking Model Helix: Multiple Viewing Angles',
            fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('grokking_helix_multiple_views.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: grokking_helix_multiple_views.png")
plt.close()

# ============================================================
# Analysis Summary
# ============================================================

print("\n" + "="*70)
print("ANALYSIS: Different Views of the Helix")
print("="*70)

print(f"""
TOP VIEW (Comp 1 vs 2):
  - Shows circular pattern ✓
  - Angle linearity: {corr_top:.4f}
  - This is the "looking down the helix" view

SIDE VIEWS (Comp 1 vs 3, Comp 2 vs 3):
  - Component 3 contribution: {(cumulative_var[2]-cumulative_var[1])*100:.1f}%
  - If helix exists, should see spiral pattern here

3D VIEW (Comp 1, 2, 3):
  - Captures {cumulative_var[2]:.1%} of variance
  - Should show full helix structure if present

KEY QUESTION:
  Do we see a helix spiraling vertically, or is the structure
  primarily circular (disc-like) rather than helical?

The difference:
  - TRUE HELIX: Circle from top + spiral from side
  - CIRCULAR DISC: Circle from top + random from side
""")

print("\nGenerated: grokking_helix_multiple_views.png")
