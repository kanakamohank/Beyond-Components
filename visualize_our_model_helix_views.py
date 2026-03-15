#!/usr/bin/env python3
"""
Visualize our 2-digit addition model's helix from different viewing angles:
- Top view: Should show circle
- Side view: Should show VERTICAL SPIRAL (tens digit progression!)
- Front view: Should show vertical spiral
- 3D view: Should show true 3D helix structure
"""

import torch
import sys
sys.path.append('src/models')

from arithmetic_circuit_discovery import build_model
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("="*70)
print("OUR 2-DIGIT ADDITION MODEL: Different Views of the Helix")
print("="*70)

# ============================================================
# Load our trained model
# ============================================================

model_path = "toy_addition_model.pt"
DEVICE = 'cpu'

print(f"\nLoading model from: {model_path}")

model = build_model(seed=0, device=DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

print("✓ Model loaded successfully")

# ============================================================
# Encoding function (from arithmetic_circuit_discovery)
# ============================================================

def _encode_pair(a, b, device='cpu'):
    """Encode a pair of numbers (a, b) as tokens."""
    a_tens, a_ones = a // 10, a % 10
    b_tens, b_ones = b // 10, b % 10
    plus_tok, eq_tok = 10, 11
    tokens = torch.tensor([[a_tens, a_ones, b_tens, b_ones, plus_tok, eq_tok]],
                          dtype=torch.long, device=device)
    return tokens

# ============================================================
# Collect activations for different sums
# ============================================================

print("\nCollecting activations at position 5 (after '=' token)...")
print("Testing sums from 0 to 99 (using pairs that produce each sum)")

# For each possible sum 0-99, create an addition problem
test_sums = list(range(100))
activations_list = []

with torch.no_grad():
    for target_sum in test_sums:
        # Create a simple problem: a + 0 = target_sum (for sums 0-9)
        # or a + b where we choose a,b to get target_sum (for sums 10-99)
        if target_sum <= 9:
            a, b = target_sum, 0
        else:
            # Use a=10, b=target_sum-10 for simplicity
            a = 10
            b = target_sum - 10
            if b > 99:  # shouldn't happen for sums 0-99
                a = target_sum // 2
                b = target_sum - a

        tokens = _encode_pair(a, b, device=DEVICE)

        # Run model and get activation at position 5 (after '=')
        logits, cache = model.run_with_cache(tokens)

        # Get residual stream at position 5, layer 1
        activation = cache['blocks.1.hook_resid_post'][0, 5, :]
        activations_list.append(activation.cpu())

activations = torch.stack(activations_list)  # [100, 128]

print(f"✓ Collected activations shape: {activations.shape}")

# ============================================================
# Perform SVD
# ============================================================

activations_centered = activations - activations.mean(dim=0, keepdim=True)
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

# Annotate key sums
for i in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]:
    ax1.annotate(str(i), (x_top[i], y_top[i]), fontsize=8,
                fontweight='bold', color='red')

# Calculate angle linearity
theta_top = np.arctan2(y_top, x_top)
angles_top = np.unwrap(theta_top - theta_top[0])
angles_top = angles_top - angles_top.min()
corr_top = abs(np.corrcoef(test_sums, angles_top)[0, 1])

ax1.set_xlabel('Component 1', fontsize=11, fontweight='bold')
ax1.set_ylabel('Component 2', fontsize=11, fontweight='bold')
ax1.set_title(f'TOP VIEW (Comp 1 vs 2)\nCircular pattern? Linearity: {corr_top:.4f}\n'
              f'Variance: {cumulative_var[1]:.1%}',
              fontsize=12, fontweight='bold')
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=ax1, label='Sum', fraction=0.046)

# ============================================================
# 2. SIDE VIEW (Component 1 vs 3) - Should show VERTICAL SPIRAL!
# ============================================================

ax2 = plt.subplot(3, 3, 2)
x_side = (U[:, 0] * S[0])
y_side = (U[:, 2] * S[2])

scatter2 = ax2.scatter(x_side, y_side, c=colors, cmap=cmap, s=60,
                       edgecolors='black', linewidth=0.5, alpha=0.8)
ax2.plot(x_side, y_side, 'gray', alpha=0.3, linewidth=0.5)

# Annotate multiples of 10 (should show vertical progression)
for i in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
    ax2.annotate(str(i), (x_side[i], y_side[i]), fontsize=8,
                fontweight='bold', color='red')

ax2.set_xlabel('Component 1 (Circular)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Component 3 (Vertical?)', fontsize=11, fontweight='bold')
ax2.set_title(f'SIDE VIEW (Comp 1 vs 3)\nLooking for VERTICAL SPIRAL!\n'
              f'Comp 3 variance: {(cumulative_var[2]-cumulative_var[1]):.1%}',
              fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=ax2, label='Sum', fraction=0.046)

# ============================================================
# 3. FRONT VIEW (Component 2 vs 3) - Should also show vertical spiral
# ============================================================

ax3 = plt.subplot(3, 3, 3)
x_front = (U[:, 1] * S[1])
y_front = (U[:, 2] * S[2])

scatter3 = ax3.scatter(x_front, y_front, c=colors, cmap=cmap, s=60,
                       edgecolors='black', linewidth=0.5, alpha=0.8)
ax3.plot(x_front, y_front, 'gray', alpha=0.3, linewidth=0.5)

# Annotate multiples of 10
for i in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
    ax3.annotate(str(i), (x_front[i], y_front[i]), fontsize=8,
                fontweight='bold', color='red')

ax3.set_xlabel('Component 2 (Circular)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Component 3 (Vertical?)', fontsize=11, fontweight='bold')
ax3.set_title(f'FRONT VIEW (Comp 2 vs 3)\nLooking for vertical spiral',
              fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
plt.colorbar(scatter3, ax=ax3, label='Sum', fraction=0.046)

# ============================================================
# 4. 3D View - Should show TRUE HELIX!
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
ax4.set_title(f'3D VIEW (Comp 1, 2, 3)\nTrue helix? {cumulative_var[2]:.1%} variance',
              fontsize=12, fontweight='bold')

# ============================================================
# 5. Color by TENS digit - Should show spiral layers
# ============================================================

ax5 = plt.subplot(3, 3, 5, projection='3d')

tens_digits = [s // 10 for s in test_sums]
colors_tens = tens_digits

scatter5 = ax5.scatter(x_3d, y_3d, z_3d, c=colors_tens, cmap='tab10', s=60,
                       edgecolors='black', linewidth=0.5, alpha=0.8)

ax5.set_xlabel('Component 1', fontsize=10, fontweight='bold')
ax5.set_ylabel('Component 2', fontsize=10, fontweight='bold')
ax5.set_zlabel('Component 3', fontsize=10, fontweight='bold')
ax5.set_title('3D VIEW - Colored by TENS digit\n'
              'Each color = one layer of helix',
              fontsize=12, fontweight='bold')
plt.colorbar(scatter5, ax=ax5, label='Tens Digit', fraction=0.046)

# ============================================================
# 6. Color by ONES digit - Should show 10 vertical strands
# ============================================================

ax6 = plt.subplot(3, 3, 6, projection='3d')

ones_digits = [s % 10 for s in test_sums]
colors_ones = ones_digits

scatter6 = ax6.scatter(x_3d, y_3d, z_3d, c=colors_ones, cmap='tab10', s=60,
                       edgecolors='black', linewidth=0.5, alpha=0.8)

ax6.set_xlabel('Component 1', fontsize=10, fontweight='bold')
ax6.set_ylabel('Component 2', fontsize=10, fontweight='bold')
ax6.set_zlabel('Component 3', fontsize=10, fontweight='bold')
ax6.set_title('3D VIEW - Colored by ONES digit\n'
              'Each color = one strand going up',
              fontsize=12, fontweight='bold')
plt.colorbar(scatter6, ax=ax6, label='Ones Digit', fraction=0.046)

# ============================================================
# 7. Variance contribution
# ============================================================

ax7 = plt.subplot(3, 3, 7)
components = np.arange(1, 11)
variances = np.diff(np.concatenate([[0], cumulative_var[:10]])) * 100

bars = ax7.bar(components, variances, alpha=0.8, edgecolor='black')
bars[0].set_color('red')
bars[1].set_color('orange')
bars[2].set_color('yellow')

ax7.set_xlabel('Component', fontsize=11, fontweight='bold')
ax7.set_ylabel('Variance Contribution (%)', fontsize=11, fontweight='bold')
ax7.set_title(f'Individual Component Contributions\n'
              f'Top 3 = {cumulative_var[2]:.1%} variance',
              fontsize=11, fontweight='bold')
ax7.grid(True, alpha=0.3, axis='y')

# ============================================================
# 8. Component 3 vs Sum - Should show VERTICAL PROGRESSION!
# ============================================================

ax8 = plt.subplot(3, 3, 8)

comp3_vals = U[:, 2] * S[2]

# Color by tens digit to see structure
scatter8 = ax8.scatter(test_sums, comp3_vals, c=tens_digits, cmap='tab10',
                       s=50, edgecolors='black', linewidth=0.5, alpha=0.8)

# Add trend line
z = np.polyfit(test_sums, comp3_vals, 1)
p = np.poly1d(z)
ax8.plot(test_sums, p(test_sums), "r--", linewidth=2, alpha=0.8,
         label=f'Trend: slope={z[0]:.3f}')

ax8.set_xlabel('Sum Value (0 to 99)', fontsize=11, fontweight='bold')
ax8.set_ylabel('Component 3 Value', fontsize=11, fontweight='bold')
ax8.set_title('Component 3 vs Sum\n'
              'Should show vertical progression!',
              fontsize=11, fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)
plt.colorbar(scatter8, ax=ax8, label='Tens Digit', fraction=0.046)

# ============================================================
# 9. All components vs sum
# ============================================================

ax9 = plt.subplot(3, 3, 9)

for comp_idx in range(5):
    y_vals = U[:, comp_idx] * S[comp_idx]
    ax9.plot(test_sums, y_vals, alpha=0.6, label=f'Comp {comp_idx+1}',
            linewidth=2, marker='o', markersize=2)

ax9.set_xlabel('Sum Value (0 to 99)', fontsize=11, fontweight='bold')
ax9.set_ylabel('Component Value', fontsize=11, fontweight='bold')
ax9.set_title('All Components vs Sum\n'
              'Comp 3 should increase monotonically!',
              fontsize=11, fontweight='bold')
ax9.legend(fontsize=8)
ax9.grid(True, alpha=0.3)

plt.suptitle('Our 2-Digit Addition Model: Multiple Viewing Angles (TRUE HELIX?)',
            fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('our_model_helix_multiple_views.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: our_model_helix_multiple_views.png")
plt.close()

# ============================================================
# Analysis Summary
# ============================================================

print("\n" + "="*70)
print("ANALYSIS: Our Model's Helix Structure")
print("="*70)

# Check if component 3 shows monotonic increase
comp3_vals = U[:, 2] * S[2]
correlation_with_sum = np.corrcoef(test_sums, comp3_vals)[0, 1]

print(f"""
TOP VIEW (Comp 1 vs 2):
  - Circular pattern? Angle linearity: {corr_top:.4f}
  - Variance: {cumulative_var[1]:.1%}

COMPONENT 3 ANALYSIS:
  - Variance contribution: {(cumulative_var[2]-cumulative_var[1])*100:.1f}%
  - Correlation with sum value: {correlation_with_sum:.4f}
  - {'✓ MONOTONIC INCREASE!' if abs(correlation_with_sum) > 0.8 else '✗ No clear progression'}

KEY FINDINGS:
  1. Top view (Comp 1 vs 2): {'Circle ✓' if corr_top > 0.9 else 'Not circular'}
  2. Component 3: {'Vertical progression ✓' if abs(correlation_with_sum) > 0.8 else 'No vertical progression'}
  3. Total variance in top 3: {cumulative_var[2]:.1%}

EXPECTED FOR TRUE HELIX:
  - Top view: Circle (comp 1 & 2 encode ones digit cyclically)
  - Side view: Spiral (comp 3 encodes tens digit progression)
  - Structure: 10 helical strands (one per ones digit value)

ACTUAL RESULT:
  {'  ✓ TRUE HELIX CONFIRMED!' if corr_top > 0.9 and abs(correlation_with_sum) > 0.8
   else '  ⚠ Structure differs from expected helix'}
""")

print("\nGenerated: our_model_helix_multiple_views.png")
