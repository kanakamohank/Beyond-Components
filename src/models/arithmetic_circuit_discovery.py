"""
arithmetic_circuit_discovery.py
================================
Complete implementation: Discovering Arithmetic Circuits in a
Toy Transformer using SVD on the OV Circuit.

Option B: Standard Digit-Wise Integer Addition (with carry)

Pipeline:
  Stage 1 → Train toy model to >99% accuracy
  Stage 2 → Find helix via activation patching + SVD + causal validation
  Stage 3 → Find carry circuit via MLP Jacobian-SVD
"""

# ─────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import numpy as np
import itertools
import random
import math
import warnings
from typing import List, Tuple, Dict, Optional
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# TransformerLens — install with: pip install transformer_lens
try:
    from transformer_lens import HookedTransformer, HookedTransformerConfig
except ImportError:
    raise ImportError(
        "transformer_lens not found.\n"
        "Install with: pip install transformer_lens"
    )

# ─────────────────────────────────────────────────────────────
# TOKEN LAYOUT  (fixed, known, no runtime tracking needed)
# ─────────────────────────────────────────────────────────────
#
#  Input  sequence  (length 6):
#    pos 0 → tens digit of A        (token 0-9)
#    pos 1 → ones digit of A        (token 0-9)
#    pos 2 → '+' operator           (token 10)
#    pos 3 → tens digit of B        (token 0-9)
#    pos 4 → ones digit of B        (token 0-9)
#    pos 5 → '=' sign               (token 11)
#
#  Answer sequence predicted autoregressively (length 3):
#    step 1 → hundreds digit of C   (token 0 or 1 only, since max=198)
#    step 2 → tens digit of C
#    step 3 → ones digit of C
#
#  Vocabulary: 12 tokens total
#    0-9  → digit literals
#    10   → '+'
#    11   → '='
#
POS_A_TENS = 0
POS_A_ONES = 1
POS_PLUS   = 2
POS_B_TENS = 3
POS_B_ONES = 4
POS_EQ     = 5
TOK_PLUS   = 10
TOK_EQ     = 11
VOCAB_SIZE = 12
context = 10
dim_mlp = 512
dim_head = 32
numof_heads = 4
dim_model = 128
nof_layers = 2
activation_fn = "relu"

# ─────────────────────────────────────────────────────────────
# DEVICE CONFIGURATION
# ─────────────────────────────────────────────────────────────
def get_device():
    """
    Automatically detect and return the best available device.
    Priority: MPS (Mac) > CUDA (NVIDIA) > CPU
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# Global device variable
DEVICE = get_device()
print(f"Using device: {DEVICE}")


# ═════════════════════════════════════════════════════════════
# STAGE 1: DATA + TRAINING
# ═════════════════════════════════════════════════════════════

def to_digits_msf(x: int, n: int) -> List[int]:
    """Most-significant-first digit decomposition.
    to_digits_msf(138, 3) → [1, 3, 8]
    """
    return [(x // (10 ** (n - 1 - i))) % 10 for i in range(n)]


def generate_addition_data(n_samples: int = 60000,
                            seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates 2-digit + 2-digit standard integer addition.

    Returns
    -------
    inputs  : LongTensor [N, 6]   — tokenized input sequence
    targets : LongTensor [N, 3]   — three answer digit tokens (MSF)
    """
    torch.manual_seed(seed)

    A = torch.randint(0, 100, (n_samples,))
    B = torch.randint(0, 100, (n_samples,))
    C = A + B   # range [0, 198]

    inputs, targets = [], []
    for a, b, c in zip(A.tolist(), B.tolist(), C.tolist()):
        inp = to_digits_msf(a, 2) + [TOK_PLUS] + to_digits_msf(b, 2) + [TOK_EQ]
        tgt = to_digits_msf(c, 3)          # [hundreds, tens, ones]
        inputs.append(inp)
        targets.append(tgt)

    return (torch.tensor(inputs,  dtype=torch.long),   # [N, 6]
            torch.tensor(targets, dtype=torch.long))    # [N, 3]


def build_model(seed: int = 0, device: torch.device = None) -> HookedTransformer:
    """
    Builds the minimal transformer for 2-digit addition.
    2 layers, 4 heads, d_model=128.
    No LayerNorm — cleaner for mechanistic analysis.
    Moves model to specified device (defaults to DEVICE global).
    """
    if device is None:
        device = DEVICE

    torch.manual_seed(seed)
    cfg = HookedTransformerConfig(
        n_layers=nof_layers,
        d_model=dim_model,
        n_heads=numof_heads,
        d_head=dim_head,
        d_mlp=dim_mlp,
        n_ctx=context,           # 6 input + 3 answer + 1 buffer
        d_vocab=VOCAB_SIZE,
        act_fn=activation_fn,
        normalization_type=None,  # No LayerNorm: simpler circuits
    )
    model = HookedTransformer(cfg)
    return model.to(device)


def train_step(model: HookedTransformer,
               X_batch: torch.Tensor,
               Y_batch: torch.Tensor,
               optimizer: torch.optim.Optimizer,
               loss_fn: nn.CrossEntropyLoss) -> float:
    """
    Single training step with teacher forcing across all answer digits.

    [FIX-1] Original code did `loss = 0` (Python int), making loss
    a non-tensor. Gradients never flowed. Fixed by initialising as
    torch.zeros(1) and using in-place addition on tensors.
    """
    model.train()
    optimizer.zero_grad()

    # [FIX-1] — initialise as tensor so .backward() works
    total_loss = torch.zeros(1, device=X_batch.device, requires_grad=False)

    for digit_pos in range(Y_batch.shape[1]):       # 0=hundreds, 1=tens, 2=ones
        if digit_pos == 0:
            inp = X_batch                           # [B, 6]
        else:
            # Teacher forcing: append correct answer digits so far
            inp = torch.cat([X_batch,
                             Y_batch[:, :digit_pos]], dim=1)  # [B, 6+digit_pos]

        logits     = model(inp)[:, -1, :]           # [B, vocab] — last position
        step_loss  = loss_fn(logits, Y_batch[:, digit_pos])
        total_loss = total_loss + step_loss         # tensor accumulation

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return total_loss.item()


@torch.no_grad()
def evaluate(model: HookedTransformer,
             loader: DataLoader) -> Dict[str, float]:
    """
    Returns per-digit accuracy and full-answer (exact-match) accuracy.
    Also breaks down accuracy by carry/no-carry cases.
    """
    model.eval()
    # Get model's device
    device = next(model.parameters()).device

    digit_correct  = [0, 0, 0]
    digit_total    = [0, 0, 0]
    exact_correct  = 0
    carry_correct  = 0;  carry_total  = 0
    nocarry_correct= 0;  nocarry_total= 0

    for X_batch, Y_batch in loader:
        # Move data to model's device
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        preds = []
        for digit_pos in range(Y_batch.shape[1]):
            if digit_pos == 0:
                inp = X_batch
            else:
                inp = torch.cat([X_batch, Y_batch[:, :digit_pos]], dim=1)
            pred = model(inp)[:, -1, :].argmax(-1)   # [B]
            preds.append(pred)
            digit_correct[digit_pos] += (pred == Y_batch[:, digit_pos]).sum().item()
            digit_total[digit_pos]   += len(pred)

        preds_tensor = torch.stack(preds, dim=1)     # [B, 3]
        exact        = (preds_tensor == Y_batch).all(dim=1)
        exact_correct += exact.sum().item()

        # Carry stratification: ones digits of A and B
        ones_a = X_batch[:, POS_A_ONES]
        ones_b = X_batch[:, POS_B_ONES]
        has_carry = (ones_a + ones_b) >= 10

        carry_correct   += exact[has_carry].sum().item()
        carry_total     += has_carry.sum().item()
        nocarry_correct += exact[~has_carry].sum().item()
        nocarry_total   += (~has_carry).sum().item()

    n = digit_total[0]
    return {
        "acc_hundreds": digit_correct[0] / digit_total[0],
        "acc_tens":     digit_correct[1] / digit_total[1],
        "acc_ones":     digit_correct[2] / digit_total[2],
        "exact_match":  exact_correct / n,
        "carry_exact":  carry_correct  / max(carry_total, 1),
        "nocarry_exact":nocarry_correct / max(nocarry_total, 1),
    }


def train_to_threshold(model: HookedTransformer,
                        train_data: TensorDataset,
                        test_data:  TensorDataset,
                        target_acc: float = 0.999,
                        max_epochs: int   = 2000,
                        batch_size: int   = 512,
                        lr:         float = 1e-3,
                        wd:         float = 1e-4,
                        save_path:  str   = "toy_addition_model_acd.pt") -> bool:
    """
    Trains until exact-match accuracy on the test set exceeds target_acc.
    Returns True if target was reached, False if max_epochs hit.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                  weight_decay=wd)
    loss_fn   = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=512, shuffle=False)

    best_acc = 0.0

    # Get model's device
    device = next(model.parameters()).device

    for epoch in range(max_epochs):
        # ── one training epoch ──
        epoch_loss = 0.0
        for X_batch, Y_batch in train_loader:
            # Move data to model's device
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            epoch_loss += train_step(model, X_batch, Y_batch,
                                     optimizer, loss_fn)

        # ── evaluate every 50 epochs ──
        if epoch % 50 == 0 or epoch == max_epochs - 1:
            metrics = evaluate(model, test_loader)
            acc     = metrics["exact_match"]
            print(f"Epoch {epoch:4d} | loss={epoch_loss/len(train_loader):.4f} | "
                  f"exact={acc:.4f} | "
                  f"carry={metrics['carry_exact']:.4f} | "
                  f"no-carry={metrics['nocarry_exact']:.4f}")

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), save_path)

            if acc >= target_acc:
                print(f"\n✓ Target accuracy {target_acc} reached at epoch {epoch}.")
                print(f"  Model saved to '{save_path}'")
                return True

    print(f"\n✗ Max epochs reached. Best accuracy: {best_acc:.4f}")
    return False


# ═════════════════════════════════════════════════════════════
# STAGE 2A: ACTIVATION PATCHING → FIND CLOCK HEADS
# ═════════════════════════════════════════════════════════════

def _encode_pair(a: int, b: int, device: torch.device = None) -> torch.Tensor:
    """Encodes a single (a, b) addition prompt as a [1, 6] token tensor."""
    if device is None:
        device = DEVICE
    return torch.tensor(
        [to_digits_msf(a, 2) + [TOK_PLUS] + to_digits_msf(b, 2) + [TOK_EQ]],
        dtype=torch.long,
        device=device
    )


@torch.no_grad()
def find_clock_heads(model:     HookedTransformer,
                     n_pairs:   int   = 300,
                     threshold: float = 0.03,
                     seed:      int   = 7) -> Tuple[List[Tuple[int,int]], torch.Tensor]:
    """
    Activation patching: change B by +1, measure which heads
    restore the correct hundreds/tens answer digit logit.

    Strategy: minimal perturbation (b → b+1) so TIE signal is sharp.

    Returns
    -------
    clock_heads : list of (layer, head) with TIE > threshold
    tie_matrix  : FloatTensor [n_layers, n_heads]
    """
    random.seed(seed)
    n_layers = model.cfg.n_layers
    n_heads  = model.cfg.n_heads
    device = next(model.parameters()).device

    tie_accumulator = torch.zeros(n_layers, n_heads)
    valid_pairs     = 0

    for _ in range(n_pairs):
        a = random.randint(10, 49)
        b = random.randint(10, 48)     # b+1 ≤ 99

        # Tokens
        clean_tokens = _encode_pair(a, b, device=device)
        corr_tokens  = _encode_pair(a, b + 1, device=device)

        # Answer token we want to recover = first non-zero digit of a+b
        correct_sum  = a + b
        # Use hundreds digit for cases where carry overflows to 3 digits
        # otherwise use tens digit — pick the most informative digit
        if correct_sum >= 100:
            answer_digit_pos = 0   # hundreds digit
            answer_token     = correct_sum // 100
        else:
            answer_digit_pos = 1   # tens digit
            answer_token     = (correct_sum // 10) % 10

        # Get clean cache
        _, clean_cache = model.run_with_cache(clean_tokens)

        # Baseline: corrupted model (no patching) probability of correct answer
        base_logits  = model(corr_tokens)
        base_prob    = torch.softmax(base_logits[0, -1, :], dim=-1)[answer_token].item()

        for layer in range(n_layers):
            for head in range(n_heads):
                # Extract clean head output: [1, seq, n_heads, d_head]
                clean_result = clean_cache[
                    f"blocks.{layer}.attn.hook_z"
                ][0, :, head, :].clone()                # [seq, d_head]

                def make_patch_hook(c_result, h_idx):
                    def hook(value, hook=None):
                        value[0, :, h_idx, :] = c_result
                        return value
                    return hook

                patched_logits = model.run_with_hooks(
                    corr_tokens,
                    fwd_hooks=[(
                        f"blocks.{layer}.attn.hook_z",
                        make_patch_hook(clean_result, head)
                    )]
                )
                patched_prob = torch.softmax(
                    patched_logits[0, -1, :], dim=-1
                )[answer_token].item()

                tie_accumulator[layer, head] += (patched_prob - base_prob)

        valid_pairs += 1

    tie_matrix  = tie_accumulator / valid_pairs
    clock_heads = [
        (l, h)
        for l in range(n_layers)
        for h in range(n_heads)
        if tie_matrix[l, h] > threshold
    ]

    # ── Report ──
    print("\n" + "═"*55)
    print("STAGE 2A: Activation Patching Results")
    print("═"*55)
    print(f"TIE matrix (averaged over {valid_pairs} prompt pairs):")
    for l in range(n_layers):
        row = "  Layer {:d}: ".format(l)
        for h in range(n_heads):
            marker = "* " if tie_matrix[l, h] > threshold else "  "
            row += f"{marker}H{h}={tie_matrix[l,h]:+.3f}  "
        print(row)
    print(f"\nClock heads (TIE > {threshold}): {clock_heads}")

    if not clock_heads:
        print("\n⚠  No clock heads found above threshold.")
        print("   → Arithmetic may be MLP-dominated. Proceed directly to Stage 3.")

    return clock_heads, tie_matrix


# ═════════════════════════════════════════════════════════════
# STAGE 2B: HELIX FIT (Approach 6 — hypothesis-driven)
# ═════════════════════════════════════════════════════════════

def collect_digit_activations(model:           HookedTransformer,
                               layer:          int,
                               digit_position: int  = POS_A_ONES,
                               fixed_b:        int  = 5,
                               digit_range:    range = range(10)
                               ) -> Tuple[torch.Tensor, List[int]]:
    """
    Collects residual stream activations at `digit_position` for
    digit values 0..9 (the ones or tens digit of A).

    [FIX-2] For digit-wise tokenisation, token positions are
    structurally fixed — no runtime index-tracking needed.

    Parameters
    ----------
    digit_position : POS_A_ONES (1) or POS_A_TENS (0)
    fixed_b        : the B operand, held constant across all prompts

    Returns
    -------
    acts_tensor : FloatTensor [N_valid, d_model]
    valid_ds    : list of digit values corresponding to rows
    """
    acts     = []
    valid_ds = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for d in digit_range:
            # Build prompt with digit d at the requested position
            if digit_position == POS_A_ONES:
                a = 10 + d    # tens=1, ones=d  (e.g., d=3 → a=13)
            elif digit_position == POS_A_TENS:
                a = d * 10 + 3  # tens=d, ones=3
            else:
                raise ValueError(f"Unexpected digit_position={digit_position}")

            inp = _encode_pair(a, fixed_b, device=device)
            _, cache = model.run_with_cache(inp)

            resid_pre = cache[f"blocks.{layer}.hook_resid_pre"]  # [1, seq, d]
            acts.append(resid_pre[0, digit_position, :].cpu())
            valid_ds.append(d)

    return torch.stack(acts), valid_ds          # [N, d_model], list[int]


def fit_helix(acts_matrix: np.ndarray,
              valid_ds:    List[int],
              periods:     List[float] = [2.0, 5.0, 10.0]
              ) -> Tuple[float, np.ndarray]:
    """
    Fits a helix model  acts ≈ H @ A  where H contains
    cos/sin features at each period, plus a bias term.

    Returns R-squared and the fitted coefficient matrix A.
    """
    def build_H(ds, periods):
        rows = []
        for d in ds:
            row = []
            for T in periods:
                row.append(math.cos(2 * math.pi * d / T))
                row.append(math.sin(2 * math.pi * d / T))
            row.append(1.0)     # bias
            rows.append(row)
        return np.array(rows)   # [N, 2*n_periods + 1]

    H   = build_H(valid_ds, periods)           # [N, 7]
    A, _, _, _ = np.linalg.lstsq(H, acts_matrix, rcond=None)

    predicted = H @ A
    ss_res    = np.sum((acts_matrix - predicted) ** 2)
    ss_tot    = np.sum((acts_matrix - acts_matrix.mean(0)) ** 2)
    r_sq      = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return r_sq, A


def stage2b_helix_fit(model:       HookedTransformer,
                       clock_heads: List[Tuple[int,int]],
                       layer_range: Optional[range] = None,
                       check_all_layers: bool = False
                      ) -> Dict[Tuple[int,int], float]:
    """
    Runs the helix fit for all clock heads (and optionally extra layers).
    Reports R² per head. Returns dict {(layer,head): r_squared}.

    If check_all_layers=True, checks residual stream at all layers regardless
    of clock_heads (to detect helix representations even when attention doesn't use them).
    """
    if layer_range is None:
        layer_range = range(model.cfg.n_layers)

    print("\n" + "═"*55)
    print("STAGE 2B: Helix Fit (R² over digit representations)")
    print("═"*55)

    if check_all_layers:
        print("  Checking ALL layers for helix (independent of attention usage)")

    results = {}
    for layer in layer_range:
        # Note: collect_digit_activations gets residual stream at layer input
        # This is the same for all heads, so we only need to compute once per layer
        acts_tensor, valid_ds = collect_digit_activations(
            model, layer, digit_position=POS_A_ONES
        )
        acts_np = acts_tensor.numpy()
        r_sq, _ = fit_helix(acts_np, valid_ds, periods=[2.0, 5.0, 10.0])

        label = ("✓ Strong helix" if r_sq > 0.70
                 else "~ Moderate"  if r_sq > 0.40
                 else "✗ Weak/none")
        print(f"  Layer {layer} (residual stream): R² = {r_sq:.4f}  {label}")

        # Store result for each head at this layer (though it's the same value)
        # This maintains compatibility with downstream code expecting (layer, head) keys
        if check_all_layers:
            for head in range(model.cfg.n_heads):
                results[(layer, head)] = r_sq
        else:
            for head in range(model.cfg.n_heads):
                if (layer, head) in clock_heads:
                    results[(layer, head)] = r_sq

    strong = [(lh, r2) for lh, r2 in results.items() if r2 > 0.70]
    if strong:
        print(f"\n  ✓ Helix found! Best: Layer {strong[0][0]}, R²={strong[0][1]:.4f}")
    else:
        print("\n  ⚠  No layer with R² > 0.70 found.")
        print("     → Digits may not form helix in residual stream.")
        print("     → MLP may use different representation.")

    return results


# ═════════════════════════════════════════════════════════════
# STAGE 2C: GEOMETRIC CIRCLE TEST (professor's diagnostic, corrected)
# ═════════════════════════════════════════════════════════════

def get_OV_circuit(model: HookedTransformer,
                   layer: int,
                   head:  int) -> torch.Tensor:
    """
    [FIX-9] W_OV = W_V @ W_O  (TransformerLens row-vector convention)

    TransformerLens stores:
      W_V : [d_model, d_head]   — projects residual stream → value vectors
      W_O : [d_head, d_model]   — projects value vectors → residual stream

    Input  x : [1, d_model]
    x → x @ W_V → x @ W_V @ W_O = x @ W_OV
    Therefore W_OV = W_V @ W_O   (NOT W_O @ W_V)
    """
    W_V  = model.W_V[layer, head]    # [d_model, d_head]
    W_O  = model.W_O[layer, head]    # [d_head,  d_model]
    return W_V @ W_O                 # [d_model, d_model]


def _safe_angle_linearity(angles: np.ndarray,
                           valid_ds: List[int]) -> float:
    """
    [FIX-4] np.unwrap fails when consecutive angle differences exceed π,
    which happens whenever the helix period T < ~4 steps.

    For digit range 0-9 with period T=10, Δθ = 2π/10 = 0.63 rad — safe.
    For period T=2, Δθ = π — boundary case, unreliable.

    We check mean |Δθ| before unwrapping and return 0.0 if unsafe.
    """
    mean_delta = np.abs(np.diff(angles)).mean()
    if mean_delta > np.pi * 0.75:
        return 0.0          # period too small for reliable unwrap

    unwrapped  = np.unwrap(angles)
    linearity  = np.corrcoef(valid_ds, unwrapped)[0, 1]
    return float(linearity)


def stage2c_geometric_test(model:       HookedTransformer,
                            clock_heads: List[Tuple[int,int]],
                            top_k:       int   = 10,
                            cv_thresh:   float = 0.25,
                            lin_thresh:  float = 0.85,
                           ) -> Dict[Tuple[int,int], List[Tuple]]:
    """
    Tests all C(top_k, 2) = 45 pairs of top-k OV singular vectors
    for helix (circle) structure.

    [FIX-8] Reports full distribution statistics alongside any BULLSEYE
    so that borderline hits are not over-interpreted.

    Returns dict mapping (layer, head) → list of found planes
    each plane is (k1, k2, radius_cv, angle_linearity).
    """
    print("\n" + "═"*55)
    print("STAGE 2C: Geometric Circle Test on OV Singular Vectors")
    print("═"*55)

    all_found = {}

    for layer, head in clock_heads:
        W_OV       = get_OV_circuit(model, layer, head)
        U, S, Vt   = torch.linalg.svd(W_OV)

        # Collect ones-digit activations (digit values 0-9)
        acts_tensor, valid_ds = collect_digit_activations(
            model, layer, digit_position=POS_A_ONES
        )

        results = []
        for k1, k2 in itertools.combinations(range(top_k), 2):
            v1 = Vt[k1].cpu()
            v2 = Vt[k2].cpu()

            coords = torch.stack([
                acts_tensor @ v1,
                acts_tensor @ v2
            ], dim=1).float()               # [N, 2]

            radii  = coords.norm(dim=1)
            angles = torch.atan2(coords[:, 1], coords[:, 0]).numpy()

            radius_mean = radii.mean().item()
            if radius_mean < 1e-8:
                continue                    # degenerate — skip

            radius_cv  = (radii.std() / radius_mean).item()
            angle_lin  = _safe_angle_linearity(angles, valid_ds)

            results.append((k1, k2, radius_cv, angle_lin))

        # Sort: low CV + high |linearity| first
        results.sort(key=lambda x: x[2] - abs(x[3]))

        found = [(k1, k2, cv, lin)
                 for k1, k2, cv, lin in results
                 if cv < cv_thresh and abs(lin) > lin_thresh]

        all_cvs  = [r[2]       for r in results]
        all_lins = [abs(r[3])  for r in results]

        print(f"\n  Layer {layer}, Head {head}:")
        print(f"    Singular values (top 5): "
              f"{S[:5].cpu().numpy().round(3)}")
        print(f"    CV   across all {len(results)} pairs: "
              f"mean={np.mean(all_cvs):.3f}, min={np.min(all_cvs):.3f}")
        print(f"    |Lin| across all {len(results)} pairs: "
              f"mean={np.mean(all_lins):.3f}, max={np.max(all_lins):.3f}")

        if found:
            print(f"    ★ BULLSEYE — {len(found)} helix plane(s) found:")
            for k1, k2, cv, lin in found[:3]:
                print(f"      Directions ({k1},{k2}): "
                      f"CV={cv:.3f}, Lin={lin:.3f}")
            all_found[(layer, head)] = found
        else:
            print(f"    No plane passed thresholds "
                  f"(CV<{cv_thresh}, |Lin|>{lin_thresh}).")
            all_found[(layer, head)] = []

    return all_found


# ═════════════════════════════════════════════════════════════
# STAGE 2D-i: OUTPUT SIDE (U) HELIX VALIDATION
# ═════════════════════════════════════════════════════════════

def stage2d_output_helix(model:     HookedTransformer,
                          layer:    int,
                          head:     int,
                          k1:       int,
                          k2:       int,
                          fixed_a:  int   = 13,
                          b_range:  range = range(5, 85)
                         ) -> Tuple[float, float]:
    """
    [FIX-6] Tests whether the OUTPUT (U) side of the OV circuit also
    encodes the SUM value helically at the answer token position.

    This distinguishes:
      "n is stored helically in the residual stream"  (V-side test, 2C)
      "addition is COMPUTED as a rotation"            (U-side test, this)

    If V-side passes but U-side fails:
      → Helix is a representation artifact, not the compute mechanism.
    If both pass:
      → Strong evidence that OV rotation implements addition.
    """
    W_OV     = get_OV_circuit(model, layer, head)
    U, S, Vt = torch.linalg.svd(W_OV)

    u1 = U[:, k1].cpu()
    u2 = U[:, k2].cpu()

    sum_coords = []
    valid_sums = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for b in b_range:
            target_sum = fixed_a + b
            if target_sum > 198:
                continue
            inp = _encode_pair(fixed_a, b, device=device)
            _, cache = model.run_with_cache(inp)

            # Residual stream AFTER this layer at the '=' position (pos 5).
            # This is the last question token — where the model assembles
            # the answer representation before autoregressive generation.
            # NOT an "answer token" (those are generated later).
            resid_post = cache[f"blocks.{layer}.hook_resid_post"]
            ans_act    = resid_post[0, POS_EQ, :].cpu()   # '=' position

            sum_coords.append(torch.stack([ans_act @ u1,
                                           ans_act @ u2]))
            valid_sums.append(target_sum)

    coords = torch.stack(sum_coords).float()    # [N, 2]
    radii  = coords.norm(dim=1)
    angles = torch.atan2(coords[:, 1], coords[:, 0]).numpy()

    radius_mean = radii.mean().item()
    radius_cv   = (radii.std() / max(radius_mean, 1e-8)).item()

    # [BUG-2 FIX] Correlate against ONES digit of sum, not the full sum.
    # The period-10 helix can only encode (sum % 10). Correlating angles
    # against full sum values (18..98) would give artificially high linearity
    # because np.unwrap unrolls a multi-cycle signal and the correlation with
    # a monotonically increasing x-axis is trivially high. We test the
    # scientifically meaningful claim: angle ∝ (sum mod 10).
    ones_of_sums = [s % 10 for s in valid_sums]
    angle_lin    = _safe_angle_linearity(angles, ones_of_sums)

    print(f"\n  [Output/U-side] Layer {layer}, Head {head}, "
          f"Plane ({k1},{k2}):")
    print(f"    Radius CV            : {radius_cv:.3f}   (want < 0.25)")
    print(f"    Angle Lin (vs sum%10): {angle_lin:.3f}   (want |·| > 0.85)")

    if radius_cv < 0.25 and abs(angle_lin) > 0.85:
        print("    ✓ Output side is helical w.r.t. SUM value.")
        print("    → Evidence: this head COMPUTES addition via rotation.")
    else:
        print("    ✗ Output side is NOT helix-structured w.r.t. SUM.")
        print("    → Head reads helically but may not compute addition here.")
        print("    → Proceed to Stage 3 (carry/MLP circuit).")

    return radius_cv, angle_lin


# ═════════════════════════════════════════════════════════════
# STAGE 2D-ii: CAUSAL PHASE-SHIFT INTERVENTION
# ═════════════════════════════════════════════════════════════

def _rotate_in_plane(h:     torch.Tensor,
                     v1:    torch.Tensor,
                     v2:    torch.Tensor,
                     theta: float) -> torch.Tensor:
    """
    Rotates vector h by angle theta in the 2D plane spanned by (v1, v2).

    h_rotated = h
              - (h·v1)v1 - (h·v2)v2          # remove original component
              + (cos θ·(h·v1) - sin θ·(h·v2))v1   # rotated component
              + (sin θ·(h·v1) + cos θ·(h·v2))v2
    """
    c1 = (h @ v1).item()
    c2 = (h @ v2).item()
    ct, st = math.cos(theta), math.sin(theta)

    return (h
            - c1 * v1 - c2 * v2
            + (ct * c1 - st * c2) * v1
            + (st * c1 + ct * c2) * v2)


def stage2d_causal_test(model:      HookedTransformer,
                         layer:     int,
                         head:      int,
                         Vt:        torch.Tensor,
                         k1:        int,
                         k2:        int,
                         n_tests:   int = 30,
                         seed:      int = 99
                        ) -> Dict[str, float]:
    """
    Causal phase-shift intervention: rotate the ones-digit of A by +delta
    in the discovered helix plane, then check if the model's predicted
    ones digit of the answer shifts by +delta (mod 10).

    [FIX-3] Uses attn.hook_v (head-specific value vectors) NOT
    hook_resid_pre (which would affect ALL heads at this layer).

    [FIX-7] Stratifies results by carry vs. no-carry cases.
    Carry failures are EXPECTED and scientifically informative —
    they motivate Stage 3's carry circuit analysis.

    The ones-digit helix has period T=10, so rotating by +delta
    corresponds to angle θ = 2π·delta/10.
    """
    random.seed(seed)
    device = next(model.parameters()).device
    # Keep v1, v2 on device for efficient operations
    v1 = Vt[k1].to(device)
    v2 = Vt[k2].to(device)

    # Build test cases: (a, b, delta) where delta ∈ {1,2,3}
    no_carry_cases, carry_cases = [], []

    for _ in range(n_tests * 4):
        if len(no_carry_cases) >= n_tests and len(carry_cases) >= n_tests:
            break
        a     = random.randint(11, 49)
        b     = random.randint(11, 49)
        delta = random.randint(1, 3)

        # Ensure rotation doesn't make ones digit go negative
        ones_a = a % 10
        if ones_a + delta > 9:
            continue    # would need carry handling in A itself; skip

        ones_sum = ones_a + (b % 10)
        case = (a, b, delta)

        if ones_sum >= 10 and len(carry_cases) < n_tests:
            carry_cases.append(case)
        elif ones_sum < 10 and len(no_carry_cases) < n_tests:
            no_carry_cases.append(case)

    print(f"\n  [Causal Phase-Shift] Layer {layer}, Head {head}, "
          f"Plane ({k1},{k2}), T=10:")

    summary = {}
    for case_type, cases in [("no_carry", no_carry_cases),
                              ("carry",    carry_cases)]:
        successes = 0
        details   = []

        for a, b, delta in cases:
            theta = 2.0 * math.pi * delta / 10.0

            inp = _encode_pair(a, b, device=device)

            with torch.no_grad():
                _, cache = model.run_with_cache(inp)

            # Get residual stream BEFORE this layer at ones digit of A
            # Keep on device for efficient operations
            resid_pre = cache[f"blocks.{layer}.hook_resid_pre"]
            h_orig    = resid_pre[0, POS_A_ONES, :]    # [d_model]

            h_rotated = _rotate_in_plane(h_orig, v1, v2, theta)

            # [FIX-3] Hook targets only this head's VALUE vectors,
            # NOT the full residual stream.
            # attn.hook_v shape: [batch, seq, n_heads, d_head]
            W_V_head = model.W_V[layer, head]    # [d_model, d_head] - keep on device

            def make_value_hook(h_rot, h_idx, pos, W_V):
                def hook(val, hook_obj):
                    # Project rotated residual into this head's value space
                    # All tensors already on same device - no transfer needed
                    new_v = h_rot @ W_V
                    val[0, pos, h_idx, :] = new_v
                    return val
                return hook

            # [BUG-1 FIX] Must predict autoregressively to reach the ones digit.
            # Feeding only the question (length 6) predicts the HUNDREDS digit,
            # not the ones digit. We need three forward passes:
            #   pass 1: question           → predict hundreds
            #   pass 2: question+hundreds  → predict tens
            #   pass 3: question+h+t       → predict ones  ← what we check

            hook_spec = (f"blocks.{layer}.attn.hook_v",
                         make_value_hook(h_rotated, head,
                                         POS_A_ONES, W_V_head))

            with torch.no_grad():
                # Pass 1: predict hundreds digit
                logits_h      = model.run_with_hooks(inp, fwd_hooks=[hook_spec])
                pred_hundreds = logits_h[0, -1, :].argmax().item()

                # Pass 2: predict tens digit
                inp_t    = torch.cat([inp,
                                      torch.tensor([[pred_hundreds]], device=device)], dim=1)
                logits_t = model.run_with_hooks(inp_t, fwd_hooks=[hook_spec])
                pred_tens = logits_t[0, -1, :].argmax().item()

                # Pass 3: predict ones digit  ← the digit we actually care about
                inp_o    = torch.cat([inp,
                                      torch.tensor([[pred_hundreds,
                                                     pred_tens]], device=device)], dim=1)
                logits_o = model.run_with_hooks(inp_o, fwd_hooks=[hook_spec])

            predicted_ones = logits_o[0, -1, :].argmax().item()
            expected_ones  = (a + b + delta) % 10

            success = (predicted_ones == expected_ones)
            successes += int(success)
            details.append(
                f"    {a}+{b} rotate+{delta}: "
                f"pred={predicted_ones} exp={expected_ones} "
                f"{'✓' if success else '✗'}"
            )

        rate = successes / max(len(cases), 1)
        summary[case_type] = rate

        print(f"\n  ── {case_type.upper()} ({len(cases)} cases) ──")
        for d in details[:6]:       # show first 6 for brevity
            print(d)
        if len(details) > 6:
            print(f"    ... ({len(details)-6} more)")
        print(f"  Success rate: {successes}/{len(cases)} = {rate:.1%}")

        if case_type == "carry":
            if rate < 0.40:
                print("  ✓ Expected: helix handles linear part only.")
                print("    Carry failures confirm Stage 3 is needed.")
            else:
                print("  ⚠ Unexpectedly high carry success — "
                      "verify test cases actually have carry.")

    return summary


# ═════════════════════════════════════════════════════════════
# STAGE 3: CARRY CIRCUIT via MLP JACOBIAN-SVD
# ═════════════════════════════════════════════════════════════

def _compute_mlp_jacobian(model: HookedTransformer,
                           layer: int,
                           inp:   torch.Tensor) -> torch.Tensor:
    """
    Computes the Jacobian of MLP_{layer} output w.r.t. its input
    at the final token position.

    J[i, j] = d(mlp_out[i]) / d(mlp_in[j])

    Returns J : FloatTensor [d_model, d_model]
    """
    mlp = model.blocks[layer].mlp

    _, cache  = model.run_with_cache(inp)

    # Try multiple possible MLP input hook names for compatibility
    # NOTE: We need the residual stream BEFORE the MLP (shape: d_model),
    #       NOT the MLP's internal hidden state (shape: d_mlp)
    mlp_hook_names = [
        f"blocks.{layer}.hook_mlp_in",       # Ideal hook name (if it exists)
        f"blocks.{layer}.hook_resid_mid",    # Residual stream after attention, before MLP
    ]

    mlp_in_np = None
    for hook_name in mlp_hook_names:
        if hook_name in cache:
            mlp_in_np = cache[hook_name][0, -1, :]
            # Validate shape: should be d_model, not d_mlp
            if mlp_in_np.shape[0] == model.cfg.d_model:
                break
            else:
                # Wrong hook - this is probably the MLP hidden layer
                mlp_in_np = None

    if mlp_in_np is None:
        # Diagnostic: show available hooks with shapes
        available_hooks = [(k, cache[k].shape) for k in cache.keys()
                          if f'blocks.{layer}' in k and ('mlp' in k.lower() or 'resid' in k.lower())]
        raise KeyError(
            f"Could not find MLP input hook for layer {layer}.\n"
            f"Tried: {mlp_hook_names}\n"
            f"Need shape [batch, seq, {model.cfg.d_model}] (d_model), NOT [{model.cfg.d_mlp}] (d_mlp).\n"
            f"Available hooks: {available_hooks}"
        )

    # Detach and re-attach requires_grad
    mlp_in = mlp_in_np.detach().clone().float().requires_grad_(True)

    # Forward pass through MLP only (unsqueeze for batch+seq dims)
    mlp_out = mlp(mlp_in.unsqueeze(0).unsqueeze(0)).squeeze()  # [d_model]

    d_model = mlp_in.shape[0]
    J       = torch.zeros(d_model, d_model, device=mlp_in.device)

    for i in range(d_model):
        grad = torch.autograd.grad(
            mlp_out[i], mlp_in,
            retain_graph=True,
            create_graph=False
        )[0]
        J[i] = grad.detach()

    return J    # [d_model, d_model]


def stage3_carry_circuit(model:       HookedTransformer,
                          layer:      int,
                          n_per_class: int = 40,
                          seed:       int  = 42
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    [FIX-10] Computes the DIFFERENCE of average MLP Jacobians:
      J_diff = E[J | carry] - E[J | no-carry]

    SVD of J_diff reveals directions the MLP uses SPECIFICALLY
    for carry-triggering inputs — the carry circuit.

    Returns U, S, Vt of J_diff.
    """
    print("\n" + "═"*55)
    print("STAGE 3: Carry Circuit — MLP Jacobian-SVD")
    print("═"*55)
    print(f"  Computing Jacobians for {n_per_class} carry + "
          f"{n_per_class} no-carry inputs at Layer {layer}...")

    random.seed(seed)
    device = next(model.parameters()).device

    carry_Js    = []
    no_carry_Js = []

    # Build pool of (a, b) pairs
    pool = [(a, b) for a in range(10, 99) for b in range(10, 99)]
    random.shuffle(pool)

    for a, b in pool:
        if (len(carry_Js) >= n_per_class and
                len(no_carry_Js) >= n_per_class):
            break

        has_carry = ((a % 10) + (b % 10)) >= 10

        if has_carry and len(carry_Js) < n_per_class:
            inp = _encode_pair(a, b, device=device)
            try:
                J = _compute_mlp_jacobian(model, layer, inp)
                carry_Js.append(J)
            except Exception as e:
                print(f"    Warning: Jacobian failed for ({a},{b}): {e}")

        elif not has_carry and len(no_carry_Js) < n_per_class:
            inp = _encode_pair(a, b, device=device)
            try:
                J = _compute_mlp_jacobian(model, layer, inp)
                no_carry_Js.append(J)
            except Exception as e:
                print(f"    Warning: Jacobian failed for ({a},{b}): {e}")

    print(f"  Collected: {len(carry_Js)} carry, {len(no_carry_Js)} no-carry Jacobians")

    if len(carry_Js) < 5 or len(no_carry_Js) < 5:
        raise RuntimeError("Too few valid Jacobians. "
                           "Check model and input encoding.")

    J_carry    = torch.stack(carry_Js).mean(0)       # [d_model, d_model]
    J_no_carry = torch.stack(no_carry_Js).mean(0)

    # Difference: what the MLP does DIFFERENTLY for carry inputs
    J_diff = J_carry - J_no_carry

    U, S, Vt = torch.linalg.svd(J_diff)

    print(f"\n  Top-10 singular values of J_diff:")
    print(f"  {S[:10].cpu().numpy().round(4)}")

    # ── Probe: do discovered directions separate carry from no-carry? ──
    print(f"\n  Carry-separability of top Jacobian directions:")
    print(f"  (|separation| > 1.5σ indicates a genuine carry direction)")

    separation_scores = []

    for k in range(min(5, len(S))):
        v = Vt[k]  # Keep on same device as model for projection

        carry_proj    = []
        no_carry_proj = []

        # Re-run a small probe set
        probe_pool = random.sample(pool, min(100, len(pool)))

        with torch.no_grad():
            for a, b in probe_pool:
                inp = _encode_pair(a, b, device=device)
                _, cache = model.run_with_cache(inp)

                # Try multiple possible MLP input hook names for compatibility
                # NOTE: We need d_model [128], not d_mlp [512]
                mlp_in = None
                for hook_name in [f"blocks.{layer}.hook_mlp_in",
                                  f"blocks.{layer}.hook_resid_mid"]:
                    if hook_name in cache:
                        candidate = cache[hook_name][0, -1, :]  # Keep on device (MPS/CUDA/CPU)
                        # Validate shape: should be d_model, not d_mlp
                        if candidate.shape[0] == model.cfg.d_model:
                            mlp_in = candidate
                            break

                if mlp_in is None:
                    raise KeyError(f"Could not find MLP input hook for layer {layer} with shape [{model.cfg.d_model}]")

                score  = (mlp_in @ v).item()

                if ((a % 10) + (b % 10)) >= 10:
                    carry_proj.append(score)
                else:
                    no_carry_proj.append(score)

        if len(carry_proj) < 3 or len(no_carry_proj) < 3:
            continue

        all_scores = carry_proj + no_carry_proj
        pooled_std = np.std(all_scores) + 1e-8
        separation = (np.mean(carry_proj) - np.mean(no_carry_proj)) / pooled_std

        separation_scores.append((k, separation, carry_proj, no_carry_proj))
        marker = "★" if abs(separation) > 1.5 else " "
        print(f"  {marker} Direction {k}: separation = {separation:+.2f} σ  "
              f"(carry mean={np.mean(carry_proj):.3f}, "
              f"no-carry mean={np.mean(no_carry_proj):.3f})")

        # Emit separation histogram for every direction
        plot_carry_separation(carry_proj, no_carry_proj,
                              direction_k=k, layer=layer)

    strong_dirs = [(k, s) for k, s, _, _ in separation_scores
                   if abs(s) > 1.5]
    if strong_dirs:
        print(f"\n  ✓ Carry circuit found: directions {[k for k,_ in strong_dirs]}")
        print(f"    These directions in MLP_{layer} distinguish carry vs. no-carry.")
    else:
        print(f"\n  ⚠ No strongly separating direction found (threshold 1.5σ).")
        print(f"    Try Layer {layer+1} or increase n_per_class.")

    return U, S, Vt


# ═════════════════════════════════════════════════════════════
# VISUALIZATION HELPERS
# ═════════════════════════════════════════════════════════════

def plot_tie_heatmap(tie_matrix: torch.Tensor,
                     save_path: str = "tie_heatmap.png"):
    """
    Heatmap of TIE scores from activation patching.
    Clock heads visually stand out as bright green cells.
    """
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(max(4, tie_matrix.shape[1]), 3))
        im = ax.imshow(tie_matrix.numpy(), cmap='RdYlGn',
                       aspect='auto', vmin=-0.05, vmax=0.15)
        ax.set_xlabel("Head Index")
        ax.set_ylabel("Layer")
        ax.set_title("Activation Patching — TIE Scores\n"
                     "(bright green = clock head)")
        ax.set_xticks(range(tie_matrix.shape[1]))
        ax.set_yticks(range(tie_matrix.shape[0]))
        for l in range(tie_matrix.shape[0]):
            for h in range(tie_matrix.shape[1]):
                ax.text(h, l, f"{tie_matrix[l,h]:.2f}",
                        ha='center', va='center', fontsize=8)
        plt.colorbar(im, ax=ax, label='TIE')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Plot saved: {save_path}")
    except Exception as e:
        print(f"  [plot_tie_heatmap] Skipped: {e}")


def plot_helix_circle(coords:   torch.Tensor,
                      valid_ds: List[int],
                      title:    str = "Helix_Plane",
                      save_path: Optional[str] = None):
    """
    Plots the 2D projection onto a singular-vector pair.
    If the helix is present, this produces a circle with digits
    arranged in order around the circumference.

    This is the single most important visual in the whole pipeline:
    a clean circle = helix confirmed; noise = helix absent.
    """
    try:
        import matplotlib.pyplot as plt
        x = coords[:, 0].numpy()
        y = coords[:, 1].numpy()

        fig, ax = plt.subplots(figsize=(6, 6))
        scatter = ax.scatter(x, y, c=valid_ds, cmap='hsv',
                             s=100, zorder=3)
        for i, d in enumerate(valid_ds):
            ax.annotate(str(d), (x[i], y[i]),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=9)

        # Draw unit-radius reference circle
        theta_ref = np.linspace(0, 2 * np.pi, 200)
        r_mean    = np.sqrt(x**2 + y**2).mean()
        ax.plot(r_mean * np.cos(theta_ref),
                r_mean * np.sin(theta_ref),
                'k--', alpha=0.3, linewidth=1)

        ax.set_aspect('equal')
        ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
        ax.axvline(0, color='k', linewidth=0.5, alpha=0.3)
        ax.set_title(title)
        plt.colorbar(scatter, ax=ax, label='Digit value')
        plt.tight_layout()

        out = save_path or f"{title.replace(' ', '_')}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Plot saved: {out}")
    except Exception as e:
        print(f"  [plot_helix_circle] Skipped: {e}")


def plot_r2_bars(r2_results: Dict[Tuple[int,int], float],
                 save_path:  str = "helix_r2.png"):
    """Bar chart of helix R² per (layer, head)."""
    try:
        import matplotlib.pyplot as plt
        labels = [f"L{l}H{h}" for l, h in r2_results]
        values = list(r2_results.values())
        colors = ['green' if v > 0.7 else 'orange' if v > 0.4
                  else 'red' for v in values]

        fig, ax = plt.subplots(figsize=(max(5, len(labels)), 4))
        ax.bar(labels, values, color=colors)
        ax.axhline(0.7, color='green', linestyle='--',
                   linewidth=1, label='Strong threshold (0.7)')
        ax.axhline(0.4, color='orange', linestyle='--',
                   linewidth=1, label='Moderate threshold (0.4)')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Helix R²")
        ax.set_title("Helix Fit Quality per Clock Head")
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Plot saved: {save_path}")
    except Exception as e:
        print(f"  [plot_r2_bars] Skipped: {e}")


def plot_carry_separation(carry_scores:    List[float],
                          no_carry_scores: List[float],
                          direction_k:     int,
                          layer:           int,
                          save_path:       Optional[str] = None):
    """
    Histogram showing separation of carry vs. no-carry projections
    onto the k-th Jacobian singular direction.
    Clear separation = this direction is the carry circuit.
    """
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        bins = np.linspace(
            min(carry_scores + no_carry_scores),
            max(carry_scores + no_carry_scores),
            30
        )
        ax.hist(carry_scores,    bins=bins, alpha=0.6,
                color='red',  label='Carry inputs')
        ax.hist(no_carry_scores, bins=bins, alpha=0.6,
                color='blue', label='No-carry inputs')
        ax.set_xlabel("Projection onto Jacobian direction")
        ax.set_ylabel("Count")
        ax.set_title(f"Layer {layer} MLP — Jacobian direction {direction_k}\n"
                     "Separation = carry circuit signature")
        ax.legend()
        plt.tight_layout()
        out = save_path or f"carry_separation_L{layer}_dir{direction_k}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Plot saved: {out}")
    except Exception as e:
        print(f"  [plot_carry_separation] Skipped: {e}")


def fft_sanity_check(acts_tensor: torch.Tensor,
                     Vt:          torch.Tensor,
                     valid_ds:    List[int],
                     layer:       int,
                     head:        int,
                     top_k:       int = 6):
    """
    FFT sanity check on top-k singular direction projections.
    Complements the helix R² fit by showing which periods are active.
    Looks for spikes at T = 2, 5, 10 (expected for digit-wise addition).

    Deliberately omitted from the main R² test but retained here as an
    optional diagnostic — call it after stage2b for extra evidence.
    """
    print(f"\n  [FFT Sanity Check] Layer {layer}, Head {head}:")
    found_any = False

    for k in range(min(top_k, Vt.shape[0])):
        v      = Vt[k].cpu()
        signal = (acts_tensor @ v).numpy()
        signal = signal - signal.mean()          # remove DC

        N        = len(signal)
        fft_vals = np.abs(np.fft.fft(signal))
        freqs    = np.fft.fftfreq(N, d=1)

        pos_mask = freqs[1:N//2] > 0
        pos_freq = freqs[1:N//2][pos_mask]
        pos_fft  = fft_vals[1:N//2][pos_mask]
        total_power = pos_fft.sum() + 1e-8

        hits = []
        for target_T in [2, 5, 10]:
            target_f  = 1.0 / target_T
            nearest   = np.argmin(np.abs(pos_freq - target_f))
            pct       = pos_fft[nearest] / total_power
            if pct > 0.12:
                hits.append(f"T={target_T}({pct:.0%})")

        if hits:
            print(f"    Dir {k}: spikes at {', '.join(hits)}")
            found_any = True

    if not found_any:
        print("    No dominant arithmetic frequency spikes found "
              "(T∈{2,5,10}).")




def run_experiment(model_path: Optional[str] = None,
                   skip_training: bool = False):
    """
    Runs the complete three-stage pipeline.

    Parameters
    ----------
    model_path    : Path to a pre-trained model checkpoint.
                    If None, trains from scratch.
    skip_training : If True and model_path is given, skips training.
    """

    print("=" * 60)
    print("ARITHMETIC CIRCUIT DISCOVERY: OPTION B")
    print("Standard Digit-Wise Integer Addition (with Carry)")
    print("=" * 60)

    # ──────────────────────────────────────────
    # STAGE 1: Train the model
    # ──────────────────────────────────────────
    print("\n" + "═"*55)
    print("STAGE 1: Data Generation and Model Training")
    print("═"*55)

    X, Y = generate_addition_data(n_samples=60000, seed=42)
    split       = int(0.8 * len(X))
    train_data  = TensorDataset(X[:split], Y[:split])
    test_data   = TensorDataset(X[split:], Y[split:])
    print(f"  Train: {len(train_data)} samples | Test: {len(test_data)} samples")

    model = build_model(seed=0, device=DEVICE)

    if skip_training and model_path is not None:
        print(f"  Loading pre-trained model from '{model_path}'")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model = model.to(DEVICE)
    else:
        save_path = model_path or "toy_addition_model.pt"
        success   = train_to_threshold(
            model, train_data, test_data,
            target_acc=0.999,
            max_epochs=2000,
            save_path=save_path
        )
        if not success:
            print("\n⚠  Model did not reach >99.9% accuracy.")
            print("   Results below may be unreliable.")
            print("   Consider increasing max_epochs or adjusting lr/wd.")

    # Final evaluation
    print("\n  Final evaluation on held-out test set:")
    test_loader = DataLoader(test_data, batch_size=512)
    metrics     = evaluate(model, test_loader)
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")

    # ──────────────────────────────────────────
    # STAGE 2: Helix analysis
    # ──────────────────────────────────────────

    # 2A: Find clock heads
    clock_heads, tie_matrix = find_clock_heads(model, n_pairs=300)
    plot_tie_heatmap(tie_matrix)           # → tie_heatmap.png

    # ──────────────────────────────────────────
    # STAGE 2B: Always check for helix in residual stream
    # (independent of whether attention uses it!)
    # ──────────────────────────────────────────
    print("\n  Checking residual stream for helix representations...")
    r2_results = stage2b_helix_fit(model, clock_heads, check_all_layers=True)
    plot_r2_bars(r2_results)           # → helix_r2.png

    # Find if any layer has strong helix
    strong_helix_layers = [(layer, r2) for (layer, head), r2 in r2_results.items()
                           if r2 > 0.70 and head == 0]  # Only check once per layer

    if strong_helix_layers:
        best_helix_layer = max(strong_helix_layers, key=lambda x: x[1])[0]
        print(f"\n  ✓ Helix representation found in residual stream at layer(s): "
              f"{[l for l, _ in strong_helix_layers]}")
    else:
        print("\n  ✗ No strong helix found in residual stream (all R² < 0.70)")
        print("     → Model may use non-helical digit representation")

    # ──────────────────────────────────────────
    # STAGE 2C/2D: Run only if we have clock heads
    # ──────────────────────────────────────────
    if not clock_heads:
        print("\n  No clock heads found (TIE ≈ 0). Skipping Stage 2C/2D.")
        print("  → Attention is not performing arithmetic (MLP-dominated)")
        print("  Proceeding directly to Stage 3 (MLP analysis).\n")
        best_layer = 1   # default: analyse last layer MLP
    else:
        # Pick best head for geometric/causal tests
        best_head = max(r2_results, key=r2_results.get,
                        default=clock_heads[0])
        best_layer, best_h = best_head
        print(f"\n  Selected head for 2C/2D: Layer {best_layer}, Head {best_h} "
              f"(R²={r2_results.get(best_head, 0):.3f})")

        # 2C: Geometric circle test
        found_planes = stage2c_geometric_test(model, clock_heads)

        # Pick the best plane for the best head (if found)
        planes_for_best = found_planes.get(best_head, [])
        if planes_for_best:
            k1, k2 = planes_for_best[0][0], planes_for_best[0][1]
        else:
            print(f"\n  No plane found in 2C for {best_head}. "
                  f"Using directions (0,1) as fallback.")
            k1, k2 = 0, 1

        # ── Visual: circle plot for the best plane ──
        W_OV_vis   = get_OV_circuit(model, best_layer, best_h)
        _, _, Vt_vis = torch.linalg.svd(W_OV_vis)
        acts_vis, valid_ds_vis = collect_digit_activations(
            model, best_layer, digit_position=POS_A_ONES
        )
        coords_vis = torch.stack([
            acts_vis @ Vt_vis[k1].cpu(),
            acts_vis @ Vt_vis[k2].cpu()
        ], dim=1).float()
        plot_helix_circle(
            coords_vis, valid_ds_vis,
            title=f"Helix Plane L{best_layer}H{best_h} dirs({k1},{k2})"
        )

        # ── FFT sanity check on best head ──
        fft_sanity_check(acts_vis, Vt_vis, valid_ds_vis,
                         best_layer, best_h)

        # 2D-i: Output (U-side) helix validation
        W_OV   = get_OV_circuit(model, best_layer, best_h)
        U_mat, S_mat, Vt_mat = torch.linalg.svd(W_OV)

        stage2d_output_helix(model, best_layer, best_h, k1, k2)

        # 2D-ii: Causal phase-shift intervention
        stage2d_causal_test(model, best_layer, best_h,
                            Vt_mat, k1, k2, n_tests=20)

    # ──────────────────────────────────────────
    # STAGE 3: Carry circuit
    # ──────────────────────────────────────────
    # Run on both layers; carry circuit is typically in layer 1 MLP
    for layer in range(model.cfg.n_layers):
        try:
            U_c, S_c, Vt_c = stage3_carry_circuit(
                model, layer=layer, n_per_class=30
            )
        except RuntimeError as e:
            print(f"  Stage 3 failed for layer {layer}: {e}")

    print("\n" + "═"*55)
    print("EXPERIMENT COMPLETE")
    print("═"*55)
    print("\nSummary of what to look for in results:")
    print("  Stage 2B: R² > 0.70  → helix present in attention input")
    print("  Stage 2C: BULLSEYE   → 2D rotational plane confirmed")
    print("  Stage 2D-i: Lin>0.85 → OV circuit writes helix for sum")
    print("  Stage 2D-ii: no-carry ✓, carry ✗  → helix = linear part only")
    print("  Stage 3: |sep|>1.5σ  → MLP direction separates carry/no-carry")
    print("\n  Two-circuit account of addition:")
    print("    Addition = Helix rotation (OV, linear)  +  "
          "Carry detection (MLP, nonlinear)")


# ═════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Arithmetic Circuit Discovery — Option B"
    )
    parser.add_argument("--model_path", type=str, default="toy_addition_model.pt",
                        help="Path to save/load model weights")
    parser.add_argument("--skip_training", action="store_true",
                        help="Load model from model_path and skip training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    run_experiment(
        model_path=args.model_path,
        skip_training=args.skip_training
    )
