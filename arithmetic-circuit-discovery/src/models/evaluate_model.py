#!/usr/bin/env python3
"""Evaluate toy addition model accuracy."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformer_lens import HookedTransformer, HookedTransformerConfig

# Device detection
def get_device():
    """Automatically detect best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

DEVICE = get_device()
print(f"Using device: {DEVICE}")

# Generate test data
def generate_addition_data(n_digits=2, n_samples=50000):
    """Generate addition data."""
    max_val = 10**n_digits - 1
    A = torch.randint(0, max_val+1, (n_samples,))
    B = torch.randint(0, max_val+1, (n_samples,))
    C = A + B

    def to_tokens(x, n_digits):
        digits = []
        for i in range(n_digits-1, -1, -1):
            digits.append((x // (10**i)) % 10)
        return digits

    inputs, targets = [], []
    for a, b, c in zip(A.tolist(), B.tolist(), C.tolist()):
        a_tok = to_tokens(a, n_digits)
        b_tok = to_tokens(b, n_digits)
        c_tok = to_tokens(c, n_digits+1)
        inp = a_tok + [10] + b_tok + [11]
        inputs.append(inp)
        targets.append(c_tok)

    return torch.tensor(inputs), torch.tensor(targets)

# Build model
cfg = HookedTransformerConfig(
    n_layers=2,
    d_model=128,
    n_heads=4,
    d_head=32,
    d_mlp=512,
    n_ctx=10,
    d_vocab=12,
    act_fn="relu",
    normalization_type=None,
)
model = HookedTransformer(cfg)
model = model.to(DEVICE)

# Load trained weights
print("\nLoading model: toy_addition_model.pt")
model.load_state_dict(torch.load("toy_addition_model.pt", map_location=DEVICE))
model.eval()

# Generate test data
print("Generating test data (12,000 samples)...")
X, Y = generate_addition_data(n_digits=2, n_samples=12000)
test_data = TensorDataset(X, Y)

# Evaluate
print("\nEvaluating model...")
correct_per_digit = [0, 0, 0]  # hundreds, tens, ones
total_per_digit = [0, 0, 0]
exact_match = 0
total = 0

with torch.no_grad():
    for X_batch, Y_batch in DataLoader(test_data, batch_size=512):
        X_batch = X_batch.to(DEVICE)
        Y_batch = Y_batch.to(DEVICE)

        predictions = []
        for digit_pos in range(Y_batch.shape[1]):
            if digit_pos == 0:
                inp = X_batch
            else:
                # Use ground truth for teacher forcing during eval
                inp = torch.cat([X_batch, Y_batch[:, :digit_pos]], dim=1)

            pred = model(inp)[:, -1, :].argmax(-1)
            predictions.append(pred)

            # Accuracy per digit
            correct = (pred == Y_batch[:, digit_pos]).sum().item()
            correct_per_digit[digit_pos] += correct
            total_per_digit[digit_pos] += len(pred)

        # Exact match (all 3 digits correct)
        pred_full = torch.stack(predictions, dim=1)  # [batch, 3]
        exact = (pred_full == Y_batch).all(dim=1).sum().item()
        exact_match += exact
        total += len(X_batch)

# Print results
print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)
print(f"Test samples: {total}")
print(f"\nPer-digit accuracy:")
print(f"  Hundreds digit: {correct_per_digit[0]/total_per_digit[0]:.4f}")
print(f"  Tens digit:     {correct_per_digit[1]/total_per_digit[1]:.4f}")
print(f"  Ones digit:     {correct_per_digit[2]/total_per_digit[2]:.4f}")
print(f"\nExact match (all 3 digits): {exact_match/total:.4f} ({exact_match}/{total})")
print(f"Overall accuracy: {sum(correct_per_digit)/sum(total_per_digit):.4f}")
print("="*60)

# Additional stats
if exact_match/total >= 0.999:
    print("✓ Model achieves ≥99.9% exact match accuracy")
elif exact_match/total >= 0.99:
    print("✓ Model achieves ≥99% exact match accuracy")
else:
    print(f"✗ Model accuracy is {exact_match/total*100:.2f}%")
