# train_toy_addition.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────
# 1. Generate Data
# ─────────────────────────────────────────
def generate_addition_data(n_digits=2, n_samples=50000):
    """
    Input:  "A + B ="  tokenized as [d1,d2, +, d1',d2', =]
    Output: digit-by-digit answer tokens
    Keeps it simple: 2-digit + 2-digit, answer ≤ 198
    """
    max_val = 10**n_digits - 1

    A = torch.randint(0, max_val+1, (n_samples,))
    B = torch.randint(0, max_val+1, (n_samples,))
    C = A + B

    # Tokenize: digits 0-9 → tokens 0-9
    # Special tokens: '+' → 10, '=' → 11
    def to_tokens(x, n_digits):
        digits = []
        for i in range(n_digits-1, -1, -1):
            digits.append((x // (10**i)) % 10)
        return digits

    inputs, targets = [], []
    for a, b, c in zip(A.tolist(), B.tolist(), C.tolist()):
        a_tok = to_tokens(a, n_digits)      # [d1, d0]
        b_tok = to_tokens(b, n_digits)      # [d1', d0']
        c_tok = to_tokens(c, n_digits+1)    # [d2, d1, d0] (answer)

        # Input sequence: a_digits + [+] + b_digits + [=]
        inp = a_tok + [10] + b_tok + [11]   # length: 2+1+2+1 = 6
        inputs.append(inp)
        targets.append(c_tok)               # length: 3

    return (torch.tensor(inputs),           # [N, 6]
            torch.tensor(targets))          # [N, 3]

X, Y = generate_addition_data(n_digits=2)

# Train/test split - held out 20% for generalization test
split = int(0.8 * len(X))
train_data = TensorDataset(X[:split], Y[:split])
test_data  = TensorDataset(X[split:], Y[split:])

# ─────────────────────────────────────────
# 2. Define Minimal Transformer
# ─────────────────────────────────────────
from transformer_lens import HookedTransformer, HookedTransformerConfig

cfg = HookedTransformerConfig(
    n_layers=2,
    d_model=128,
    n_heads=4,
    d_head=32,
    d_mlp=512,
    n_ctx=10,
    d_vocab=12,        # 0-9 digits + '+' + '='
    act_fn="relu",
    normalization_type=None,  # No layernorm: cleaner for analysis
)
toy_model = HookedTransformer(cfg)

# ─────────────────────────────────────────
# 3. Train to >99% Accuracy
# ─────────────────────────────────────────
optimizer = torch.optim.Adam(toy_model.parameters(), lr=1e-3,
                             weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

def train_epoch(model, loader, optimizer):
    model.train()
    total_correct = 0
    for X_batch, Y_batch in loader:
        # Predict each answer digit autoregressively
        loss = 0
        for digit_pos in range(Y_batch.shape[1]):
            # Input = question + answer digits so far
            if digit_pos == 0:
                inp = X_batch
            else:
                inp = torch.cat([X_batch, Y_batch[:, :digit_pos]], dim=1)

            logits = model(inp)[:, -1, :]   # [B, vocab]
            loss += loss_fn(logits, Y_batch[:, digit_pos])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Train until >99% test accuracy - typically 1000-3000 steps
for epoch in range(500):
    train_epoch(toy_model,
                DataLoader(train_data, batch_size=512, shuffle=True),
                optimizer)

    if epoch % 50 == 0:
        # Eval
        correct = 0
        toy_model.eval()
        with torch.no_grad():
            for X_batch, Y_batch in DataLoader(test_data, batch_size=512):
                for digit_pos in range(Y_batch.shape[1]):
                    if digit_pos == 0:
                        inp = X_batch
                    else:
                        inp = torch.cat([X_batch, Y_batch[:, :digit_pos]],
                                        dim=1)
                    pred = toy_model(inp)[:, -1, :].argmax(-1)
                    correct += (pred == Y_batch[:, digit_pos]).sum().item()

        acc = correct / (len(test_data) * Y_batch.shape[1])
        print(f"Epoch {epoch}: test accuracy = {acc:.4f}")

        if acc > 0.999:
            print("Target accuracy reached. Saving model.")
            torch.save(toy_model.state_dict(), "toy_addition_model.pt")
            break