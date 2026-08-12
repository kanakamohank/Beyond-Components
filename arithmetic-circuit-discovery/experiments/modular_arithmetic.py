"""
Modular Arithmetic Circuit Analysis
====================================
Train a small transformer on (a + b) mod p, then analyze:
  1. Fourier structure in embeddings (circular representation)
  2. Which components are responsible (attention vs MLP)
  3. How MLP layers compute addition on the circle
  4. Whether modular arithmetic defeats polysemanticity

Based on: Nanda et al., "Progress measures for grokking via mechanistic interpretability"

Key insight: For mod-p addition, the model learns:
  - Embed numbers as points on a circle: cos(2πka/p), sin(2πka/p)
  - Attention moves operand representations together
  - MLP computes trig identity: cos(k(a+b)) = cos(ka)cos(kb) - sin(ka)sin(kb)
  - Each MLP neuron handles ONE frequency k → monosemantic!
"""

import math
import os
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Model: Minimal transformer for modular addition
# ============================================================================

class ModularArithmeticTransformer(nn.Module):
    """
    1-layer transformer for (a + b) mod p.
    
    Architecture mirrors Nanda et al.'s grokking setup:
      - Token embedding + positional embedding (3 positions: a, b, =)
      - Single attention layer (multi-head)
      - MLP with GELU activation
      - Unembedding to p classes
    
    No layer norm (simplifies analysis). Bias terms included.
    """
    
    def __init__(self, p: int, d_model: int = 128, n_heads: int = 4, d_mlp: int = 512):
        super().__init__()
        self.p = p
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_mlp = d_mlp
        
        # Embeddings: p tokens for numbers + 1 for '=' operator
        self.tok_embed = nn.Embedding(p + 1, d_model)  # tokens 0..p-1 = numbers, p = '='
        self.pos_embed = nn.Embedding(3, d_model)       # positions: 0=a, 1=b, 2='='
        
        # Attention
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        # MLP
        self.mlp_in = nn.Linear(d_model, d_mlp)
        self.mlp_out = nn.Linear(d_mlp, d_model)
        
        # Unembed: project to p output classes (one per residue)
        self.unembed = nn.Linear(d_model, p)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    
    def forward(self, tokens, return_cache=False):
        """
        Args:
            tokens: (batch, 3) — [a, b, p] where p is the '=' token
            return_cache: if True, return intermediate activations for analysis
        
        Returns:
            logits: (batch, p) — prediction at the '=' position
            cache: dict of intermediate activations (if return_cache=True)
        """
        B, T = tokens.shape  # T=3
        cache = {}
        
        # Embedding
        tok_emb = self.tok_embed(tokens)                    # (B, 3, d_model)
        pos_emb = self.pos_embed(torch.arange(T, device=tokens.device))  # (3, d_model)
        x = tok_emb + pos_emb                               # (B, 3, d_model)
        cache['embed'] = x.detach()
        
        # Attention
        Q = self.W_Q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d_head)
        K = self.W_K(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        attn_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, H, T, T)
        attn_probs = F.softmax(attn_scores, dim=-1)
        cache['attn_patterns'] = attn_probs.detach()
        
        attn_out = (attn_probs @ V).transpose(1, 2).reshape(B, T, self.d_model)  # (B, T, d_model)
        attn_out = self.W_O(attn_out)
        cache['attn_out'] = attn_out.detach()
        
        x = x + attn_out  # Residual connection
        cache['resid_mid'] = x.detach()
        
        # MLP
        mlp_pre = self.mlp_in(x)                           # (B, T, d_mlp)
        cache['mlp_pre_act'] = mlp_pre.detach()
        mlp_act = F.gelu(mlp_pre)                          # GELU activation
        cache['mlp_post_act'] = mlp_act.detach()
        mlp_out = self.mlp_out(mlp_act)                    # (B, T, d_model)
        cache['mlp_out'] = mlp_out.detach()
        
        x = x + mlp_out  # Residual connection
        cache['resid_post'] = x.detach()
        
        # Unembed at position 2 (the '=' position)
        logits = self.unembed(x[:, 2, :])                  # (B, p)
        
        if return_cache:
            return logits, cache
        return logits


# ============================================================================
# Data generation
# ============================================================================

def make_modular_addition_data(p: int, device='cpu'):
    """Generate full dataset of (a + b) mod p for all a, b in [0, p).
    
    Returns:
        tokens: (p*p, 3) — each row is [a, b, p_token]
        labels: (p*p,) — correct answer (a+b) mod p
    """
    eq_token = p  # Use p as the '=' token
    
    all_a = []
    all_b = []
    all_labels = []
    
    for a in range(p):
        for b in range(p):
            all_a.append(a)
            all_b.append(b)
            all_labels.append((a + b) % p)
    
    tokens = torch.stack([
        torch.tensor(all_a),
        torch.tensor(all_b),
        torch.full((p * p,), eq_token),
    ], dim=1).to(device)  # (p*p, 3)
    
    labels = torch.tensor(all_labels).to(device)  # (p*p,)
    
    return tokens, labels


# ============================================================================
# Training with grokking
# ============================================================================

def train_modular_arithmetic(
    p: int = 113,
    d_model: int = 128,
    n_heads: int = 4,
    d_mlp: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    epochs: int = 50000,
    train_frac: float = 0.3,
    seed: int = 42,
    log_interval: int = 500,
    save_dir: str = "modular_arithmetic_logs",
    device: str = None,
):
    """Train a transformer on (a+b) mod p until grokking.
    
    Grokking = model first memorizes training data, then suddenly generalizes
    to test data after many more epochs. Weight decay is critical for this.
    
    Returns:
        model: Trained model
        history: Dict with training metrics
    """
    if device is None:
        device = 'mps' if torch.backends.mps.is_available() else \
                 'cuda' if torch.cuda.is_available() else 'cpu'
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate data
    tokens, labels = make_modular_addition_data(p, device=device)
    n_total = len(tokens)
    n_train = int(n_total * train_frac)
    
    # Random train/test split
    perm = torch.randperm(n_total)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    
    train_tokens, train_labels = tokens[train_idx], labels[train_idx]
    test_tokens, test_labels = tokens[test_idx], labels[test_idx]
    
    logger.info(f"Modular arithmetic: ({p}-prime) | train={n_train}, test={n_total-n_train}")
    logger.info(f"Model: d_model={d_model}, n_heads={n_heads}, d_mlp={d_mlp}")
    logger.info(f"Device: {device}")
    
    # Model
    model = ModularArithmeticTransformer(p, d_model, n_heads, d_mlp).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training loop
    history = {
        'train_loss': [], 'test_loss': [],
        'train_acc': [], 'test_acc': [],
        'epochs': [],
    }
    
    best_test_acc = 0.0
    grokking_epoch = None
    
    train_loader = DataLoader(
        TensorDataset(train_tokens, train_labels),
        batch_size=min(512, n_train),
        shuffle=True,
    )
    
    for epoch in range(epochs):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        for batch_tokens, batch_labels in train_loader:
            logits = model(batch_tokens)
            loss = F.cross_entropy(logits, batch_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch_labels)
            epoch_correct += (logits.argmax(dim=-1) == batch_labels).sum().item()
            epoch_total += len(batch_labels)
        
        train_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total
        
        # --- Evaluate ---
        if epoch % log_interval == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                test_logits = model(test_tokens)
                test_loss = F.cross_entropy(test_logits, test_labels).item()
                test_acc = (test_logits.argmax(dim=-1) == test_labels).float().mean().item()
            
            history['epochs'].append(epoch)
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            
            logger.info(
                f"Epoch {epoch:6d} | "
                f"Train: loss={train_loss:.4f} acc={train_acc:.3f} | "
                f"Test: loss={test_loss:.4f} acc={test_acc:.3f}"
            )
            
            # Detect grokking
            if test_acc > 0.95 and grokking_epoch is None:
                grokking_epoch = epoch
                logger.info(f"🎉 GROKKING detected at epoch {epoch}! Test acc={test_acc:.3f}")
                # Save grokked model
                torch.save(model.state_dict(), os.path.join(save_dir, 'model_grokked.pt'))
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
            
            # Early stop if fully grokked
            if test_acc > 0.99:
                logger.info(f"Test accuracy > 99% — stopping training.")
                torch.save(model.state_dict(), os.path.join(save_dir, 'model_final.pt'))
                break
    
    # Save final model and history
    torch.save(model.state_dict(), os.path.join(save_dir, 'model_final.pt'))
    
    history['grokking_epoch'] = grokking_epoch
    history['best_test_acc'] = best_test_acc
    history['config'] = {
        'p': p, 'd_model': d_model, 'n_heads': n_heads, 'd_mlp': d_mlp,
        'lr': lr, 'weight_decay': weight_decay, 'train_frac': train_frac,
    }
    
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    return model, history


# ============================================================================
# Fourier Analysis: The Circle Representation
# ============================================================================

def compute_fourier_basis(p: int):
    """Compute the discrete Fourier basis for Z/pZ.
    
    For mod-p arithmetic, the natural basis is:
        cos(2πk·n/p) and sin(2πk·n/p) for k=0,...,p-1
    
    This is the "circle" — each number n is represented as a point on
    p//2 different circles (one per frequency k).
    
    Returns:
        fourier_basis: (p, p) — rows are basis vectors, columns are number positions
        freq_labels: list of str labels for each basis vector
    """
    basis = np.zeros((p, p))
    labels = []
    
    # DC component (k=0): constant
    basis[0] = 1.0 / np.sqrt(p)
    labels.append('const')
    
    idx = 1
    for k in range(1, (p + 1) // 2):
        # Cosine component
        cos_vec = np.array([np.cos(2 * np.pi * k * n / p) for n in range(p)])
        basis[idx] = cos_vec * np.sqrt(2.0 / p)
        labels.append(f'cos_{k}')
        idx += 1
        
        # Sine component
        if k < p // 2 or p % 2 == 1:
            sin_vec = np.array([np.sin(2 * np.pi * k * n / p) for n in range(p)])
            basis[idx] = sin_vec * np.sqrt(2.0 / p)
            labels.append(f'sin_{k}')
            idx += 1
    
    # Handle Nyquist frequency for even p
    if p % 2 == 0:
        k = p // 2
        cos_vec = np.array([np.cos(2 * np.pi * k * n / p) for n in range(p)])
        basis[idx] = cos_vec * np.sqrt(1.0 / p)
        labels.append(f'cos_{k}')
    
    return basis, labels


def analyze_fourier_structure(model, p: int, save_dir: str = "modular_arithmetic_logs"):
    """Analyze the Fourier structure of the trained model.
    
    Key analyses:
    1. Project token embeddings onto Fourier basis → shows circular representation
    2. Analyze MLP neuron activations → each should respond to one frequency
    3. Compute "Fourier power spectrum" of each component
    
    This is the core evidence for monosemanticity in modular arithmetic.
    """
    model.eval()
    device = next(model.parameters()).device
    
    fourier_basis, freq_labels = compute_fourier_basis(p)
    fourier_basis_t = torch.tensor(fourier_basis, dtype=torch.float32)  # (p, p)
    
    results = {}
    
    # ======== 1. Token Embedding Fourier Analysis ========
    # Get embeddings for number tokens 0..p-1
    num_embeds = model.tok_embed.weight[:p].detach().cpu()  # (p, d_model)
    
    # Project each embedding dimension onto Fourier basis
    # fourier_coeffs[k, d] = how much frequency k appears in dimension d
    fourier_coeffs = fourier_basis_t @ num_embeds  # (p, d_model)
    
    # Power spectrum: sum of squared coefficients across embedding dims
    fourier_power = (fourier_coeffs ** 2).sum(dim=1).numpy()  # (p,)
    
    # Find dominant frequencies
    top_freq_idx = np.argsort(fourier_power)[::-1][:10]
    
    logger.info("\n" + "="*60)
    logger.info("FOURIER ANALYSIS OF TOKEN EMBEDDINGS")
    logger.info("="*60)
    logger.info(f"Top frequencies in embeddings (out of {p} possible):")
    for i, idx in enumerate(top_freq_idx):
        logger.info(f"  {i+1}. {freq_labels[idx]:10s} — power={fourier_power[idx]:.4f}")
    
    results['embedding_fourier_power'] = {
        freq_labels[i]: float(fourier_power[i]) for i in range(len(freq_labels))
    }
    results['top_embedding_frequencies'] = [freq_labels[i] for i in top_freq_idx]
    
    # ======== 2. MLP Neuron Fourier Analysis ========
    # Run all p*p inputs through the model and cache MLP activations
    tokens, labels = make_modular_addition_data(p, device=device)
    
    with torch.no_grad():
        # Process in batches to avoid OOM
        batch_size = min(2048, len(tokens))
        all_mlp_pre = []
        all_mlp_post = []
        
        for i in range(0, len(tokens), batch_size):
            batch = tokens[i:i+batch_size]
            _, cache = model(batch, return_cache=True)
            # Get MLP activations at the '=' position (position 2)
            all_mlp_pre.append(cache['mlp_pre_act'][:, 2, :].cpu())
            all_mlp_post.append(cache['mlp_post_act'][:, 2, :].cpu())
        
        mlp_pre = torch.cat(all_mlp_pre, dim=0)   # (p*p, d_mlp)
        mlp_post = torch.cat(all_mlp_post, dim=0)  # (p*p, d_mlp)
    
    # Reshape to (p, p, d_mlp) — indexed by (a, b, neuron)
    mlp_pre_2d = mlp_pre.reshape(p, p, -1)
    mlp_post_2d = mlp_post.reshape(p, p, -1)
    
    # For each neuron, compute its 2D Fourier transform over (a, b)
    # A neuron that computes cos(k(a+b)) will have power concentrated at frequency (k, k)
    n_neurons = mlp_post_2d.shape[2]
    
    neuron_freq_a = np.zeros(n_neurons, dtype=int)  # Dominant freq w.r.t. operand a
    neuron_freq_b = np.zeros(n_neurons, dtype=int)  # Dominant freq w.r.t. operand b
    neuron_max_power = np.zeros(n_neurons)
    neuron_total_power = np.zeros(n_neurons)
    neuron_selectivity = np.zeros(n_neurons)  # How concentrated the power is
    
    logger.info("\n" + "="*60)
    logger.info("MLP NEURON FOURIER ANALYSIS")
    logger.info("="*60)
    
    for neuron_idx in range(n_neurons):
        # Get this neuron's activation pattern over all (a, b) pairs
        act_2d = mlp_post_2d[:, :, neuron_idx].numpy()  # (p, p)
        
        # 2D DFT
        fft_2d = np.fft.fft2(act_2d)
        power_2d = np.abs(fft_2d) ** 2
        
        # Find dominant frequency
        # Exclude DC component (0, 0)
        power_2d_no_dc = power_2d.copy()
        power_2d_no_dc[0, 0] = 0
        
        max_idx = np.unravel_index(np.argmax(power_2d_no_dc), power_2d.shape)
        neuron_freq_a[neuron_idx] = max_idx[0]
        neuron_freq_b[neuron_idx] = max_idx[1]
        neuron_max_power[neuron_idx] = power_2d_no_dc[max_idx]
        neuron_total_power[neuron_idx] = power_2d_no_dc.sum()
        
        # Selectivity: fraction of total power in the dominant frequency
        if neuron_total_power[neuron_idx] > 0:
            neuron_selectivity[neuron_idx] = (
                neuron_max_power[neuron_idx] / neuron_total_power[neuron_idx]
            )
    
    # ======== 3. Monosemanticity Analysis ========
    # A neuron is "monosemantic" if its selectivity is high (power concentrated at one freq)
    # AND its dominant freq_a == freq_b (responds to same frequency in both operands)
    
    is_monosemantic = (neuron_selectivity > 0.3) & (neuron_freq_a == neuron_freq_b)
    n_monosemantic = is_monosemantic.sum()
    
    # Neurons where freq_a == freq_b → they compute functions of (a+b) mod p
    same_freq = (neuron_freq_a == neuron_freq_b)
    n_same_freq = same_freq.sum()
    
    # Active neurons (significant total power)
    power_threshold = np.percentile(neuron_total_power, 50)
    is_active = neuron_total_power > power_threshold
    n_active = is_active.sum()
    
    # Among active neurons, what fraction are monosemantic?
    active_and_mono = (is_active & is_monosemantic).sum()
    mono_frac = active_and_mono / max(n_active, 1)
    
    logger.info(f"Total MLP neurons: {n_neurons}")
    logger.info(f"Active neurons (power > median): {n_active}")
    logger.info(f"Same-frequency neurons (freq_a == freq_b): {n_same_freq}")
    logger.info(f"Monosemantic neurons (selectivity > 0.3 & same freq): {n_monosemantic}")
    logger.info(f"Monosemanticity rate (among active): {mono_frac:.1%}")
    
    # Show top neurons by power
    logger.info(f"\nTop 20 MLP neurons by power:")
    top_neurons = np.argsort(neuron_total_power)[::-1][:20]
    for i, n_idx in enumerate(top_neurons):
        fa, fb = neuron_freq_a[n_idx], neuron_freq_b[n_idx]
        sel = neuron_selectivity[n_idx]
        mono = "MONO" if is_monosemantic[n_idx] else "poly"
        logger.info(
            f"  Neuron {n_idx:4d}: freq=({fa:3d},{fb:3d}) "
            f"selectivity={sel:.3f} power={neuron_total_power[n_idx]:.1f} [{mono}]"
        )
    
    # ======== 4. Frequency Distribution ========
    # Which frequencies are most used?
    from collections import Counter
    active_freqs = neuron_freq_a[is_active & same_freq]
    freq_counter = Counter(active_freqs.tolist())
    
    logger.info(f"\nFrequency usage among same-freq active neurons:")
    for freq, count in freq_counter.most_common(15):
        logger.info(f"  Frequency k={freq}: {count} neurons")
    
    results['mlp_analysis'] = {
        'n_neurons': int(n_neurons),
        'n_active': int(n_active),
        'n_monosemantic': int(n_monosemantic),
        'monosemanticity_rate': float(mono_frac),
        'n_same_freq': int(n_same_freq),
        'frequency_usage': {str(k): v for k, v in freq_counter.most_common(20)},
        'top_neurons': [
            {
                'neuron': int(n_idx),
                'freq_a': int(neuron_freq_a[n_idx]),
                'freq_b': int(neuron_freq_b[n_idx]),
                'selectivity': float(neuron_selectivity[n_idx]),
                'power': float(neuron_total_power[n_idx]),
                'is_monosemantic': bool(is_monosemantic[n_idx]),
            }
            for n_idx in top_neurons
        ],
    }
    
    # ======== 5. Attention Pattern Analysis ========
    logger.info("\n" + "="*60)
    logger.info("ATTENTION PATTERN ANALYSIS")
    logger.info("="*60)
    
    with torch.no_grad():
        # Use a subset of inputs
        sample_tokens = tokens[:min(1000, len(tokens))]
        _, cache = model(sample_tokens, return_cache=True)
        attn_patterns = cache['attn_patterns'].cpu()  # (B, H, 3, 3)
    
    # Average attention pattern per head
    avg_attn = attn_patterns.mean(dim=0)  # (H, 3, 3)
    
    logger.info("Average attention patterns (at '=' position, attending to [a, b, =]):")
    for h in range(avg_attn.shape[0]):
        a_weight = avg_attn[h, 2, 0].item()
        b_weight = avg_attn[h, 2, 1].item()
        eq_weight = avg_attn[h, 2, 2].item()
        logger.info(f"  Head {h}: a={a_weight:.3f}, b={b_weight:.3f}, ={eq_weight:.3f}")
    
    results['attention'] = {
        'avg_pattern_at_eq': [
            {
                'head': h,
                'attn_to_a': float(avg_attn[h, 2, 0]),
                'attn_to_b': float(avg_attn[h, 2, 1]),
                'attn_to_eq': float(avg_attn[h, 2, 2]),
            }
            for h in range(avg_attn.shape[0])
        ]
    }
    
    # ======== 6. The Circle: How Addition Happens ========
    logger.info("\n" + "="*60)
    logger.info("THE CIRCLE: HOW MODULAR ADDITION HAPPENS")
    logger.info("="*60)
    
    # For the dominant frequency k, show how MLP computes cos(k(a+b))
    if len(freq_counter) > 0:
        dominant_freq = freq_counter.most_common(1)[0][0]
        
        # Find neurons tuned to this frequency
        tuned_neurons = np.where(
            (neuron_freq_a == dominant_freq) & 
            (neuron_freq_b == dominant_freq) &
            is_active
        )[0]
        
        if len(tuned_neurons) > 0:
            best_neuron = tuned_neurons[np.argmax(neuron_selectivity[tuned_neurons])]
            
            logger.info(f"\nDominant frequency: k={dominant_freq}")
            logger.info(f"Neurons tuned to k={dominant_freq}: {len(tuned_neurons)}")
            logger.info(f"Best neuron: {best_neuron} (selectivity={neuron_selectivity[best_neuron]:.3f})")
            
            # Show this neuron's activation pattern
            act_2d = mlp_post_2d[:, :, best_neuron].numpy()
            
            # The ideal pattern is cos(2π·k·(a+b)/p)
            ideal = np.zeros((p, p))
            for a in range(p):
                for b in range(p):
                    ideal[a, b] = np.cos(2 * np.pi * dominant_freq * (a + b) / p)
            
            # Correlation between actual and ideal
            corr = np.corrcoef(act_2d.flatten(), ideal.flatten())[0, 1]
            logger.info(f"Correlation with cos(2π·{dominant_freq}·(a+b)/{p}): {corr:.4f}")
            
            results['circle_computation'] = {
                'dominant_frequency': int(dominant_freq),
                'n_tuned_neurons': int(len(tuned_neurons)),
                'best_neuron': int(best_neuron),
                'best_selectivity': float(neuron_selectivity[best_neuron]),
                'cosine_correlation': float(corr),
            }
            
            logger.info(f"\nThe model represents numbers on a CIRCLE:")
            logger.info(f"  Number n → point at angle 2π·{dominant_freq}·n/{p} on the unit circle")
            logger.info(f"  Addition (a+b) mod {p} → rotation on the circle")
            logger.info(f"  MLP neuron {best_neuron} computes: cos(2π·{dominant_freq}·(a+b)/{p})")
            logger.info(f"  This is the trig identity:")
            logger.info(f"    cos(k(a+b)) = cos(ka)·cos(kb) - sin(ka)·sin(kb)")
    
    # ======== 7. Polysemanticity Score ========
    logger.info("\n" + "="*60)
    logger.info("POLYSEMANTICITY ANALYSIS")
    logger.info("="*60)
    
    # For each active neuron, compute how many frequencies it responds to
    # (number of Fourier components above 10% of its max power)
    n_features_per_neuron = np.zeros(n_neurons)
    for n_idx in range(n_neurons):
        if not is_active[n_idx]:
            continue
        act_2d = mlp_post_2d[:, :, n_idx].numpy()
        fft_2d = np.fft.fft2(act_2d)
        power_2d = np.abs(fft_2d) ** 2
        power_2d[0, 0] = 0  # Remove DC
        max_pow = power_2d.max()
        if max_pow > 0:
            n_features_per_neuron[n_idx] = (power_2d > 0.1 * max_pow).sum()
    
    active_features = n_features_per_neuron[is_active]
    
    logger.info(f"Features per active neuron (mean): {active_features.mean():.1f}")
    logger.info(f"Features per active neuron (median): {np.median(active_features):.1f}")
    logger.info(f"Neurons with 1-2 features (monosemantic): {(active_features <= 2).sum()}/{n_active}")
    logger.info(f"Neurons with 3-5 features: {((active_features > 2) & (active_features <= 5)).sum()}/{n_active}")
    logger.info(f"Neurons with 6+ features (polysemantic): {(active_features > 5).sum()}/{n_active}")
    
    poly_rate = (active_features > 5).sum() / max(n_active, 1)
    mono_rate_strict = (active_features <= 2).sum() / max(n_active, 1)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"VERDICT: Does modular arithmetic defeat polysemanticity?")
    logger.info(f"{'='*60}")
    logger.info(f"Monosemantic rate (≤2 features): {mono_rate_strict:.1%}")
    logger.info(f"Polysemantic rate (>5 features): {poly_rate:.1%}")
    if mono_rate_strict > 0.5:
        logger.info(f"YES — Majority of neurons are monosemantic!")
        logger.info(f"Each neuron responds to a single Fourier frequency,")
        logger.info(f"encoding one 'direction' on the circle.")
    else:
        logger.info(f"MIXED — Some monosemantic structure, but polysemanticity remains.")
    
    results['polysemanticity'] = {
        'mean_features_per_neuron': float(active_features.mean()),
        'median_features_per_neuron': float(np.median(active_features)),
        'monosemantic_rate': float(mono_rate_strict),
        'polysemantic_rate': float(poly_rate),
        'n_monosemantic_strict': int((active_features <= 2).sum()),
        'n_polysemantic': int((active_features > 5).sum()),
    }
    
    # Save all results
    results_path = os.path.join(save_dir, 'fourier_analysis.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_path}")
    
    return results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Modular Arithmetic Circuit Analysis")
    parser.add_argument('--p', type=int, default=113, help='Prime modulus')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--d_mlp', type=int, default=512, help='MLP hidden dimension')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1.0, help='Weight decay (critical for grokking)')
    parser.add_argument('--epochs', type=int, default=50000, help='Max epochs')
    parser.add_argument('--train_frac', type=float, default=0.3, help='Fraction of data for training')
    parser.add_argument('--save_dir', type=str, default='modular_arithmetic_logs', help='Save directory')
    parser.add_argument('--analyze_only', type=str, default=None, help='Path to existing model to analyze')
    parser.add_argument('--device', type=str, default=None, help='Device override')
    
    args = parser.parse_args()
    
    if args.analyze_only:
        # Load existing model and analyze
        model = ModularArithmeticTransformer(
            args.p, args.d_model, args.n_heads, args.d_mlp
        )
        device = args.device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        model.load_state_dict(torch.load(args.analyze_only, map_location=device))
        model = model.to(device)
        logger.info(f"Loaded model from {args.analyze_only}")
        
        results = analyze_fourier_structure(model, args.p, args.save_dir)
    else:
        # Train and analyze
        model, history = train_modular_arithmetic(
            p=args.p,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_mlp=args.d_mlp,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            train_frac=args.train_frac,
            save_dir=args.save_dir,
            device=args.device,
        )
        
        if history.get('best_test_acc', 0) > 0.9:
            logger.info("\nModel grokked! Running Fourier analysis...")
            results = analyze_fourier_structure(model, args.p, args.save_dir)
        else:
            logger.info(f"\nModel did NOT grok (best test acc={history.get('best_test_acc', 0):.3f})")
            logger.info("Try more epochs or adjust hyperparameters.")
