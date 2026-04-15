"""
Analyze a saved checkpoint to extract surviving circuit components.
Loads mask values and reports which attention heads and MLP layers survived pruning.
"""
import torch
import numpy as np
import sys
import os

def clamp_mask_fn(x):
    return torch.clamp(x, 0.0, 1.0)

def analyze_checkpoint(checkpoint_path, threshold=0.1):
    """
    Analyze masks from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        threshold: Mask values above this are considered "active/surviving"
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print(f"\n{'='*70}")
    print(f"CHECKPOINT ANALYSIS")
    print(f"{'='*70}")
    
    # Print basic info
    if 'step' in checkpoint:
        print(f"Step: {checkpoint['step']}")
    if 'val_loss' in checkpoint:
        print(f"Val Loss: {checkpoint['val_loss']:.4f}")
    if 'val_kl' in checkpoint:
        print(f"Val KL: {checkpoint['val_kl']:.4f}")
    if 'l1_norm' in checkpoint:
        print(f"L1 Norm: {checkpoint['l1_norm']:.1f}")
    
    # Extract masks
    qk_masks = checkpoint.get('qk_masks', {})
    ov_masks = checkpoint.get('ov_masks', {})
    mlp_in_masks = checkpoint.get('mlp_in_masks', {})
    mlp_out_masks = checkpoint.get('mlp_out_masks', {})
    
    # ==================== ATTENTION HEADS ====================
    print(f"\n{'='*70}")
    print(f"ATTENTION HEAD ANALYSIS (threshold={threshold})")
    print(f"{'='*70}")
    
    # Parse attention head keys and group by layer
    attn_heads = {}
    for key in sorted(set(list(ov_masks.keys()) + list(qk_masks.keys()))):
        # Key format: differential_head_{layer}_{head}
        parts = key.split('_')
        layer = int(parts[2])
        head = int(parts[3])
        
        ov_raw = ov_masks.get(key)
        qk_raw = qk_masks.get(key)
        
        ov_vals = clamp_mask_fn(ov_raw).detach().numpy() if ov_raw is not None else None
        qk_vals = clamp_mask_fn(qk_raw).detach().numpy() if qk_raw is not None else None
        
        if layer not in attn_heads:
            attn_heads[layer] = []
        
        # Compute summary stats
        ov_mean = float(np.mean(ov_vals)) if ov_vals is not None else None
        ov_max = float(np.max(ov_vals)) if ov_vals is not None else None
        ov_active = int(np.sum(ov_vals > threshold)) if ov_vals is not None else 0
        ov_total = len(ov_vals) if ov_vals is not None else 0
        
        qk_mean = float(np.mean(qk_vals)) if qk_vals is not None else None
        qk_max = float(np.max(qk_vals)) if qk_vals is not None else None
        qk_active = int(np.sum(qk_vals > threshold)) if qk_vals is not None else 0
        qk_total = len(qk_vals) if qk_vals is not None else 0
        
        attn_heads[layer].append({
            'head': head,
            'ov_mean': ov_mean,
            'ov_max': ov_max,
            'ov_active': ov_active,
            'ov_total': ov_total,
            'qk_mean': qk_mean,
            'qk_max': qk_max,
            'qk_active': qk_active,
            'qk_total': qk_total,
        })
    
    # Print attention head summary
    surviving_heads = []
    dead_heads = []
    
    for layer in sorted(attn_heads.keys()):
        print(f"\n--- Layer {layer} ---")
        for h in sorted(attn_heads[layer], key=lambda x: x['head']):
            ov_status = f"OV: mean={h['ov_mean']:.4f}, max={h['ov_max']:.4f}, active={h['ov_active']}/{h['ov_total']}" if h['ov_mean'] is not None else "OV: N/A"
            qk_status = f"QK: mean={h['qk_mean']:.4f}, max={h['qk_max']:.4f}, active={h['qk_active']}/{h['qk_total']}" if h['qk_mean'] is not None else "QK: N/A"
            
            # A head "survives" if its OV mask has significant active directions
            is_alive = (h['ov_mean'] is not None and h['ov_mean'] > threshold) or \
                       (h['qk_mean'] is not None and h['qk_mean'] > threshold)
            
            status_icon = "✅" if is_alive else "❌"
            print(f"  Head {h['head']:2d} {status_icon}  {ov_status}  |  {qk_status}")
            
            if is_alive:
                surviving_heads.append((layer, h['head'], h['ov_mean'], h['qk_mean']))
            else:
                dead_heads.append((layer, h['head']))
    
    # ==================== MLP LAYERS ====================
    print(f"\n{'='*70}")
    print(f"MLP LAYER ANALYSIS (threshold={threshold})")
    print(f"{'='*70}")
    
    mlp_layers = {}
    for key in sorted(set(list(mlp_in_masks.keys()) + list(mlp_out_masks.keys()))):
        # Key format: mlp_{layer}
        layer = int(key.split('_')[1])
        
        in_raw = mlp_in_masks.get(key)
        out_raw = mlp_out_masks.get(key)
        
        in_vals = clamp_mask_fn(in_raw).detach().numpy() if in_raw is not None else None
        out_vals = clamp_mask_fn(out_raw).detach().numpy() if out_raw is not None else None
        
        in_mean = float(np.mean(in_vals)) if in_vals is not None else None
        in_max = float(np.max(in_vals)) if in_vals is not None else None
        in_active = int(np.sum(in_vals > threshold)) if in_vals is not None else 0
        in_total = len(in_vals) if in_vals is not None else 0
        
        out_mean = float(np.mean(out_vals)) if out_vals is not None else None
        out_max = float(np.max(out_vals)) if out_vals is not None else None
        out_active = int(np.sum(out_vals > threshold)) if out_vals is not None else 0
        out_total = len(out_vals) if out_vals is not None else 0
        
        mlp_layers[layer] = {
            'in_mean': in_mean, 'in_max': in_max, 'in_active': in_active, 'in_total': in_total,
            'out_mean': out_mean, 'out_max': out_max, 'out_active': out_active, 'out_total': out_total,
            'in_vals': in_vals, 'out_vals': out_vals,
        }
    
    surviving_mlps = []
    dead_mlps = []
    
    for layer in sorted(mlp_layers.keys()):
        m = mlp_layers[layer]
        in_status = f"IN: mean={m['in_mean']:.4f}, max={m['in_max']:.4f}, active={m['in_active']}/{m['in_total']}" if m['in_mean'] is not None else "IN: N/A"
        out_status = f"OUT: mean={m['out_mean']:.4f}, max={m['out_max']:.4f}, active={m['out_active']}/{m['out_total']}" if m['out_mean'] is not None else "OUT: N/A"
        
        is_alive = (m['in_mean'] is not None and m['in_mean'] > threshold) or \
                   (m['out_mean'] is not None and m['out_mean'] > threshold)
        
        status_icon = "✅" if is_alive else "❌"
        print(f"  MLP Layer {layer:2d} {status_icon}  {in_status}  |  {out_status}")
        
        if is_alive:
            surviving_mlps.append((layer, m['in_mean'], m['out_mean'], m['in_active'], m['in_total'], m['out_active'], m['out_total']))
        else:
            dead_mlps.append(layer)
    
    # ==================== SUMMARY ====================
    print(f"\n{'='*70}")
    print(f"CIRCUIT SUMMARY")
    print(f"{'='*70}")
    
    total_heads = len(surviving_heads) + len(dead_heads)
    total_mlps = len(surviving_mlps) + len(dead_mlps)
    
    print(f"\nSurviving Attention Heads: {len(surviving_heads)}/{total_heads}")
    for layer, head, ov_mean, qk_mean in surviving_heads:
        ov_str = f"OV={ov_mean:.4f}" if ov_mean is not None else ""
        qk_str = f"QK={qk_mean:.4f}" if qk_mean is not None else ""
        print(f"  L{layer}H{head} ({ov_str} {qk_str})")
    
    print(f"\nSurviving MLP Layers: {len(surviving_mlps)}/{total_mlps}")
    for layer, in_mean, out_mean, in_active, in_total, out_active, out_total in surviving_mlps:
        in_str = f"IN={in_mean:.4f} ({in_active}/{in_total} dirs)" if in_mean is not None else ""
        out_str = f"OUT={out_mean:.4f} ({out_active}/{out_total} dirs)" if out_mean is not None else ""
        print(f"  MLP L{layer} ({in_str} | {out_str})")
    
    print(f"\nDead (pruned) Attention Heads: {len(dead_heads)}/{total_heads}")
    print(f"Dead (pruned) MLP Layers: {len(dead_mlps)}/{total_mlps}")
    
    # ==================== TOP SVD DIRECTIONS ====================
    print(f"\n{'='*70}")
    print(f"TOP SVD DIRECTIONS (per surviving component)")
    print(f"{'='*70}")
    
    # For surviving attention heads, show top-k mask values
    print("\n--- Surviving Attention Heads: Top SVD directions ---")
    for layer, head, ov_mean, qk_mean in surviving_heads:
        key = f'differential_head_{layer}_{head}'
        ov_raw = ov_masks.get(key)
        if ov_raw is not None:
            ov_vals = clamp_mask_fn(ov_raw).detach().numpy()
            top_k = min(10, len(ov_vals))
            top_indices = np.argsort(ov_vals)[::-1][:top_k]
            top_values = ov_vals[top_indices]
            dirs_str = ", ".join([f"d{i}={v:.3f}" for i, v in zip(top_indices, top_values)])
            print(f"  L{layer}H{head} OV top-{top_k}: {dirs_str}")
    
    # For surviving MLPs, show top-k mask values
    print("\n--- Surviving MLP Layers: Top SVD directions ---")
    for layer, in_mean, out_mean, in_active, in_total, out_active, out_total in surviving_mlps:
        key = f'mlp_{layer}'
        in_raw = mlp_in_masks.get(key)
        out_raw = mlp_out_masks.get(key)
        
        if in_raw is not None:
            in_vals = clamp_mask_fn(in_raw).detach().numpy()
            top_k = min(10, len(in_vals))
            top_indices = np.argsort(in_vals)[::-1][:top_k]
            top_values = in_vals[top_indices]
            dirs_str = ", ".join([f"d{i}={v:.3f}" for i, v in zip(top_indices, top_values)])
            print(f"  MLP L{layer} IN top-{top_k}: {dirs_str}")
        
        if out_raw is not None:
            out_vals = clamp_mask_fn(out_raw).detach().numpy()
            top_k = min(10, len(out_vals))
            top_indices = np.argsort(out_vals)[::-1][:top_k]
            top_values = out_vals[top_indices]
            dirs_str = ", ".join([f"d{i}={v:.3f}" for i, v in zip(top_indices, top_values)])
            print(f"  MLP L{layer} OUT top-{top_k}: {dirs_str}")
    
    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "logs/arithmetic_circuit_constrained_phi3_best.pt"
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        # Try to find it
        for root, dirs, files in os.walk("logs"):
            for f in files:
                if f.endswith('.pt'):
                    print(f"  Found: {os.path.join(root, f)}")
        sys.exit(1)
    
    analyze_checkpoint(checkpoint_path, threshold)
