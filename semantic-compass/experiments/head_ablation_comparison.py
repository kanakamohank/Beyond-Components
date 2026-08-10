"""Head-ablation comparison for the semantic compass.

Quantifies how much of a category signal (e.g., masc vs fem context)
is carried by the target head's OV top-2 plane, compared to the same
plane ablation applied to every other head in the same layer, and to
matching heads in nearby layers.

Ablation: project out span(u_0, u_1) of head h's W_OV from its per-head
output at the last token. This zeros that head's write-out only in the
candidate plane, leaving all other heads (and the rest of its own
write) untouched.

Signal: mean logit-diff on plus-context prompts minus mean logit-diff
on minus-context prompts. A head whose plane carries the category
moves this gap toward zero.

Usage
-----
python experiments/head_ablation_comparison.py \
    --model gpt2 --layer 9 --head 7 --dims 1 2 \
    --tok_plus " he" --tok_minus " she" \
    --prompt_plus  "The man laced up his boots because" \
    --prompt_plus  "The father waved to the crowd and" \
    --prompt_plus  "The king announced that" \
    --prompt_minus "The woman laced up her boots because" \
    --prompt_minus "The mother waved to the crowd and" \
    --prompt_minus "The queen announced that" \
    --other_layers 8 10 \
    --out_prefix head_ablation_gpt2
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer

warnings.filterwarnings("ignore")


def plane_projector(model, layer, head, d1, d2, device, dtype):
    """Return (u1, u2) on device/dtype for W_OV of head."""
    W_V = model.W_V[layer, head].detach().float().cpu()
    W_O = model.W_O[layer, head].detach().float().cpu()
    _U, _S, Vt = torch.linalg.svd(W_V @ W_O, full_matrices=False)
    u1 = Vt[d1, :].to(device).to(dtype)
    u2 = Vt[d2, :].to(device).to(dtype)
    return u1, u2


def ld_with_ablation(model, prompts, tok_plus_id, tok_minus_id,
                     ablate_layer=None, ablate_head=None,
                     u1=None, u2=None, full_head=False):
    """Mean logit(tok_plus - tok_minus) over prompts.
    If full_head=True, zero the head's entire output at the last
    position. Otherwise project out span(u1, u2) from head
    `ablate_head`'s output at the last position."""
    scores = []
    if ablate_layer is not None:
        hook_name = f"blocks.{ablate_layer}.attn.hook_result"

        def hook(res, hook):
            if full_head:
                res[:, -1, ablate_head, :] = 0
                return res
            v = res[:, -1, ablate_head, :]
            c1 = (v @ u1).unsqueeze(-1) * u1
            c2 = (v @ u2).unsqueeze(-1) * u2
            res[:, -1, ablate_head, :] = v - c1 - c2
            return res

    for p in prompts:
        toks = model.to_tokens(p)
        with torch.no_grad():
            if ablate_layer is None:
                logits = model(toks)
            else:
                logits = model.run_with_hooks(
                    toks, fwd_hooks=[(hook_name, hook)])
        last = logits[0, -1, :]
        scores.append(float(last[tok_plus_id] - last[tok_minus_id]))
    return float(np.mean(scores))


def signal(model, plus_prompts, minus_prompts, tok_plus_id,
           tok_minus_id, ablate_layer=None, ablate_head=None,
           u1=None, u2=None, full_head=False):
    ld_plus = ld_with_ablation(
        model, plus_prompts, tok_plus_id, tok_minus_id,
        ablate_layer, ablate_head, u1, u2, full_head)
    ld_minus = ld_with_ablation(
        model, minus_prompts, tok_plus_id, tok_minus_id,
        ablate_layer, ablate_head, u1, u2, full_head)
    return ld_plus - ld_minus, ld_plus, ld_minus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--head", type=int, required=True)
    ap.add_argument("--dims", type=int, nargs=2, required=True)
    ap.add_argument("--tok_plus", required=True)
    ap.add_argument("--tok_minus", required=True)
    ap.add_argument("--prompt_plus", action="append", required=True)
    ap.add_argument("--prompt_minus", action="append", required=True)
    ap.add_argument("--other_layers", type=int, nargs="*", default=[])
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--out_dir", default="helix_usage_validated")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    dtype = torch.bfloat16
    print(f"Loading {args.model} on {device}...")
    model = HookedTransformer.from_pretrained(
        args.model, device=device, dtype=dtype)
    model.set_use_attn_result(True)

    tok_plus_id = model.to_single_token(args.tok_plus)
    tok_minus_id = model.to_single_token(args.tok_minus)
    n_heads = model.cfg.n_heads
    print(f"n_heads={n_heads}")

    # ---- Baseline signal ----
    base_sig, base_plus, base_minus = signal(
        model, args.prompt_plus, args.prompt_minus,
        tok_plus_id, tok_minus_id)
    print(f"\nBASELINE signal = LD_plus - LD_minus = "
          f"{base_plus:+.3f} - ({base_minus:+.3f}) = {base_sig:+.3f}")

    # ---- Target head ablation (plane only) ----
    u1, u2 = plane_projector(
        model, args.layer, args.head, args.dims[0], args.dims[1],
        device, dtype)
    tgt_sig, tgt_p, tgt_m = signal(
        model, args.prompt_plus, args.prompt_minus, tok_plus_id,
        tok_minus_id, args.layer, args.head, u1, u2)
    tgt_drop = base_sig - tgt_sig

    # ---- Target head full-head ablation (calibration) ----
    full_sig, full_p, full_m = signal(
        model, args.prompt_plus, args.prompt_minus, tok_plus_id,
        tok_minus_id, args.layer, args.head, full_head=True)
    full_drop = base_sig - full_sig

    log = [
        f"Model: {args.model}",
        f"Target: L{args.layer} H{args.head} dims "
        f"({args.dims[0]},{args.dims[1]})",
        f"tokens: {args.tok_plus!r}={tok_plus_id}  "
        f"{args.tok_minus!r}={tok_minus_id}",
        "",
        f"BASELINE   LD_plus={base_plus:+.3f}  "
        f"LD_minus={base_minus:+.3f}  signal={base_sig:+.3f}",
        "",
        f"TARGET PLANE-ONLY ABLATION L{args.layer} H{args.head} "
        f"dims ({args.dims[0]},{args.dims[1]}): "
        f"LD_plus={tgt_p:+.3f}  LD_minus={tgt_m:+.3f}  "
        f"signal={tgt_sig:+.3f}  drop={tgt_drop:+.3f}  "
        f"({tgt_drop/base_sig:+.1%} of baseline)",
        f"TARGET FULL-HEAD ABLATION L{args.layer} H{args.head}: "
        f"LD_plus={full_p:+.3f}  LD_minus={full_m:+.3f}  "
        f"signal={full_sig:+.3f}  drop={full_drop:+.3f}  "
        f"({full_drop/base_sig:+.1%} of baseline)",
        f"PLANE SHARE OF HEAD: "
        f"{tgt_drop/full_drop if abs(full_drop) > 1e-6 else 0:+.1%}",
        "",
        f"SAME-LAYER SWEEP (layer {args.layer}, plane = top-2 "
        f"OV-SVD of each head):",
        f"  {'head':>5} {'LD+':>8} {'LD-':>8} {'signal':>8} "
        f"{'drop':>8} {'% of tgt':>10}",
        "  " + "-" * 55,
    ]

    same_layer_drops = []
    for h in range(n_heads):
        u1h, u2h = plane_projector(
            model, args.layer, h, 0, 1, device, dtype)
        sig_h, lp, lm = signal(
            model, args.prompt_plus, args.prompt_minus, tok_plus_id,
            tok_minus_id, args.layer, h, u1h, u2h)
        drop = base_sig - sig_h
        marker = "  *" if h == args.head else "   "
        log.append(
            f"{marker}{h:>5} {lp:+8.3f} {lm:+8.3f} {sig_h:+8.3f} "
            f"{drop:+8.3f} "
            f"{drop/tgt_drop if abs(tgt_drop) > 1e-6 else 0:+10.1%}")
        same_layer_drops.append((h, drop))

    # ---- Other layers: target head index, same dims ----
    log.append("")
    log.append(
        f"OTHER-LAYER SWEEP (head {args.head}, top-2 plane):")
    log.append(
        f"  {'layer':>6} {'LD+':>8} {'LD-':>8} {'signal':>8} "
        f"{'drop':>8} {'% of tgt':>10}")
    log.append("  " + "-" * 55)
    for L in args.other_layers:
        if L >= model.cfg.n_layers:
            continue
        u1L, u2L = plane_projector(
            model, L, args.head, 0, 1, device, dtype)
        sig_L, lp, lm = signal(
            model, args.prompt_plus, args.prompt_minus, tok_plus_id,
            tok_minus_id, L, args.head, u1L, u2L)
        drop = base_sig - sig_L
        log.append(
            f"   {L:>6} {lp:+8.3f} {lm:+8.3f} {sig_L:+8.3f} "
            f"{drop:+8.3f} "
            f"{drop/tgt_drop if abs(tgt_drop) > 1e-6 else 0:+10.1%}")

    # ---- Summary ----
    other_drops = [d for h, d in same_layer_drops if h != args.head]
    max_other = max(other_drops, key=abs) if other_drops else 0
    med_other = float(np.median([abs(d) for d in other_drops])) \
        if other_drops else 0
    ratio = (abs(tgt_drop) / abs(max_other)
             if abs(max_other) > 1e-6 else float("inf"))
    log.append("")
    log.append("SUMMARY")
    log.append(f"  target head drop               = {tgt_drop:+.3f}  "
               f"({tgt_drop/base_sig:+.1%} of baseline)")
    log.append(f"  max other-head drop (layer L)  = {max_other:+.3f}")
    log.append(f"  median |other-head drop|       = {med_other:+.3f}")
    log.append(f"  target / max-other ratio       = {ratio:.2f}x")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.out_prefix}.txt"
    out_path.write_text("\n".join(log))
    print("\n".join(log))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
