"""Introspect why GPT-2 / Llama respond more than Gemma / Phi-3 to
compass injections.

For each model, at each ensemble member's injection layer:
  - residual stream norm (||resid_pre|| at the last token of a neutral prompt)
  - primary head's top singular value sigma_1
  - injection magnitude: alpha * sigma_1 (the paper convention)
  - SNR proxy: (alpha * sigma_1) / ||resid_pre||
  - presence of logit softcap / attention softcap
  - n_kv_heads (GQA collapse factor)

Prints a short table so we can see which models' injections are
'loud' relative to the stream vs which are 'quiet'.
"""
from __future__ import annotations

import warnings
import torch
from transformer_lens import HookedTransformer

warnings.filterwarnings("ignore")


PROMPT = "The doctor said that"


def primary_head_spec(name):
    return {
        "gpt2":        (10, 9,  [(9,7,(1,3)),(10,9,(0,3)),(11,1,(1,3)),(7,6,(1,2))]),
        "google/gemma-2-2b":        (21, 4, [(21,4,(0,1)),(20,5,(0,2)),(16,2,(1,2)),(24,5,(1,2))]),
        "microsoft/Phi-3-mini-4k-instruct": (28, 1, [(24,14,(0,2)),(28,26,(0,1)),(26,24,(2,3)),(27,16,(0,2))]),
        "meta-llama/Llama-3.2-3B": (26, 14,[(22,12,(2,3)),(22,14,(0,3)),(26,23,(0,1)),(26,21,(0,1))]),
    }[name]


def introspect(name, alphas):
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    dtype = torch.bfloat16
    print(f"\n=========== {name} ===========")
    print(f"  loading on {device} dtype={dtype}...")
    model = HookedTransformer.from_pretrained(name, device=device, dtype=dtype)
    cfg = model.cfg
    print(f"  n_layers={cfg.n_layers} n_heads={cfg.n_heads} "
          f"d_model={cfg.d_model} d_head={cfg.d_head} "
          f"d_vocab={cfg.d_vocab}")
    kv = getattr(cfg, "n_key_value_heads", None)
    if kv is None:
        kv = getattr(cfg, "n_kv_heads", None)
    if kv is None:
        kv = cfg.n_heads
    print(f"  n_kv_heads (GQA): {kv}   "
          f"gqa_factor = n_heads/n_kv_heads = {cfg.n_heads / max(1,kv):.1f}x")
    for attr in ("final_logit_softcap", "attn_logit_softcap",
                 "attn_softcap", "use_attn_softcap"):
        if hasattr(cfg, attr):
            print(f"  cfg.{attr} = {getattr(cfg, attr)}")

    # cache residual stream at all layers
    toks = model.to_tokens(PROMPT)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks)

    primary_L, primary_H, ensemble = primary_head_spec(name)

    # tabulate per (L,H,d1,d2)
    header = (f"{'L':>3} {'H':>3} {'d1,d2':>6} {'sigma_1':>8} {'sigma_d1':>8} "
              f"{'sigma_d2':>8} {'||resid||':>10} "
              + " ".join(f"snr@a={a}".rjust(10) for a in alphas))
    print("  " + header)
    for (L, H, plane) in ensemble:
        d1, d2 = plane
        W_V = model.W_V[L, H].detach().float().cpu()
        W_O = model.W_O[L, H].detach().float().cpu()
        _U, S, _Vt = torch.linalg.svd(W_V @ W_O, full_matrices=False)
        sig1 = float(S[0])
        sigd1 = float(S[d1])
        sigd2 = float(S[d2])
        resid = cache[f"blocks.{L}.hook_resid_pre"][0, -1, :].float().cpu()
        resid_norm = float(resid.norm())
        snrs = [ (a * max(sigd1, sigd2)) / resid_norm for a in alphas ]
        print(f"  {L:>3} {H:>3} {d1},{d2:>2}  {sig1:>8.3f} {sigd1:>8.3f} "
              f"{sigd2:>8.3f} {resid_norm:>10.2f}  "
              + "  ".join(f"{s:>8.3f}" for s in snrs))

    del model
    if device == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    # alphas matched to what we actually used in each eval
    introspect("gpt2", alphas=[0.5, 1.0, 1.5])
    introspect("meta-llama/Llama-3.2-3B", alphas=[5.0, 10.0, 20.0])
    introspect("microsoft/Phi-3-mini-4k-instruct", alphas=[0.5, 1.0, 1.5])
    introspect("google/gemma-2-2b", alphas=[5.0, 10.0, 20.0])
