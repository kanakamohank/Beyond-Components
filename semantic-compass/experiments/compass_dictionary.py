"""Compass dictionary: decode SV directions of all passing compass heads
across 3 models, producing a gender/pronoun dial dictionary.

For each (model, head), compute SVD of W_OV and decode Vt[k] through W_U
for k = 0..3.  Show top/bottom tokens along +v and -v of each SV axis.

Outputs (one file per model plus a combined markdown table):
  helix_usage_validated/compass_dict_gpt2.txt
  helix_usage_validated/compass_dict_phi3.txt
  helix_usage_validated/compass_dict_gemma.txt
  helix_usage_validated/compass_dict_all.md   (combined, paper-ready)

Usage:
  .venv/bin/python experiments/compass_dictionary.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

warnings.filterwarnings("ignore")


MODEL_HEADS = {
    "gpt2": [(10, 9), (9, 7), (11, 1)],
    "microsoft/Phi-3-mini-4k-instruct": [
        (28, 1), (27, 16), (24, 10), (26, 22), (25, 24)],
    "google/gemma-2-2b": [(21, 4)],
}

SV_MAX = 8           # hard cap on axes decoded per head
SV_MIN = 4           # always decode at least this many (scan used top-4)
VAR_TARGET = 0.90    # extend past SV_MIN until cumulative σ² ≥ 0.90
TOP_K = 15           # top/bottom tokens per side
OUT = Path("helix_usage_validated")


def decode_head(model, layer, head, device):
    W_V = model.W_V[layer, head].detach().float()
    W_O = model.W_O[layer, head].detach().float()
    W_OV = W_V @ W_O               # (d_model, d_model)
    U, S, Vt = torch.linalg.svd(W_OV, full_matrices=False)

    # Cumulative variance explained by σ_k² / Σσ²
    s2 = (S ** 2)
    total = float(s2.sum())
    cum_var = torch.cumsum(s2, dim=0) / total  # (r,)

    # Pick K: at least SV_MIN, extend until VAR_TARGET, cap at SV_MAX
    K = SV_MIN
    while K < min(SV_MAX, Vt.shape[0]) and float(cum_var[K - 1]) < VAR_TARGET:
        K += 1

    W_U = model.W_U.detach().float()   # (d_model, vocab)
    # Mean-center unembedding rows so generic high-frequency tokens
    # don't dominate: bias = W_U^T @ ones / N
    bias = W_U.mean(dim=1, keepdim=True)
    W_U_c = W_U - bias

    results = []
    for k in range(K):
        v = Vt[k]                    # (d_model,)
        logits_pos = v @ W_U_c       # (vocab,)
        logits_neg = -v @ W_U_c

        top_pos = torch.topk(logits_pos, TOP_K)
        top_neg = torch.topk(logits_neg, TOP_K)
        toks_pos = [model.to_string([int(i)]).strip()
                    or repr(model.to_string([int(i)]))
                    for i in top_pos.indices]
        toks_neg = [model.to_string([int(i)]).strip()
                    or repr(model.to_string([int(i)]))
                    for i in top_neg.indices]
        results.append(dict(
            k=k, sigma=float(S[k]),
            cum_var=float(cum_var[k]),
            pos=toks_pos, neg=toks_neg,
        ))
    return results, K, float(cum_var[K - 1]), int(S.shape[0])


def run_model(model_name, heads, device):
    short = {
        "gpt2": "gpt2",
        "microsoft/Phi-3-mini-4k-instruct": "phi3",
        "google/gemma-2-2b": "gemma",
    }[model_name]
    print(f"\n=== {short} ({model_name}) ===")
    model = HookedTransformer.from_pretrained(
        model_name, device=device, dtype=torch.float32)

    lines = [f"Model: {model_name}", ""]
    md_rows = []

    for (L, H) in heads:
        print(f"  decoding L{L} H{H}...")
        res, K, cum_K, rank = decode_head(model, L, H, device)
        lines.append(f"=== L{L} H{H} ===")
        lines.append(
            f"  decoded K={K}/{rank} axes  cum_var@K={cum_K:.3f}  "
            f"(target={VAR_TARGET}, min={SV_MIN}, max={SV_MAX})")
        for r in res:
            lines.append(
                f"  SV{r['k']}  sigma={r['sigma']:.3f}  "
                f"cum_var={r['cum_var']:.3f}")
            lines.append(f"    +v : {' | '.join(r['pos'])}")
            lines.append(f"    -v : {' | '.join(r['neg'])}")
            md_rows.append(dict(
                model=short, head=f"L{L}H{H}", sv=r['k'],
                sigma=r['sigma'], cum_var=r['cum_var'],
                pos=", ".join(r['pos'][:8]),
                neg=", ".join(r['neg'][:8]),
            ))
        lines.append("")

    out_path = OUT / f"compass_dict_{short}.txt"
    out_path.write_text("\n".join(lines))
    print(f"  wrote {out_path}")
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return md_rows


def main():
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    OUT.mkdir(exist_ok=True)

    all_rows = []
    for model_name, heads in MODEL_HEADS.items():
        all_rows.extend(run_model(model_name, heads, device))

    # Combined markdown table
    md = ["# Compass Dictionary (across GPT-2, Phi-3, Gemma-2)", ""]
    md.append("Decoded via W_U @ V^T_k with per-row centering of W_U.")
    md.append("Top tokens along +v/-v for each decoded SV axis of each "
              "passing compass head.")
    md.append("")
    md.append(f"Axes per head: at least {SV_MIN}, extended until cumulative "
              f"σ² ≥ {VAR_TARGET}, capped at {SV_MAX}.")
    md.append("")
    md.append("| Model | Head | SV | σ | cum σ² | +v direction (top-8) | −v direction (top-8) |")
    md.append("|---|---|---:|---:|---:|---|---|")
    for r in all_rows:
        md.append(
            f"| {r['model']} | {r['head']} | {r['sv']} | "
            f"{r['sigma']:.2f} | {r['cum_var']:.2f} | "
            f"{r['pos']} | {r['neg']} |")
    md_path = OUT / "compass_dict_all.md"
    md_path.write_text("\n".join(md))
    print(f"\nwrote {md_path}")


if __name__ == "__main__":
    main()
