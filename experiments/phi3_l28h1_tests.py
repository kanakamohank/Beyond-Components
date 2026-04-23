"""Phi-3 L28H1 vs L24H10: principal angles + decode.

Two tests in one script, Vt (write-direction) SVD convention.

Test 1: principal angles between L28H1 and L24H10 OV write subspaces.
        Small angles => same dial.  Large angles => independent circuits.
Test 2: decode top-4 write directions of L28H1 via W_U; report top-k
        tokens promoted at each axis.

Run:
    .venv/bin/python experiments/phi3_l28h1_tests.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer

warnings.filterwarnings("ignore")


def head_svd(model, layer, head):
    W_V = model.W_V[layer, head].detach().float().cpu()
    W_O = model.W_O[layer, head].detach().float().cpu()
    _U, S, Vt = torch.linalg.svd(W_V @ W_O, full_matrices=False)
    return S, Vt


def principal_angles(A, B):
    QA, _ = np.linalg.qr(A.T)
    QB, _ = np.linalg.qr(B.T)
    sv = np.linalg.svd(QA.T @ QB, compute_uv=False)
    return np.degrees(np.arccos(np.clip(sv, -1.0, 1.0)))


def decode_direction(v, W_U, tokenizer, top_k=20):
    scores = v @ W_U
    top_pos = torch.topk(scores, top_k)
    top_neg = torch.topk(-scores, top_k)
    pos = [tokenizer.decode([int(i)]).strip()
           for i in top_pos.indices.tolist()]
    neg = [tokenizer.decode([int(i)]).strip()
           for i in top_neg.indices.tolist()]
    return pos, neg


def main():
    out_dir = Path("helix_usage_validated")
    out_dir.mkdir(exist_ok=True)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    dtype = torch.bfloat16
    print(f"Loading Phi-3 on {device}...")
    model = HookedTransformer.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct", device=device, dtype=dtype)
    W_U = model.W_U.detach().float().cpu()

    log = ["=" * 70,
           "Phi-3 L28H1 vs L24H10: principal angles + decode",
           "=" * 70, ""]

    # ---- Test 1: principal angles ------------------------------------
    S10, Vt10 = head_svd(model, 24, 10)
    S1, Vt1 = head_svd(model, 28, 1)
    log.append("[1] Principal angles (write-subspace overlap)")

    # paper's plane: L24H10 SV(0,1); scan's top plane: L28H1 SV(0,3)
    pA = Vt10[[0, 1], :].numpy()
    pB = Vt1[[0, 3], :].numpy()
    ang2 = principal_angles(pA, pB)
    log.append(f"  L24H10(0,1) vs L28H1(0,3)  2D x 2D  "
               f"angles=[{ang2[0]:5.1f}, {ang2[1]:5.1f}]")

    # top-4 write subspaces (dominant OV capacity of each head)
    pA4 = Vt10[:4, :].numpy()
    pB4 = Vt1[:4, :].numpy()
    ang4 = principal_angles(pA4, pB4)
    log.append(f"  L24H10 top-4 vs L28H1 top-4  4D x 4D  "
               f"angles=[{', '.join(f'{a:5.1f}' for a in ang4)}]")

    log.append(f"  L24H10 top-4 sigma = "
               f"{[round(float(x), 2) for x in S10[:4]]}")
    log.append(f"  L28H1  top-4 sigma = "
               f"{[round(float(x), 2) for x in S1[:4]]}")

    if ang4.max() < 20:
        tag = "SAME dial (all 4 principal angles <20 deg)"
    elif ang4.min() < 30:
        tag = "PARTIAL overlap (some shared axes)"
    else:
        tag = "INDEPENDENT subspaces"
    log.append(f"  verdict: {tag}")
    log.append("")

    # ---- Test 2: decode L28H1 top write directions -------------------
    log.append("[2] L28H1 write-direction decode through W_U (top-20)")
    for k in range(4):
        v = Vt1[k, :].float()
        pos, neg = decode_direction(v, W_U, model.tokenizer, top_k=20)
        log.append(f"  SV{k} (sigma={float(S1[k]):.3f})")
        log.append(f"    +v : {' '.join(pos)}")
        log.append(f"    -v : {' '.join(neg)}")
    log.append("")

    txt = "\n".join(log)
    print(txt)
    (out_dir / "phi3_l28h1_tests.txt").write_text(txt)
    print(f"\nSaved helix_usage_validated/phi3_l28h1_tests.txt")


if __name__ == "__main__":
    main()
