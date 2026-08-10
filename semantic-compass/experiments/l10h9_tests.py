"""L10H9 vs L9H7: principal-angles + decode + causal sweep.

Three tests in one script, all using the Vt (write-direction) SVD
convention and in-context injection at the head's own layer.

Test 1: principal angles between 2D and 4D OV write subspaces of
        GPT-2 L9H7 and L10H9. Small angles => same dial.
Test 2: decode top-4 write directions of L10H9 via W_U; report
        top-30 tokens promoted at each axis.
Test 3: causal alpha-sweep at L10H9 SV(0,3) in-context, so we have
        a directly-comparable amplitude/phase row for Table 1.

Run:
    .venv/bin/python experiments/l10h9_tests.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer

warnings.filterwarnings("ignore")


NEUTRAL_PROMPTS = [
    "The person said that",
    "Then they said that",
    "Afterwards, the speaker said that",
]
MASC_PROMPTS = [
    "The king rode into battle. When the crowd cheered,",
    "My father picked up the phone. When the caller spoke,",
    "The boy opened his new book. As he started reading,",
]
FEM_PROMPTS = [
    "The queen rode into battle. When the crowd cheered,",
    "My mother picked up the phone. When the caller spoke,",
    "The girl opened her new book. As she started reading,",
]


def head_svd(model, layer, head):
    W_V = model.W_V[layer, head].detach().float().cpu()
    W_O = model.W_O[layer, head].detach().float().cpu()
    _U, S, Vt = torch.linalg.svd(W_V @ W_O, full_matrices=False)
    return S, Vt


def principal_angles(A, B):
    """Principal angles (deg) between row-spans of A and B (each kxd)."""
    QA, _ = np.linalg.qr(A.T)
    QB, _ = np.linalg.qr(B.T)
    sv = np.linalg.svd(QA.T @ QB, compute_uv=False)
    return np.degrees(np.arccos(np.clip(sv, -1.0, 1.0)))


def decode_direction(v, W_U, tokenizer, top_k=30):
    """Top-k tokens promoted by direction v (and -v)."""
    scores = v @ W_U
    top_pos = torch.topk(scores, top_k)
    top_neg = torch.topk(-scores, top_k)
    pos = [tokenizer.decode([int(i)]).strip()
           for i in top_pos.indices.tolist()]
    neg = [tokenizer.decode([int(i)]).strip()
           for i in top_neg.indices.tolist()]
    return pos, neg


def fit_sinusoid(angles_deg, y):
    th = np.radians(angles_deg)
    N = len(th)
    mu = y.mean()
    c = ((y - mu) * np.cos(th)).sum() * 2.0 / N
    s = ((y - mu) * np.sin(th)).sum() * 2.0 / N
    A = float(np.hypot(c, s))
    phi = float(np.degrees(np.arctan2(s, c)))
    return mu, A, phi


def causal_sweep(model, layer, head, d1, d2, tok_plus_id, tok_minus_id,
                 device, dtype, alphas=(1.0, 3.0, 10.0), n_angles=36):
    S, Vt = head_svd(model, layer, head)
    u1 = Vt[d1, :].to(device).to(dtype)
    u2 = Vt[d2, :].to(device).to(dtype)
    sig1, sig2 = float(S[d1]), float(S[d2])
    hook_name = f"blocks.{layer}.hook_resid_pre"
    angles = np.linspace(0, 360, n_angles, endpoint=False)

    def ld(prompt, edit):
        toks = model.to_tokens(prompt)
        with torch.no_grad():
            if edit is None:
                logits = model(toks)
            else:
                def _h(r, hook):
                    r[0, -1, :] = r[0, -1, :] + edit
                    return r
                logits = model.run_with_hooks(
                    toks, fwd_hooks=[(hook_name, _h)])
        last = logits[0, -1, :]
        return float(last[tok_plus_id] - last[tok_minus_id])

    def mean_ld(prompts, edit):
        return float(np.mean([ld(p, edit) for p in prompts]))

    base = {
        "neutral": mean_ld(NEUTRAL_PROMPTS, None),
        "masc": mean_ld(MASC_PROMPTS, None),
        "fem": mean_ld(FEM_PROMPTS, None),
    }

    fits = {}
    for a in alphas:
        lds = np.zeros(n_angles)
        for i, deg in enumerate(angles):
            th = np.radians(deg)
            vec = (a * sig1 * np.cos(th)) * u1 + \
                  (a * sig2 * np.sin(th)) * u2
            lds[i] = mean_ld(NEUTRAL_PROMPTS, vec)
        mu, A, phi = fit_sinusoid(angles, lds)
        fits[a] = dict(mu=mu, A=A, phi=phi,
                       argmax=float(angles[int(np.argmax(lds))]),
                       argmin=float(angles[int(np.argmin(lds))]))
    return dict(S=S, base=base, fits=fits)


def main():
    out_dir = Path("helix_usage_validated")
    out_dir.mkdir(exist_ok=True)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    dtype = torch.bfloat16
    print(f"Loading gpt2 on {device}...")
    model = HookedTransformer.from_pretrained(
        "gpt2", device=device, dtype=dtype)
    W_U = model.W_U.detach().float().cpu()

    tok_plus_id = model.to_single_token(" he")
    tok_minus_id = model.to_single_token(" she")

    log = ["=" * 70,
           "L10H9 vs L9H7: principal angles + decode + causal sweep",
           "=" * 70, ""]

    # ---- Test 1: principal angles ------------------------------------
    S7, Vt7 = head_svd(model, 9, 7)
    S9, Vt9 = head_svd(model, 10, 9)
    log.append("[1] Principal angles (write-subspace overlap)")

    # paper's plane: L9H7 SV(1,2); scan's top plane: L10H9 SV(0,3)
    pA = Vt7[[1, 2], :].numpy()
    pB = Vt9[[0, 3], :].numpy()
    ang2 = principal_angles(pA, pB)
    log.append(f"  L9H7(1,2) vs L10H9(0,3)  2D x 2D  "
               f"angles=[{ang2[0]:5.1f}, {ang2[1]:5.1f}]")

    # top-4 write subspaces (dominant OV capacity of each head)
    pA4 = Vt7[:4, :].numpy()
    pB4 = Vt9[:4, :].numpy()
    ang4 = principal_angles(pA4, pB4)
    log.append(f"  L9H7 top-4 vs L10H9 top-4  4D x 4D  "
               f"angles=[{', '.join(f'{a:5.1f}' for a in ang4)}]")

    # singular values for context
    log.append(f"  L9H7 top-4 sigma = "
               f"{[round(float(x), 2) for x in S7[:4]]}")
    log.append(f"  L10H9 top-4 sigma = "
               f"{[round(float(x), 2) for x in S9[:4]]}")

    if ang4.max() < 20:
        tag = "SAME dial (all 4 principal angles <20 deg)"
    elif ang4.min() < 30:
        tag = "PARTIAL overlap (some shared axes)"
    else:
        tag = "INDEPENDENT subspaces"
    log.append(f"  verdict: {tag}")
    log.append("")

    # ---- Test 2: decode L10H9 top write directions ------------------
    log.append("[2] L10H9 write-direction decode through W_U (top-30)")
    for k in range(4):
        v = Vt9[k, :].float()
        pos, neg = decode_direction(v, W_U, model.tokenizer, top_k=20)
        log.append(f"  SV{k} (sigma={float(S9[k]):.3f})")
        log.append(f"    +v : {' '.join(pos)}")
        log.append(f"    -v : {' '.join(neg)}")
    log.append("")

    # ---- Test 3: causal sweep on L10H9 SV(0,3) ----------------------
    log.append("[3] Causal alpha-sweep: L10H9 SV(0,3) in-context")
    res = causal_sweep(model, 10, 9, 0, 3,
                       tok_plus_id, tok_minus_id, device, dtype)
    base = res["base"]
    log.append(f"  baseline LD: neutral={base['neutral']:+.3f}  "
               f"masc={base['masc']:+.3f}  fem={base['fem']:+.3f}")
    phis = []
    for a, f in res["fits"].items():
        log.append(f"  alpha={a:5.1f}  A={f['A']:6.3f}  "
                   f"phi={f['phi']:+7.2f} deg  "
                   f"argmax={f['argmax']:5.1f}  "
                   f"argmin={f['argmin']:5.1f}")
        phis.append(f["phi"])
    # phase drift across alphas
    phi_arr = np.array(phis)
    dphi = float(np.ptp(((phi_arr - phi_arr[0] + 180) % 360) - 180))
    log.append(f"  phase drift across alpha = {dphi:.2f} deg")

    # quick compare row for the paper
    res7 = causal_sweep(model, 9, 7, 1, 2,
                        tok_plus_id, tok_minus_id, device, dtype)
    log.append("")
    log.append("  COMPARE (paper's L9H7 SV(1,2) under same harness)")
    for a, f in res7["fits"].items():
        log.append(f"    alpha={a:5.1f}  A={f['A']:6.3f}  "
                   f"phi={f['phi']:+7.2f} deg")
    log.append("")

    txt = "\n".join(log)
    print(txt)
    (out_dir / "l10h9_tests.txt").write_text(txt)
    print(f"\nSaved helix_usage_validated/l10h9_tests.txt")


if __name__ == "__main__":
    main()
