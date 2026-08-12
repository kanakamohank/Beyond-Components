"""Four-way harmonic probe for compass planes.

Injects v(theta, alpha) = alpha * (sigma_i * cos(theta) * u_i +
                                    sigma_j * sin(theta) * u_j)
into the residual stream and records logits of FOUR probe tokens
nominally placed at 0, 90, 180, 270 degrees on the compass.

The fitted response is
    f(theta) = <logit-vector, 4-way basis>(theta)
where we use the k=1 and k=2 DFT amplitudes of
    g(theta) = logit(t_0) - logit(t_180)
    h(theta) = logit(t_90) - logit(t_270)
and directly the 2-harmonic fit of the combined score
    s(theta) = logit(t_0) cos(0) + logit(t_90) cos(-90)
             + logit(t_180) cos(-180) + logit(t_270) cos(-270)
equivalently  s = (l_0 - l_180) = g.

We report per-alpha:
  A1, phi1 : first-harmonic amplitude/phase of g(theta)
  A2, phi2 : second-harmonic amplitude/phase of g(theta)
  B1, B2   : same harmonics measured from h(theta), 90 deg-shifted axis
  A2 / A1  : four-way structure indicator

Usage:
    python experiments/fourway_probe.py \
        --model microsoft/Phi-3-mini-4k-instruct \
        --layer 24 --head 17 --dims 1 2 \
        --tokens month upon hour year \
        --prompt "The event took about one " \
        --prompt "They waited nearly one " \
        --prompt "The project finished within one " \
        --out_prefix phi3_temporal_fourway
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer

warnings.filterwarnings("ignore")


def dft_harmonic(y, angles_deg, k):
    th = np.radians(angles_deg)
    N = len(th)
    c = (y * np.cos(k * th)).sum() * 2.0 / N
    s = (y * np.sin(k * th)).sum() * 2.0 / N
    return float(np.hypot(c, s)), float(np.degrees(np.arctan2(s, c)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--head", type=int, required=True)
    ap.add_argument("--dims", type=int, nargs=2, required=True)
    ap.add_argument("--tokens", nargs=4, required=True,
                    help="Four tokens at 0, 90, 180, 270 deg")
    ap.add_argument("--prompt", action="append", required=True)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[1.0, 3.0, 10.0])
    ap.add_argument("--n_angles", type=int, default=36)
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

    L, H = args.layer, args.head
    d1, d2 = args.dims
    W_V = model.W_V[L, H].detach().float().cpu()
    W_O = model.W_O[L, H].detach().float().cpu()
    _U, S, Vt = torch.linalg.svd(W_V @ W_O, full_matrices=False)
    u1 = Vt[d1, :].to(device).to(dtype)
    u2 = Vt[d2, :].to(device).to(dtype)
    sig1, sig2 = float(S[d1]), float(S[d2])
    print(f"L{L}H{H} plane (SV{d1},SV{d2}): "
          f"sigma={sig1:.3f},{sig2:.3f}")

    tok_ids = [model.to_single_token(t) for t in args.tokens]
    print("Probe tokens:")
    for i, (t, tid) in enumerate(zip(args.tokens, tok_ids)):
        print(f"  {i * 90:3d} deg: {t!r} = {tid}")

    hook = f"blocks.{L}.hook_resid_pre"

    def logits_at(prompt, edit):
        toks = model.to_tokens(prompt)
        with torch.no_grad():
            if edit is None:
                out = model(toks)
            else:
                def _h(r, hook):
                    r[0, -1, :] = r[0, -1, :] + edit
                    return r
                out = model.run_with_hooks(
                    toks, fwd_hooks=[(hook, _h)])
        return out[0, -1, :].detach().float().cpu().numpy()

    angles = np.linspace(0, 360, args.n_angles, endpoint=False)
    log = [
        f"Model: {args.model}   L{L}H{H} SV{d1},{d2}  "
        f"sigma={sig1:.3f},{sig2:.3f}",
        f"tokens: "
        + "  ".join(f"{t}={tid}" for t, tid in
                    zip(args.tokens, tok_ids)),
        f"prompts: {len(args.prompt)}",
        "",
        f"{'alpha':>6} "
        f"{'g_A1':>8} {'g_phi1':>8} {'g_A2':>8} {'g_phi2':>8} "
        f"{'g_A2/A1':>9} "
        f"{'h_A1':>8} {'h_phi1':>8} {'h_A2':>8} {'h_phi2':>8} "
        f"{'h_A2/A1':>9}",
        "-" * 110,
    ]
    per_alpha_curves = {}
    for alpha in args.alphas:
        print(f"\nSWEEP alpha={alpha}")
        g = np.zeros(len(angles))
        h = np.zeros(len(angles))
        for i, deg in enumerate(angles):
            th = np.radians(deg)
            vec = (alpha * sig1 * np.cos(th)) * u1 + \
                  (alpha * sig2 * np.sin(th)) * u2
            logs_0 = np.mean([logits_at(p, vec)[tok_ids[0]]
                              for p in args.prompt])
            logs_90 = np.mean([logits_at(p, vec)[tok_ids[1]]
                               for p in args.prompt])
            logs_180 = np.mean([logits_at(p, vec)[tok_ids[2]]
                                for p in args.prompt])
            logs_270 = np.mean([logits_at(p, vec)[tok_ids[3]]
                                for p in args.prompt])
            g[i] = logs_0 - logs_180
            h[i] = logs_90 - logs_270
        per_alpha_curves[alpha] = (angles.copy(), g.copy(), h.copy())

        gA1, gphi1 = dft_harmonic(g, angles, 1)
        gA2, gphi2 = dft_harmonic(g, angles, 2)
        hA1, hphi1 = dft_harmonic(h, angles, 1)
        hA2, hphi2 = dft_harmonic(h, angles, 2)
        gratio = gA2 / gA1 if gA1 > 1e-9 else float("nan")
        hratio = hA2 / hA1 if hA1 > 1e-9 else float("nan")
        print(f"  g(0-180): A1={gA1:.3f} @ {gphi1:+.1f}  "
              f"A2={gA2:.3f} @ {gphi2:+.1f}  A2/A1={gratio:.3f}")
        print(f"  h(90-270): A1={hA1:.3f} @ {hphi1:+.1f}  "
              f"A2={hA2:.3f} @ {hphi2:+.1f}  A2/A1={hratio:.3f}")
        log.append(
            f"{alpha:>6.1f} "
            f"{gA1:>8.3f} {gphi1:>+8.1f} {gA2:>8.3f} {gphi2:>+8.1f} "
            f"{gratio:>9.3f} "
            f"{hA1:>8.3f} {hphi1:>+8.1f} {hA2:>8.3f} {hphi2:>+8.1f} "
            f"{hratio:>9.3f}"
        )

    log.append("")
    log.append("Raw curves per alpha:")
    for a, (ang, g, h) in per_alpha_curves.items():
        log.append(f"  alpha={a}")
        log.append(
            f"    {'deg':>6} {'g=l0-l180':>11} {'h=l90-l270':>12}")
        for deg, gv, hv in zip(ang, g, h):
            log.append(f"    {deg:>6.1f} {gv:>+11.3f} {hv:>+12.3f}")

    Path(args.out_dir).mkdir(exist_ok=True)
    p = Path(args.out_dir) / f"{args.out_prefix}.txt"
    p.write_text("\n".join(log))
    print(f"\nSaved {p}")


if __name__ == "__main__":
    main()
