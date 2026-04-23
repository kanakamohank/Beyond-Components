"""Automated compass scanner (gender axis, single model).

Iterates over all (layer, head, SV-pair) for a given model, injects
v(theta, alpha) = alpha * (sigma_i cos(theta) u_i + sigma_j sin(theta) u_j)
into blocks.{L}.hook_resid_pre on a set of neutral prompts, and fits
LD(theta) = mu + A cos(theta - phi) at two alpha settings.

Score per plane:
  linearity = R^2(A vs alpha, forced through origin)
  amp_slope = A(alpha_hi) / alpha_hi
  phase_drift = |phi(alpha_hi) - phi(alpha_lo)|  (minimum-arc degrees)

Pass criterion (two-pillar, same as Table 1):
  linearity > lin_thresh AND phase_drift < phase_thresh AND
  amp_slope > amp_thresh

Also records the random-plane null for each head (random orthonormal
2D subspace drawn from the head's W_OV top-8 SV subspace) so we can
report how much the score exceeds chance.

Output:
  {out_dir}/{out_prefix}_scan.txt   -- per-plane scores
  {out_dir}/{out_prefix}_top.txt    -- top-K planes by amp_slope among
                                       passing entries
  {out_dir}/{out_prefix}_heatmap.png -- layer x head pass-rate heatmap

Usage (GPT-2, gender):
    python experiments/compass_scan.py --model gpt2 \
        --tok_plus " he" --tok_minus " she" \
        --top_svs 4 --alphas 3 10 --n_angles 12 \
        --out_prefix gpt2_scan_gender
"""
from __future__ import annotations

import argparse
import itertools
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


def fit_sinusoid(angles_deg, y):
    th = np.radians(angles_deg)
    N = len(th)
    mu = y.mean()
    c = ((y - mu) * np.cos(th)).sum() * 2.0 / N
    s = ((y - mu) * np.sin(th)).sum() * 2.0 / N
    A = float(np.hypot(c, s))
    phi = float(np.degrees(np.arctan2(s, c)))
    return mu, A, phi


def circ_diff(a_deg, b_deg):
    d = (a_deg - b_deg + 180.0) % 360.0 - 180.0
    return abs(d)


def logits_at(model, toks, hook_name, edit_vec):
    with torch.no_grad():
        if edit_vec is None:
            out = model(toks)
        else:
            def _h(r, hook):
                r[0, -1, :] = r[0, -1, :] + edit_vec
                return r
            out = model.run_with_hooks(
                toks, fwd_hooks=[(hook_name, _h)])
    return out[0, -1, :].detach().float().cpu()


def score_plane(model, L, u1, u2, sig1, sig2,
                prompts, tok_plus_id, tok_minus_id,
                alphas, angles, device, dtype):
    """Return (lin_r2, amp_slope, phase_drift, amp_lo, amp_hi)."""
    hook_name = f"blocks.{L}.hook_resid_pre"
    tokss = [model.to_tokens(p) for p in prompts]

    amps = []
    phis = []
    for a in alphas:
        lds = np.zeros(len(angles))
        for i, deg in enumerate(angles):
            th = np.radians(deg)
            vec = (a * sig1 * float(np.cos(th))) * u1 \
                + (a * sig2 * float(np.sin(th))) * u2
            ld_sum = 0.0
            for toks in tokss:
                lg = logits_at(model, toks, hook_name, vec)
                ld_sum += float(lg[tok_plus_id] - lg[tok_minus_id])
            lds[i] = ld_sum / len(tokss)
        _, A, phi = fit_sinusoid(angles, lds)
        amps.append(A)
        phis.append(phi)

    amps = np.array(amps)
    if amps.max() < 1e-6:
        lin_r2 = 0.0
    else:
        alpha_arr = np.array(alphas, dtype=float)
        denom = (alpha_arr ** 2).sum()
        slope = (alpha_arr * amps).sum() / denom
        resid = amps - slope * alpha_arr
        ss_res = (resid ** 2).sum()
        ss_tot = (amps ** 2).sum()
        lin_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    amp_slope = float(amps[-1] / alphas[-1])
    phase_drift = float(circ_diff(phis[-1], phis[0]))
    return lin_r2, amp_slope, phase_drift, amps, phis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tok_plus", default=" he")
    ap.add_argument("--tok_minus", default=" she")
    ap.add_argument("--top_svs", type=int, default=4,
                    help="Pairs drawn from top-k SVs of W_OV per head.")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[3.0, 10.0])
    ap.add_argument("--n_angles", type=int, default=12)
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="Subset of layers to scan; default = all.")
    ap.add_argument("--heads", type=int, nargs="*", default=None,
                    help="Subset of heads to scan; default = all.")
    ap.add_argument("--null_seeds", type=int, default=3,
                    help="Random-plane control runs per head.")
    ap.add_argument("--null_mode", default="top4",
                    choices=["top4", "full_ov"],
                    help="top4: random 2D from head's top-k SV subspace "
                         "(strict, same as paper). full_ov: random 2D "
                         "from the full W_OV range (weaker null).")
    ap.add_argument("--lin_thresh", type=float, default=0.95)
    ap.add_argument("--phase_thresh", type=float, default=10.0)
    ap.add_argument("--amp_thresh", type=float, default=0.2)
    ap.add_argument("--top_k", type=int, default=20)
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

    cfg = model.cfg
    n_layers = cfg.n_layers
    n_heads = cfg.n_heads
    print(f"Model: n_layers={n_layers}  n_heads={n_heads}")

    layers = args.layers if args.layers else list(range(n_layers))
    heads = args.heads if args.heads else list(range(n_heads))

    tok_plus_id = model.to_single_token(args.tok_plus)
    tok_minus_id = model.to_single_token(args.tok_minus)

    angles = np.linspace(0, 360, args.n_angles, endpoint=False)
    sv_pairs = list(itertools.combinations(range(args.top_svs), 2))
    n_planes = len(layers) * len(heads) * len(sv_pairs)
    print(f"Planes to scan: {n_planes}  "
          f"(layers={len(layers)}, heads={len(heads)}, "
          f"SV-pairs={len(sv_pairs)})")
    print(f"Forward passes: ~{n_planes * len(args.alphas) * args.n_angles * len(NEUTRAL_PROMPTS):,}")

    results = []
    null_results = []
    rng = np.random.default_rng(0)

    for L in layers:
        for H in heads:
            W_V = model.W_V[L, H].detach().float().cpu()
            W_O = model.W_O[L, H].detach().float().cpu()
            _U, S, Vt = torch.linalg.svd(W_V @ W_O, full_matrices=False)
            top_k = min(args.top_svs, Vt.shape[0])
            V_top = Vt[:top_k, :].transpose(0, 1)  # [d_model, top_k]
            S_top = S[:top_k]

            for (d1, d2) in sv_pairs:
                if d1 >= top_k or d2 >= top_k:
                    continue
                u1 = Vt[d1, :].to(device).to(dtype)
                u2 = Vt[d2, :].to(device).to(dtype)
                sig1, sig2 = float(S[d1]), float(S[d2])
                lin_r2, amp_slope, phase_drift, amps, phis = \
                    score_plane(model, L, u1, u2, sig1, sig2,
                                NEUTRAL_PROMPTS,
                                tok_plus_id, tok_minus_id,
                                args.alphas, angles, device, dtype)
                row = dict(L=L, H=H, d1=d1, d2=d2,
                           sig1=sig1, sig2=sig2,
                           lin_r2=lin_r2, amp_slope=amp_slope,
                           phase_drift=phase_drift,
                           A_lo=float(amps[0]), A_hi=float(amps[-1]),
                           phi_lo=float(phis[0]), phi_hi=float(phis[-1]))
                results.append(row)
                print(f"L{L:>2}H{H:>2} SV({d1},{d2}) "
                      f"A({args.alphas[0]})={amps[0]:5.2f} "
                      f"A({args.alphas[-1]})={amps[-1]:5.2f} "
                      f"slope={amp_slope:5.3f} "
                      f"R2={lin_r2:5.3f} dphi={phase_drift:5.1f}")

            # random-plane null per head. Two modes:
            #  - top4: random 2D inside the head's top-k SV subspace
            #  - full_ov: random 2D inside the full W_OV row space
            if args.null_mode == "full_ov":
                V_full = Vt.transpose(0, 1)  # [d_model, d_head]
                null_basis = V_full
                sig_eff = float(S.mean())
            else:
                null_basis = V_top
                sig_eff = float(S_top.mean())
            null_dim = null_basis.shape[1]
            for seed in range(args.null_seeds):
                rs = rng.normal(size=(null_dim, 2))
                q, _ = np.linalg.qr(rs)
                coef = torch.from_numpy(q).float()
                u_rand = null_basis @ coef  # [d_model, 2]
                u_rand = u_rand / u_rand.norm(dim=0, keepdim=True)
                u1 = u_rand[:, 0].to(device).to(dtype)
                u2 = u_rand[:, 1].to(device).to(dtype)
                lin_r2, amp_slope, phase_drift, amps, phis = \
                    score_plane(model, L, u1, u2, sig_eff, sig_eff,
                                NEUTRAL_PROMPTS,
                                tok_plus_id, tok_minus_id,
                                args.alphas, angles, device, dtype)
                null_results.append(dict(
                    L=L, H=H, seed=seed,
                    lin_r2=lin_r2, amp_slope=amp_slope,
                    phase_drift=phase_drift))

    # scoring
    def passes(r):
        return (r["lin_r2"] >= args.lin_thresh
                and r["phase_drift"] <= args.phase_thresh
                and r["amp_slope"] >= args.amp_thresh)

    passed = [r for r in results if passes(r)]
    passed.sort(key=lambda r: r["amp_slope"], reverse=True)

    null_amp = np.array([r["amp_slope"] for r in null_results])
    null_pass = sum(
        1 for r in null_results
        if r["lin_r2"] >= args.lin_thresh
        and r["phase_drift"] <= args.phase_thresh
        and r["amp_slope"] >= args.amp_thresh)

    Path(args.out_dir).mkdir(exist_ok=True)
    log = [
        f"Model: {args.model}",
        f"Planes scanned: {len(results)}  "
        f"(layers={layers[0]}..{layers[-1]}, heads={heads[0]}..{heads[-1]}, "
        f"SV-pairs from top-{args.top_svs})",
        f"Alphas: {args.alphas}  n_angles={args.n_angles}  "
        f"prompts={len(NEUTRAL_PROMPTS)}",
        f"Thresholds: lin>{args.lin_thresh}  dphi<{args.phase_thresh}  "
        f"amp_slope>{args.amp_thresh}",
        f"Random-plane null mode: {args.null_mode}",
        f"Random-plane null: {len(null_results)} planes "
        f"({args.null_seeds} seeds x {len(layers) * len(heads)} heads) -- "
        f"{null_pass} pass ({100.0*null_pass/max(1,len(null_results)):.1f}%)",
        f"Null amp_slope: mean={null_amp.mean():.3f} "
        f"p95={np.percentile(null_amp, 95):.3f} "
        f"max={null_amp.max():.3f}",
        f"PASSED planes: {len(passed)} / {len(results)} "
        f"({100.0*len(passed)/max(1,len(results)):.1f}%)",
        "",
        f"{'rank':>4} {'L':>3} {'H':>3} {'SV':>7} "
        f"{'A_lo':>6} {'A_hi':>6} {'slope':>6} {'R2':>6} {'dphi':>6} "
        f"{'phi_hi':>7}",
        "-" * 70,
    ]
    for i, r in enumerate(passed[:args.top_k]):
        log.append(
            f"{i+1:>4} {r['L']:>3} {r['H']:>3} "
            f"({r['d1']},{r['d2']}) "
            f"{r['A_lo']:>6.2f} {r['A_hi']:>6.2f} "
            f"{r['amp_slope']:>6.3f} {r['lin_r2']:>6.3f} "
            f"{r['phase_drift']:>6.1f} {r['phi_hi']:>+7.1f}")

    log.append("")
    log.append("All results (sorted by amp_slope desc):")
    all_sorted = sorted(results, key=lambda r: r["amp_slope"],
                        reverse=True)
    log.append(
        f"{'L':>3} {'H':>3} {'SV':>7} "
        f"{'A_lo':>6} {'A_hi':>6} {'slope':>6} {'R2':>6} {'dphi':>6} "
        f"{'pass':>5}")
    for r in all_sorted:
        log.append(
            f"{r['L']:>3} {r['H']:>3} "
            f"({r['d1']},{r['d2']}) "
            f"{r['A_lo']:>6.2f} {r['A_hi']:>6.2f} "
            f"{r['amp_slope']:>6.3f} {r['lin_r2']:>6.3f} "
            f"{r['phase_drift']:>6.1f} "
            f"{'YES' if passes(r) else 'no':>5}")

    p = Path(args.out_dir) / f"{args.out_prefix}_scan.txt"
    p.write_text("\n".join(log))
    print(f"\nSaved {p}")

    # heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        H_mat = np.zeros((n_layers, n_heads))
        for r in results:
            if passes(r):
                H_mat[r["L"], r["H"]] += 1
        fig, ax = plt.subplots(figsize=(max(6, n_heads * 0.25), max(4, n_layers * 0.25)))
        im = ax.imshow(H_mat, aspect="auto", cmap="magma",
                       interpolation="nearest")
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(f"{args.model}: passing planes per (L,H)")
        plt.colorbar(im, ax=ax, label="Count")
        for (i, j), v in np.ndenumerate(H_mat):
            if v > 0:
                ax.text(j, i, int(v), ha="center", va="center",
                        color="white", fontsize=6)
        pp = Path(args.out_dir) / f"{args.out_prefix}_heatmap.png"
        plt.tight_layout()
        plt.savefig(pp, dpi=150)
        print(f"Saved {pp}")
    except Exception as e:
        print(f"(heatmap skipped: {e})")


if __name__ == "__main__":
    main()
