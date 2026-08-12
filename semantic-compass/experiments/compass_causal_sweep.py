"""Compass-causal alpha sweep + plots.

Replicates the analysis behind phi3_compass_causal_*.png and
gpt2_compass_causal_*.png for an arbitrary (model, layer, head, dims,
token-pair).  For each alpha in {1, 3, 10}, injects
alpha * sigma * (cos theta * u1 + sin theta * u2) into the residual
stream just before the target layer, then measures how
logit(tok_plus - tok_minus) varies with theta across 36 angles.

Usage:
    python experiments/compass_causal_sweep.py \\
        --model google/gemma-2-2b --layer 21 --head 4 --dims 0 2 \\
        --tok_plus " run" --tok_minus " stone" \\
        --prompt_neutral "They decided to" \\
        --prompt_plus "They decided to quickly" \\
        --prompt_minus "They picked up a heavy" \\
        --out_prefix gemma2b_compass_causal
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformer_lens import HookedTransformer

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fp32", action="store_true",
                    help="Load model in fp32 instead of bf16.")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--head", type=int, required=True)
    ap.add_argument("--dims", type=int, nargs=2, required=True)
    ap.add_argument("--tok_plus", required=True)
    ap.add_argument("--tok_minus", required=True)
    ap.add_argument("--prompt_neutral", action="append", required=True)
    ap.add_argument("--prompt_plus", action="append", required=True)
    ap.add_argument("--prompt_minus", action="append", required=True)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[1.0, 3.0, 10.0])
    ap.add_argument("--n_angles", type=int, default=36)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--out_dir", default="helix_usage_validated")
    ap.add_argument("--inject_layer", type=int, default=None,
                    help="Layer index for injection site; default = --layer.")
    ap.add_argument("--inject_site", default="hook_resid_pre",
                    choices=["hook_resid_pre", "hook_resid_mid",
                            "hook_resid_post"],
                    help="Hook site on --inject_layer.")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    print(f"Loading {args.model} on {device}...")
    dtype = torch.float32 if args.fp32 else torch.bfloat16
    model = HookedTransformer.from_pretrained(
        args.model, device=device, dtype=dtype)

    layer, head = args.layer, args.head
    d1, d2 = args.dims

    W_V = model.W_V[layer, head].detach().float().cpu()
    W_O = model.W_O[layer, head].detach().float().cpu()
    W_VO = W_V @ W_O
    _U, S, Vt = torch.linalg.svd(W_VO, full_matrices=False)
    u1 = Vt[d1, :].to(device).to(dtype)
    u2 = Vt[d2, :].to(device).to(dtype)
    sigma1, sigma2 = float(S[d1]), float(S[d2])
    print(f"L{layer}H{head} plane SV{d1},SV{d2}: "
          f"sigma_{d1}={sigma1:.3f}  sigma_{d2}={sigma2:.3f}")

    tok_plus_id = model.to_single_token(args.tok_plus)
    tok_minus_id = model.to_single_token(args.tok_minus)
    print(f"  tokens: {args.tok_plus!r}={tok_plus_id}  "
          f"{args.tok_minus!r}={tok_minus_id}")

    prompts = {
        "neutral": args.prompt_neutral,
        "plus": args.prompt_plus,
        "minus": args.prompt_minus,
    }

    inject_layer = args.inject_layer if args.inject_layer is not None else layer
    hook_name = f"blocks.{inject_layer}.{args.inject_site}"
    print(f"Injection site: {hook_name}")

    def ld(prompt, edit):
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            if edit is None:
                logits = model(tokens)
            else:
                def hook(resid, hook):  # noqa: F811
                    resid[0, -1, :] = resid[0, -1, :] + edit
                    return resid

                logits = model.run_with_hooks(
                    tokens, fwd_hooks=[(hook_name, hook)])
        last = logits[0, -1, :]
        return float(last[tok_plus_id] - last[tok_minus_id])

    def mean_ld(prompt_list, edit):
        return float(np.mean([ld(p, edit) for p in prompt_list]))

    base = {k: mean_ld(v, None) for k, v in prompts.items()}
    print(f"\nBASELINE logit({args.tok_plus} - {args.tok_minus}):")
    for k, v in base.items():
        print(f"  {k:<8}: {v:+.3f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    log_lines = [
        f"Model: {args.model}",
        f"L{layer}H{head} plane (SV{d1}, SV{d2}): "
        f"sigma_{d1}={sigma1:.3f}  sigma_{d2}={sigma2:.3f}",
        f"tokens: {args.tok_plus!r}={tok_plus_id}  "
        f"{args.tok_minus!r}={tok_minus_id}",
        "",
        "BASELINE:",
        *[f"  {k}: {v:+.3f}" for k, v in base.items()],
        "",
    ]

    angles = np.linspace(0, 360, args.n_angles, endpoint=False)
    curves_by_alpha = {}
    fit_info = {}
    for alpha in args.alphas:
        print(f"\nSWEEP alpha={alpha}")
        log_lines.append(f"SWEEP alpha={alpha}")
        log_lines.append(
            f"    deg | {'neutral':>9} {'plus':>9} {'minus':>9}")
        log_lines.append(
            "  ----- | --------- --------- ---------")
        curves = {"neutral": [], "plus": [], "minus": []}
        for deg in angles:
            th = np.radians(deg)
            vec = (alpha * sigma1 * np.cos(th)) * u1 + \
                  (alpha * sigma2 * np.sin(th)) * u2
            for k in curves:
                curves[k].append(mean_ld(prompts[k], vec))
            log_lines.append(
                f"  {deg:5.1f} | {curves['neutral'][-1]:+9.3f} "
                f"{curves['plus'][-1]:+9.3f} "
                f"{curves['minus'][-1]:+9.3f}"
            )
        curves_by_alpha[alpha] = curves

        neutral = np.array(curves["neutral"])
        mean_val = float(neutral.mean())
        theta_rad = np.radians(angles)
        c = (neutral * np.cos(theta_rad)).sum() * 2.0 / len(angles)
        s = (neutral * np.sin(theta_rad)).sum() * 2.0 / len(angles)
        amp = float(np.hypot(c, s))
        phase_deg = float(np.degrees(np.arctan2(s, c)))
        argmax_deg = float(angles[np.argmax(neutral)])
        argmin_deg = float(angles[np.argmin(neutral)])
        fit_info[alpha] = dict(mean=mean_val, amp=amp,
                               phase=phase_deg,
                               argmax=argmax_deg, argmin=argmin_deg)
        log_lines.append(
            f"  fit: amp={amp:.3f}  phase={phase_deg:.1f} deg  "
            f"argmin={argmin_deg:.1f}  argmax={argmax_deg:.1f}"
        )
        log_lines.append("")

    mid_alpha = args.alphas[len(args.alphas) // 2]
    _plot_curves(angles, curves_by_alpha, out_dir,
                 f"{args.out_prefix}_curves.png",
                 args.tok_plus, args.tok_minus)
    _plot_polar(angles, curves_by_alpha[mid_alpha], mid_alpha,
                out_dir, f"{args.out_prefix}_polar.png",
                args.tok_plus, args.tok_minus)
    _plot_linearity(args.alphas, fit_info, out_dir,
                    f"{args.out_prefix}_linearity.png")

    for a, r in fit_info.items():
        print(f"  alpha={a}: amp={r['amp']:.3f}  "
              f"phase={r['phase']:.1f}  "
              f"argmin={r['argmin']:.1f}  argmax={r['argmax']:.1f}")

    txt_path = out_dir / f"{args.out_prefix}.txt"
    txt_path.write_text("\n".join(log_lines))
    print(f"\nSaved {txt_path}")


def _plot_curves(angles, curves_by_alpha, out_dir, name,
                 tok_plus, tok_minus):
    fig, axes = plt.subplots(1, len(curves_by_alpha),
                             figsize=(5 * len(curves_by_alpha), 4),
                             sharey=False)
    if len(curves_by_alpha) == 1:
        axes = [axes]
    for ax, (alpha, c) in zip(axes, curves_by_alpha.items()):
        ax.plot(angles, c["neutral"], "k-o", label="neutral", ms=3)
        ax.plot(angles, c["plus"], "b-s", label=f"{tok_plus}-ctx",
                ms=3)
        ax.plot(angles, c["minus"], "r-^", label=f"{tok_minus}-ctx",
                ms=3)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("injection angle (deg)")
        ax.set_ylabel(f"logit({tok_plus} - {tok_minus})")
        ax.set_title(f"alpha = {alpha}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / name, dpi=120)
    plt.close(fig)
    print(f"  saved {out_dir/name}")


def _plot_polar(angles, curves, alpha, out_dir, name,
                tok_plus, tok_minus):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="polar")
    th = np.radians(angles)

    neutral = np.array(curves["neutral"])
    argmax_deg = float(angles[np.argmax(neutral)])
    argmin_deg = float(angles[np.argmin(neutral)])

    r_max = max(
        max(curves["neutral"]), max(curves["plus"]), max(curves["minus"]))
    r_min = min(
        min(curves["neutral"]), min(curves["plus"]), min(curves["minus"]))
    pad = 0.05 * (r_max - r_min + 1e-9)
    r_lo = r_min - pad
    r_hi = r_max + pad

    half = np.radians(45.0)
    plus_center = np.radians(argmax_deg)
    minus_center = np.radians(argmin_deg)
    th_plus = np.linspace(plus_center - half, plus_center + half, 60)
    th_minus = np.linspace(
        minus_center - half, minus_center + half, 60)
    ax.fill_between(
        th_plus, r_lo, r_hi, color="blue", alpha=0.08,
        label=f"predicts{tok_plus}")
    ax.fill_between(
        th_minus, r_lo, r_hi, color="red", alpha=0.08,
        label=f"predicts{tok_minus}")

    for key, style, label in [
        ("neutral", "k-o", "neutral"),
        ("plus", "b-s", f"{tok_plus}-ctx"),
        ("minus", "r-^", f"{tok_minus}-ctx"),
    ]:
        ax.plot(th, curves[key], style, label=label, ms=3)

    ax.set_ylim(r_lo, r_hi)
    ax.set_title(
        f"Compass dial (alpha={alpha})\n"
        f"radius = logit({tok_plus} - {tok_minus}), "
        f"angle = injection theta\n"
        f"blue sector: compass forces{tok_plus}"
        f"  |  red sector: compass forces{tok_minus}")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0))
    fig.tight_layout()
    fig.savefig(out_dir / name, dpi=120)
    plt.close(fig)
    print(f"  saved {out_dir/name}")


def _plot_linearity(alphas, fit_info, out_dir, name):
    amps = [fit_info[a]["amp"] for a in alphas]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(alphas, amps, "ko-")
    ax.set_xlabel("alpha")
    ax.set_ylabel("fit amplitude")
    ax.set_title("Linearity: amplitude vs alpha")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / name, dpi=120)
    plt.close(fig)
    print(f"  saved {out_dir/name}")


if __name__ == "__main__":
    main()
