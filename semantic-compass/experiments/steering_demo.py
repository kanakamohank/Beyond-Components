"""Steering demo: rotate the compass plane to flip pronoun predictions.

Given a compass plane (u_i, u_j) at layer L, head h, and a target
angle theta_tgt, we compute the current coefficient of the final-token
residual on that plane and reinsert it rotated to theta_tgt while
preserving magnitude. This is a genuine in-plane rotation, not an
additive push.

For each prompt:
  1. Cache the original resid_pre at layer L.
  2. Project r onto (u_i, u_j), getting (a, b); compute
     magnitude m = sqrt(a^2 + b^2) and original angle theta_cur.
  3. Replace the plane contribution with m * (cos(theta_tgt) u_i +
     sin(theta_tgt) u_j), i.e.,  r' = r - (a u_i + b u_j) +
     m (cos theta_tgt u_i + sin theta_tgt u_j).
  4. Optionally multiply m by a boost factor (default 1.0).
  5. Forward pass from layer L onward with r' at the last token.

Baseline: run unmodified, report top-1 token.
Steered : rotate to theta_tgt, report top-1 token.

Usage:
    python experiments/steering_demo.py \
        --model google/gemma-2-2b --layer 21 --head 4 --dims 0 2 \
        --theta_he 180 --theta_she 0 \
        --boost 3.0 \
        --out_prefix steering_gemma
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer

warnings.filterwarnings("ignore")


MASC_PROMPTS = [
    "The king rode into battle. When the crowd cheered,",
    "My father picked up the phone. When the caller spoke,",
    "The boy opened his new book. As he started reading,",
    "Grandpa settled into his armchair. Slowly",
    "The waiter set the plate down. Politely",
]

FEM_PROMPTS = [
    "The queen rode into battle. When the crowd cheered,",
    "My mother picked up the phone. When the caller spoke,",
    "The girl opened her new book. As she started reading,",
    "Grandma settled into her armchair. Slowly",
    "The waitress set the plate down. Politely",
]

# For Phi-3 L24H10, the compass is (his vs their) = singular vs
# plural possessive. Use prompts that naturally produce these.
SING_PROMPTS = [
    "The boy reached into",
    "The scientist presented",
    "The old man admired",
    "The artist showed off",
    "The student raised",
]

PLUR_PROMPTS = [
    "The boys reached into",
    "The scientists presented",
    "The old men admired",
    "The artists showed off",
    "The students raised",
]


def top_token(model, logits):
    tid = int(logits.argmax().item())
    return model.tokenizer.decode([tid]).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--head", type=int, required=True)
    ap.add_argument("--dims", type=int, nargs=2, required=True)
    ap.add_argument("--tok_plus", default=" he")
    ap.add_argument("--tok_minus", default=" she")
    ap.add_argument("--theta_plus", type=float, default=180.0,
                    help="Angle that favors tok_plus.")
    ap.add_argument("--theta_minus", type=float, default=0.0,
                    help="Angle that favors tok_minus.")
    ap.add_argument("--boost", type=float, default=1.0,
                    help="Multiply the restored magnitude by this.")
    ap.add_argument("--mode", choices=["rotate", "inject"],
                    default="inject",
                    help="rotate: in-place plane rotation; "
                    "inject: additive alpha*sigma push at target angle.")
    ap.add_argument("--alpha", type=float, default=5.0,
                    help="Injection scale for --mode inject.")
    ap.add_argument("--axis", choices=["gender", "number"],
                    default="gender",
                    help="gender: he/she prompts. "
                    "number: his/their (singular vs plural) prompts.")
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

    hook_name = f"blocks.{L}.hook_resid_pre"
    tok_plus_id = model.to_single_token(args.tok_plus)
    tok_minus_id = model.to_single_token(args.tok_minus)

    def run(prompt, theta_tgt=None):
        toks = model.to_tokens(prompt)
        if theta_tgt is None:
            with torch.no_grad():
                logits = model(toks)
            return logits[0, -1, :].float().cpu()

        def hook(r, hook):
            v = r[0, -1, :]
            th = np.radians(theta_tgt)
            if args.mode == "rotate":
                a = (v * u1).sum()
                b = (v * u2).sum()
                m = torch.sqrt(a * a + b * b) * args.boost
                v_new = v - a * u1 - b * u2 \
                    + (m * float(np.cos(th))) * u1 \
                    + (m * float(np.sin(th))) * u2
            else:  # inject
                push = (args.alpha * sig1 * float(np.cos(th))) * u1 \
                    + (args.alpha * sig2 * float(np.sin(th))) * u2
                v_new = v + push
            r[0, -1, :] = v_new
            return r

        with torch.no_grad():
            logits = model.run_with_hooks(
                toks, fwd_hooks=[(hook_name, hook)])
        return logits[0, -1, :].float().cpu()

    rows = []
    log = [
        f"Model: {args.model}   L{L}H{H} SV{d1},{d2}",
        f"Mode={args.mode}  alpha={args.alpha}  boost={args.boost}",
        f"theta_plus={args.theta_plus} ({args.tok_plus})  "
        f"theta_minus={args.theta_minus} ({args.tok_minus})",
        "",
        f"{'idx':>3} {'ctx':<4} {'base_top1':<14} {'steered_top1':<14} "
        f"{'base_LD':>9} {'steer_LD':>9} {'sign_flip?':>9} "
        f"{'top_flip?':>8}",
        "-" * 86,
    ]

    def process(prompts, ctx_label, theta_tgt):
        top_flipped = 0
        sign_flipped = 0
        for i, p in enumerate(prompts):
            base = run(p, None)
            steer = run(p, theta_tgt)
            btop = top_token(model, base)
            stop = top_token(model, steer)
            b_ld = float(base[tok_plus_id] - base[tok_minus_id])
            s_ld = float(steer[tok_plus_id] - steer[tok_minus_id])
            top_flip = int(
                (ctx_label == "masc" and stop.lower().startswith("she"))
                or (ctx_label == "fem" and stop.lower().startswith("he")
                    and not stop.lower().startswith("her")))
            sign_flip = int(np.sign(b_ld) != np.sign(s_ld)
                            and abs(s_ld) > 0.1)
            top_flipped += top_flip
            sign_flipped += sign_flip
            rows.append(dict(prompt=p, ctx=ctx_label, base_top=btop,
                             steer_top=stop, base_ld=b_ld,
                             steer_ld=s_ld, top_flip=top_flip,
                             sign_flip=sign_flip))
            log.append(
                f"{i:>3} {ctx_label:<4} {btop:<14} {stop:<14} "
                f"{b_ld:>+9.3f} {s_ld:>+9.3f} "
                f"{'YES' if sign_flip else 'no':>9} "
                f"{'YES' if top_flip else 'no':>8}"
            )
        return top_flipped, sign_flipped

    if args.axis == "gender":
        plus_prompts, minus_prompts = MASC_PROMPTS, FEM_PROMPTS
        plus_label, minus_label = "masc", "fem"
    else:
        plus_prompts, minus_prompts = SING_PROMPTS, PLUR_PROMPTS
        plus_label, minus_label = "sing", "plur"

    print(f"\n{plus_label} prompts -> force {args.tok_minus}")
    tm, sm = process(plus_prompts, plus_label, args.theta_minus)
    print(f"\n{minus_label} prompts -> force {args.tok_plus}")
    tf, sf = process(minus_prompts, minus_label, args.theta_plus)

    log.append("")
    n = len(plus_prompts) + len(minus_prompts)
    log.append(
        f"SIGN-FLIP (LD crosses 0): "
        f"{sm}/{len(plus_prompts)} {plus_label}->{minus_label}   "
        f"{sf}/{len(minus_prompts)} {minus_label}->{plus_label}   "
        f"TOTAL {sm + sf}/{n}")
    log.append(
        f"TOP-1 FLIP (argmax switches): "
        f"{tm}/{len(plus_prompts)} {plus_label}->{minus_label}   "
        f"{tf}/{len(minus_prompts)} {minus_label}->{plus_label}   "
        f"TOTAL {tm + tf}/{n}")
    bp = np.mean([r["base_ld"] for r in rows
                  if r["ctx"] == plus_label])
    sp = np.mean([r["steer_ld"] for r in rows
                  if r["ctx"] == plus_label])
    bm = np.mean([r["base_ld"] for r in rows
                  if r["ctx"] == minus_label])
    sm_ = np.mean([r["steer_ld"] for r in rows
                   if r["ctx"] == minus_label])
    log.append(f"{plus_label} avg LD: base={bp:+.3f}  "
               f"steered={sp:+.3f}  delta={sp - bp:+.3f}")
    log.append(f"{minus_label} avg LD: base={bm:+.3f}  "
               f"steered={sm_:+.3f}  delta={sm_ - bm:+.3f}")

    Path(args.out_dir).mkdir(exist_ok=True)
    p = Path(args.out_dir) / f"{args.out_prefix}.txt"
    p.write_text("\n".join(log))
    print("\n".join(log))
    print(f"\nSaved {p}")


if __name__ == "__main__":
    main()
