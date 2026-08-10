"""Baseline comparison for the semantic compass.

Compare the compass plane against four standard feature-direction methods
on the SAME causal alpha-sweep harness used by compass_causal_sweep.py.

Methods
-------
1. compass          : (u_i, u_j) from SVD of W_OV at the target head.
2. linear_probe     : logistic regression on cached residual activations
                      (plus vs minus contexts) at the target layer.
3. actadd           : mean(resid_plus) - mean(resid_minus) at the target
                      layer (Turner et al.).
4. contrastive_grad : gradient of (logit_+ - logit_-) w.r.t. the
                      pre-layer residual, averaged across contrastive
                      prompts.
5. fisher_top2      : top-2 eigenvectors of E[g g^T] where g is the
                      contrastive gradient per prompt.

All 1D directions are paired with a fixed random orthonormal partner
to form a 2D plane; every method then gets the same 36-angle * 3-alpha
sweep into blocks.{layer}.hook_resid_pre.  Every direction is
unit-normalized before injection so magnitudes are comparable.

Usage
-----
python experiments/baseline_comparison.py \
    --model gpt2 --layer 9 --head 7 --dims 1 2 \
    --tok_plus " he" --tok_minus " she" \
    --prompt_neutral "The person said that" \
    --prompt_neutral "Then they said that" \
    --prompt_neutral "Afterwards, the speaker said that" \
    --prompt_plus  "The man laced up his boots because" \
    --prompt_plus  "The father waved to the crowd and" \
    --prompt_plus  "The king announced that" \
    --prompt_minus "The woman laced up her boots because" \
    --prompt_minus "The mother waved to the crowd and" \
    --prompt_minus "The queen announced that" \
    --out_prefix baselines_gpt2
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer

warnings.filterwarnings("ignore")


# ---------- shared helpers --------------------------------------------------


def unit(x: torch.Tensor) -> torch.Tensor:
    n = torch.linalg.norm(x)
    return x if n < 1e-12 else x / n


def orth_partner(v: torch.Tensor, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(v.shape, generator=g, dtype=torch.float32)
    r = r - (r @ v) * v
    return unit(r)


def fit_sin(curve: np.ndarray, angles_deg: np.ndarray):
    th = np.radians(angles_deg)
    c = (curve * np.cos(th)).sum() * 2.0 / len(angles_deg)
    s = (curve * np.sin(th)).sum() * 2.0 / len(angles_deg)
    amp = float(np.hypot(c, s))
    phase = float(np.degrees(np.arctan2(s, c)))
    return float(curve.mean()), amp, phase


# ---------- direction extractors -------------------------------------------


def compass_dirs(model, layer, head, d1, d2):
    W_V = model.W_V[layer, head].detach().float().cpu()
    W_O = model.W_O[layer, head].detach().float().cpu()
    _U, S, Vt = torch.linalg.svd(W_V @ W_O, full_matrices=False)
    return unit(Vt[d1, :]), unit(Vt[d2, :])


def cache_resid(model, prompts, layer):
    hook = f"blocks.{layer}.hook_resid_pre"
    out = []
    for p in prompts:
        toks = model.to_tokens(p)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                toks, names_filter=[hook])
        out.append(cache[hook][0, -1, :].detach().float().cpu())
    return torch.stack(out)


def probe_dir(resid_plus, resid_minus, steps=400, lr=0.05):
    X = torch.cat([resid_plus, resid_minus], dim=0)
    y = torch.cat([torch.ones(len(resid_plus)),
                   torch.zeros(len(resid_minus))])
    d = X.shape[1]
    w = torch.zeros(d, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        logits = X @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y)
        loss.backward()
        opt.step()
    return unit(w.detach())


def actadd_dir(resid_plus, resid_minus):
    return unit(resid_plus.mean(0) - resid_minus.mean(0))


def contrastive_grads(model, prompts, layer, tok_plus_id, tok_minus_id):
    hook = f"blocks.{layer}.hook_resid_pre"
    grads = []
    for p in prompts:
        toks = model.to_tokens(p)
        captured = {}

        def fwd(resid, hook):
            resid.requires_grad_(True)
            resid.retain_grad()
            captured["r"] = resid
            return resid

        model.reset_hooks()
        model.add_hook(hook, fwd)
        try:
            logits = model(toks)
            diff = (logits[0, -1, tok_plus_id]
                    - logits[0, -1, tok_minus_id])
            diff.backward()
            g = captured["r"].grad[0, -1, :].detach().float().cpu()
        finally:
            model.reset_hooks()
        grads.append(g)
    return torch.stack(grads)


def fisher_top2(grads):
    G = grads - grads.mean(0, keepdim=True)
    C = G.T @ G / max(len(G) - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(C)
    v2 = unit(eigvecs[:, -1])
    v1 = unit(eigvecs[:, -2])
    return v1, v2


# ---------- sweep driver ---------------------------------------------------


def run_sweep(model, layer, prompts_neutral, tok_plus_id, tok_minus_id,
              e1, e2, alphas, n_angles, dtype, device, scale=1.0):
    hook = f"blocks.{layer}.hook_resid_pre"
    e1 = unit(e1.to(torch.float32))
    e2 = unit(e2.to(torch.float32))
    e1_d = e1.to(dtype).to(device)
    e2_d = e2.to(dtype).to(device)
    angles = np.linspace(0, 360, n_angles, endpoint=False)

    def ld(prompt, edit):
        toks = model.to_tokens(prompt)
        with torch.no_grad():
            if edit is None:
                logits = model(toks)
            else:
                def _hook(resid, hook):
                    resid[0, -1, :] = resid[0, -1, :] + edit
                    return resid
                logits = model.run_with_hooks(
                    toks, fwd_hooks=[(hook, _hook)])
        last = logits[0, -1, :]
        return float(last[tok_plus_id] - last[tok_minus_id])

    def mean_ld(edit):
        return float(np.mean([ld(p, edit) for p in prompts_neutral]))

    per_alpha = {}
    for a in alphas:
        curve = []
        for deg in angles:
            th = np.radians(deg)
            vec = (a * scale * np.cos(th)) * e1_d + \
                  (a * scale * np.sin(th)) * e2_d
            curve.append(mean_ld(vec))
        curve = np.array(curve)
        mu, amp, phase = fit_sin(curve, angles)
        per_alpha[a] = dict(mu=mu, amp=amp, phase=phase, curve=curve)
    return per_alpha, angles


# ---------- main ----------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
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
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    dtype = torch.bfloat16
    print(f"Loading {args.model} on {device}...")
    model = HookedTransformer.from_pretrained(
        args.model, device=device, dtype=dtype)
    layer, head = args.layer, args.head
    d1, d2 = args.dims

    tok_plus_id = model.to_single_token(args.tok_plus)
    tok_minus_id = model.to_single_token(args.tok_minus)

    # ---- extract five direction sets on CPU float32 ----
    print("Extracting compass directions (SVD of W_OV)...")
    c1, c2 = compass_dirs(model, layer, head, d1, d2)

    print("Caching residuals for probe / actadd...")
    resid_plus = cache_resid(model, args.prompt_plus, layer)
    resid_minus = cache_resid(model, args.prompt_minus, layer)

    print("Training linear probe...")
    w_probe = probe_dir(resid_plus, resid_minus)

    print("Computing ActAdd mean-diff...")
    w_act = actadd_dir(resid_plus, resid_minus)

    print("Collecting contrastive gradients...")
    all_prompts = (args.prompt_plus + args.prompt_minus
                   + args.prompt_neutral)
    grads = contrastive_grads(
        model, all_prompts, layer, tok_plus_id, tok_minus_id)
    w_grad = unit(grads.mean(0))
    f1, f2 = fisher_top2(grads)

    methods = {
        "compass":          (c1, c2),
        "linear_probe":     (w_probe, orth_partner(w_probe, args.seed)),
        "actadd":           (w_act,   orth_partner(w_act, args.seed + 1)),
        "contrastive_grad": (w_grad,  orth_partner(w_grad, args.seed + 2)),
        "fisher_top2":      (f1, f2),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    log = [
        f"Model: {args.model}   Layer {layer}  Head {head}  "
        f"dims ({d1},{d2})",
        f"tokens: {args.tok_plus!r}={tok_plus_id}  "
        f"{args.tok_minus!r}={tok_minus_id}",
        f"alphas: {args.alphas}   angles: {args.n_angles}",
        "",
        "All directions unit-normalized; 1D directions paired with "
        "a fixed random orthonormal partner.",
        "",
        f"{'method':<18} {'A(1)':>8} {'A(3)':>8} {'A(10)':>8} "
        f"{'slope':>8} {'phi(1)':>9} {'phi(3)':>9} {'phi(10)':>9} "
        f"{'Δφ':>6}",
        "-" * 96,
    ]

    results = {}
    for name, (e1, e2) in methods.items():
        print(f"\n--- {name} ---")
        per_alpha, _ = run_sweep(
            model, layer, args.prompt_neutral, tok_plus_id,
            tok_minus_id, e1, e2, args.alphas, args.n_angles,
            dtype, device)
        amps = [per_alpha[a]["amp"] for a in args.alphas]
        phases = [per_alpha[a]["phase"] for a in args.alphas]
        slope = amps[-1] / args.alphas[-1] if args.alphas[-1] > 0 else 0.0
        drift = max(phases) - min(phases)
        for a, r in per_alpha.items():
            print(f"  alpha={a}: A={r['amp']:.3f}  "
                  f"phi={r['phase']:+.1f}  mu={r['mu']:+.3f}")
        results[name] = dict(amps=amps, phases=phases, slope=slope,
                             drift=drift)
        log.append(
            f"{name:<18} "
            f"{amps[0]:8.3f} {amps[1]:8.3f} {amps[2]:8.3f} "
            f"{slope:8.3f} "
            f"{phases[0]:+9.1f} {phases[1]:+9.1f} {phases[2]:+9.1f} "
            f"{drift:6.1f}"
        )

    log.append("")
    log.append("Interpretation:")
    log.append("  A(alpha) growing linearly with alpha => causal plane.")
    log.append("  Δφ small across alpha => phase-stable (causal).")
    log.append("  slope = A(10)/10 (direction-pair sensitivity).")
    txt = out_dir / f"{args.out_prefix}.txt"
    txt.write_text("\n".join(log))
    print(f"\nSaved {txt}")


if __name__ == "__main__":
    main()
