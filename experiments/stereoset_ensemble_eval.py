"""Stage 3+4: multi-plane ensemble injection on StereoSet.

Picks top-K (L, H) heads from the Stage-2 scan, retrieves each head's
best passing plane (d1, d2) and fitted phi, then evaluates StereoSet
intrasentence with ADDITIVE injections of
    alpha * sigma_i * cos(theta_k) * u_{d1}
  + alpha * sigma_j * sin(theta_k) * u_{d2}
at each head's own resid_pre hook, summed across the k heads.

theta_k is set to (phi_k + 180 deg): the anti-stereotype pole of each
plane, rotated 180 deg from its fitted stereotype phase.

Baselines reported alongside ensemble:
  - unsteered baseline
  - single-head L10H9 SV0 (paper's storage head for gender)

Outputs:
  helix_usage_validated/stereoset_ensemble_gpt2.csv
  helix_usage_validated/stereoset_ensemble_gpt2.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer

warnings.filterwarnings("ignore")


OUT_DIR = Path("helix_usage_validated")
SCAN_PATH = OUT_DIR / "stereoset_scan_gpt2.jsonl"
PRONOUNS = {"he", "she", "her", "his", "him", "hers", "himself", "herself",
            "He", "She", "His", "Her", "Him", "Hers", "himself", "herself"}


def load_scan_rows():
    return [json.loads(l) for l in SCAN_PATH.open()]


def pick_ensemble(scan_rows, K=4, prefer_non_pronoun_phi=True):
    """Return a list of K dicts: (L, H, d1, d2, sigma_i, sigma_j, phi)
    picked as the best-slope passing plane per head, ordered by total
    pass count across probes. phi is the fitted phi_hi of the chosen
    plane's best (non-pronoun if available) passing probe."""
    counts = Counter()
    for r in scan_rows:
        if r["passed"]:
            counts[(r["L"], r["H"])] += 1
    topK = [lh for lh, _ in counts.most_common(K)]

    best = []
    for (L, H) in topK:
        cands = [r for r in scan_rows
                 if r["passed"] and r["L"] == L and r["H"] == H]
        if prefer_non_pronoun_phi:
            non_pron = [r for r in cands
                        if r["stereo"] not in PRONOUNS
                        and r["anti"] not in PRONOUNS]
            pool = non_pron or cands
        else:
            pool = cands
        pool.sort(key=lambda r: r["amp_slope"], reverse=True)
        best.append(pool[0])
    return best


def sentence_logprob(model, text: str, fwd_hooks):
    tokens = model.to_tokens(text)
    with torch.no_grad():
        if fwd_hooks is None:
            logits = model(tokens)[0]
        else:
            logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)[0]
    log_probs = torch.log_softmax(logits[:-1].float(), dim=-1)
    targets = tokens[0, 1:]
    lp = log_probs[torch.arange(targets.shape[0]), targets]
    return float(lp.sum())


def build_ensemble_hooks(model, ensemble, alpha, device, dtype):
    """For each head in `ensemble`, return a (hook_name, fn) pair that
    adds v = alpha * sigma_i * cos(theta) u_i + alpha * sigma_j * sin(theta) u_j
    to the FINAL-token resid_pre at that head's layer. theta is set to
    (phi_best + 180 deg) to push toward the anti-stereo pole.

    Uses the head-specific W_OV SVD to get u_{d1}, u_{d2}, sigma_i, sigma_j.
    """
    hooks = []
    for h in ensemble:
        L, H = h["L"], h["H"]
        d1, d2 = h["d1"], h["d2"]
        phi = h["phi_hi"]
        theta = math.radians(phi + 180.0)

        W_V = model.W_V[L, H].detach().float().cpu()
        W_O = model.W_O[L, H].detach().float().cpu()
        _U, S, Vt = torch.linalg.svd(W_V @ W_O, full_matrices=False)
        u1 = Vt[d1, :].to(device).to(dtype)
        u2 = Vt[d2, :].to(device).to(dtype)
        s1, s2 = float(S[d1]), float(S[d2])
        vec = (alpha * s1 * float(math.cos(theta))) * u1 \
            + (alpha * s2 * float(math.sin(theta))) * u2
        hook_name = f"blocks.{L}.hook_resid_pre"

        def _hook(r, hook, _v=vec):
            r[0, :, :] = r[0, :, :] + _v
            return r

        hooks.append((hook_name, _hook))
    return hooks


def build_single_compass_hooks(model, L, H, d1, d2, phi, alpha, device, dtype):
    """Single-head compass hook at (L, H, SVpair (d1,d2)) with theta = phi+180."""
    theta = math.radians(phi + 180.0)
    W_V = model.W_V[L, H].detach().float().cpu()
    W_O = model.W_O[L, H].detach().float().cpu()
    _U, S, Vt = torch.linalg.svd(W_V @ W_O, full_matrices=False)
    u1 = Vt[d1, :].to(device).to(dtype)
    u2 = Vt[d2, :].to(device).to(dtype)
    s1, s2 = float(S[d1]), float(S[d2])
    vec = (alpha * s1 * float(math.cos(theta))) * u1 \
        + (alpha * s2 * float(math.sin(theta))) * u2
    hook_name = f"blocks.{L}.hook_resid_pre"

    def _hook(r, hook):
        r[0, :, :] = r[0, :, :] + vec
        return r
    return [(hook_name, _hook)]


def score_example(model, example, fwd_hooks):
    sents = example["sentences"]["sentence"]
    labels = example["sentences"]["gold_label"]
    out = {0: None, 1: None, 2: None}
    for text, lab in zip(sents, labels):
        out[int(lab)] = sentence_logprob(model, text, fwd_hooks)
    return out


def compute_metrics(model, rows, fwd_hooks):
    lms_num = lms_den = ss_num = ss_den = 0
    skipped = 0
    for ex in rows:
        scores = score_example(model, ex, fwd_hooks)
        if any(v is None for v in scores.values()):
            skipped += 1
            continue
        anti, stereo, unrel = scores[0], scores[1], scores[2]
        if stereo > unrel:
            lms_num += 1
        lms_den += 1
        if anti > unrel:
            lms_num += 1
        lms_den += 1
        if stereo > anti:
            ss_num += 1
        ss_den += 1
    lms = 100.0 * lms_num / max(1, lms_den)
    ss = 100.0 * ss_num / max(1, ss_den)
    icat = lms * min(ss, 100 - ss) / 50.0
    return dict(lms=lms, ss=ss, icat=icat,
                lms_num=lms_num, lms_den=lms_den,
                ss_num=ss_num, ss_den=ss_den, skipped=skipped)


CSV_COLS = ["condition", "alpha", "domain", "lms", "ss", "icat",
            "n_rows", "skipped"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4,
                    help="Top-K heads to ensemble.")
    ap.add_argument("--alphas", default="0.5,1.0,1.5",
                    help="Ensemble alphas to try.")
    ap.add_argument("--domains", default="gender,race,profession,religion")
    ap.add_argument("--sanity_n", type=int, default=0)
    ap.add_argument("--out_prefix", default="stereoset_ensemble_gpt2")
    args = ap.parse_args()

    t0 = time.time()
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    dtype = torch.bfloat16
    print(f"[{time.time()-t0:6.1f}s] Loading gpt2 ...", flush=True)
    model = HookedTransformer.from_pretrained("gpt2", device=device, dtype=dtype)
    print(f"[{time.time()-t0:6.1f}s]   loaded", flush=True)

    scan_rows = load_scan_rows()
    ensemble = pick_ensemble(scan_rows, K=args.K, prefer_non_pronoun_phi=True)
    print(f"[{time.time()-t0:6.1f}s] Ensemble (K={args.K}):", flush=True)
    for h in ensemble:
        print(f"   L{h['L']:>2}H{h['H']:<2}  SV({h['d1']},{h['d2']})  "
              f"phi={h['phi_hi']:+.1f}  slope={h['amp_slope']:.3f}  "
              f"best-probe='{h['stereo']}/{h['anti']}' ({h['domain']})",
              flush=True)

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]

    print(f"[{time.time()-t0:6.1f}s] Loading StereoSet ...", flush=True)
    ds = load_dataset("McGill-NLP/stereoset", "intrasentence",
                      split="validation")
    rows_all = [dict(r) for r in ds if r["bias_type"] in domains]
    print(f"[{time.time()-t0:6.1f}s]   {len(rows_all)} rows "
          f"(domains={domains})", flush=True)
    if args.sanity_n > 0:
        rows_all = rows_all[: args.sanity_n]
        print(f"[{time.time()-t0:6.1f}s]   [sanity] -> {len(rows_all)}",
              flush=True)

    by_domain = {d: [r for r in rows_all if r["bias_type"] == d]
                 for d in domains}

    # single-head paper primary for GPT-2 gender
    gpt2_primary_L, gpt2_primary_H = 10, 9
    gpt2_primary_plane = (0, 3)
    # find its fitted phi from scan (best non-pronoun passing probe, if any)
    prim_rows = [r for r in scan_rows
                 if r["passed"] and r["L"] == gpt2_primary_L
                 and r["H"] == gpt2_primary_H
                 and r["d1"] == gpt2_primary_plane[0]
                 and r["d2"] == gpt2_primary_plane[1]]
    prim_rows.sort(key=lambda r: r["amp_slope"], reverse=True)
    if prim_rows:
        gpt2_primary_phi = prim_rows[0]["phi_hi"]
    else:
        gpt2_primary_phi = 0.0
    print(f"[{time.time()-t0:6.1f}s] Single-head baseline: "
          f"L{gpt2_primary_L}H{gpt2_primary_H} SV{gpt2_primary_plane}  "
          f"phi={gpt2_primary_phi:+.1f}", flush=True)

    runs = []
    # baseline
    print(f"[{time.time()-t0:6.1f}s] Running baseline ...", flush=True)
    for d in domains:
        m = compute_metrics(model, by_domain[d], None)
        runs.append(dict(condition="baseline", alpha=0.0, domain=d,
                         lms=m["lms"], ss=m["ss"], icat=m["icat"],
                         n_rows=len(by_domain[d]),
                         skipped=m["skipped"]))
        print(f"    baseline/{d:12s}  LMS={m['lms']:6.2f}  "
              f"SS={m['ss']:6.2f}  ICAT={m['icat']:6.2f}", flush=True)

    # single-head compass at each alpha
    for a in alphas:
        hooks = build_single_compass_hooks(
            model, gpt2_primary_L, gpt2_primary_H,
            gpt2_primary_plane[0], gpt2_primary_plane[1],
            gpt2_primary_phi, a, device, dtype)
        print(f"[{time.time()-t0:6.1f}s] Running single-head alpha={a} ...",
              flush=True)
        for d in domains:
            m = compute_metrics(model, by_domain[d], hooks)
            runs.append(dict(
                condition=f"single_L{gpt2_primary_L}H{gpt2_primary_H}",
                alpha=a, domain=d,
                lms=m["lms"], ss=m["ss"], icat=m["icat"],
                n_rows=len(by_domain[d]), skipped=m["skipped"]))
            print(f"    single   a={a}  {d:12s}  "
                  f"LMS={m['lms']:6.2f}  SS={m['ss']:6.2f}  "
                  f"ICAT={m['icat']:6.2f}", flush=True)

    # ensemble at each alpha
    for a in alphas:
        hooks = build_ensemble_hooks(model, ensemble, a, device, dtype)
        print(f"[{time.time()-t0:6.1f}s] Running ensemble(K={args.K}) "
              f"alpha={a} ...", flush=True)
        for d in domains:
            m = compute_metrics(model, by_domain[d], hooks)
            runs.append(dict(
                condition=f"ensemble_K{args.K}",
                alpha=a, domain=d,
                lms=m["lms"], ss=m["ss"], icat=m["icat"],
                n_rows=len(by_domain[d]), skipped=m["skipped"]))
            print(f"    ens K{args.K} a={a}  {d:12s}  "
                  f"LMS={m['lms']:6.2f}  SS={m['ss']:6.2f}  "
                  f"ICAT={m['icat']:6.2f}", flush=True)

    # write outputs
    OUT_DIR.mkdir(exist_ok=True)
    csv_path = OUT_DIR / f"{args.out_prefix}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for r in runs:
            w.writerow({k: r.get(k, "") for k in CSV_COLS})
    lines = [
        f"Ensemble: K={args.K}  alphas={alphas}  domains={domains}",
        f"Ensemble heads:",
    ]
    for h in ensemble:
        lines.append(f"   L{h['L']:>2}H{h['H']:<2} SV({h['d1']},{h['d2']}) "
                     f"phi={h['phi_hi']:+.1f}")
    lines += [
        "",
        f"{'cond':<22} {'alpha':>6} {'domain':<12} "
        f"{'LMS':>7} {'SS':>7} {'ICAT':>7}",
        "-" * 64,
    ]
    for r in runs:
        lines.append(
            f"{r['condition']:<22} {r['alpha']:>6.2f} {r['domain']:<12} "
            f"{r['lms']:>7.2f} {r['ss']:>7.2f} {r['icat']:>7.2f}")
    txt_path = OUT_DIR / f"{args.out_prefix}.txt"
    txt_path.write_text("\n".join(lines))
    print(f"[{time.time()-t0:6.1f}s] wrote {csv_path}", flush=True)
    print(f"[{time.time()-t0:6.1f}s] wrote {txt_path}", flush=True)


if __name__ == "__main__":
    main()