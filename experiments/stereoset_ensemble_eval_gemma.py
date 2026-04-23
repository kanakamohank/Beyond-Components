"""Stage 3+4 (Gemma variant): multi-plane ensemble injection on
StereoSet for Gemma-2-2b.

Mirrors experiments/stereoset_ensemble_eval.py but:
  - Loads google/gemma-2-2b.
  - Reads Gemma scan from helix_usage_validated/stereoset_scan_gemma.jsonl
  - Pronoun filter is GPT-2-ish ASCII; fine for English pronouns either way.
  - If fewer than K heads strictly pass the two-pillar test, falls back
    to top-K by amp_slope so ensemble still has K heads. We print a
    warning when fallback is used.
  - Single-head baseline head is L21H4 SV(0,2) (paper primary for Gemma).

Outputs:
  helix_usage_validated/stereoset_ensemble_gemma.csv
  helix_usage_validated/stereoset_ensemble_gemma.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer

warnings.filterwarnings("ignore")


OUT_DIR = Path("helix_usage_validated")
SCAN_PATH = OUT_DIR / "stereoset_scan_gemma.jsonl"
MODEL_NAME = "google/gemma-2-2b"
PRONOUNS = {"he", "she", "her", "his", "him", "hers", "himself", "herself",
            "He", "She", "His", "Her", "Him", "Hers"}


def load_scan_rows():
    return [json.loads(l) for l in SCAN_PATH.open()]


def pick_ensemble(scan_rows, K=4, prefer_non_pronoun_phi=True):
    """Return K heads. First pass: top heads by strict-pass count.
    If fewer than K heads have strict passes, fall back to top-K by
    amp_slope (using the strongest plane per head)."""
    passing = Counter()
    for r in scan_rows:
        if r["passed"]:
            passing[(r["L"], r["H"])] += 1

    chosen = [lh for lh, _ in passing.most_common(K)]
    fallback_used = False
    if len(chosen) < K:
        fallback_used = True
        # include the best-slope row per head (even if not passing),
        # rank heads by their best amp_slope.
        best_by_head = {}
        for r in scan_rows:
            lh = (r["L"], r["H"])
            if lh not in best_by_head or r["amp_slope"] > best_by_head[lh]["amp_slope"]:
                best_by_head[lh] = r
        ranked = sorted(best_by_head.values(),
                        key=lambda r: r["amp_slope"], reverse=True)
        for r in ranked:
            lh = (r["L"], r["H"])
            if lh in chosen:
                continue
            chosen.append(lh)
            if len(chosen) >= K:
                break

    best = []
    for (L, H) in chosen:
        cands = [r for r in scan_rows if r["L"] == L and r["H"] == H]
        if prefer_non_pronoun_phi:
            non_pron = [r for r in cands
                        if r["stereo"] not in PRONOUNS
                        and r["anti"] not in PRONOUNS]
            pool = non_pron or cands
        else:
            pool = cands
        pool.sort(key=lambda r: r["amp_slope"], reverse=True)
        best.append(pool[0])
    return best, fallback_used


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
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--alphas", default="0.5,1.0,1.5")
    ap.add_argument("--domains", default="gender,race,profession,religion")
    ap.add_argument("--sanity_n", type=int, default=0)
    ap.add_argument("--out_prefix", default="stereoset_ensemble_gemma")
    args = ap.parse_args()

    t0 = time.time()
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    dtype = torch.bfloat16
    print(f"[{time.time()-t0:6.1f}s] Loading {MODEL_NAME} ...", flush=True)
    model = HookedTransformer.from_pretrained(
        MODEL_NAME, device=device, dtype=dtype)
    print(f"[{time.time()-t0:6.1f}s]   loaded "
          f"(n_layers={model.cfg.n_layers} n_heads={model.cfg.n_heads})",
          flush=True)

    scan_rows = load_scan_rows()
    ensemble, fallback = pick_ensemble(scan_rows, K=args.K,
                                       prefer_non_pronoun_phi=True)
    tag = " [FALLBACK: fewer than K strict passes]" if fallback else ""
    print(f"[{time.time()-t0:6.1f}s] Ensemble (K={args.K}):{tag}", flush=True)
    for h in ensemble:
        print(f"   L{h['L']:>2}H{h['H']:<2}  SV({h['d1']},{h['d2']})  "
              f"phi={h['phi_hi']:+.1f}  slope={h['amp_slope']:.3f}  "
              f"best-probe='{h['stereo']}/{h['anti']}' ({h['domain']})  "
              f"passed={h['passed']}",
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

    # single-head paper primary for Gemma-2-2b gender
    primary_L, primary_H = 21, 4
    primary_plane = (0, 2)
    prim_rows = [r for r in scan_rows
                 if r["L"] == primary_L
                 and r["H"] == primary_H
                 and r["d1"] == primary_plane[0]
                 and r["d2"] == primary_plane[1]]
    prim_rows.sort(key=lambda r: r["amp_slope"], reverse=True)
    if prim_rows:
        primary_phi = prim_rows[0]["phi_hi"]
    else:
        primary_phi = 0.0
    print(f"[{time.time()-t0:6.1f}s] Single-head baseline: "
          f"L{primary_L}H{primary_H} SV{primary_plane}  "
          f"phi={primary_phi:+.1f}", flush=True)

    runs = []
    print(f"[{time.time()-t0:6.1f}s] Running baseline ...", flush=True)
    for d in domains:
        m = compute_metrics(model, by_domain[d], None)
        runs.append(dict(condition="baseline", alpha=0.0, domain=d,
                         lms=m["lms"], ss=m["ss"], icat=m["icat"],
                         n_rows=len(by_domain[d]),
                         skipped=m["skipped"]))
        print(f"    baseline/{d:12s}  LMS={m['lms']:6.2f}  "
              f"SS={m['ss']:6.2f}  ICAT={m['icat']:6.2f}", flush=True)

    for a in alphas:
        hooks = build_single_compass_hooks(
            model, primary_L, primary_H,
            primary_plane[0], primary_plane[1],
            primary_phi, a, device, dtype)
        print(f"[{time.time()-t0:6.1f}s] Running single-head alpha={a} ...",
              flush=True)
        for d in domains:
            m = compute_metrics(model, by_domain[d], hooks)
            runs.append(dict(
                condition=f"single_L{primary_L}H{primary_H}",
                alpha=a, domain=d,
                lms=m["lms"], ss=m["ss"], icat=m["icat"],
                n_rows=len(by_domain[d]), skipped=m["skipped"]))
            print(f"    single   a={a}  {d:12s}  "
                  f"LMS={m['lms']:6.2f}  SS={m['ss']:6.2f}  "
                  f"ICAT={m['icat']:6.2f}", flush=True)

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

    OUT_DIR.mkdir(exist_ok=True)
    csv_path = OUT_DIR / f"{args.out_prefix}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for r in runs:
            w.writerow({k: r.get(k, "") for k in CSV_COLS})
    lines = [
        f"Model: {MODEL_NAME}",
        f"Ensemble: K={args.K}  alphas={alphas}  domains={domains}"
        + (" [FALLBACK: fewer than K strict passes]" if fallback else ""),
        f"Ensemble heads:",
    ]
    for h in ensemble:
        lines.append(f"   L{h['L']:>2}H{h['H']:<2} SV({h['d1']},{h['d2']}) "
                     f"phi={h['phi_hi']:+.1f}  slope={h['amp_slope']:.3f}  "
                     f"passed={h['passed']}")
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
