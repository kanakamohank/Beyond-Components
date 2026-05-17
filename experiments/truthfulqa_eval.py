"""Capability eval: TruthfulQA MC1 under compass ensemble injection.

Same ensemble heads we selected on StereoSet scans -> evaluated on
TruthfulQA MC1. MC1 = argmax log-likelihood over the answer choices,
with exactly one correct answer per question (so baseline is accuracy).

We compare:
  - baseline (no hooks)
  - ensemble at user-specified alpha(s) (default = each model's best-alpha
    from StereoSet)

A capability drop of >3-5 pts at our debiasing alpha would be a red flag.

Usage:
  python -u experiments/truthfulqa_eval.py --model gpt2  --alphas 1.5
  python -u experiments/truthfulqa_eval.py --model gemma --alphas 20.0
  python -u experiments/truthfulqa_eval.py --model phi3  --alphas 10.0
  python -u experiments/truthfulqa_eval.py --model llama --alphas 20.0

Outputs:
  helix_usage_validated/truthfulqa_{tag}.csv
  helix_usage_validated/truthfulqa_{tag}.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
import warnings
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer

warnings.filterwarnings("ignore")

OUT_DIR = Path("helix_usage_validated")
SCAN_DIR = OUT_DIR

PRONOUNS = {"he", "she", "her", "his", "him", "hers", "himself", "herself",
            "He", "She", "His", "Her", "Him", "Hers"}

MODEL_SPECS = {
    "gpt2":  ("gpt2",                        "gpt2"),
    "gemma": ("google/gemma-2-2b",           "gemma"),
    "phi3":  ("microsoft/Phi-3-mini-4k-instruct", "phi3"),
    "llama": ("meta-llama/Llama-3.2-3B",     "llama"),
}

DEFAULT_ALPHAS = {
    "gpt2":  "1.5",
    "gemma": "20.0",
    "phi3":  "10.0",
    "llama": "20.0",
}


def load_scan_rows(scan_tag: str):
    path = SCAN_DIR / f"stereoset_scan_{scan_tag}.jsonl"
    return [json.loads(l) for l in path.open()]


def pick_ensemble(scan_rows, K=4):
    passing = Counter()
    for r in scan_rows:
        if r["passed"]:
            passing[(r["L"], r["H"])] += 1

    chosen = [lh for lh, _ in passing.most_common(K)]
    fallback_used = False
    if len(chosen) < K:
        fallback_used = True
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
        non_pron = [r for r in cands
                    if r["stereo"] not in PRONOUNS
                    and r["anti"] not in PRONOUNS]
        pool = non_pron or cands
        pool.sort(key=lambda r: r["amp_slope"], reverse=True)
        best.append(pool[0])
    return best, fallback_used


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


def choice_logprob(model, prompt: str, choice: str, fwd_hooks):
    """Log-prob of `choice` tokens conditioned on `prompt`.

    We tokenize (prompt + " " + choice) and sum log-probs over the
    choice-only tokens so we don't penalize longer prompts.
    """
    full = prompt.rstrip() + " " + choice.strip()
    prompt_toks = model.to_tokens(prompt.rstrip())
    full_toks = model.to_tokens(full)
    n_prompt = prompt_toks.shape[1]
    if full_toks.shape[1] <= n_prompt:
        return float("-inf")
    with torch.no_grad():
        if fwd_hooks is None:
            logits = model(full_toks)[0]
        else:
            logits = model.run_with_hooks(full_toks, fwd_hooks=fwd_hooks)[0]
    log_probs = torch.log_softmax(logits[:-1].float(), dim=-1)
    targets = full_toks[0, 1:]
    lp_per_tok = log_probs[torch.arange(targets.shape[0]), targets]
    # choice tokens start at index (n_prompt-1) in the shifted array
    choice_lp = lp_per_tok[n_prompt - 1:]
    return float(choice_lp.sum())


def mc1_accuracy(model, rows, fwd_hooks):
    correct = total = 0
    skipped = 0
    for ex in rows:
        q = ex["question"]
        try:
            choices = ex["mc1_targets"]["choices"]
            labels = ex["mc1_targets"]["labels"]
        except (KeyError, TypeError):
            skipped += 1
            continue
        if not choices or 1 not in labels:
            skipped += 1
            continue
        prompt = f"Q: {q}\nA:"
        lps = [choice_logprob(model, prompt, c, fwd_hooks) for c in choices]
        pred = int(max(range(len(lps)), key=lambda i: lps[i]))
        gold = int(labels.index(1))
        if pred == gold:
            correct += 1
        total += 1
    acc = 100.0 * correct / max(1, total)
    return dict(acc=acc, n=total, correct=correct, skipped=skipped)


CSV_COLS = ["condition", "alpha", "mc1_acc", "n", "correct", "skipped"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_SPECS.keys()), required=True)
    ap.add_argument("--alphas", default="")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--sanity_n", type=int, default=0)
    args = ap.parse_args()

    hf_name, scan_tag = MODEL_SPECS[args.model]
    alphas_str = args.alphas or DEFAULT_ALPHAS[args.model]
    alphas = [float(a) for a in alphas_str.split(",") if a.strip()]

    t0 = time.time()
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    dtype = torch.bfloat16
    print(f"[{time.time()-t0:6.1f}s] Loading {hf_name} on {device} ...",
          flush=True)
    model = HookedTransformer.from_pretrained(hf_name, device=device,
                                              dtype=dtype)

    scan_rows = load_scan_rows(scan_tag)
    ensemble, fallback = pick_ensemble(scan_rows, K=args.K)
    tag = " [FALLBACK: fewer than K strict passes]" if fallback else ""
    print(f"[{time.time()-t0:6.1f}s] Ensemble (K={args.K}):{tag}", flush=True)
    for h in ensemble:
        print(f"   L{h['L']:>2}H{h['H']:<2} SV({h['d1']},{h['d2']}) "
              f"phi={h['phi_hi']:+.1f} slope={h['amp_slope']:.3f} "
              f"passed={h['passed']}", flush=True)

    print(f"[{time.time()-t0:6.1f}s] Loading TruthfulQA MC1 ...", flush=True)
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    rows_all = [dict(r) for r in ds]
    print(f"[{time.time()-t0:6.1f}s]   {len(rows_all)} rows", flush=True)
    if args.sanity_n > 0:
        rows_all = rows_all[: args.sanity_n]
        print(f"[{time.time()-t0:6.1f}s]   [sanity] -> {len(rows_all)}",
              flush=True)

    runs = []

    print(f"[{time.time()-t0:6.1f}s] baseline ...", flush=True)
    m = mc1_accuracy(model, rows_all, None)
    runs.append(dict(condition="baseline", alpha=0.0,
                     mc1_acc=m["acc"], n=m["n"], correct=m["correct"],
                     skipped=m["skipped"]))
    print(f"    baseline  MC1_acc={m['acc']:6.2f}  n={m['n']}  "
          f"skipped={m['skipped']}", flush=True)

    for a in alphas:
        hooks = build_ensemble_hooks(model, ensemble, a, device, dtype)
        print(f"[{time.time()-t0:6.1f}s] ensemble K={args.K} alpha={a} ...",
              flush=True)
        m = mc1_accuracy(model, rows_all, hooks)
        runs.append(dict(condition=f"ensemble_K{args.K}", alpha=a,
                         mc1_acc=m["acc"], n=m["n"], correct=m["correct"],
                         skipped=m["skipped"]))
        print(f"    ens a={a}  MC1_acc={m['acc']:6.2f}  n={m['n']}  "
              f"skipped={m['skipped']}", flush=True)

    OUT_DIR.mkdir(exist_ok=True)
    csv_path = OUT_DIR / f"truthfulqa_{args.model}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for r in runs:
            w.writerow({k: r.get(k, "") for k in CSV_COLS})
    lines = [
        f"Model: {hf_name}",
        f"Ensemble: K={args.K}  alphas={alphas}"
        + (" [FALLBACK]" if fallback else ""),
        "Ensemble heads:",
    ]
    for h in ensemble:
        lines.append(f"   L{h['L']:>2}H{h['H']:<2} SV({h['d1']},{h['d2']}) "
                     f"phi={h['phi_hi']:+.1f}  slope={h['amp_slope']:.3f}  "
                     f"passed={h['passed']}")
    lines += [
        "",
        f"{'cond':<22} {'alpha':>6} {'MC1_acc':>8} {'n':>6} {'corr':>6}",
        "-" * 52,
    ]
    for r in runs:
        lines.append(
            f"{r['condition']:<22} {r['alpha']:>6.2f} "
            f"{r['mc1_acc']:>8.2f} {r['n']:>6} {r['correct']:>6}")
    (OUT_DIR / f"truthfulqa_{args.model}.txt").write_text("\n".join(lines))
    print(f"[{time.time()-t0:6.1f}s] wrote {csv_path}", flush=True)
    print(f"[{time.time()-t0:6.1f}s] wrote "
          f"{OUT_DIR / f'truthfulqa_{args.model}.txt'}", flush=True)


if __name__ == "__main__":
    main()
