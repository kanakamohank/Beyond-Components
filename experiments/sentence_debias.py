"""SentenceDebias (Liang et al. 2020) baseline for gender / race
debiasing on gpt2, phi3, gemma, llama.

Method
------
1. Build a labeled set of sentences from StereoSet intrasentence fills
   for the requested domain (same labels as inlp_debias.py).
2. Cache the last-token residual at one target layer for each sentence.
3. Center each class, stack the centered vectors, take the top-k
   principal components of that matrix. These span the bias subspace V.
4. At inference, apply the residual-stream hook
       h  ->  h - V V^T h
   i.e. orthogonal projection onto the complement of V.
5. Re-run the CrowS-Pairs scoring loop with that hook, log debiased SS
   per domain alongside the baseline. Same evaluator and split as
   inlp_debias.py so tables are directly comparable.

Usage
-----
  python -u experiments/sentence_debias.py --model phi3  --domain gender --layer 24
  python -u experiments/sentence_debias.py --model phi3  --domain race   --layer 24
  python -u experiments/sentence_debias.py --model llama --domain gender --layer 22
  python -u experiments/sentence_debias.py --model llama --domain race   --layer 22
  python -u experiments/sentence_debias.py --model gpt2  --domain gender --layer 10
  python -u experiments/sentence_debias.py --model gpt2  --domain race   --layer 10
  python -u experiments/sentence_debias.py --model gemma --domain gender --layer 21
  python -u experiments/sentence_debias.py --model gemma --domain race   --layer 21

Outputs
-------
  helix_usage_validated/sentdebias_<model>_<domain>_L<layer>.pt
  helix_usage_validated/sentdebias_<model>_<domain>_L<layer>.json
  helix_usage_validated/crowspairs_sentdebias_<model>_<domain>_L<layer>.{csv,txt}
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer

warnings.filterwarnings("ignore")

OUT_DIR = Path("helix_usage_validated")
CROWS_CSV = Path("data/crows_pairs_anonymized.csv")
CROWS_DOMAIN_MAP = {
    "gender": "gender",
    "race-color": "race",
    "religion": "religion",
    "socioeconomic": "profession",
}

MODEL_SPECS = {
    "gpt2":  "gpt2",
    "gemma": "google/gemma-2-2b",
    "phi3":  "microsoft/Phi-3-mini-4k-instruct",
    "llama": "meta-llama/Llama-3.2-3B",
}

MALE_WORDS = {
    "man", "men", "male", "males", "he", "him", "his", "himself",
    "father", "dad", "brother", "son", "husband", "uncle", "nephew",
    "boy", "boys", "guy", "guys", "gentleman", "gentlemen", "sir",
    "mr", "grandfather", "grandpa", "groom", "king", "prince", "actor",
    "waiter", "steward", "host", "lord", "monk", "priest",
}
FEMALE_WORDS = {
    "woman", "women", "female", "females", "she", "her", "hers", "herself",
    "mother", "mom", "sister", "daughter", "wife", "aunt", "niece",
    "girl", "girls", "gal", "gals", "lady", "ladies", "madam",
    "mrs", "miss", "ms", "grandmother", "grandma", "bride",
    "queen", "princess", "actress", "waitress", "stewardess", "hostess",
    "nun", "nurse",
}
BLACK_WORDS = {
    "black", "african", "nigerian", "ethiopian", "ghanaian", "kenyan",
    "somali", "jamaican", "haitian", "afro",
}
WHITE_WORDS = {
    "white", "european", "caucasian", "british", "irish", "german",
    "swedish", "norwegian", "russian", "polish", "australian",
}


def _keyword_label(text, pos, neg):
    words = {w.lower().strip(".,!?;:") for w in text.split()}
    has_pos = bool(words & pos)
    has_neg = bool(words & neg)
    if has_pos and not has_neg:
        return 1
    if has_neg and not has_pos:
        return 0
    return None


def load_stereoset_examples(domain):
    if domain == "gender":
        pos, neg = MALE_WORDS, FEMALE_WORDS
        bias_type = "gender"
    elif domain == "race":
        pos, neg = BLACK_WORDS, WHITE_WORDS
        bias_type = "race"
    else:
        raise ValueError(domain)
    ds = load_dataset("McGill-NLP/stereoset", "intrasentence",
                      split="validation")
    items = []
    for row in ds:
        if row["bias_type"] != bias_type:
            continue
        ctx = row["context"]
        if "BLANK" not in ctx:
            continue
        for s in row["sentences"]["sentence"]:
            lab = _keyword_label(s, pos, neg)
            if lab is not None:
                items.append((s, lab))
    return items


def cache_residuals(model, items, layer, t0):
    d = model.cfg.d_model
    X = torch.zeros(len(items), d, dtype=torch.float32)
    y = torch.zeros(len(items), dtype=torch.long)
    hook_name = f"blocks.{layer}.hook_resid_pre"
    for i, (text, lab) in enumerate(items):
        toks = model.to_tokens(text)
        with torch.no_grad():
            _logits, cache = model.run_with_cache(
                toks, names_filter=[hook_name])
        resid = cache[hook_name][0, -1, :].detach().float().cpu()
        X[i] = resid
        y[i] = lab
        if (i + 1) % 500 == 0:
            print(f"[{time.time()-t0:6.1f}s]   cached {i+1}/{len(items)}",
                  flush=True)
    return X, y


def compute_bias_subspace(X, y, k=1):
    """Class-centered PCA (Liang 2020): subtract per-class mean, stack,
    take top-k PCs. k=1 recovers the simple mean-diff direction up to
    sign. Returns V with shape (d, k) orthonormal.
    """
    X_np = X.numpy()
    y_np = y.numpy()
    centered = np.zeros_like(X_np)
    for c in (0, 1):
        mask = y_np == c
        if mask.sum() == 0:
            continue
        centered[mask] = X_np[mask] - X_np[mask].mean(axis=0, keepdims=True)
    # PCA via SVD on centered rows
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    V = Vt[:k].T  # (d, k)
    V = V / np.linalg.norm(V, axis=0, keepdims=True)
    return torch.from_numpy(V).float(), S[:k].tolist()


def sentence_logprob(model, text, fwd_hooks):
    tokens = model.to_tokens(text)
    with torch.no_grad():
        if fwd_hooks is None:
            logits = model(tokens)[0]
        else:
            logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)[0]
    logp = torch.log_softmax(logits[:-1].float(), dim=-1)
    targets = tokens[0, 1:]
    return float(logp[torch.arange(targets.shape[0]), targets].sum())


def score_crowspairs(model, V, layer, device, dtype, only_domain, t0):
    """V is (d, k) orthonormal on the bias subspace.  Hook projects the
    residual onto the complement:  h -> h - V V^T h.
    """
    hook_name = f"blocks.{layer}.hook_resid_pre"
    V_dev = V.to(device).to(dtype)

    def proj_hook(r, hook):
        # r: (1, T, d).  r @ V: (1, T, k); @ V^T -> (1, T, d)
        r[0, :, :] = r[0, :, :] - (r[0, :, :] @ V_dev) @ V_dev.T
        return r

    rows_all = []
    with CROWS_CSV.open() as f:
        for row in csv.DictReader(f):
            if row["bias_type"] in CROWS_DOMAIN_MAP:
                rows_all.append(row)
    by_dom = {d: [] for d in ("gender", "race", "profession", "religion")}
    for r in rows_all:
        by_dom[CROWS_DOMAIN_MAP[r["bias_type"]]].append(r)

    runs = []
    for cond_name, fwd in [("baseline", None),
                           ("sentdebias", [(hook_name, proj_hook)])]:
        total_pro = total_n = 0
        print(f"[{time.time()-t0:6.1f}s] scoring {cond_name} ...", flush=True)
        for d in ("gender", "race", "profession", "religion"):
            if only_domain == "gender" and d not in ("gender", "profession"):
                continue
            if only_domain == "race" and d != "race":
                continue
            rows = by_dom[d]
            pro = tot = skip = 0
            for ex in rows:
                try:
                    lp_m = sentence_logprob(model, ex["sent_more"], fwd)
                    lp_l = sentence_logprob(model, ex["sent_less"], fwd)
                except Exception:
                    skip += 1
                    continue
                direction = ex.get("stereo_antistereo", "stereo")
                stereo_is_more = direction == "stereo"
                if stereo_is_more and lp_m > lp_l:
                    pro += 1
                if (not stereo_is_more) and lp_l > lp_m:
                    pro += 1
                tot += 1
            ss = 100.0 * pro / max(1, tot)
            runs.append(dict(condition=cond_name, domain=d, ss=ss, n=tot,
                             skipped=skip))
            total_pro += pro
            total_n += tot
            print(f"    {cond_name}  {d:12s}  SS={ss:6.2f}  n={tot}",
                  flush=True)
        overall = 100.0 * total_pro / max(1, total_n)
        runs.append(dict(condition=cond_name, domain="overall",
                         ss=overall, n=total_n, skipped=0))
        print(f"    {cond_name}  overall       SS={overall:6.2f}  "
              f"n={total_n}", flush=True)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_SPECS.keys()), required=True)
    ap.add_argument("--domain", choices=["gender", "race"], required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--k", type=int, default=1,
                    help="PCA rank of the bias subspace (Liang default = 1).")
    args = ap.parse_args()

    t0 = time.time()
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    dtype = torch.bfloat16
    hf_name = MODEL_SPECS[args.model]
    print(f"[{time.time()-t0:6.1f}s] Loading {hf_name} on {device}",
          flush=True)
    model = HookedTransformer.from_pretrained(
        hf_name, device=device, dtype=dtype)

    print(f"[{time.time()-t0:6.1f}s] Loading StereoSet domain={args.domain}",
          flush=True)
    items = load_stereoset_examples(args.domain)
    print(f"[{time.time()-t0:6.1f}s]   {len(items)} labeled examples",
          flush=True)

    print(f"[{time.time()-t0:6.1f}s] Caching residuals at L{args.layer}",
          flush=True)
    X, y = cache_residuals(model, items, args.layer, t0)

    print(f"[{time.time()-t0:6.1f}s] Computing bias subspace (k={args.k})",
          flush=True)
    V, sigmas = compute_bias_subspace(X, y, k=args.k)
    print(f"[{time.time()-t0:6.1f}s]   top singular values: "
          f"{[f'{s:.2f}' for s in sigmas]}", flush=True)

    OUT_DIR.mkdir(exist_ok=True)
    tag = f"{args.model}_{args.domain}_L{args.layer}"
    pt_path = OUT_DIR / f"sentdebias_{tag}.pt"
    torch.save({"V": V, "sigmas": sigmas, "layer": args.layer,
                "model": hf_name, "domain": args.domain, "k": args.k},
               pt_path)
    json_path = OUT_DIR / f"sentdebias_{tag}.json"
    json_path.write_text(json.dumps({
        "model": hf_name, "domain": args.domain, "layer": args.layer,
        "k": args.k, "subspace_singular_values": sigmas,
        "n_items": len(items),
    }, indent=2))
    print(f"[{time.time()-t0:6.1f}s] wrote {pt_path}  {json_path}",
          flush=True)

    print(f"[{time.time()-t0:6.1f}s] Scoring CrowS-Pairs under SentenceDebias "
          f"hook", flush=True)
    runs = score_crowspairs(model, V, args.layer, device, dtype,
                            args.domain, t0)

    csv_path = OUT_DIR / f"crowspairs_sentdebias_{tag}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "domain", "ss",
                                          "n", "skipped"])
        w.writeheader()
        for r in runs:
            w.writerow(r)
    txt_path = OUT_DIR / f"crowspairs_sentdebias_{tag}.txt"
    lines = [f"Model: {hf_name}",
             f"SentenceDebias: domain={args.domain}  layer={args.layer}  "
             f"k={args.k}",
             f"Subspace singular values: "
             f"{['%.2f' % s for s in sigmas]}",
             "",
             f"{'cond':<12} {'domain':<12} {'SS':>7} {'n':>6}",
             "-" * 40]
    for r in runs:
        lines.append(f"{r['condition']:<12} {r['domain']:<12} "
                     f"{r['ss']:>7.2f} {r['n']:>6}")
    txt_path.write_text("\n".join(lines))
    print(f"[{time.time()-t0:6.1f}s] wrote {csv_path}  {txt_path}",
          flush=True)


if __name__ == "__main__":
    main()
