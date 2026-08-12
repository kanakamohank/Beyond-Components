"""INLP (Iterative Null-space Projection, Ravfogel 2020) baseline for
gender / race debiasing on gpt2 and gemma-2-2b.

Pipeline
--------
1.  Build a labeled classification task from StereoSet intrasentence
    stereo/anti-stereo fills for the requested domain.
2.  Cache the last-token residual at one target layer for each labeled
    sentence.  Cache once per model; INLP iterations never re-forward.
3.  Iteratively train a PyTorch linear classifier on MPS, extract its
    direction w, build the null-space projector P = I - ww^T / ||w||^2,
    apply P to the residual cache, repeat until probe test accuracy
    drops below --target_acc (default 0.55).
4.  Save the product projection P (d x d) and the per-iteration
    accuracies.
5.  Apply P as a residual-stream hook at the target layer and re-run
    the CrowS-Pairs scoring loop, logging the debiased SS per domain
    alongside the baseline.

Usage
-----
  python -u experiments/inlp_debias.py --model gpt2  --domain gender --layer 10
  python -u experiments/inlp_debias.py --model gpt2  --domain race   --layer 10
  python -u experiments/inlp_debias.py --model gemma --domain gender --layer 21
  python -u experiments/inlp_debias.py --model gemma --domain race   --layer 21

Outputs
-------
  helix_usage_validated/inlp_<model>_<domain>_L<layer>.pt
  helix_usage_validated/inlp_<model>_<domain>_L<layer>.json
  helix_usage_validated/crowspairs_inlp_<model>_<domain>_L<layer>.{csv,txt}
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
import torch.nn as nn
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
    """Return list[(text, label)] with attribute labels.

    gender: 1=male mention, 0=female mention
    race:   1=black mention, 0=white mention
    Sentences mentioning both or neither are skipped.
    """
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


def train_probe(X_tr, y_tr, X_te, y_te, device, max_epochs=200, lr=1e-2):
    d = X_tr.shape[1]
    clf = nn.Linear(d, 1, bias=True).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    X_tr = X_tr.to(device)
    y_tr = y_tr.float().to(device)
    for _ in range(max_epochs):
        opt.zero_grad()
        logits = clf(X_tr).squeeze(-1)
        loss = bce(logits, y_tr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        preds_te = (clf(X_te.to(device)).squeeze(-1) > 0).long().cpu()
        acc = (preds_te == y_te).float().mean().item()
    w = clf.weight.detach().squeeze(0).cpu().float().numpy()
    return w, acc


def inlp_iterate(X, y, device, target_acc=0.55, max_iters=25, split_seed=0,
                 t0=None):
    rng = np.random.default_rng(split_seed)
    N = X.shape[0]
    idx = rng.permutation(N)
    n_tr = int(0.8 * N)
    tr, te = idx[:n_tr], idx[n_tr:]
    X = X.clone()
    y = y.clone()
    d = X.shape[1]
    P = torch.eye(d, dtype=torch.float32)
    accs = []
    for it in range(max_iters):
        X_tr, y_tr = X[tr], y[tr]
        X_te, y_te = X[te], y[te]
        w, acc = train_probe(X_tr, y_tr, X_te, y_te, device)
        accs.append(float(acc))
        tag = "" if t0 is None else f"[{time.time()-t0:6.1f}s] "
        print(f"{tag}  iter {it+1}/{max_iters}  probe_acc={acc:.3f}",
              flush=True)
        if acc < target_acc:
            break
        w_t = torch.from_numpy(w).float()
        w_norm = w_t / (w_t.norm() + 1e-8)
        Pi = torch.eye(d, dtype=torch.float32) - torch.outer(w_norm, w_norm)
        X = X @ Pi
        P = P @ Pi
    return P, accs


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


def score_crowspairs(model, P, layer, device, dtype, only_domain, t0):
    hook_name = f"blocks.{layer}.hook_resid_pre"
    P_dev = P.to(device).to(dtype)

    def proj_hook(r, hook):
        r[0, :, :] = r[0, :, :] @ P_dev
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
                           ("inlp", [(hook_name, proj_hook)])]:
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
    ap.add_argument("--max_iters", type=int, default=25)
    ap.add_argument("--target_acc", type=float, default=0.55)
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

    print(f"[{time.time()-t0:6.1f}s] INLP iterations ...", flush=True)
    P, accs = inlp_iterate(X, y, device, target_acc=args.target_acc,
                           max_iters=args.max_iters, t0=t0)

    OUT_DIR.mkdir(exist_ok=True)
    tag = f"{args.model}_{args.domain}_L{args.layer}"
    pt_path = OUT_DIR / f"inlp_{tag}.pt"
    torch.save({"P": P, "accs": accs, "layer": args.layer,
                "model": hf_name, "domain": args.domain}, pt_path)
    json_path = OUT_DIR / f"inlp_{tag}.json"
    json_path.write_text(json.dumps({
        "model": hf_name, "domain": args.domain, "layer": args.layer,
        "probe_accs_per_iter": accs,
        "final_rank": int(np.linalg.matrix_rank(P.numpy())),
    }, indent=2))
    print(f"[{time.time()-t0:6.1f}s] wrote {pt_path}  {json_path}",
          flush=True)

    print(f"[{time.time()-t0:6.1f}s] Scoring CrowS-Pairs under INLP hook",
          flush=True)
    runs = score_crowspairs(model, P, args.layer, device, dtype,
                            args.domain, t0)

    csv_path = OUT_DIR / f"crowspairs_inlp_{tag}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "domain", "ss",
                                          "n", "skipped"])
        w.writeheader()
        for r in runs:
            w.writerow(r)
    txt_path = OUT_DIR / f"crowspairs_inlp_{tag}.txt"
    lines = [f"Model: {hf_name}",
             f"INLP: domain={args.domain}  layer={args.layer}  "
             f"iters_used={len(accs)}  final_probe_acc={accs[-1]:.3f}",
             f"Probe acc trajectory: {['%.3f' % a for a in accs]}",
             "",
             f"{'cond':<10} {'domain':<12} {'SS':>7} {'n':>6}",
             "-" * 40]
    for r in runs:
        lines.append(f"{r['condition']:<10} {r['domain']:<12} "
                     f"{r['ss']:>7.2f} {r['n']:>6}")
    txt_path.write_text("\n".join(lines))
    print(f"[{time.time()-t0:6.1f}s] wrote {csv_path}  {txt_path}",
          flush=True)


if __name__ == "__main__":
    main()
