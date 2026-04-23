"""WinoGender α × head sweep: extends winogender_eval.py.

For one model, runs:
  - baseline (once)
  - compass α-sweep on a primary head
  - actadd at α=1.0 on the primary head (anchor for the sweep)
  - compass + actadd at α=1.0 on extra heads

Adds a second bias metric alongside stereo_corr:
  stereo_delta = mean(logit_gap on male-dominated occs)
               - mean(logit_gap on female-dominated occs)
  (0 = unbiased, large positive = model favors he for male-dom occs)

Produces:
  {out_prefix}.csv  — full matrix
  {out_prefix}.txt  — human-readable summary

Usage (serial per model — MPS contention):
  .venv/bin/python experiments/winogender_sweep.py \\
      --model gpt2 \\
      --primary_layer 10 --primary_head 9 --primary_svd 0 \\
      --extra_heads "9,7,0;11,1,1" \\
      --alphas "0.25,0.5,1.0,1.5,2.0" \\
      --out_prefix winogender_gpt2_sweep
"""
from __future__ import annotations

import argparse
import csv
import math
import warnings
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer

warnings.filterwarnings("ignore")


BLS_STATS_PATH = Path("data/winogender_occupations_stats.tsv")


ACTADD_PROMPTS_HE = [
    "He walked into the room and",
    "He said that",
    "He picked up the phone and",
    "He looked at the sign and",
    "He opened the book and",
    "He sat down at the table and",
    "He smiled and",
    "He thought about it and",
    "He closed the door and",
    "He held the cup and",
    "He stood up and",
    "He nodded and",
]
ACTADD_PROMPTS_SHE = [
    "She walked into the room and",
    "She said that",
    "She picked up the phone and",
    "She looked at the sign and",
    "She opened the book and",
    "She sat down at the table and",
    "She smiled and",
    "She thought about it and",
    "She closed the door and",
    "She held the cup and",
    "She stood up and",
    "She nodded and",
]


def load_bls_stats(path=BLS_STATS_PATH):
    stats = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            stats[row["occupation"]] = dict(
                bls_pct_female=float(row["bls_pct_female"]),
                bergsma_pct_female=float(row["bergsma_pct_female"]),
            )
    return stats


def load_winogender():
    ds = load_dataset("oskarvanderwal/winogender", "all", split="test")
    return ds


def pronoun_token_ids(model):
    ids = {}
    for p in ("he", "she"):
        for variant in (f" {p}", p):
            toks = model.to_tokens(variant, prepend_bos=False)[0]
            if len(toks) == 1:
                ids[p] = (int(toks[0].item()), variant)
                break
        else:
            raise ValueError(
                f"No single-token form of pronoun {p!r} in this tokenizer")
    return ids


def make_compass_hook(layer, u_dir, alpha):
    hook_name = f"blocks.{layer}.hook_resid_post"
    u_dir = u_dir / u_dir.norm()

    def _h(r, hook):
        proj = (r * u_dir).sum(dim=-1, keepdim=True)
        return r - alpha * proj * u_dir

    return [(hook_name, _h)]


def make_actadd_hook(layer, steer_dir, alpha):
    hook_name = f"blocks.{layer}.hook_resid_post"
    steer_dir = steer_dir / steer_dir.norm()

    def _h(r, hook):
        proj = (r * steer_dir).sum(dim=-1, keepdim=True)
        return r - alpha * proj * steer_dir

    return [(hook_name, _h)]


def build_actadd_direction(model, layer, device):
    hook_name = f"blocks.{layer}.hook_resid_post"

    def capture(buf):
        def _h(r, hook):
            buf.append(r[0, -1, :].detach().cpu().float())
            return r
        return _h

    acts_he, acts_she = [], []
    for s in ACTADD_PROMPTS_HE:
        tokens = model.to_tokens(s)
        with torch.no_grad():
            model.run_with_hooks(
                tokens, fwd_hooks=[(hook_name, capture(acts_he))])
    for s in ACTADD_PROMPTS_SHE:
        tokens = model.to_tokens(s)
        with torch.no_grad():
            model.run_with_hooks(
                tokens, fwd_hooks=[(hook_name, capture(acts_she))])

    mu_he = torch.stack(acts_he).mean(0)
    mu_she = torch.stack(acts_she).mean(0)
    return (mu_he - mu_she).to(device)


def get_compass_direction(model, layer, head, svd_dim, device, dtype):
    W_V = model.W_V[layer, head].detach().float().cpu()
    W_O = model.W_O[layer, head].detach().float().cpu()
    _U, S, Vt = torch.linalg.svd(W_V @ W_O, full_matrices=False)
    u = Vt[svd_dim, :].to(device).to(dtype)
    return u, float(S[svd_dim])


def last_logits(model, prefix, fwd_hooks):
    tokens = model.to_tokens(prefix)
    with torch.no_grad():
        if fwd_hooks is None:
            logits = model(tokens)
        else:
            logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
    return logits[0, -1, :]


def prefix_before_pronoun(sentence, pronoun):
    candidates = [f" {pronoun} ", f" {pronoun}.", f" {pronoun},",
                  f" {pronoun}'"]
    best = -1
    for c in candidates:
        idx = sentence.find(c)
        if idx >= 0 and (best < 0 or idx < best):
            best = idx
    if best < 0:
        return None
    return sentence[: best].rstrip()


def compute_bias(model, rows, pronoun_ids, bls_stats, fwd_hooks):
    """Returns raw_gap, stereo_corr, stereo_delta."""
    id_he, _ = pronoun_ids["he"]
    id_she, _ = pronoun_ids["she"]

    per_occ_gaps = {}
    all_abs_gaps = []
    skipped = 0

    for row in rows:
        prefix = prefix_before_pronoun(row["sentence"], row["pronoun"])
        if prefix is None:
            skipped += 1
            continue
        logits = last_logits(model, prefix, fwd_hooks)
        gap = float(logits[id_he] - logits[id_she])
        all_abs_gaps.append(abs(gap))
        per_occ_gaps.setdefault(row["occupation"], []).append(gap)

    if not all_abs_gaps:
        return dict(raw_gap=float("nan"),
                    stereo_corr=float("nan"),
                    stereo_delta=float("nan"),
                    n_rows=0, n_occ=0, skipped=skipped)

    raw_gap = float(np.mean(all_abs_gaps))

    # stereo_corr: Pearson of pct_male vs mean gap across occupations
    xs, ys = [], []
    male_gaps, female_gaps = [], []
    for occ, gaps in per_occ_gaps.items():
        if occ not in bls_stats:
            continue
        pct_female = bls_stats[occ]["bls_pct_female"]
        pct_male = 100.0 - pct_female
        mean_gap = float(np.mean(gaps))
        xs.append(pct_male)
        ys.append(mean_gap)
        if pct_female < 50.0:
            male_gaps.append(mean_gap)
        else:
            female_gaps.append(mean_gap)

    if len(xs) < 3 or np.std(xs) == 0 or np.std(ys) == 0:
        stereo_corr = float("nan")
    else:
        stereo_corr = float(np.corrcoef(xs, ys)[0, 1])

    if male_gaps and female_gaps:
        stereo_delta = float(np.mean(male_gaps) - np.mean(female_gaps))
    else:
        stereo_delta = float("nan")

    return dict(raw_gap=raw_gap,
                stereo_corr=stereo_corr,
                stereo_delta=stereo_delta,
                n_rows=len(all_abs_gaps),
                n_occ=len(per_occ_gaps),
                skipped=skipped)


def compute_wikitext_ppl(model, device, fwd_hooks=None,
                         n_chunks=16, chunk_len=512):
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1",
                          split="validation")
    except Exception as e:
        print(f"  [warn] could not load wikitext-2: {e}")
        return float("nan")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    tokens = model.to_tokens(text)[0]
    total_nll = 0.0
    total_n = 0
    for i in range(n_chunks):
        start = i * chunk_len
        end = start + chunk_len + 1
        if end > tokens.shape[0]:
            break
        chunk = tokens[start:end].unsqueeze(0)
        with torch.no_grad():
            if fwd_hooks is None:
                logits = model(chunk)[0]
            else:
                logits = model.run_with_hooks(
                    chunk, fwd_hooks=fwd_hooks)[0]
        log_probs = torch.log_softmax(logits[:-1], dim=-1)
        targets = chunk[0, 1:]
        nll = -log_probs[torch.arange(targets.shape[0]), targets]
        total_nll += float(nll.sum())
        total_n += targets.shape[0]
    if total_n == 0:
        return float("nan")
    return math.exp(total_nll / total_n)


def sanity_checks(model, rows, pronoun_ids, bls_stats):
    assert len(rows) > 0, "no WinoGender rows loaded"
    assert "he" in pronoun_ids and "she" in pronoun_ids
    assert len(bls_stats) >= 58, (
        f"BLS stats only has {len(bls_stats)} occupations; "
        "expected all 60 WinoGender occupations")
    ok = 0
    for row in rows:
        if prefix_before_pronoun(row["sentence"], row["pronoun"]) is not None:
            ok += 1
    frac = ok / len(rows)
    assert frac > 0.8, (
        f"prefix_before_pronoun resolves for only {100*frac:.1f}% of rows")
    assert pronoun_ids["he"][0] != pronoun_ids["she"][0], (
        "tokenizer mapped he and she to the same id")
    print(f"  sanity: {ok}/{len(rows)} rows have resolvable pronouns "
          f"({100*frac:.1f}%)")


def parse_extra_heads(s: str):
    if not s.strip():
        return []
    out = []
    for triple in s.split(";"):
        triple = triple.strip()
        if not triple:
            continue
        parts = [int(x) for x in triple.split(",")]
        assert len(parts) == 3, (
            f"extra_heads triple must be L,H,SV; got {triple!r}")
        out.append(tuple(parts))
    return out


def parse_alphas(s: str):
    return [float(x) for x in s.split(",") if x.strip()]


CSV_COLS = [
    "condition", "layer", "head", "svd", "alpha", "sigma",
    "raw_gap", "stereo_corr", "stereo_delta",
    "ppl", "ppl_delta", "n_rows", "n_occ", "skipped",
]


def _row(condition, layer, head, svd, alpha, sigma, res, ppl, ppl_delta):
    return dict(
        condition=condition,
        layer=layer, head=head, svd=svd, alpha=alpha, sigma=sigma,
        raw_gap=res.get("raw_gap", float("nan")),
        stereo_corr=res.get("stereo_corr", float("nan")),
        stereo_delta=res.get("stereo_delta", float("nan")),
        ppl=ppl, ppl_delta=ppl_delta,
        n_rows=res.get("n_rows", 0),
        n_occ=res.get("n_occ", 0),
        skipped=res.get("skipped", 0),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--primary_layer", type=int, required=True)
    ap.add_argument("--primary_head", type=int, required=True)
    ap.add_argument("--primary_svd", type=int, default=0)
    ap.add_argument("--extra_heads", default="",
                    help="Semicolon-separated L,H,SV triples")
    ap.add_argument("--alphas", default="0.25,0.5,1.0,1.5,2.0")
    ap.add_argument("--n_ppl_chunks", type=int, default=16)
    ap.add_argument("--sanity_n", type=int, default=0,
                    help="If >0, subset WinoGender rows to N for fast check")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--out_dir", default="helix_usage_validated")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    dtype = torch.float32 if args.fp32 else torch.bfloat16
    print(f"Loading {args.model} on {device} (dtype={dtype})...")
    model = HookedTransformer.from_pretrained(
        args.model, device=device, dtype=dtype)

    ds = load_winogender()
    rows = [dict(r) for r in ds]
    if args.sanity_n > 0:
        rows = rows[: args.sanity_n]
        print(f"  [sanity mode] subset rows to {len(rows)}")
    print(f"  WinoGender rows: {len(rows)}")

    bls_stats = load_bls_stats()
    print(f"  BLS occupations: {len(bls_stats)}")

    pronoun_ids = pronoun_token_ids(model)
    print(f"  pronoun token ids: "
          f"he={pronoun_ids['he']}  she={pronoun_ids['she']}")

    sanity_checks(model, rows, pronoun_ids, bls_stats)

    alphas = parse_alphas(args.alphas)
    extra_heads = parse_extra_heads(args.extra_heads)
    all_heads = [(args.primary_layer, args.primary_head, args.primary_svd)]
    all_heads += extra_heads
    print(f"  alphas: {alphas}")
    print(f"  heads: {all_heads}")

    # Baseline
    print("  baseline WikiText-2 PPL...")
    ppl_base = compute_wikitext_ppl(
        model, device, fwd_hooks=None, n_chunks=args.n_ppl_chunks)
    print(f"    PPL = {ppl_base:.3f}")
    print("  baseline bias...")
    base_res = compute_bias(model, rows, pronoun_ids, bls_stats, None)

    runs = []
    runs.append(_row("baseline", -1, -1, -1, 0.0, float("nan"),
                     base_res, ppl_base, 0.0))

    # Primary head: α-sweep for compass + α=1.0 anchor for ActAdd
    L_pri, H_pri, SV_pri = all_heads[0]
    u_pri, sigma_pri = get_compass_direction(
        model, L_pri, H_pri, SV_pri, device, dtype)
    print(f"  primary compass L{L_pri}H{H_pri} SV{SV_pri} "
          f"sigma={sigma_pri:.3f}")

    print(f"  building ActAdd direction at layer {L_pri}...")
    actadd_pri = build_actadd_direction(model, L_pri, device).to(dtype)
    print(f"    |d_actadd|={float(actadd_pri.norm()):.3f}")

    cos_pri = float(
        (actadd_pri / actadd_pri.norm()).dot(u_pri / u_pri.norm()))
    print(f"    cos(d_actadd, u_compass_primary) = {cos_pri:+.4f}")

    for alpha in alphas:
        print(f"  compass α={alpha} on primary head...")
        hook = make_compass_hook(L_pri, u_pri, alpha)
        res = compute_bias(model, rows, pronoun_ids, bls_stats, hook)
        ppl = compute_wikitext_ppl(
            model, device, hook, n_chunks=args.n_ppl_chunks)
        runs.append(_row("compass", L_pri, H_pri, SV_pri,
                         alpha, sigma_pri, res, ppl, ppl - ppl_base))

    # ActAdd anchor on primary layer
    print(f"  actadd α=1.0 on primary layer L{L_pri}...")
    hook = make_actadd_hook(L_pri, actadd_pri, 1.0)
    res = compute_bias(model, rows, pronoun_ids, bls_stats, hook)
    ppl = compute_wikitext_ppl(
        model, device, hook, n_chunks=args.n_ppl_chunks)
    runs.append(_row("actadd", L_pri, -1, -1, 1.0, float("nan"),
                     res, ppl, ppl - ppl_base))

    # Extra heads at α=1.0 (compass + actadd at that layer)
    for (L, H, SV) in extra_heads:
        print(f"  extra head L{L}H{H} SV{SV}...")
        u, sigma = get_compass_direction(model, L, H, SV, device, dtype)
        if L == L_pri:
            actadd_dir = actadd_pri  # reuse
        else:
            print(f"    building ActAdd direction at layer {L}...")
            actadd_dir = build_actadd_direction(model, L, device).to(dtype)

        hook = make_compass_hook(L, u, 1.0)
        res = compute_bias(model, rows, pronoun_ids, bls_stats, hook)
        ppl = compute_wikitext_ppl(
            model, device, hook, n_chunks=args.n_ppl_chunks)
        runs.append(_row("compass", L, H, SV, 1.0, sigma,
                         res, ppl, ppl - ppl_base))

        hook = make_actadd_hook(L, actadd_dir, 1.0)
        res = compute_bias(model, rows, pronoun_ids, bls_stats, hook)
        ppl = compute_wikitext_ppl(
            model, device, hook, n_chunks=args.n_ppl_chunks)
        runs.append(_row("actadd", L, -1, -1, 1.0, float("nan"),
                         res, ppl, ppl - ppl_base))

    # Outputs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / f"{args.out_prefix}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for r in runs:
            w.writerow({k: r.get(k, "") for k in CSV_COLS})

    lines = [
        f"Model: {args.model}",
        f"Primary compass: L{L_pri} H{H_pri} SV{SV_pri}  "
        f"sigma={sigma_pri:.3f}",
        f"Extra heads: {extra_heads}",
        f"Alphas: {alphas}",
        f"WinoGender rows: {len(rows)}",
        f"Baseline PPL: {ppl_base:.3f}",
        "",
        f"{'cond':<8} {'L':>3} {'H':>3} {'SV':>3} {'alpha':>6} "
        f"{'sigma':>7} {'raw_gap':>8} {'stereo_corr':>12} "
        f"{'stereo_delta':>13} {'ppl':>8} {'ppl_delta':>10} {'n':>5}",
        "-" * 100,
    ]
    for r in runs:
        lines.append(
            f"{r['condition']:<8} {r['layer']:>3} {r['head']:>3} "
            f"{r['svd']:>3} {r['alpha']:>6.2f} "
            f"{r['sigma']:>7.3f} "
            f"{r['raw_gap']:>8.3f} {r['stereo_corr']:>12.3f} "
            f"{r['stereo_delta']:>13.3f} "
            f"{r['ppl']:>8.3f} {r['ppl_delta']:>+10.3f} "
            f"{r['n_rows']:>5}")
    txt_path = out_dir / f"{args.out_prefix}.txt"
    txt_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nSaved {csv_path}")
    print(f"Saved {txt_path}")


if __name__ == "__main__":
    main()
