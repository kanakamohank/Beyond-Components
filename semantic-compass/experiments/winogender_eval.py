"""WinoGender evaluation: compass-edit vs ActAdd vs baseline.

Measures whether projecting out the compass head's SV(dim) direction
reduces gender bias on WinoGender more than a generic ActAdd "he - she"
direction, while preserving Wikitext-2 perplexity.

Metrics
-------
  raw_gap:
      mean_r |logit(he) - logit(she)| at the position right before the
      pronoun, over all test sentences.  A pronoun-aware model scores
      high; too-aggressive debiasing suppresses this.

  stereo_corr:
      Pearson corr between BLS percent-male (100 - bls_pct_female) for
      each occupation and the model's mean (logit(he) - logit(she)) on
      test sentences using that occupation, evaluated before the pronoun.
      A model with no occupation bias scores 0; a biased model scores
      positive.

Three intervention conditions
-----------------------------
  baseline : no intervention.
  actadd   : steering direction = mean residual (paired minimal "he/she"
             prompts) at the pronoun position of an *independent* prompt
             set.  Applied as - alpha * <r, d> d at a fixed layer's
             hook_resid_post.  This is the Turner-2023 ActAdd recipe.
  compass  : at L.H, subtract alpha * <r, u> u where u = Vt[svd_dim]
             (the SV(dim) write-direction of W_OV for head H, layer L).

Usage
-----
  .venv/bin/python experiments/winogender_eval.py \\
      --model gpt2 --layer 10 --head 9 --svd_dim 0 \\
      --compass_alpha 1.0 --actadd_alpha 1.0
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


# Paired minimal prompts for ActAdd direction-building.  Each pair
# differs only by the pronoun.  Residual captured at the last token so
# the ActAdd direction encodes the pronoun signal at the edit site.
# No overlap with WinoGender sentences.
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
    """Return ids for single-token variants of 'he' and 'she'.  Tries
    the space-prefixed form first (common), else bare form."""
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
    """Zeroes the alpha-scaled projection of each token's residual onto
    the compass direction, at blocks.{layer}.hook_resid_post."""
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
    """ActAdd-style steering vector: mean (resid on he-prompts) -
    mean (resid on she-prompts) at the final token of each paired prompt,
    captured at blocks.{layer}.hook_resid_post."""
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


def last_logits(model, prefix, fwd_hooks):
    tokens = model.to_tokens(prefix)
    with torch.no_grad():
        if fwd_hooks is None:
            logits = model(tokens)
        else:
            logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
    return logits[0, -1, :]


def prefix_before_pronoun(sentence, pronoun):
    """Return the sentence truncated to end just before the pronoun,
    without a trailing space.  Returns None if the pronoun cannot be
    located with a word boundary."""
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
    """Return raw_gap and stereo_corr for a row set under one edit."""
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
                    n_rows=0, n_occ=0, skipped=skipped)

    raw_gap = float(np.mean(all_abs_gaps))

    xs, ys = [], []
    for occ, gaps in per_occ_gaps.items():
        if occ not in bls_stats:
            continue
        pct_male = 100.0 - bls_stats[occ]["bls_pct_female"]
        xs.append(pct_male)
        ys.append(float(np.mean(gaps)))
    if len(xs) < 3 or np.std(xs) == 0 or np.std(ys) == 0:
        stereo_corr = float("nan")
    else:
        stereo_corr = float(np.corrcoef(xs, ys)[0, 1])

    return dict(raw_gap=raw_gap, stereo_corr=stereo_corr,
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
    """Hard-fail on preconditions that would silently break the eval."""
    assert len(rows) > 0, "no WinoGender rows loaded"
    assert "he" in pronoun_ids and "she" in pronoun_ids
    assert len(bls_stats) >= 58, (
        f"BLS stats only has {len(bls_stats)} occupations; "
        "expected all 60 WinoGender occupations")
    # At least 80% of rows must have resolvable pronoun positions.
    ok = 0
    for row in rows:
        if prefix_before_pronoun(row["sentence"], row["pronoun"]) is not None:
            ok += 1
    frac = ok / len(rows)
    assert frac > 0.8, (
        f"prefix_before_pronoun resolves for only {100*frac:.1f}% of "
        f"rows; check pronoun-location logic")
    # Pronoun tokens must differ.
    assert pronoun_ids["he"][0] != pronoun_ids["she"][0], (
        "tokenizer mapped he and she to the same id")
    print(f"  sanity: {ok}/{len(rows)} rows have resolvable pronouns "
          f"({100*frac:.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--head", type=int, default=9)
    ap.add_argument("--svd_dim", type=int, default=0)
    ap.add_argument("--actadd_alpha", type=float, default=1.0)
    ap.add_argument("--compass_alpha", type=float, default=1.0)
    ap.add_argument("--n_ppl_chunks", type=int, default=16)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--out_prefix", default="winogender_gpt2_l10h9")
    ap.add_argument("--out_dir", default="helix_usage_validated")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    dtype = torch.float32 if args.fp32 else torch.bfloat16
    print(f"Loading {args.model} on {device} (dtype={dtype})...")
    model = HookedTransformer.from_pretrained(
        args.model, device=device, dtype=dtype)

    # Compass direction
    W_V = model.W_V[args.layer, args.head].detach().float().cpu()
    W_O = model.W_O[args.layer, args.head].detach().float().cpu()
    _U, S, Vt = torch.linalg.svd(W_V @ W_O, full_matrices=False)
    u_compass = Vt[args.svd_dim, :].to(device).to(dtype)
    print(f"  compass direction: L{args.layer}H{args.head} "
          f"SV{args.svd_dim} (sigma={float(S[args.svd_dim]):.3f})")

    # Data
    ds = load_winogender()
    rows = [dict(r) for r in ds]
    print(f"  WinoGender rows: {len(rows)}")

    bls_stats = load_bls_stats()
    print(f"  BLS occupations: {len(bls_stats)}")

    pronoun_ids = pronoun_token_ids(model)
    print(f"  pronoun token ids: "
          f"he={pronoun_ids['he']}  she={pronoun_ids['she']}")

    # Sanity checks — abort with assertion if preconditions fail.
    sanity_checks(model, rows, pronoun_ids, bls_stats)

    # ActAdd direction (built from paired minimal prompts, NOT from
    # WinoGender, so no data leak).
    print(f"  building ActAdd direction at layer {args.layer}...")
    actadd_dir = build_actadd_direction(model, args.layer, device).to(dtype)
    actadd_norm = float(actadd_dir.norm())
    compass_norm = float(u_compass.norm())
    print(f"    |d_actadd|={actadd_norm:.3f}   "
          f"|u_compass|={compass_norm:.3f}")
    assert actadd_norm > 1e-6, "ActAdd direction is zero"
    assert compass_norm > 1e-6, "compass direction is zero"
    # Cosine between the two directions, for the record:
    cos = float((actadd_dir / actadd_norm).dot(u_compass / compass_norm))
    print(f"    cos(d_actadd, u_compass) = {cos:+.4f}")

    # Baseline PPL
    print("  baseline WikiText-2 PPL...")
    ppl_base = compute_wikitext_ppl(
        model, device, fwd_hooks=None, n_chunks=args.n_ppl_chunks)
    print(f"    PPL = {ppl_base:.3f}")

    conditions = [
        ("baseline", None),
        ("actadd", make_actadd_hook(
            args.layer, actadd_dir, args.actadd_alpha)),
        ("compass", make_compass_hook(
            args.layer, u_compass, args.compass_alpha)),
    ]

    log = [
        f"Model: {args.model}",
        f"Compass: L{args.layer} H{args.head} SV{args.svd_dim}  "
        f"sigma={float(S[args.svd_dim]):.3f}",
        f"alphas: actadd={args.actadd_alpha}  "
        f"compass={args.compass_alpha}",
        f"WinoGender rows: {len(rows)}",
        f"Baseline PPL (wikitext-2, n_chunks={args.n_ppl_chunks}): "
        f"{ppl_base:.3f}",
        "",
        f"{'condition':<10} {'raw_gap':>8} {'stereo_corr':>12} "
        f"{'ppl':>8} {'ppl_delta':>10} {'n':>5}",
        "-" * 60,
    ]

    for name, edit in conditions:
        print(f"  condition: {name}")
        res = compute_bias(model, rows, pronoun_ids, bls_stats, edit)
        ppl = (ppl_base if edit is None
               else compute_wikitext_ppl(
                   model, device, fwd_hooks=edit,
                   n_chunks=args.n_ppl_chunks))
        log.append(
            f"{name:<10} {res['raw_gap']:>8.3f} "
            f"{res['stereo_corr']:>12.3f} "
            f"{ppl:>8.3f} {ppl - ppl_base:>+10.3f} "
            f"{res['n_rows']:>5}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{args.out_prefix}.txt"
    out.write_text("\n".join(log))
    print("\n".join(log))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
