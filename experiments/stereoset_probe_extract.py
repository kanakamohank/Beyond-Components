"""Stage 1: extract (stereo, anti-stereo) probe pairs from StereoSet
intrasentence across gender/race/profession/religion.

For each example we diff the 'stereotype' and 'anti-stereotype' filled
sentences against the BLANK context, take the content-word fill for
each, and --- where both sides tokenize to a single GPT-2 token (tried
with and without a leading space) --- record the pair. We also print
examples where exactly one side is single-token so you can eyeball
whether a synonym / morphological variant on the other side could be
swapped in.

Outputs:
  helix_usage_validated/stereoset_probe_pairs.tsv
  helix_usage_validated/stereoset_probe_halfmatch.tsv
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from transformers import GPT2TokenizerFast


OUT_DIR = Path("helix_usage_validated")
OUT_DIR.mkdir(exist_ok=True)

DOMAINS = ("gender", "race", "profession", "religion")


def diff_fill(context: str, filled: str) -> str | None:
    """Recover the fill word from (context-with-BLANK, filled-sentence)."""
    if "BLANK" not in context:
        return None
    prefix, suffix = context.split("BLANK", 1)
    if not filled.startswith(prefix):
        return None
    rest = filled[len(prefix):]
    if suffix and not rest.endswith(suffix):
        return None
    fill = rest[: len(rest) - len(suffix)] if suffix else rest
    fill = fill.strip()
    fill = re.sub(r"[^A-Za-z'\-]", "", fill)
    return fill or None


def tokenize_single(tok: GPT2TokenizerFast, word: str) -> tuple[int | None, str | None]:
    """Return (token_id, used_variant) if either `word` or ` word` is
    one-token. Otherwise (None, None)."""
    for variant in (" " + word, word):
        ids = tok.encode(variant, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0], variant
    return None, None


def main():
    print("Loading StereoSet intrasentence ...")
    ds = load_dataset("McGill-NLP/stereoset", "intrasentence", split="validation")
    print(f"  total rows: {len(ds)}")

    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    pairs = []
    halfmatch = []
    per_domain_total = Counter()
    per_domain_pair = Counter()
    per_domain_half = Counter()

    for ex in ds:
        domain = ex["bias_type"]
        if domain not in DOMAINS:
            continue
        per_domain_total[domain] += 1

        context = ex["context"]
        sents = ex["sentences"]["sentence"]
        labels = ex["sentences"]["gold_label"]
        by_label = {int(lab): s for lab, s in zip(labels, sents)}
        if 0 not in by_label or 1 not in by_label:
            continue
        anti_fill = diff_fill(context, by_label[0])
        stereo_fill = diff_fill(context, by_label[1])
        if not anti_fill or not stereo_fill:
            continue
        if anti_fill.lower() == stereo_fill.lower():
            continue

        s_id, s_var = tokenize_single(tok, stereo_fill)
        a_id, a_var = tokenize_single(tok, anti_fill)

        if s_id is not None and a_id is not None:
            pairs.append(dict(
                domain=domain,
                target=ex["target"],
                context=context,
                stereo=stereo_fill,
                anti=anti_fill,
                stereo_token_id=s_id,
                anti_token_id=a_id,
                stereo_variant=s_var,
                anti_variant=a_var,
            ))
            per_domain_pair[domain] += 1
        elif s_id is not None or a_id is not None:
            halfmatch.append(dict(
                domain=domain,
                target=ex["target"],
                context=context,
                stereo=stereo_fill,
                anti=anti_fill,
                stereo_single=s_id is not None,
                anti_single=a_id is not None,
            ))
            per_domain_half[domain] += 1

    # Deduplicate: (domain, stereo_token_id, anti_token_id)
    seen = set()
    uniq_pairs = []
    for p in pairs:
        key = (p["domain"], p["stereo_token_id"], p["anti_token_id"])
        if key in seen:
            continue
        seen.add(key)
        uniq_pairs.append(p)

    print()
    print("=== per-domain summary ===")
    for d in DOMAINS:
        print(f"  {d:12s} total={per_domain_total[d]:4d}  "
              f"both-single={per_domain_pair[d]:4d}  "
              f"one-single={per_domain_half[d]:4d}")
    print(f"  unique pairs (dedup by token ids): {len(uniq_pairs)}")

    # Write pair table
    pair_path = OUT_DIR / "stereoset_probe_pairs.tsv"
    with pair_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "domain", "target", "stereo", "anti",
            "stereo_token_id", "anti_token_id",
            "stereo_variant", "anti_variant", "context"], delimiter="\t")
        w.writeheader()
        for p in uniq_pairs:
            w.writerow(p)
    print(f"  wrote {pair_path}")

    # Write halfmatches for eyeball review
    half_path = OUT_DIR / "stereoset_probe_halfmatch.tsv"
    with half_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "domain", "target", "stereo", "anti",
            "stereo_single", "anti_single", "context"], delimiter="\t")
        w.writeheader()
        for p in halfmatch:
            w.writerow(p)
    print(f"  wrote {half_path}")

    # Show the first 20 pairs per domain for human review
    print()
    print("=== sample pairs (first 8 per domain) ===")
    for d in DOMAINS:
        dpairs = [p for p in uniq_pairs if p["domain"] == d]
        print(f"\n-- {d} ({len(dpairs)} unique) --")
        for p in dpairs[:8]:
            print(f"  target={p['target']:20s} "
                  f"stereo='{p['stereo']}' ({p['stereo_variant']})  "
                  f"anti='{p['anti']}' ({p['anti_variant']})")

    # And a handful of halfmatches per domain, as eyeball aids
    print()
    print("=== sample HALFMATCH (one side single-token) ===")
    for d in DOMAINS:
        dhalf = [p for p in halfmatch if p["domain"] == d]
        print(f"\n-- {d} halfmatch ({len(dhalf)}) --")
        for p in dhalf[:8]:
            side = "stereo" if p["stereo_single"] else "anti"
            missing = "anti" if p["stereo_single"] else "stereo"
            print(f"  target={p['target']:20s} "
                  f"single={side}:'{p[side]}'  "
                  f"multi={missing}:'{p[missing]}'")


if __name__ == "__main__":
    main()
