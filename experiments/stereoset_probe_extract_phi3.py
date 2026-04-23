"""Stage 1 (Phi-3 variant): extract StereoSet probe pairs using Phi-3
tokenizer.

Same logic as stereoset_probe_extract_gemma.py; swaps tokenizer to
microsoft/Phi-3-mini-4k-instruct (SentencePiece-based).

Outputs:
  helix_usage_validated/stereoset_probe_pairs_phi3.tsv
  helix_usage_validated/stereoset_probe_halfmatch_phi3.tsv
"""
from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


OUT_DIR = Path("helix_usage_validated")
OUT_DIR.mkdir(exist_ok=True)

DOMAINS = ("gender", "race", "profession", "religion")
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"


def diff_fill(context: str, filled: str) -> str | None:
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


def tokenize_single(tok, word: str) -> tuple[int | None, str | None]:
    for variant in (" " + word, word):
        ids = tok.encode(variant, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0], variant
    return None, None


def main():
    print("Loading StereoSet intrasentence ...")
    ds = load_dataset("McGill-NLP/stereoset", "intrasentence",
                      split="validation")
    print(f"  total rows: {len(ds)}")

    print(f"Loading {MODEL_NAME} tokenizer ...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

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

    seen = set()
    uniq_pairs = []
    for p in pairs:
        key = (p["domain"], p["stereo_token_id"], p["anti_token_id"])
        if key in seen:
            continue
        seen.add(key)
        uniq_pairs.append(p)

    print()
    print("=== per-domain summary (Phi-3 tokenizer) ===")
    for d in DOMAINS:
        print(f"  {d:12s} total={per_domain_total[d]:4d}  "
              f"both-single={per_domain_pair[d]:4d}  "
              f"one-single={per_domain_half[d]:4d}")
    print(f"  unique pairs (dedup by token ids): {len(uniq_pairs)}")

    pair_path = OUT_DIR / "stereoset_probe_pairs_phi3.tsv"
    with pair_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "domain", "target", "stereo", "anti",
            "stereo_token_id", "anti_token_id",
            "stereo_variant", "anti_variant", "context"], delimiter="\t")
        w.writeheader()
        for p in uniq_pairs:
            w.writerow(p)
    print(f"  wrote {pair_path}")

    half_path = OUT_DIR / "stereoset_probe_halfmatch_phi3.tsv"
    with half_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "domain", "target", "stereo", "anti",
            "stereo_single", "anti_single", "context"], delimiter="\t")
        w.writeheader()
        for p in halfmatch:
            w.writerow(p)
    print(f"  wrote {half_path}")


if __name__ == "__main__":
    main()
