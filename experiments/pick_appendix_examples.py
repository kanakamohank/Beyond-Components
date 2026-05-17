"""Curate representative appendix rows from qualitative_spotcheck CSVs.

Selection policy per (model, domain):
  1. Drop rows where either sentence's *only* lexical change is a proper
     name (likely pure identity/name confound, not stereotype content).
  2. Drop rows that are pronouns-only swaps.
  3. Prefer verdicts in {FLIP -> anti-stereo, anti-stereo stronger}
     with large |debias_shift|.
  4. Also emit 1-2 representative REGRESSIONs per model for honesty
     (FLIP -> stereo with large |debias_shift|).

Output:
  helix_usage_validated/appendix_examples.md
  helix_usage_validated/appendix_examples.csv
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

OUT_DIR = Path("helix_usage_validated")
MODELS = ["gpt2", "phi3", "gemma", "llama"]
DOMAINS = ["gender", "race", "profession", "religion"]
WIN_VERDICTS = {"FLIP -> anti-stereo", "anti-stereo stronger"}
REG_VERDICTS = {"FLIP -> stereo (regression)", "stereo stronger"}

# Very common given-names that dominate CrowS-Pairs; presence in the sole
# differing token means it's likely a name-swap confound.
COMMON_NAMES = {
    "manuel", "jeff", "anita", "maria", "tyrone", "brad", "jamal",
    "connor", "darnell", "liam", "brett", "dante", "latoya", "ashley",
    "deshawn", "cody", "chad", "devonte", "jose", "juan", "carlos",
    "aisha", "jamal", "shaniqua", "kirby", "hakim", "abdul", "omar",
    "luis", "miguel", "angel", "pedro", "lupe", "felicia", "keisha",
    "jerome", "tyrell", "tamika", "darius", "shonda",
}


def token_set(s):
    return [w.lower().strip(".,!?;:\"'()") for w in s.split()]


def sole_diff_is_name(sent_more, sent_less):
    a = token_set(sent_more)
    b = token_set(sent_less)
    if len(a) != len(b):
        return False
    diffs = [(i, a[i], b[i]) for i in range(len(a)) if a[i] != b[i]]
    if not diffs:
        return False
    return all(x in COMMON_NAMES or y in COMMON_NAMES for _, x, y in diffs)


def is_pronoun_only_swap(sent_more, sent_less):
    pronouns = {"he", "she", "him", "her", "his", "hers", "himself",
                "herself", "man", "woman", "men", "women", "boy", "girl"}
    a = token_set(sent_more)
    b = token_set(sent_less)
    if len(a) != len(b):
        return False
    diffs = [(a[i], b[i]) for i in range(len(a)) if a[i] != b[i]]
    return diffs and all(x in pronouns and y in pronouns for x, y in diffs)


def load(m):
    path = OUT_DIR / f"qualitative_spotcheck_{m}.csv"
    with path.open() as f:
        return list(csv.DictReader(f))


def pick_wins(rows, k=1):
    elig = [r for r in rows
            if r["verdict"] in WIN_VERDICTS
            and not sole_diff_is_name(r["sent_more"], r["sent_less"])
            and not is_pronoun_only_swap(r["sent_more"], r["sent_less"])]
    elig.sort(key=lambda r: float(r["debias_shift"]), reverse=True)
    return elig[:k]


def pick_regressions(rows, k=1):
    elig = [r for r in rows
            if r["verdict"] in REG_VERDICTS
            and not sole_diff_is_name(r["sent_more"], r["sent_less"])
            and not is_pronoun_only_swap(r["sent_more"], r["sent_less"])]
    elig.sort(key=lambda r: float(r["debias_shift"]))  # most-negative first
    return elig[:k]


def main():
    chosen = []
    for m in MODELS:
        rows = load(m)
        by_dom = defaultdict(list)
        for r in rows:
            by_dom[r["domain"]].append(r)
        for d in DOMAINS:
            wins = pick_wins(by_dom[d], k=1)
            for r in wins:
                out = dict(r)
                out["model"] = m
                out["kind"] = "win"
                chosen.append(out)
        # one honest regression per model, drawn from whichever domain has
        # the worst offender
        reg = pick_regressions(rows, k=1)
        for r in reg:
            out = dict(r)
            out["model"] = m
            out["kind"] = "regression"
            chosen.append(out)

    csv_cols = ["model", "domain", "kind", "idx", "stereo_antistereo",
                "sent_more", "sent_less",
                "gap_base", "gap_routed", "shift", "debias_shift",
                "verdict"]
    csv_path = OUT_DIR / "appendix_examples.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        for r in chosen:
            w.writerow({c: r[c] for c in csv_cols})
    print(f"wrote {csv_path}  ({len(chosen)} rows)")

    md = [
        "# Appendix-ready qualitative examples",
        "",
        "Selection: top-shift anti-stereo wins per (model, domain), plus "
        "one honest regression per model. Rows whose only lexical "
        "difference is a given-name or a pronoun have been filtered out "
        "so the example isolates stereotype content, not identity "
        "swapping. `gap = logp(sent_more) − logp(sent_less)`; "
        "`debias_shift > 0` means the routed intervention pushed "
        "probability mass *away from* the stereotyped sentence on that "
        "row.",
        "",
    ]
    for m in MODELS:
        md.append(f"## {m}")
        md.append("")
        md.append("| domain | verdict | gap (base → routed) "
                  "| Δ debias | sent_more / sent_less |")
        md.append("|---|---|---|---|---|")
        for r in chosen:
            if r["model"] != m:
                continue
            gb = float(r["gap_base"])
            gr = float(r["gap_routed"])
            ds = float(r["debias_shift"])
            md.append(
                f"| {r['domain']} ({r['kind']}) | {r['verdict']} | "
                f"{gb:+.2f} → {gr:+.2f} | {ds:+.2f} | "
                f"*more:* {r['sent_more']}<br>*less:* {r['sent_less']} |"
            )
        md.append("")
    md_path = OUT_DIR / "appendix_examples.md"
    md_path.write_text("\n".join(md))
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
