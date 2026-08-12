#!/usr/bin/env python3
"""Materialize the two per-paper trees from SPLIT_MANIFEST.tsv.

Paths are preserved verbatim relative to the repo root so that every path
reference inside the cookbooks and the LaTeX sources stays valid.
"""
import shutil
from pathlib import Path

ROOT = Path("/home/user/Beyond-Components")
TARGETS = {
    "compass": ROOT / "semantic-compass",
    "arithmetic": ROOT / "arithmetic-circuit-discovery",
}

rows = [
    line.split("\t")
    for line in (ROOT / "SPLIT_MANIFEST.tsv").read_text().splitlines()[1:]
]

for d in TARGETS.values():
    if d.exists():
        shutil.rmtree(d)

counts = {k: 0 for k in TARGETS}
unclassified = [p for p, b, _ in rows if b not in {"compass", "arithmetic", "both", "drop"}]
if unclassified:
    raise SystemExit(
        f"{len(unclassified)} file(s) unclassified; re-run build_manifest.py and add a rule:\n  "
        + "\n  ".join(unclassified[:10])
    )

for path, bucket, _reason in rows:
    if bucket == "drop":
        continue
    dests = list(TARGETS) if bucket == "both" else [bucket]
    for d in dests:
        out = TARGETS[d] / path
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, out)
        counts[d] += 1

for k, v in counts.items():
    print(f"{TARGETS[k].name:<30} {v} files")
