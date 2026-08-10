#!/usr/bin/env python3
"""Emit a per-repo SPLIT_NOTES.md documenting provenance of the split."""
from collections import Counter
from pathlib import Path

ROOT = Path("/home/user/Beyond-Components")
rows = [l.split("\t") for l in (ROOT / "SPLIT_MANIFEST.tsv").read_text().splitlines()[1:]]
counts = Counter(b for _, b, _ in rows)

COMMON = """
## How the split was decided

The source repository carried three papers' worth of material in one flat tree.
Each of the 873 tracked files was classified into exactly one bucket —
`compass`, `arithmetic`, `both`, or `drop` — using these signals, in priority
order:

1. **Explicit cookbook references.** `COMPASS_COOKBOOK.md` and
   `ARITHMETIC_CIRCUIT_PLAN.md` name their scripts by path. Between them they
   account for most of `experiments/`, and their reference sets do not overlap.
   That gave a clean seam for the bulk of the code. Cookbook references are
   authoritative about *intent*, not about behaviour — see the caveat below.
2. **What a script actually writes.** Two plotting scripts are listed in the
   compass cookbook but write the *arithmetic* paper's figures into
   `mathematical_toolkit_results/paper_plots/`. Trusting the cookbook alone put
   them on the wrong side; they are shared.
3. **LaTeX figure references** (`\\includegraphics`, resolved through each
   document's `\\graphicspath`) for the figure directories.
4. **The import graph**, for shared `src/` modules.
5. **Module docstrings**, for the 27 scripts neither cookbook names.
6. **Filename conventions**, for the artifact directories only.

The full per-file decision table, with a one-line reason for every file, is in
`SPLIT_MANIFEST.tsv` at the root of this repository. It is identical in both
split repositories, so any classification can be audited from either side.

### Bucket totals

| Bucket | Files | Meaning |
|---|---:|---|
| `compass` | {compass} | Semantic Compasses only |
| `arithmetic` | {arithmetic} | Fourier digit arithmetic only |
| `both` | {both} | Genuinely shared — duplicated into both repositories |
| `drop` | {drop} | Beyond Components only — in neither repository |
| **total** | **{total}** | |

Each repository additionally carries three files that exist in neither source
nor manifest: its own `README.md`, this `SPLIT_NOTES.md`, and a copy of
`SPLIT_MANIFEST.tsv`. So `git ls-files | wc -l` gives {compass_tracked} for
`semantic-compass` and {arith_tracked} for `arithmetic-circuit-discovery`.

### What is shared, and why

{both} files are duplicated rather than assigned to one side:

- `src/models/masked_transformer_circuit.py`, `src/utils/`,
  `src/data/data_loader.py`, `experiments/train.py`, `configs/gp_config.yaml` —
  Stage-1 learnable-mask direction discovery. Inherited from Beyond Components;
  both papers build on it.
- `investigate_helix_usage_validated.py` — genuinely dual-purpose. It is the
  compass paper's nine-test falsification battery *and* the cross-task helix
  investigation used on the arithmetic side.
- `experiments/generate_paper_plots.py`, `experiments/generate_missing_plots.py`
  — listed in the compass cookbook, but they render five of the arithmetic
  paper's six figures.
- Cross-model helix scan logs in `helix_usage_validated/` (`*_sweep_output`,
  `*_trace_output`), which are the raw material behind
  `svd_stats_ov_helix_circuit.md` and `helix_cross_model_analysis.md`.
- `knowledge/` literature summaries that both papers cite, plus
  `requirements.txt`, `setup.py`, `.gitignore`, `CLAUDE.md`.

### What was left behind

{drop} files belong to **Beyond Components: Singular Vector-Based
Interpretability of Transformer Circuits** ([arXiv:2511.20273](https://arxiv.org/abs/2511.20273))
and appear in neither repository: the original `README.md`, the Greater-Than
task config, `svd_logs/` (a 42 MB IOI circuit-discovery run),
`images/intervention.png`, `experiments/analyze_checkpoint.py`, and the
sigma-amplification scripts (`experiments/ablation/comprehensive_sigma_test.py`,
`experiments/evaluation/generate_sigma_table.py`).

Some Beyond-Components-lineage files *were* kept, because an in-scope cookbook
depends on them: the Stage-1 files listed above, plus
`experiments/ablation/intervention.py` and `run_ablation.py` (arithmetic),
`experiments/evaluation/comprehensive_metrics_table.py` (compass), and
`configs/ioi_config.yaml` (named in the arithmetic plan's Phase 0).

### Divergence from the source files

Copies are byte-identical to the source repository with exactly two exceptions,
both applied by `tools/split/postprocess.py` so they stay auditable:

- `semantic-compass/COMPASS_COOKBOOK.md` — the opening paragraph described the
  combined two-paper repo; it now points at the separate arithmetic repository.
- `arithmetic-circuit-discovery/.gitignore` — the inherited `*_results/*` rule
  would have silently swallowed `fourier_results/` (tracked in the source repo)
  and the paper's figure directory. Both are now explicitly un-ignored.

Every other path is preserved verbatim, so path references inside the cookbooks
and the LaTeX sources still resolve. Two caveats: references to generated
directories that were never tracked (`data/data_main/`,
`mathematical_toolkit_results/`) did not resolve in the source repo either, and
a handful of `COMPASS_COOKBOOK.md` artifact paths (`bc/*.png`,
`helix_usage_validated/workshop_suite_results.json`) were already stale before
the split.

## How the split was verified

`tools/split/verify.py` in the source repository checks:

| Check | Result |
|---|---|
| Every tracked file classified | 873/873, no file unassigned |
| Python syntax | all files parse in both repositories |
| Intra-repo `src.*` / `experiments.*` imports | all resolve, no split-induced breakage |
| Files survive `git add` | no manifest file swallowed by an inherited ignore rule |
| Every referenced figure is committed or has a producer script | passes in both |
| LaTeX `\\includegraphics` | compass 45/48 resolve; the 3 misses are `example-image-*` placeholders from the ACL template |
| Test suite (arithmetic repo only) | 224 passed — identical to the same run on the source repository |

`semantic-compass` has no `tests/` directory: the source repo's entire test
suite targets arithmetic and Fourier code, as `COMPASS_COOKBOOK.md` itself
notes.

Three classification bugs were caught this way and fixed:
`experiments/train.py` was filed arithmetic-only because the arithmetic cookbook
names it, which broke `run_train.py` on the compass side; the two plotting
scripts described above were filed compass-only, leaving the arithmetic paper
unable to rebuild five of its six figures; and `fourier_results/` was copied to
disk but silently untracked.

## Known pre-existing gaps

These were carried over unchanged rather than silently repaired:

- Seven modules referenced by `ARITHMETIC_CIRCUIT_PLAN.md` were never committed
  (`src/models/arithmetic_pipeline.py`, `src/analysis/circuit_identification.py`,
  `src/analysis/geometric_interpreter.py`, `src/analysis/neuron_analyzer.py`,
  `src/data/arithmetic_data.py`, `src/utils/helix_visualization.py`,
  `experiments/arithmetic_validation.py`). They are missing on the source
  repository's `main` too.
- The arithmetic paper's six figures live under the gitignored
  `mathematical_toolkit_results/paper_plots/` and were never tracked, so
  `paper/` does not compile as-is. The arithmetic repository's README maps each
  figure to the script that renders it and the data step it reads.
- Git history was not carried across. The source repository's 35 commits are
  interleaved across all three papers, so each split repository starts from a
  single commit. The source repository remains the historical record — and the
  only place holding `tools/split/`, so re-auditing or regenerating the
  manifest is done from there, not from either split repository.
"""

HEADERS = {
    "semantic-compass": """# Split notes — semantic-compass

This repository was split out of the combined `Beyond-Components` research
repository, which held three papers in one tree. This repository contains the
**Semantic Compasses** paper and its implementation: {compass} compass-specific
files plus {both} shared files.

The arithmetic half lives in `arithmetic-circuit-discovery`.
""",
    "arithmetic-circuit-discovery": """# Split notes — arithmetic-circuit-discovery

This repository was split out of the combined `Beyond-Components` research
repository, which held three papers in one tree. This repository contains the
**Fourier Basis of Digit Arithmetic** paper and its implementation:
{arithmetic} arithmetic-specific files plus {both} shared files.

The compass half lives in `semantic-compass`.
""",
}

fields = dict(
    compass=counts["compass"],
    arithmetic=counts["arithmetic"],
    both=counts["both"],
    drop=counts["drop"],
    total=sum(counts.values()),
    compass_tracked=counts['compass'] + counts['both'] + 3,
    arith_tracked=counts['arithmetic'] + counts['both'] + 3,
)

for repo, header in HEADERS.items():
    (ROOT / repo / "SPLIT_NOTES.md").write_text(
        header.format(**fields) + COMMON.format(**fields)
    )
    # the audit table travels with both repos
    (ROOT / repo / "SPLIT_MANIFEST.tsv").write_text((ROOT / "SPLIT_MANIFEST.tsv").read_text())
    print("wrote", repo + "/SPLIT_NOTES.md")
