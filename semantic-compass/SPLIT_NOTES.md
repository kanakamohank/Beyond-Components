# Split notes — semantic-compass

This repository was split out of the combined `Beyond-Components` research
repository, which held three papers in one tree. This repository contains the
**Semantic Compasses** paper and its implementation: 594 compass-specific
files plus 43 shared files.

The arithmetic half lives in `arithmetic-circuit-discovery`.

## How the split was decided

The source repository carried three papers' worth of material in one flat tree.
Each of the 873 tracked files was classified into exactly one bucket —
`compass`, `arithmetic`, `both`, or `drop` — using these signals, in priority
order:

1. **Explicit cookbook references.** `COMPASS_COOKBOOK.md` and
   `ARITHMETIC_CIRCUIT_PLAN.md` name their scripts by path. Between them they
   account for most of `experiments/`, and — importantly — their reference sets
   do not overlap at all. That gave a clean seam for the bulk of the code.
2. **LaTeX figure references** (`\includegraphics`, resolved through each
   document's `\graphicspath`) for the figure directories.
3. **The import graph**, for shared `src/` modules.
4. **Module docstrings**, for the 27 scripts neither cookbook names.
5. **Filename conventions**, for the artifact directories only.

The full per-file decision table, with a one-line reason for every file, is in
`SPLIT_MANIFEST.tsv` at the root of this repository. It is identical in both
split repositories, so any classification can be audited from either side.

### Bucket totals

| Bucket | Files | Meaning |
|---|---:|---|
| `compass` | 594 | Semantic Compasses only |
| `arithmetic` | 171 | Fourier digit arithmetic only |
| `both` | 43 | Genuinely shared — duplicated into both repositories |
| `drop` | 65 | Beyond Components only — in neither repository |
| **total** | **873** | |

### What is shared, and why

43 files are duplicated rather than assigned to one side:

- `src/models/masked_transformer_circuit.py`, `src/utils/`,
  `src/data/data_loader.py`, `experiments/train.py`, `configs/gp_config.yaml` —
  Stage-1 learnable-mask direction discovery. Inherited from Beyond Components;
  both papers build on it.
- `investigate_helix_usage_validated.py` — genuinely dual-purpose. It is the
  compass paper's nine-test falsification battery *and* the cross-task helix
  investigation used on the arithmetic side.
- `knowledge/` literature summaries that both papers cite, plus
  `requirements.txt`, `setup.py`, `.gitignore`, `CLAUDE.md`.

### What was left behind

65 files belong to **Beyond Components: Singular Vector-Based
Interpretability of Transformer Circuits** ([arXiv:2511.20273](https://arxiv.org/abs/2511.20273))
and appear in neither repository: the original `README.md`, the IOI and
Greater-Than task configs, `svd_logs/` (a 42 MB IOI circuit-discovery run),
`images/intervention.png`, and the sigma-amplification scripts. Three
Beyond-Components-lineage files *were* kept, because the cookbooks depend on
them: `experiments/ablation/intervention.py`,
`experiments/evaluation/comprehensive_metrics_table.py`, and the Stage-1 files
listed above.

Paths are preserved verbatim from the source repository, so every path
reference inside the cookbooks and the LaTeX sources remains valid.

## How the split was verified

| Check | Result |
|---|---|
| Every tracked file classified | 873/873, no file unassigned |
| Python syntax | all files parse in both repositories |
| Intra-repo `src.*` / `experiments.*` imports | all resolve, no split-induced breakage |
| LaTeX `\includegraphics` | compass 45/48 resolve; the 3 misses are `example-image-*` placeholders from the ACL template package |
| Test suite | 224 passed — byte-identical to the same run on the source repository |

The one classification bug this caught: `experiments/train.py` was initially
filed as arithmetic-only because the arithmetic cookbook names it, which broke
`run_train.py` in the compass repository. It is shared, and is now in both.

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
  `paper/` does not compile as-is. The arithmetic repository's README lists the
  script that regenerates each one.
- Git history was not carried across. The source repository's 35 commits are
  interleaved across all three papers, so each split repository starts from a
  single commit. The source repository remains the historical record.
