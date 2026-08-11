# Split notes — semantic-compass

This repository was split out of the combined `Beyond-Components` research
repository, which held three papers in one tree. This repository contains the
**Semantic Compasses** paper and its implementation: 592 compass-specific
files plus 29 shared files.

The arithmetic half lives in `arithmetic-circuit-discovery`.

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
3. **LaTeX figure references** (`\includegraphics`, resolved through each
   document's `\graphicspath`) for the figure directories.
4. **The import graph**, for shared `src/` modules.
5. **Module docstrings**, for the 27 scripts neither cookbook names.
6. **Filename conventions**, for the artifact directories only.

The full per-file decision table, with a one-line reason for every file, is in
`SPLIT_MANIFEST.tsv` at the root of this repository. It is identical in both
split repositories, so any classification can be audited from either side.

### Bucket totals

| Bucket | Files | Meaning |
|---|---:|---|
| `compass` | 592 | Semantic Compasses only |
| `arithmetic` | 164 | Fourier digit arithmetic only |
| `both` | 29 | Genuinely shared — duplicated into both repositories |
| `drop` | 88 | Beyond Components only — in neither repository |
| **total** | **873** | |

Each repository additionally carries three files that exist in neither source
nor manifest: its own `README.md`, this `SPLIT_NOTES.md`, and a copy of
`SPLIT_MANIFEST.tsv`. So `git ls-files | wc -l` gives 624 for
`semantic-compass` and 196 for `arithmetic-circuit-discovery`.

### What is shared, and why

29 files are duplicated rather than assigned to one side:

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

### What was left behind: Beyond Components

88 files belong to **Beyond Components: Singular Vector-Based
Interpretability of Transformer Circuits** ([arXiv:2511.20273](https://arxiv.org/abs/2511.20273))
and appear in neither repository. Neither paper's pipeline uses that code:

- **The compass** derives every direction at run time from the model's own
  weights — `torch.linalg.svd(W_V @ W_O)` in `compass_causal_sweep.py:71-74`.
  No learned masks, no training step, no checkpoints. An AST scan of the
  compass tree found exactly one file importing Beyond Components code
  (`experiments/train.py`, itself the Beyond Components trainer) plus two
  `__init__.py` shims re-exporting it. Nothing in the pipeline reached them.
- **The arithmetic paper's method** is the 15 numbered steps in Phases A-F of
  `ARITHMETIC_CIRCUIT_PLAN.md`. None of them import Beyond Components code. It
  survived only in a superseded branch that the plan itself files under
  `SUPPLEMENTARY SCRIPTS — Not in Main Pipeline`, where S5 is titled
  "(Old Pipeline)".

So the dependency was dropped rather than carried or vendored. That also
retires the CC BY-SA attribution obligation that shipping the source would
carry — an obligation no README wording can remove, which is why the code had
to go rather than the prose.

Removed: the Beyond Components model (`masked_transformer_circuit.py`) and its
helpers (`src/utils/{utils,visualization,constants}.py`, the IOI/GP/GT
`data_loader.py`), its entry points (`experiments/train.py`, `run_train.py`,
`experiments/ablation/intervention.py`, `run_ablation.py`,
`experiments/evaluation/comprehensive_metrics_table.py`, which reads
`checkpoints/{ioi,gt,gp}`), its task configs (`gp_config.yaml`,
`ioi_config.yaml`, `gt_config.yaml`), the superseded arithmetic branch
(`analyze_fourier_circuits.py`, `analyze_svd_directions.py`,
`analyze_sum_encoding.py`, `causal_validation.py`, `run_fourier_discovery.py`,
`helix_circuit_discovery.py`, `run_helix_analysis.py`), the original
Beyond Components `README.md`, `svd_logs/` (a 42 MB IOI run),
`images/intervention.png`, and the sigma-amplification scripts.

Three test files and one test class went with them —
`test_fourier_circuits.py` (22), `test_causal_validation.py` (17),
`test_arithmetic_training_integration.py` (36) and `TestUtilsIntegration` (4),
totalling 79. The suite goes 224 -> 145, and every removed test was asserting
against removed Beyond Components code.

All of it remains recoverable from the source repository's history.

### Divergence from the source files

Copies are byte-identical to the source repository except for the following,
all applied by `tools/split/postprocess.py` so they stay auditable:

- `semantic-compass/COMPASS_COOKBOOK.md` — the opening paragraph described the
  combined two-paper repo; it now points at the separate arithmetic repository.
- `arithmetic-circuit-discovery/.gitignore` — the inherited `*_results/*` rule
  would have silently swallowed `fourier_results/` (tracked in the source repo)
  and the paper's figure directory. Both are now explicitly un-ignored.
- `arithmetic-circuit-discovery/src/{,data/,utils/}__init__.py` — trimmed to
  stop re-exporting the removed Beyond Components modules.
- `arithmetic-circuit-discovery/tests/test_arithmetic_dataset.py` —
  `TestUtilsIntegration` removed; it asserted against the removed column registry.
- `arithmetic-circuit-discovery/ARITHMETIC_CIRCUIT_PLAN.md` — a scope note marks
  the retired mask-learning sections as historical.
- `semantic-compass/COMPASS_COOKBOOK.md` — the Stage-1 citation note replaced
  with how directions are actually computed, and two rows dropped from the
  paper-to-code table that pointed at removed scripts *and* at paper labels
  (`tab:directions`, `tab:bc_full`, `app:bc_results`) present in neither
  compass paper.

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
| No Beyond Components import survives | zero matches in either repository |
| LaTeX `\includegraphics` | compass 45/48 resolve; the 3 misses are `example-image-*` placeholders from the ACL template |
| Test suite (arithmetic repo only) | 145 passed (224 minus the 79 Beyond Components tests removed with their code) |

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
