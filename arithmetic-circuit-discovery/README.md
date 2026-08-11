# The Fourier Basis of Digit Arithmetic

Implementation and paper source for:

> **The Fourier Basis of Digit Arithmetic: Mechanistic Interpretability of
> Addition Circuits in Language Models**

Pre-trained language models represent digits in a **Fourier basis of ℤ/10ℤ**.
Across Gemma 2B, Phi-3 Mini, and LLaMA 3.2-3B, the digit subspace at the
computation layer decomposes into exactly nine directions — two each for
frequencies k=1..4 and one for k=5 (parity) — and that subspace is causally
necessary: zeroing it destroys addition accuracy, while rotating its phase
shifts the model's answer by a predictable amount mod 10.

The repository covers encoding characterization, causal validation, component
attribution, verification of the computation mechanism, and generalization
tests.

**Start here: [`ARITHMETIC_CIRCUIT_PLAN.md`](ARITHMETIC_CIRCUIT_PLAN.md)** — the
14-step execution pipeline, in order, with the exact command for each step and
what to look for in its output.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.8+ and PyTorch 2.0+. A CUDA-capable GPU is recommended;
the pipeline also runs on Apple MPS (`--device mps`), which is what the
recorded runs used.

## Quickstart

The pipeline is sequential — Step 1 determines the `comp-layer` that every
later step needs.

```bash
# Step 1 — layer scan + unembed patching. MUST RUN FIRST.
python experiments/arithmetic_circuit_scan_updated.py \
    --model gemma-2b --device mps --n-per-digit 100 --n-test 150

# Step 2 — verify the digit encoding is a perfect Fourier basis of Z/10Z.
python experiments/eigenvector_dft.py --model gemma-2b --comp-layer 19 --device mps

# Step 4 — causal knockout: zero the 9D Fourier subspace, measure the damage.
python experiments/fourier_knockout.py --model gemma-2b --comp-layer 19 --device mps
```

Results are written to `mathematical_toolkit_results/`. Adding a new model is
described in the "Quick Reference" section at the top of the plan.

## Layout

```
arithmetic-circuit-discovery/
├── ARITHMETIC_CIRCUIT_PLAN.md     # the 14-step pipeline — read this first
├── paper/                         # paper source (main.tex + sections/)
│
├── experiments/                   # ~40 scripts, all named in the plan
│   ├── arithmetic_circuit_scan_updated.py  # Step 1 — layer scan, unembed/Fisher patching
│   ├── eigenvector_dft.py                  # Step 2 — DFT of SVD directions
│   ├── fourier_decomposition.py            # Step 3 — per-layer Fourier sweep
│   ├── fourier_knockout.py                 # Step 4 — necessity via knockout
│   ├── fourier_phase_rotation.py           # phase rotation → answer shift mod 10
│   ├── fourier_head_attribution.py         # which heads/MLPs write the subspace
│   ├── carry_stratification.py             # carry vs. no-carry behaviour
│   ├── crt_sanity_check.py                 # Chinese Remainder Theorem alignment
│   ├── multidigit_circuit.py               # beyond single digits
│   ├── generalization_tests.py             # transfer to unseen operand ranges
│   └── mathematical_toolkit.py             # Fisher / ICA / tensor-decomposition toolkit
│
├── src/
│   ├── analysis/fourier_discovery.py       # DFT power spectra, Fourier basis
│   ├── data/arithmetic_dataset.py          # prompt generation
│   ├── models/{online,offline}_svd_scanner.py
│   └── utils/model_registry.py             # per-model layer/head specs
│
├── tests/                         # unit tests for the pipeline
├── configs/arithmetic_*.yaml
├── images/                        # helix / FFT / geometry figures across models
├── fourier_results/               # recorded Fourier discovery outputs
└── requirements.txt
```

## Supporting write-ups

| Document | Contents |
|---|---|
| `MATHEMATICAL_TOOLKIT_PROPOSAL.md` | The five mathematical approaches behind `mathematical_toolkit.py` |
| `FOURIER_PHASE_ROTATION_FINDINGS.md` | Cross-model phase-rotation results |
| `Diagnosing Fisher Patching.md` | Debug log for the Fisher-patching subspace |
| `helix_cross_model_analysis.md` | What helix/circle patterns actually do, across models |
| `svd_stats_ov_helix_circuit.md` | OV-SVD helix statistics per model variant |
| `experiments/circuit_synthesis.md` | Multi-model synthesis of the discovered circuit |
| `knowledge/` | Summaries of the related literature (grokking, clock/pizza, CRT) |

## Running the tests

```bash
pip install pytest
pytest tests/ --ignore=tests/test_arithmetic_pipeline.py
```

`tests/test_arithmetic_pipeline.py` is excluded because it imports
`src/models/arithmetic_pipeline.py`, which was never committed — see below.

## Known gaps

These predate this repository and were carried over as-is rather than papered
over:

**1. Paper figures are not committed.** `paper/main.tex` sets
`\graphicspath{{../mathematical_toolkit_results/paper_plots/}}`, and that
directory is generated output that was never tracked. Six figures are
therefore missing, and `paper/` will not compile until they are regenerated.

The plotting scripts read the JSON that the analysis steps write into
`mathematical_toolkit_results/`, so each figure needs its data step run first:

| Figure | Plotting script | Data step it reads |
|---|---|---|
| `layer_scan_curves.png` | `generate_paper_plots.py`, `generate_missing_plots.py` | Step 1 `arithmetic_circuit_scan_updated.py` |
| `fourier_heatmap_cross_model.png` | `generate_paper_plots.py` | Step 3 `fourier_decomposition.py` (layer sweep) |
| `energy_explosion.png` | `generate_paper_plots.py` | Step 3 `fourier_decomposition.py` |
| `ablation_curves.png` | `generate_missing_plots.py` | Step 4 `fourier_knockout.py`, `multilayer_freq_ablation.py` |
| `neuron_frequency_tuning.png` | `generate_missing_plots.py` | `neuron_trig_analysis.py` |
| `eigenvector_fourier_cross_model.png` | `plot_eigenvector_dft.py` | Step 2 `eigenvector_dft.py` |

All six scripts are in `experiments/`. Note that the data steps themselves emit
no images — they only write JSON; the three plotting scripts do all the
rendering.

`.gitignore` has been adjusted so `mathematical_toolkit_results/paper_plots/`
is trackable once those figures exist.

**2. Six referenced modules do not exist.** `ARITHMETIC_CIRCUIT_PLAN.md`
cites `src/models/arithmetic_pipeline.py`,
`src/analysis/{circuit_identification,geometric_interpreter,neuron_analyzer}.py`,
`src/data/arithmetic_data.py`, and `experiments/arithmetic_validation.py`. None
were ever committed, so `run_geometric_pipeline.py` and
`tests/test_arithmetic_pipeline.py` cannot run. This is unchanged from the
source repository.

**3. The plan documents a superseded pipeline that is no longer shipped.** The
sections describing the old mask-learning route — Phases 1-5 and
`SUPPLEMENTARY SCRIPTS` S5 ("Old Pipeline") — reference scripts that were
removed with the Beyond Components dependency (see `SPLIT_NOTES.md`). The 15
main steps in Phases A-F are unaffected and are the paper's actual method.

## Provenance

This repository was split out of a combined research repository that held three
papers. See [`SPLIT_NOTES.md`](SPLIT_NOTES.md) for what was included, what was
left behind, and how the split was verified.

## License

[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
