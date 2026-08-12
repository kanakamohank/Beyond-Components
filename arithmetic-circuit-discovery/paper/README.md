# Paper: The Fourier Basis of Digit Arithmetic

## Compilation

### With NeurIPS style (recommended)
1. Download `neurips_2024.sty` from the [NeurIPS 2024 style files](https://neurips.cc/Conferences/2024/PaperInformation/StyleFiles) and place it in this directory.
2. Compile:
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Without NeurIPS style (fallback)
The paper will compile with standard article class formatting if `neurips_2024.sty` is not found. Just run the same commands above.

## Structure

- `main.tex` — Main document (imports sections)
- `sections/abstract.tex` — Abstract
- `sections/introduction.tex` — Introduction + contributions
- `sections/background.tex` — Related work + Fourier/CRT background
- `sections/methods.tex` — Models, Fourier extraction, causal methods, steering
- `sections/results.tex` — All experimental results (5 subsections)
- `sections/discussion.tex` — Discussion, limitations, conclusion
- `sections/appendix.tex` — MPS artifact, component attribution, ablation data
- `references.bib` — Bibliography

## Figures

All figures are loaded from `../mathematical_toolkit_results/paper_plots/`. Key plots:
- `layer_scan_curves.png` — Full-patch transfer across layers (3 models)
- `ablation_curves.png` — Causal ablation (per-freq, cumulative, knockout)
- `neuron_frequency_tuning.png` — Per-neuron Fourier frequency distributions
- `fourier_heatmap_cross_model.png` — Fourier energy spectra across layers
- `eigenvector_fourier_cross_model.png` — SVD directions as DFT modes
- `fisher_dimension_sweep.png` — Fisher patching dimension sweep
- `energy_explosion.png` — Energy amplification at computation layers
