# Semantic Compasses

Implementation and paper source for:

> **Semantic Compasses: Rank-2 Causal Dials in Attention-Head OV Singular Planes**

A *semantic compass* is a two-dimensional plane spanned by a pair of singular
directions of an attention head's OV matrix. Injecting a rotating vector in that
plane moves a model's output logits along a smooth, single-frequency sinusoid —
a continuous, causally effective "dial" over a semantic contrast such as
he/she, past/future, or person/place.

The repository covers the full pipeline: discovering candidate planes, decoding
what each plane means, validating that the dial is causal rather than
correlational, falsifying it against nine null tests, and applying it downstream
on StereoSet, CrowS-Pairs, Winogender, and TruthfulQA.

**Start here: [`COMPASS_COOKBOOK.md`](COMPASS_COOKBOOK.md)** — the reference
document mapping every paper claim to the script that produces it and the
artifact it lands in.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.8+ and PyTorch 2.0+. A CUDA-capable GPU is recommended;
the smaller GPT-2 experiments run on CPU or MPS.

## Quickstart

Five commands, each regenerating one reviewable artifact. Full details and the
remaining flags are in §1 of the cookbook.

```bash
# 1. Figure 2 — GPT-2 L9H7 gender compass, 36-angle causal sweep.
python experiments/compass_causal_sweep.py --model gpt2 --layer 9 --head 7 --dims 1 2 \
    --tok_plus " he" --tok_minus " she" --out_prefix gpt2_quickstart

# 2. Decode a head's compass dictionary.
python experiments/compass_dictionary.py

# 3. Blind causal scan (Fig. 3 heatmap, Table 3 pass rates).
python experiments/compass_scan.py --model gpt2 --tok_plus " he" --tok_minus " she" \
    --top_svs 4 --alphas 3 10 --n_angles 12 --out_prefix gpt2_quickstart_scan

# 4. Nine-test falsification battery.
python investigate_helix_usage_validated.py --test all-must-have gpt2

# 5. Routed CrowS-Pairs at calibrated alpha (Table 7).
python experiments/crowspairs_routed_eval.py --model gpt2 \
    --alpha_json helix_usage_validated/per_domain_alpha_gpt2.json
```

## Layout

```
semantic-compass/
├── COMPASS_COOKBOOK.md            # paper claim → script → artifact, end to end
├── paper_compass/                 # ICLR submission (main.tex + sections/ + figures/)
├── paper_compass_acl/             # EMNLP/ACL submission of the same work
├── acl-style-files-master/        # ACL style files for the above
│
├── experiments/                   # ~50 scripts, all named in the cookbook
│   ├── compass_causal_sweep.py    # L2 — α-sweep on a single head (Fig. 2)
│   ├── compass_scan.py            # L1 — blind scan over heads/planes (Fig. 3)
│   ├── compass_dictionary.py      # decode SVD axes to vocabulary
│   ├── stereoset_*.py             # StereoSet probe / scan / ensemble eval
│   ├── crowspairs_*.py            # CrowS-Pairs, incl. routed evaluation
│   ├── winogender_*.py            # Winogender sweep and eval
│   ├── inlp_debias.py             # INLP baseline
│   ├── sentence_debias.py         # SentenceDebias baseline
│   ├── baseline_comparison.py     # compass vs. four standard direction methods
│   └── run_*.sh                   # queue drivers for multi-model runs
│
├── investigate_helix_usage_validated.py   # L3 — nine-test falsification battery
├── helix_usage_validated/         # all experiment artifacts (logs, plots, CSVs)
│
├── src/                           # shared Stage-1 direction-discovery code
│   ├── models/masked_transformer_circuit.py
│   ├── data/data_loader.py
│   └── utils/
├── configs/gp_config.yaml         # Gender Pronoun config for Stage-1
└── requirements.txt
```

## On Stage 1 and prior work

Stage-1 direction discovery — learnable masks over singular directions on the
Gender Pronoun task — is method and infrastructure from **Beyond Components:
Singular Vector-Based Interpretability of Transformer Circuits**
(Ahmad, Joshi & Modi; [arXiv:2511.20273](https://arxiv.org/abs/2511.20273)).
This work extends that line; it does not claim direction discovery as its own
contribution. The relevant files (`src/models/masked_transformer_circuit.py`,
`src/utils/`, `experiments/train.py`, `configs/gp_config.yaml`) are carried here
so the pipeline runs end to end.

## Provenance

This repository was split out of a combined research repository that held three
papers. See [`SPLIT_NOTES.md`](SPLIT_NOTES.md) for what was included, what was
left behind, and the known gaps.

## License

[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
