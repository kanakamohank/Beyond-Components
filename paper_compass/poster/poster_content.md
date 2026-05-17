# COMPASS — poster content (8 panels)

Three columns, A0 landscape; each panel shows one figure (or one
small table) plus at most three bullets.

All figure paths are relative to the repo root. The LaTeX template
picks them up directly from `paper_compass/figures/`,
`helix_usage_validated/`, and `paper_compass/poster/images/`.

Palette (ICLR-neutral):
- Background: `#FAFAFA`
- Accent 1 (method): `#2E5EAA` (deep blue)
- Accent 2 (causal): `#C0392B` (brick red)
- Text: `#111111`
- Panel borders: `#D0D0D0`, 0.5 pt

---

## Title bar (full width, 12% height)

**COMPASS: Two-Dimensional Semantic Dials Inside a Single Attention
Head**

Sub-line (italic, smaller):
*A rank-2 causal editing tool that reads, writes, and routes per-domain
bias — tested on GPT-2, Phi-3-mini, Gemma-2-2B, Llama-3.2-3B.*

Author block (right-justified): Name · Affiliation · ICLR 2026 *under review*

---

## Column 1 — What a compass is

### Panel 1.  Motivation (no image, 3 bullets)

- Mechanistic interp usually goes *feature → causal* one attribute,
  one head at a time.
- We take the **OV-SVD of one head**, pick **two right-singular
  vectors** $(u_i, u_j)$, and treat the 2D plane as a dial:
  $v(\theta,\alpha)=\alpha\sigma_i\cos\theta\,u_i+\alpha\sigma_j\sin\theta\,u_j.$
- Gender / race / temporal / entity all surface as the same kind of
  plane in $\ge 2\%$ of heads across four models.

### Panel 2.  Sinusoidal response (figure)

- **Image:** `helix_usage_validated/phi3_l28h1_causal_polar.png`
- **Caption:** Phi-3 Mini L28H1 gender compass. Radius =
  logit($he-she$), angle = injection $\theta$.
- **Bullet:** Poles at $0^\circ / 180^\circ$ decode to
  *he / his / him* vs. *she / her / hers* — the plane is causal,
  not just correlated.

### Panel 3.  Linearity + phase (figure)

- **Image:** `paper_compass/figures/phi3_linearity.png`
  *(amplitude vs. $\alpha$ for Phi-3 L24H10)*
- Amplitude grows **linearly** with $\alpha$  ($R^2 \ge 0.993$).
- Phase is **stable** across $\alpha\in\{1,3,10\}$ (drift $\le 3^\circ$).
- Random planes and permuted labels kill both — two-pillar test.

---

## Column 2 — That it generalizes

### Panel 4.  Non-pronoun compasses (figure)

- **Image:** `paper_compass/figures/phi3_polar.png`
  *(his / their compass — the axis is number, not gender)*
- **Bullet:** Same two-vector recipe finds planes for
  temporal (*was / is*), entity (*people / things*), and
  plural-person axes in the same models — compasses are not a
  gender artifact.

### Panel 5.  Pipeline (figure)

- **Image:** `paper_compass/poster/images/pipeline_schematic.png`
- OV-SVD → $\alpha$-sweep + 9 null tests → route by pass-count
  → SNR-matched per-domain $\alpha_d=\tau/\overline{\mathrm{SNR}}_d$.
- Blind-scan pass rates (95% CI): GPT-2 2.06%, Phi-3 1.18%,
  Gemma 2.02%, Llama 0.22% — all above the full-OV null.

### Panel 6.  Four models, same structure (table)

| Model | head | $A(10)$ | $\Delta\phi$ |
|---|---|---|---|
| GPT-2 Small  | L10H9  | 14.16 | $0.3^\circ$ |
| Phi-3 Mini   | L28H1  | 12.85 | $1.3^\circ$ |
| Gemma-2-2B   | L21H4  |  2.19 | $1.3^\circ$ |
| Llama-3.2-3B | L26H14 |  3.01 | $2.4^\circ$ |

- Amplitudes differ by $\sim 5\times$ across families, but the
  same two-pillar recipe discovers the plane in every model —
  independent of depth, width, vocab size.

---

## Column 3 — That it can drive behaviour

### Panel 7.  Head–domain routing (figure)

- **Image:** `helix_usage_validated/head_domain_heatmap_phi3.png`
- **Bullet:** Each passing head specializes for one of
  {gender, race, profession, religion}. We inject only on rows
  whose `bias_type` matches.
- Routing + SNR-matched $\alpha$ replaces a single global scale.

### Panel 8.  Debiasing beats INLP / SentenceDebias (figure)

- **Image:** `paper_compass/poster/images/crowspairs_compare.png`
  *(CrowS-Pairs $|SS-50|$, gender & race, 4 models × 5 methods)*
- **Bullet:** routed + matched-strength compass beats **INLP** on
  7/8 cells and **SentenceDebias** on 6/8, without moving
  TruthfulQA MC1 by more than $\pm 3$ pts.

---

## Take-home strip (bottom full-width, 10% height)

Three bullets, large font:

1. Any attention head that writes a categorical axis contains a
   **2D compass plane** you can find by looking at one SVD.
2. The plane is **causal, linear, and phase-stable** — it isn't just
   a correlation.
3. Routing + per-domain $\alpha$ calibration turns those planes into
   a debiasing stack that matches or beats INLP / SentenceDebias
   while preserving capability.

**Code + artifacts:** `github.com/<...>/Beyond-Components` (replace
with final URL)

---

## Building the two bespoke images

```bash
.venv/bin/python paper_compass/poster/make_poster_figs.py
# → paper_compass/poster/images/pipeline_schematic.png
# → paper_compass/poster/images/crowspairs_compare.png
```

Everything else in the poster is a pre-existing artifact — no
experiments need to re-run.
