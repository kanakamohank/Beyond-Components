# Building the COMPASS poster

## Files in this folder

- `poster_content.md` — full narrative + image plan. Use this if you
  want to build the poster in PowerPoint, Keynote, Figma, Canva, or
  Google Slides — it's the same 12 panels plus title bar and
  take-home strip, just text / image hooks.
- `poster.tex` — A0 landscape `beamerposter` template. Compile with
  `pdflatex poster.tex` from this directory. All image paths are
  relative, pointing at `../figures/` and
  `../helix_usage_validated/`.

## Option A: LaTeX (reproducible, version-controlled)

```bash
cd paper_compass/poster
pdflatex -interaction=nonstopmode poster.tex
# → poster.pdf (A0 landscape, ~1189 × 841 mm)
```

**Prereqs:**
- TeX Live 2022+ with `beamerposter` and `beamer`
  (`tlmgr install beamerposter` if missing)
- `xcolor`, `graphicx`, `booktabs` (default TL)

**Before you build:**
1. Replace `<your-account>` in the take-home block with the real URL.
2. (Optional) Render the grouped debias bar chart from the snippet in
   `poster_content.md` §"Figures you still need to render" — save as
   `paper_compass/poster/images/crowspairs_compare.png` and swap it
   into Panel 9 of `poster.tex`.

## Option B: Overleaf (no local TeX install)

1. Create a new project on [overleaf.com](https://www.overleaf.com),
   upload `poster.tex`.
2. Upload the PNGs / PDFs referenced in `poster.tex` — copy from
   `paper_compass/figures/` and
   `helix_usage_validated/`.
3. Click Recompile. Overleaf ships `beamerposter` by default.

Starter templates you can graft our content into if you'd rather start
from a nicer-looking poster skeleton:
- `overleaf.com/latex/templates/a0poster-landscape-poster/` — plain
  A0 with section blocks.
- `overleaf.com/latex/templates/gemini-poster-theme/` — modern three-
  column theme, slightly fancier than the default beamerposter look.
- `overleaf.com/latex/templates/better-poster-latex-template/` — Mike
  Morrison's "Better Poster" layout (single-finding big-text style);
  would work well with the take-home strip as the central message.

## Option C: Non-LaTeX (Canva / PowerPoint / Google Slides / Figma)

Fastest path if you're not married to LaTeX. Follow
`poster_content.md` directly:

1. **Canva**  → `canva.com/posters/templates/` — filter by A0 or
   "research poster." 12 text+image frames correspond to our 12
   panels.
2. **PowerPoint** → Design → Slide Size → Custom (48" × 36" or A0).
   Draw 3 columns × 4 rows of text boxes; paste bullets from
   `poster_content.md` into each and drop the PNGs.
3. **Google Slides** → same as PowerPoint; free, browser-based.
4. **Figma** → `figma.com/community/tag/academic%20poster`; heavy
   snap-to-grid and exports straight to PDF.

## Option D: One-shot online converters

If you want a "turn my Markdown into a poster" pipeline:

- **[md2pdf](https://md2pdf.netlify.app)** — paste
  `poster_content.md`, tweak the CSS to 2 or 3 columns, export PDF.
  Not truly poster-sized but good for a quick draft.
- **[marp.app](https://marp.app)** — Markdown → slides → PDF. Set a
  very large canvas (e.g. 1189 × 841 mm) in the front-matter and each
  panel becomes one section.
- **[Decktape](https://github.com/astefanutti/decktape)** — if you
  build an HTML/CSS poster with e.g. Beamer-style HTML frameworks
  (Reveal.js), Decktape converts any slide page to PDF at arbitrary
  resolution.

## Printing

Give the print shop a single-page PDF at the **final physical size**
(A0 = 1189 × 841 mm, or 48"×36"). `pdfinfo poster.pdf` should show
`Page size: 3370 × 2384 pts` for A0. If it doesn't, the
`beamerposter` size option in `poster.tex` line 9 (`size=a0`) was
overridden — delete any `scale=` on that same line and rebuild.

## If you need help

- Swap any image path in `poster.tex` without re-running experiments;
  all PNG/PDF artifacts are already in the repo.
- The pipeline diagram (Panel 7) is intentionally text-only in the
  LaTeX version; if you want a "real" diagram, draw it in
  PowerPoint, export as PNG, drop it in `poster/images/pipeline.png`,
  and replace the `enumerate` block in Panel 7 with a single
  `\panelimg{images/pipeline.png}`.
