#!/usr/bin/env python3
"""Post-copy edits applied to the materialized trees.

materialize.py copies manifest rows byte-for-byte. Everything that *differs*
from the source repository is applied here, so the divergence is auditable in
one place rather than hand-applied and undocumented.
"""
from pathlib import Path

ROOT = Path("/home/user/Beyond-Components")
COMPASS = ROOT / "semantic-compass"
ARITH = ROOT / "arithmetic-circuit-discovery"
TPL = ROOT / "tools" / "split" / "templates"


def edit(path: Path, old: str, new: str, label: str) -> None:
    s = path.read_text()
    # An empty `new` is a deletion: idempotence is "the anchor is gone",
    # since "" is trivially a substring of everything.
    if (old not in s) if new == "" else (new in s):
        print(f"  = {label} (already applied)")
        return
    assert old in s, f"anchor not found for {label} in {path}"
    path.write_text(s.replace(old, new, 1))
    print(f"  + {label}")


print("post-copy edits:")

# 1. The compass cookbook opened by describing the combined two-paper repo.
edit(
    COMPASS / "COMPASS_COOKBOOK.md",
    """This is the reference bible for the **Geometric Compass** half of the
`Beyond-Components` repo. The repo contains two papers; this cookbook is
about the second one (`paper_compass/`). The first one (`paper/`,
arithmetic circuit) is out of scope here.""",
    """This is the reference bible for the **Geometric Compass** paper
(`paper_compass/`, `paper_compass_acl/`). It was written when the compass
lived alongside the arithmetic-circuit paper in the combined
`Beyond-Components` repo; that half now lives in the separate
`arithmetic-circuit-discovery` repository and is out of scope here.""",
    "COMPASS_COOKBOOK.md intro rewritten for the standalone repo",
)

# 2. The inherited `*_results/*` rule would silently swallow two directories
#    that this repo must ship: the paper's figures, and the tracked Fourier
#    discovery outputs (which were tracked in the source repo despite the rule).
edit(
    ARITH / ".gitignore",
    "*_results/*\n",
    "*_results/*\n"
    "# ...but these two must stay tracked: the paper's figures, and the\n"
    "# recorded Fourier discovery outputs (tracked in the source repo too).\n"
    "!mathematical_toolkit_results/\n"
    "!mathematical_toolkit_results/paper_plots/\n"
    "!mathematical_toolkit_results/paper_plots/**\n"
    "!fourier_results/\n"
    "!fourier_results/**\n",
    ".gitignore un-ignores paper_plots/ and fourier_results/",
)

# 3. Per-repo READMEs. The source README is Beyond Components' and ships to
#    neither repo.
for repo, tpl in ((COMPASS, "README.semantic-compass.md"), (ARITH, "README.arithmetic-circuit-discovery.md")):
    (repo / "README.md").write_text((TPL / tpl).read_text())
    print(f"  + {repo.name}/README.md")


# 4. Trim the package roots that re-exported the removed Beyond Components
#    modules, so `import src.data` / `import src.utils` still work.
(ARITH / "src" / "data" / "__init__.py").write_text(
    """from .arithmetic_dataset import (
    load_arithmetic_dataset,
    ArithmeticDataset,
    ArithmeticPromptGenerator,
    generate_arithmetic_prompts,
)

__all__ = [
    'load_arithmetic_dataset',
    'ArithmeticDataset',
    'ArithmeticPromptGenerator',
    'generate_arithmetic_prompts',
]
"""
)
print("  + src/data/__init__.py trimmed to the arithmetic dataset")

(ARITH / "src" / "utils" / "__init__.py").write_text(
    """from .model_registry import get_model_spec

__all__ = ['get_model_spec']
"""
)
print("  + src/utils/__init__.py trimmed to the model registry")

(ARITH / "src" / "__init__.py").write_text(
    '"""Fourier-basis analysis of digit arithmetic circuits in language models."""\n\n'
    '__version__ = "0.1.0"\n'
)
print("  + src/__init__.py docstring no longer describes the BC method")

# 5. Drop the test class that asserted against the removed BC column registry.
tests = ARITH / "tests" / "test_arithmetic_dataset.py"
src = tests.read_text()
marker = "# ======================================================================\n# Integration with src/utils/utils.py column name functions"
if marker in src:
    tests.write_text(src[: src.index(marker)].rstrip() + "\n")
    print("  + tests/test_arithmetic_dataset.py: dropped TestUtilsIntegration (tested BC helpers)")
else:
    print("  = TestUtilsIntegration already removed")

# 6. Documentation that referenced the removed Beyond Components code.
edit(
    COMPASS / "COMPASS_COOKBOOK.md",
    """> **One-line citation note.** Stage-1 of the compass paper (learnable-mask
> direction discovery on the Gender Pronoun task) is method and
> infrastructure from **Beyond Components (Ahmad et al.)**. The compass
> paper extends that line of work; it does not claim direction discovery
> as its own. See `paper_compass/sections/results.tex` §3.1 (currently
> needs a citation pass — flagged for the next paper revision).""",
    """> **Note on direction discovery.** Compass planes are computed directly
> from each head's OV matrix at run time (`torch.linalg.svd(W_V @ W_O)`,
> see §2.1) — there is no training step and no learned mask. Earlier
> revisions of this cookbook described a Stage-1 mask-learning stage
> adapted from Beyond Components (Ahmad et al.); that code is not part of
> this repository and no script here depends on it.""",
    "COMPASS_COOKBOOK.md: Stage-1 note replaced with the actual SVD route",
)

# Two rows of the paper<->code table pointed at removed scripts *and* at paper
# labels (tab:directions, tab:bc_full, app:bc_results) that exist in neither
# compass paper -- stale on both counts.
edit(
    COMPASS / "COMPASS_COOKBOOK.md",
    """| `tab:directions` | results §3.1 | `src/models/masked_transformer_circuit.py` (Beyond Components method) | (Stage-1 mask training output) |
| `tab:bc_full` | appendix `app:bc_results` | Stage-1 sweep + `experiments/evaluation/comprehensive_metrics_table.py` | (Stage-1 GP eval) |
""",
    "",
    "COMPASS_COOKBOOK.md: dropped 2 stale rows (labels absent from both papers)",
)

edit(
    ARITH / "ARITHMETIC_CIRCUIT_PLAN.md",
    """# Arithmetic Circuit Discovery — Execution Pipeline

## Overview""",
    """# Arithmetic Circuit Discovery — Execution Pipeline

> **Scope note.** The paper's method is the 15 numbered steps in Phases A-F
> below; those are complete and supported. This document also retains
> sections describing an earlier mask-learning route — "What Already
> Exists", "Phase 0", and `SUPPLEMENTARY SCRIPTS` S5 ("Old Pipeline") —
> which was built on the Beyond Components `MaskedTransformerCircuit`.
> That dependency has been removed and those scripts are no longer shipped,
> so commands in those sections will not run. They are kept as a record of
> how the work developed. See `SPLIT_NOTES.md`.

## Overview""",
    "ARITHMETIC_CIRCUIT_PLAN.md: scope note on the retired mask-learning route",
)

# 7. Per-paper experiment notebooks. Generated here rather than shipped as
#    manifest rows because materialize.py rmtree's the target directories.
import subprocess as _sp
_sp.run([__import__("sys").executable, str(ROOT / "tools" / "build_notebooks.py")], check=True)

print("done")
