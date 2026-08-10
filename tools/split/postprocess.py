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
    if new in s:
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

print("done")
