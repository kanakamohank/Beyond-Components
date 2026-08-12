#!/usr/bin/env python3
"""Verify each split repo: syntax, intra-repo src.* imports, LaTeX figures."""
import ast
import re
import sys
from pathlib import Path

ROOT = Path("/home/user/Beyond-Components")
REPOS = [ROOT / "semantic-compass", ROOT / "arithmetic-circuit-discovery"]

# Modules the ORIGINAL repo already fails to provide (pre-existing breakage).
PREEXISTING_MISSING = {
    "src.analysis.circuit_identification",
    "src.analysis.geometric_interpreter",
    "src.analysis.neuron_analyzer",
    "src.data.arithmetic_data",
    "src.models.arithmetic_pipeline",
    "src.utils.helix_visualization",
}

failed = False

for repo in REPOS:
    print(f"\n{'=' * 64}\n{repo.name}\n{'=' * 64}")
    pys = sorted(repo.rglob("*.py"))

    # ---- 1. syntax ----
    syntax_errors = []
    trees = {}
    for f in pys:
        try:
            trees[f] = ast.parse(f.read_text(errors="replace"), filename=str(f))
        except SyntaxError as e:
            syntax_errors.append(f"{f.relative_to(repo)}:{e.lineno}: {e.msg}")
    print(f"syntax          : {len(pys) - len(syntax_errors)}/{len(pys)} parse clean")
    for e in syntax_errors:
        print("   SYNTAX", e)
        failed = True

    # ---- 2. intra-repo src.* imports ----
    def module_present(mod: str) -> bool:
        rel = mod.replace(".", "/")
        return (repo / f"{rel}.py").exists() or (repo / rel / "__init__.py").exists()

    missing, preexisting = set(), set()
    for f, tree in trees.items():
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module]
            for m in mods:
                if not m.startswith("src"):
                    continue
                if module_present(m):
                    continue
                (preexisting if m in PREEXISTING_MISSING else missing).add(
                    (m, str(f.relative_to(repo)))
                )

    if missing:
        failed = True
        print(f"src imports     : {len(missing)} UNRESOLVED (split-induced)")
        for m, f in sorted(missing):
            print(f"   MISSING {m}  <- {f}")
    else:
        print("src imports     : all resolve within this repo")
    if preexisting:
        print(f"                  ({len(preexisting)} pre-existing gaps, absent on main too)")
        for m, f in sorted(preexisting):
            print(f"   pre-existing  {m}  <- {f}")

    # ---- 3. sibling experiments/ imports ----
    exp_missing = set()
    for f, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                m = node.module
                if m.startswith("experiments") and not module_present(m):
                    exp_missing.add((m, str(f.relative_to(repo))))
    if exp_missing:
        failed = True
        print(f"experiments imports: {len(exp_missing)} UNRESOLVED")
        for m, f in sorted(exp_missing):
            print(f"   MISSING {m}  <- {f}")
    else:
        print("experiments imp.: all resolve within this repo")

    # ---- 4. LaTeX \includegraphics ----
    GFX = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
    EXTS = ["", ".pdf", ".png", ".jpg", ".jpeg", ".eps"]
    tex_missing = []
    n_refs = 0
    for tex in sorted(repo.rglob("*.tex")):
        base = tex.parent
        # section files live in <paper>/sections/, graphics resolve from <paper>/
        roots = [base, base.parent]
        # honour \\graphicspath declared in the document root's main.tex
        for docroot in (base, base.parent):
            mt = docroot / "main.tex"
            if mt.exists():
                for gp in re.findall(r"\\graphicspath\{(.+?)\}\s*$",
                                     mt.read_text(errors="replace"), re.M):
                    for sub in re.findall(r"\{([^}]*)\}", gp):
                        roots.append(docroot / sub)
        for ref in GFX.findall(tex.read_text(errors="replace")):
            n_refs += 1
            if not any((r / (ref + e)).exists() for r in roots for e in EXTS):
                tex_missing.append((str(tex.relative_to(repo)), ref))
    if tex_missing:
        print(f"latex figures   : {n_refs - len(tex_missing)}/{n_refs} resolve")
        for t, ref in tex_missing:
            print(f"   MISSING FIG  {ref}   <- {t}")
    else:
        print(f"latex figures   : {n_refs}/{n_refs} resolve")


# ---- 5. every materialized file actually survives `git add` ----
# A .gitignore rule inherited from the source repo can silently swallow files
# that were tracked there. Compare the manifest's expectation against what git
# would really track.
import subprocess
from collections import Counter

rows = [l.split("\t") for l in (ROOT / "SPLIT_MANIFEST.tsv").read_text().splitlines()[1:]]
counts = Counter(b for _, b, _ in rows)
NEW_FILES = {"README.md", "SPLIT_NOTES.md", "SPLIT_MANIFEST.tsv", "notebooks/*.ipynb"}

print(f"\n{'=' * 64}\ngit-tracking survival\n{'=' * 64}")
for repo, bucket in ((ROOT / "semantic-compass", "compass"),
                     (ROOT / "arithmetic-circuit-discovery", "arithmetic")):
    expected = {p for p, b, _ in rows if b in (bucket, "both")}
    ignored = sorted(
        p for p in expected
        if subprocess.run(["git", "check-ignore", "-q", f"{repo.name}/{p}"],
                          cwd=ROOT).returncode == 0
    )
    n_exp = len(expected) + len(NEW_FILES)
    if ignored:
        failed = True
        print(f"{repo.name}: {len(ignored)} file(s) SWALLOWED by .gitignore")
        for p in ignored[:10]:
            print(f"   IGNORED {p}")
    else:
        print(f"{repo.name}: all {len(expected)} manifest files tracked "
              f"(+{len(NEW_FILES)} new = {n_exp} expected)")

# ---- 6. every figure a paper needs has a producer in the SAME repo ----
# Referencing a figure is not enough: the repo must also contain the script
# that writes it, or the paper can never be rebuilt from this repo alone.
print(f"\n{'=' * 64}\nfigure producers\n{'=' * 64}")
for repo in REPOS:
    needed = set()
    for tex in repo.rglob("*.tex"):
        if "acl-style-files" in str(tex):
            continue  # template placeholders, not real figures
        for ref in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",
                              tex.read_text(errors="replace")):
            needed.add(Path(ref).name)
    orphans = []
    for fig in sorted(needed):
        if any((repo / "images").rglob(fig)) or list(repo.rglob(f"**/figures/{fig}")):
            continue  # figure is committed; no producer required
        if not any(fig in f.read_text(errors="replace")
                   for f in repo.rglob("experiments/*.py")):
            orphans.append(fig)
    if orphans:
        failed = True
        print(f"{repo.name}: {len(orphans)} figure(s) with NO producer script")
        for f in orphans:
            print(f"   ORPHAN {f}")
    else:
        print(f"{repo.name}: every referenced figure is committed or has a producer")


# ---- 7. no Beyond Components code survives in either repo ----
# Neither paper's pipeline uses it; shipping it would also re-incur the
# CC BY-SA attribution obligation. Assert it stays gone.
BC_MODULES = re.compile(
    r"^\s*(?:from|import)\s+.*(?:masked_transformer_circuit"
    r"|src\.utils\.(?:utils|visualization|constants)"
    r"|src\.data\.data_loader)",
    re.M,
)
BC_FILES = {
    "masked_transformer_circuit.py", "train.py", "intervention.py",
    "gp_config.yaml", "ioi_config.yaml", "gt_config.yaml",
    "comprehensive_metrics_table.py",
}
print(f"\n{'=' * 64}\nBeyond Components removal\n{'=' * 64}")
for repo in REPOS:
    bad_imports = [
        str(f.relative_to(repo)) for f in repo.rglob("*.py")
        if BC_MODULES.search(f.read_text(errors="replace"))
    ]
    bad_files = [
        str(f.relative_to(repo)) for f in repo.rglob("*")
        if f.is_file() and f.name in BC_FILES
    ]
    if bad_imports or bad_files:
        failed = True
        print(f"{repo.name}: Beyond Components RESIDUE")
        for x in bad_imports:
            print(f"   IMPORT {x}")
        for x in bad_files:
            print(f"   FILE   {x}")
    else:
        print(f"{repo.name}: no Beyond Components imports or files")

sys.exit(1 if failed else 0)
