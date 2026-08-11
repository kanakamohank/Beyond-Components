#!/usr/bin/env python3
"""Execute each per-paper notebook and report per-cell status.

Each notebook is run with cwd set to its OWN repository, which is also the test
that its find_root() marker logic works without the sibling repo present.
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = [
    ROOT / "semantic-compass" / "notebooks" / "semantic_compass_experiments.ipynb",
    ROOT / "arithmetic-circuit-discovery" / "notebooks" / "arithmetic_circuit_experiments.ipynb",
]

overall = 0
for nb_path in NOTEBOOKS:
    nb = nbformat.read(nb_path, as_version=4)
    NotebookClient(
        nb, timeout=900, kernel_name="python3",
        resources={"metadata": {"path": str(nb_path.parent)}},
        allow_errors=True,
    ).execute()

    failures, n = [], 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        n += 1
        errs = [o for o in cell.get("outputs", []) if o.get("output_type") == "error"]
        if errs:
            e = errs[0]
            failures.append((n, e.get("ename"), (e.get("evalue") or "")[:180]))

    print(f"\n{'=' * 70}")
    print(f"{nb_path.relative_to(ROOT)}  --  {n} code cells")
    print("=" * 70)
    if failures:
        overall = 1
        for i, ename, ev in failures:
            print(f"  cell {i:>2}: {ename}: {ev}")
    else:
        print("  all code cells executed without error")

sys.exit(overall)
