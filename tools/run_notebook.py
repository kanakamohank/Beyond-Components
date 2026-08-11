#!/usr/bin/env python3
"""Execute experiments_end_to_end.ipynb and report per-cell status.

Runs every cell in order and does NOT stop at the first error, so one pass
surfaces every broken cell rather than one at a time.
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path("/home/user/Beyond-Components")
NB = ROOT / "experiments_end_to_end.ipynb"

nb = nbformat.read(NB, as_version=4)
client = NotebookClient(
    nb, timeout=900, kernel_name="python3",
    resources={"metadata": {"path": str(ROOT)}},
    allow_errors=True,          # keep going; collect every failure
)
client.execute()

failures, empties = [], []
idx = 0
for cell in nb.cells:
    if cell.cell_type != "code":
        continue
    idx += 1
    errs = [o for o in cell.get("outputs", []) if o.get("output_type") == "error"]
    text = "".join(
        o.get("text", "") for o in cell.get("outputs", []) if o.get("output_type") == "stream"
    )
    if errs:
        e = errs[0]
        failures.append((idx, e.get("ename"), (e.get("evalue") or "")[:200],
                         [l for l in e.get("traceback", []) if "line" in l][-1:]))
    elif not text.strip() and not any(
        o.get("output_type") == "display_data" for o in cell.get("outputs", [])
    ):
        empties.append(idx)

nbformat.write(nb, NB.with_suffix(".executed.ipynb"))

print(f"\n{'=' * 70}\nEXECUTION REPORT — {idx} code cells\n{'=' * 70}")
if failures:
    print(f"\n{len(failures)} FAILED:\n")
    for n, ename, eval_, tb in failures:
        print(f"  cell {n:>2}: {ename}: {eval_}")
        for l in tb:
            print(f"           {l.strip()[:160]}")
else:
    print("\nAll code cells executed without error.")
if empties:
    print(f"\n{len(empties)} produced no output: {empties}")
print()
sys.exit(1 if failures else 0)
