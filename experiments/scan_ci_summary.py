"""Agresti-Coull 95% CIs for the pass-rate of each compass scan.

Reads the header lines of existing *_scan.txt files and prints a
comparison table: observed passes / total, point estimate, and 95%
Agresti-Coull interval, for both the compass planes and the random
null planes.

Usage:
    .venv/bin/python experiments/scan_ci_summary.py \\
        helix_usage_validated/gpt2_scan_gender_scan.txt \\
        helix_usage_validated/phi3_scan_gender_scan.txt \\
        helix_usage_validated/gemma_scan_gender_scan.txt
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path


Z = 1.96  # 95% normal quantile


def agresti_coull(x, n, z=Z):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    n_tilde = n + z * z
    p_tilde = (x + z * z / 2.0) / n_tilde
    half = z * math.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
    return (p_tilde, max(0.0, p_tilde - half), min(1.0, p_tilde + half))


PASS_RE = re.compile(
    r"PASSED planes:\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
NULL_RE = re.compile(
    r"Random-plane null:\s+(\d+)\s+planes.*?--\s+(\d+)\s+pass",
    re.IGNORECASE | re.DOTALL)
MODE_RE = re.compile(r"null mode:\s*(\w+)", re.IGNORECASE)


def parse_scan(path):
    text = Path(path).read_text()
    head = text[: 2048]
    m_pass = PASS_RE.search(head)
    m_null = NULL_RE.search(head)
    m_mode = MODE_RE.search(head)
    if not (m_pass and m_null):
        raise ValueError(f"Could not parse counts from {path}")
    x_pass, n_pass = int(m_pass.group(1)), int(m_pass.group(2))
    n_null_planes = int(m_null.group(1))
    x_null = int(m_null.group(2))
    mode = m_mode.group(1) if m_mode else "top4"
    return dict(name=Path(path).stem,
                x_pass=x_pass, n_pass=n_pass,
                x_null=x_null, n_null=n_null_planes,
                null_mode=mode)


def fmt_row(label, x, n, ci):
    p, lo, hi = ci
    return (f"{label:<40} {x:>5}/{n:<5}  "
            f"{100*p:6.2f}%  [{100*lo:5.2f}%, {100*hi:5.2f}%]")


def main():
    paths = sys.argv[1:]
    if not paths:
        print("Usage: scan_ci_summary.py <scan1.txt> [scan2.txt ...]")
        sys.exit(1)

    print("=" * 82)
    print("Compass-scan pass-rate with 95% Agresti-Coull CIs")
    print("=" * 82)
    print(f"{'scan':<40} {'x/n':>11}  {'rate':>6}  {'95% CI':>18}")
    print("-" * 82)

    for p in paths:
        r = parse_scan(p)
        ci_pass = agresti_coull(r["x_pass"], r["n_pass"])
        ci_null = agresti_coull(r["x_null"], r["n_null"])
        tag_base = r["name"].replace("_scan", "")
        print(fmt_row(tag_base + "  compass", r["x_pass"], r["n_pass"], ci_pass))
        print(fmt_row(
            f"  null ({r['null_mode']})",
            r["x_null"], r["n_null"], ci_null))
        p_c = r["x_pass"] / max(1, r["n_pass"])
        p_n = r["x_null"] / max(1, r["n_null"])
        excess = p_c - p_n
        print(f"{'  excess (compass - null)':<40} "
              f"{'':>11}  {100*excess:+6.2f}%")
        print()


if __name__ == "__main__":
    main()
