"""Second-harmonic fit for compass curves.

Re-reads saved alpha-sweep logs (neutral column) and fits
  LD(theta) = mu + A1 cos(theta - phi1) + A2 cos(2 theta - phi2)
using the first two DFT bins. Reports A1, A2, A2/A1, phi1, phi2 at
each alpha for the three Phi-3 compasses (gender, temporal, entity).

A2/A1 >> 0 indicates four-way categorical structure; A2/A1 ~ 0
indicates a pure antipodal dial.

Usage:
    python experiments/second_harmonic_fit.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def parse_sweeps(txt_path: Path):
    """Return dict {alpha: (angles, neutral_curve)} from a saved log."""
    text = txt_path.read_text().splitlines()
    sweeps = {}
    cur_alpha = None
    cur = []
    in_table = False
    for line in text:
        m = re.match(r"\s*SWEEP alpha=([\d.]+)", line)
        if m:
            if cur_alpha is not None:
                sweeps[cur_alpha] = np.array(cur)
            cur_alpha = float(m.group(1))
            cur = []
            in_table = False
            continue
        if cur_alpha is not None and "deg |" in line:
            in_table = True
            continue
        if in_table and "---" in line:
            continue
        if in_table:
            m2 = re.match(r"\s*([\d.]+)\s*\|\s*([+-][\d.]+)\s+"
                          r"([+-][\d.]+)\s+([+-][\d.]+)", line)
            if m2:
                deg = float(m2.group(1))
                neutral = float(m2.group(2))
                cur.append((deg, neutral))
            elif line.strip().startswith("fit:"):
                in_table = False
    if cur_alpha is not None and cur:
        sweeps[cur_alpha] = np.array(cur)
    return sweeps


def harmonic_fit(angles_deg, y):
    th = np.radians(angles_deg)
    N = len(th)
    mu = float(np.mean(y))
    c1 = (y * np.cos(th)).sum() * 2.0 / N
    s1 = (y * np.sin(th)).sum() * 2.0 / N
    c2 = (y * np.cos(2 * th)).sum() * 2.0 / N
    s2 = (y * np.sin(2 * th)).sum() * 2.0 / N
    A1 = float(np.hypot(c1, s1))
    phi1 = float(np.degrees(np.arctan2(s1, c1)))
    A2 = float(np.hypot(c2, s2))
    phi2 = float(np.degrees(np.arctan2(s2, c2)))
    ratio = A2 / A1 if A1 > 1e-9 else float("nan")
    return mu, A1, phi1, A2, phi2, ratio


def main():
    root = Path("helix_usage_validated")
    targets = [
        ("Phi-3 L24H10 (gender)",     "phi3_compass_causal.txt"),
        ("Phi-3 L24H17 (temporal)",   "phi3_temporal_compass_causal.txt"),
        ("Phi-3 L24H28 (entity)",     "phi3_entity_compass_causal.txt"),
    ]
    out_lines = [
        "SECOND-HARMONIC FIT",
        "LD(theta) = mu + A1 cos(theta - phi1) + A2 cos(2 theta - phi2)",
        "",
        f"{'compass':<28} {'alpha':>6} {'mu':>8} "
        f"{'A1':>8} {'phi1':>8} {'A2':>8} {'phi2':>8} "
        f"{'A2/A1':>8}",
        "-" * 96,
    ]
    for name, fname in targets:
        p = root / fname
        if not p.exists():
            print(f"missing {p}")
            continue
        sweeps = parse_sweeps(p)
        for alpha in sorted(sweeps):
            arr = sweeps[alpha]
            angles = arr[:, 0]
            y = arr[:, 1]
            mu, A1, phi1, A2, phi2, ratio = harmonic_fit(angles, y)
            out_lines.append(
                f"{name:<28} {alpha:>6.1f} {mu:+8.3f} "
                f"{A1:>8.3f} {phi1:>+8.1f} "
                f"{A2:>8.3f} {phi2:>+8.1f} "
                f"{ratio:>8.3f}"
            )
        out_lines.append("")

    out_lines.append("Interpretation:")
    out_lines.append(
        "  A2 / A1 ~ 0       -> pure antipodal dial "
        "(two-way categorical)")
    out_lines.append(
        "  A2 / A1 >~ 0.2    -> meaningful four-way structure "
        "(peaks every 90 deg)")
    out_lines.append(
        "  phi2 encodes the orientation of the four-way cross.")
    out = root / "second_harmonic_fit.txt"
    out.write_text("\n".join(out_lines))
    print("\n".join(out_lines))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
