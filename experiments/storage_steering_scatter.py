"""Storage-vs-steering scatter for the 5 primary compass heads.

Hard-coded values pulled from existing logs:
  - Y (steering, A(10)/10):  amp at alpha=10 / 10  from
      gpt2_compass_causal.txt, phi3_compass_causal.txt,
      phi3_l28h1_causal.txt, gemma2b_compass_causal.txt,
      llama32_3b_compass_causal.txt.
  - X (storage, plane-ablation drop %):  TARGET PLANE-ONLY ABLATION
      drop from head_ablation_{gpt2,phi3,phi3_l28h1,gemma,llama}.txt
      and tab:headshare in paper_compass/sections/results.tex.

Output: helix_usage_validated/storage_steering_scatter.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path("helix_usage_validated/storage_steering_scatter.png")

# Each row: label, x (plane-ablation drop %), y (A(10)/10), source files
HEADS = [
    ("GPT-2 L9H7",     0.6,  1.170,  "gpt2_compass_causal.txt / tab:headshare"),
    ("Phi-3 L24H10",  -0.6,  1.437,  "phi3_compass_causal.txt / head_ablation_phi3.txt"),
    ("Phi-3 L28H1",   34.5,  1.285,  "phi3_l28h1_causal.txt / head_ablation_phi3_l28h1.txt"),
    ("Gemma L21H4",   14.3,  0.271,  "gemma2b_compass_causal.txt / tab:headshare"),
    ("Llama L26H14",  34.2,  0.936,  "llama32_3b_compass_causal.txt / head_ablation_llama.txt"),
]


def main():
    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    # Quadrant divider: storage threshold = 5% (matches paper text:
    # "≤0.6% plane ablation drop" defines steering).
    ax.axvline(5.0, color="lightgray", linestyle="--", linewidth=1)

    for label, x, y, _src in HEADS:
        ax.scatter(x, y, s=120, edgecolor="black", linewidth=1.0, zorder=3)
        ax.annotate(label, (x, y), xytext=(8, 6),
                    textcoords="offset points", fontsize=10)

    ax.set_xlabel("Storage:  plane-ablation signal drop (% of baseline)")
    ax.set_ylabel("Steering:  causal-sweep slope  A(10)/10")
    ax.set_title("Storage vs. steering for the 5 primary compass heads")

    # Quadrant labels
    ax.text(-3, 1.55, "steering-only", fontsize=9, color="gray",
            style="italic")
    ax.text(20, 1.55, "storage + steering", fontsize=9, color="gray",
            style="italic")

    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 40)
    ax.set_ylim(0, 1.7)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
