"""Single-model compass dictionary.

Reads the scan output to find all passing compass heads, then decodes
top-8 SV axes of each through the unembedding matrix (mean-centred).

Usage:
  .venv/bin/python experiments/compass_dictionary_single.py \\
      --model meta-llama/Llama-3.2-3B --tag llama32_3b \\
      --heads_json helix_usage_validated/llama32_3b_top_head.json
"""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

warnings.filterwarnings("ignore")

SV_MIN = 4
SV_MAX = 8
VAR_TARGET = 0.90
TOP_K = 15


def parse_passing_heads(scan_txt: Path) -> list[tuple[int, int]]:
    """Return unique (layer, head) pairs from scan's PASSED table."""
    row_re = re.compile(
        r"^\s*\d+\s+(\d+)\s+(\d+)\s+\(\d+,\d+\)"
    )
    seen = set()
    out = []
    for line in scan_txt.read_text().splitlines():
        m = row_re.match(line)
        if not m:
            continue
        L, H = int(m[1]), int(m[2])
        if (L, H) in seen:
            continue
        seen.add((L, H))
        out.append((L, H))
    return out


def decode_head(model, layer, head):
    W_V = model.W_V[layer, head].detach().float()
    W_O = model.W_O[layer, head].detach().float()
    _U, S, Vt = torch.linalg.svd(W_V @ W_O, full_matrices=False)

    s2 = S ** 2
    cum_var = torch.cumsum(s2, dim=0) / float(s2.sum())
    K = SV_MIN
    while K < min(SV_MAX, Vt.shape[0]) and float(cum_var[K - 1]) < VAR_TARGET:
        K += 1

    W_U = model.W_U.detach().float()
    bias = W_U.mean(dim=1, keepdim=True)
    W_U_c = W_U - bias

    results = []
    for k in range(K):
        v = Vt[k]
        logits_pos = v @ W_U_c
        logits_neg = -v @ W_U_c
        top_pos = torch.topk(logits_pos, TOP_K)
        top_neg = torch.topk(logits_neg, TOP_K)
        toks_pos = [model.to_string([int(i)]).strip()
                    or repr(model.to_string([int(i)]))
                    for i in top_pos.indices]
        toks_neg = [model.to_string([int(i)]).strip()
                    or repr(model.to_string([int(i)]))
                    for i in top_neg.indices]
        results.append(dict(
            k=k, sigma=float(S[k]),
            cum_var=float(cum_var[k]),
            pos=toks_pos, neg=toks_neg,
        ))
    return results, K, float(cum_var[K - 1]), int(S.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--heads_json", default=None,
                    help="optional: JSON with single {layer,head} fallback")
    ap.add_argument("--out_dir", default="helix_usage_validated")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    scan_txt = out_dir / f"{args.tag}_scan_gender_scan.txt"
    heads: list[tuple[int, int]] = []
    if scan_txt.exists():
        heads = parse_passing_heads(scan_txt)
        print(f"Found {len(heads)} passing heads in {scan_txt}")
    if not heads and args.heads_json:
        import json
        j = json.loads(Path(args.heads_json).read_text())
        heads = [(j["layer"], j["head"])]
        print(f"Fallback: using single head from JSON: {heads}")
    if not heads:
        raise SystemExit("no heads to decode")

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    print(f"Loading {args.model} on {device} (fp32)...")
    model = HookedTransformer.from_pretrained(
        args.model, device=device, dtype=torch.float32)

    lines = [f"Model: {args.model}", ""]
    for (L, H) in heads:
        print(f"  decoding L{L} H{H}...")
        res, K, cum_K, rank = decode_head(model, L, H)
        lines.append(f"=== L{L} H{H} ===")
        lines.append(
            f"  decoded K={K}/{rank} axes  cum_var@K={cum_K:.3f}  "
            f"(target={VAR_TARGET}, min={SV_MIN}, max={SV_MAX})")
        for r in res:
            lines.append(
                f"  SV{r['k']}  sigma={r['sigma']:.3f}  "
                f"cum_var={r['cum_var']:.3f}")
            lines.append(f"    +v : {' | '.join(r['pos'])}")
            lines.append(f"    -v : {' | '.join(r['neg'])}")
        lines.append("")

    out_txt = out_dir / f"compass_dict_{args.tag}.txt"
    out_txt.write_text("\n".join(lines))
    print(f"wrote {out_txt}")


if __name__ == "__main__":
    main()
