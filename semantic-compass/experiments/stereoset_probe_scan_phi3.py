"""Stage 2 (Phi-3 variant): scan Phi-3-mini-4k-instruct's top-4 OV-SVD
planes against every Phi-3-tokenized StereoSet probe pair.

Same two-stage filter as stereoset_probe_scan_gemma.py.

Output:
  helix_usage_validated/stereoset_scan_phi3.jsonl
  helix_usage_validated/stereoset_scan_phi3_summary.txt
"""
from __future__ import annotations

import csv
import itertools
import json
import time
import warnings
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer

warnings.filterwarnings("ignore")


NEUTRAL_PROMPTS = [
    "The person said that",
    "Then they said that",
    "Afterwards, the speaker said that",
]
OUT_DIR = Path("helix_usage_validated")
PAIRS_PATH = OUT_DIR / "stereoset_probe_pairs_phi3.tsv"
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

TOP_SVS = 4
ALPHAS = [3.0, 10.0]
N_ANGLES = 12
LIN_THRESH = 0.95
PHASE_THRESH = 10.0
AMP_THRESH = 0.2

# Phi-3 vocab is 32k, same order as GPT-2; keep 500.
DECODE_TOPK = 500


def fit_sinusoid(angles_deg, y):
    th = np.radians(angles_deg)
    N = len(th)
    mu = y.mean()
    c = ((y - mu) * np.cos(th)).sum() * 2.0 / N
    s = ((y - mu) * np.sin(th)).sum() * 2.0 / N
    A = float(np.hypot(c, s))
    phi = float(np.degrees(np.arctan2(s, c)))
    return mu, A, phi


def circ_diff(a, b):
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def load_probes():
    pairs = []
    with PAIRS_PATH.open() as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            pairs.append(dict(
                domain=row["domain"],
                stereo=row["stereo"],
                anti=row["anti"],
                stereo_id=int(row["stereo_token_id"]),
                anti_id=int(row["anti_token_id"]),
            ))
    return pairs


def logits_vec(model, toks, hook_name, edit_vec):
    with torch.no_grad():
        if edit_vec is None:
            out = model(toks)
        else:
            def _h(r, hook):
                r[0, -1, :] = r[0, -1, :] + edit_vec
                return r
            out = model.run_with_hooks(toks, fwd_hooks=[(hook_name, _h)])
    return out[0, -1, :].detach().float().cpu()


def decode_topk_ids(W_U_T, u, k):
    scores = (u.float() @ W_U_T).cpu().numpy()
    pos = set(np.argpartition(-scores, k)[:k].tolist())
    neg = set(np.argpartition(scores, k)[:k].tolist())
    return pos, neg


def main():
    t_start = time.time()
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    dtype = torch.bfloat16
    print(f"[{time.time()-t_start:6.1f}s] Loading {MODEL_NAME} on {device} ...",
          flush=True)
    model = HookedTransformer.from_pretrained(
        MODEL_NAME, device=device, dtype=dtype)
    cfg = model.cfg
    n_layers, n_heads = cfg.n_layers, cfg.n_heads
    print(f"[{time.time()-t_start:6.1f}s]   n_layers={n_layers} "
          f"n_heads={n_heads}  d_model={cfg.d_model}  d_vocab={cfg.d_vocab}",
          flush=True)

    probes = load_probes()
    P = len(probes)
    all_probe_ids = torch.tensor(
        [[p["stereo_id"], p["anti_id"]] for p in probes],
        dtype=torch.long)
    print(f"[{time.time()-t_start:6.1f}s]   probes loaded: {P}",
          flush=True)

    W_U = model.W_U.detach().float().cpu()

    prompts_toks = [model.to_tokens(p) for p in NEUTRAL_PROMPTS]
    angles = np.linspace(0, 360, N_ANGLES, endpoint=False)
    sv_pairs = list(itertools.combinations(range(TOP_SVS), 2))

    print(f"[{time.time()-t_start:6.1f}s] Stage 2a: decode pre-filter ...",
          flush=True)
    candidate_map = {}
    decode_candidates_counter = Counter()
    probe_s_ids = all_probe_ids[:, 0].numpy()
    probe_a_ids = all_probe_ids[:, 1].numpy()

    for L in range(n_layers):
        for H in range(n_heads):
            W_V = model.W_V[L, H].detach().float().cpu()
            W_O = model.W_O[L, H].detach().float().cpu()
            _U, S, Vt = torch.linalg.svd(W_V @ W_O, full_matrices=False)
            top = min(TOP_SVS, Vt.shape[0])
            pos_sets = []
            neg_sets = []
            for d in range(top):
                u = Vt[d, :]
                pos, neg = decode_topk_ids(W_U, u, DECODE_TOPK)
                pos_sets.append(pos)
                neg_sets.append(neg)
            for (d1, d2) in sv_pairs:
                if d1 >= top or d2 >= top:
                    continue
                cand = []
                for p in range(P):
                    s_id = int(probe_s_ids[p])
                    a_id = int(probe_a_ids[p])
                    hit = False
                    for d in (d1, d2):
                        pos, neg = pos_sets[d], neg_sets[d]
                        if (s_id in pos and a_id in neg) or \
                           (a_id in pos and s_id in neg):
                            hit = True
                            break
                    if hit:
                        cand.append(p)
                if cand:
                    candidate_map[(L, H, d1, d2)] = np.array(cand, dtype=np.int32)
                    decode_candidates_counter[(L, H)] += len(cand)
        if (L + 1) % 4 == 0:
            print(f"[{time.time()-t_start:6.1f}s]   stage2a: L{L} scanned",
                  flush=True)

    n_planes_total = n_layers * n_heads * len(sv_pairs)
    n_planes_cand = len(candidate_map)
    n_pairs_cand = sum(len(v) for v in candidate_map.values())
    print(f"[{time.time()-t_start:6.1f}s]   planes with candidates: "
          f"{n_planes_cand}/{n_planes_total}", flush=True)
    print(f"[{time.time()-t_start:6.1f}s]   (plane,probe) candidates: "
          f"{n_pairs_cand} / {n_planes_total * P} = "
          f"{100.0 * n_pairs_cand / (n_planes_total * P):.2f}%",
          flush=True)
    print(f"[{time.time()-t_start:6.1f}s]   Top 10 (L,H) by candidate count:",
          flush=True)
    for (L, H), n in decode_candidates_counter.most_common(10):
        print(f"      L{L:>2}H{H:<2} {n} candidate (plane,probe) hits",
              flush=True)

    print(f"[{time.time()-t_start:6.1f}s] Stage 2b: causal sweep on "
          f"{n_planes_cand} planes ...", flush=True)
    rows = []
    done = 0
    total_fwd = n_planes_cand * len(ALPHAS) * N_ANGLES * len(NEUTRAL_PROMPTS)
    print(f"[{time.time()-t_start:6.1f}s]   forward passes: "
          f"~{total_fwd:,}", flush=True)

    for L in range(n_layers):
        heads_in_layer = [H for H in range(n_heads)
                          if any((L, H, d1, d2) in candidate_map
                                 for (d1, d2) in sv_pairs)]
        if not heads_in_layer:
            continue
        for H in heads_in_layer:
            W_V = model.W_V[L, H].detach().float().cpu()
            W_O = model.W_O[L, H].detach().float().cpu()
            _U, S, Vt = torch.linalg.svd(W_V @ W_O, full_matrices=False)
            top = min(TOP_SVS, Vt.shape[0])
            for (d1, d2) in sv_pairs:
                if d1 >= top or d2 >= top:
                    continue
                key = (L, H, d1, d2)
                if key not in candidate_map:
                    continue
                cand_idx = candidate_map[key]
                Pc = len(cand_idx)
                probe_ids = all_probe_ids[cand_idx]

                u1 = Vt[d1, :].to(device).to(dtype)
                u2 = Vt[d2, :].to(device).to(dtype)
                sig1, sig2 = float(S[d1]), float(S[d2])
                hook_name = f"blocks.{L}.hook_resid_pre"
                ld = np.zeros((len(ALPHAS), N_ANGLES, Pc), dtype=np.float32)
                for ia, a in enumerate(ALPHAS):
                    for ig, deg in enumerate(angles):
                        th = np.radians(deg)
                        vec = (a * sig1 * float(np.cos(th))) * u1 \
                            + (a * sig2 * float(np.sin(th))) * u2
                        diff_sum = np.zeros(Pc, dtype=np.float32)
                        for toks in prompts_toks:
                            lg = logits_vec(model, toks, hook_name, vec)
                            plus = lg[probe_ids[:, 0]].numpy()
                            minus = lg[probe_ids[:, 1]].numpy()
                            diff_sum += (plus - minus)
                        ld[ia, ig, :] = diff_sum / len(prompts_toks)

                alpha_arr = np.array(ALPHAS, dtype=float)
                denom = (alpha_arr ** 2).sum()
                for pi in range(Pc):
                    amps = np.zeros(len(ALPHAS))
                    phis = np.zeros(len(ALPHAS))
                    for ia in range(len(ALPHAS)):
                        _, A, phi = fit_sinusoid(angles, ld[ia, :, pi])
                        amps[ia] = A
                        phis[ia] = phi
                    if amps.max() < 1e-6:
                        lin_r2 = 0.0
                    else:
                        slope = (alpha_arr * amps).sum() / denom
                        resid = amps - slope * alpha_arr
                        ss_res = (resid ** 2).sum()
                        ss_tot = (amps ** 2).sum()
                        lin_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
                    amp_slope = float(amps[-1] / ALPHAS[-1])
                    dphi = float(circ_diff(phis[-1], phis[0]))
                    passed = bool(lin_r2 >= LIN_THRESH
                                  and dphi <= PHASE_THRESH
                                  and amp_slope >= AMP_THRESH)
                    pr = probes[int(cand_idx[pi])]
                    rows.append(dict(
                        L=L, H=H, d1=d1, d2=d2,
                        domain=pr["domain"],
                        stereo=pr["stereo"], anti=pr["anti"],
                        stereo_id=pr["stereo_id"], anti_id=pr["anti_id"],
                        lin_r2=float(lin_r2), amp_slope=float(amp_slope),
                        phase_drift=float(dphi),
                        A_lo=float(amps[0]), A_hi=float(amps[-1]),
                        phi_hi=float(phis[-1]),
                        passed=passed))
                done += 1
                if done % 20 == 0:
                    el = time.time() - t_start
                    eta = el * (n_planes_cand - done) / max(1, done)
                    print(f"[{el:6.1f}s]   plane {done}/{n_planes_cand}  "
                          f"eta={eta:.1f}s  L{L}H{H} SV({d1},{d2})  "
                          f"cand={Pc}", flush=True)

    OUT_DIR.mkdir(exist_ok=True)
    jsonl_path = OUT_DIR / "stereoset_scan_phi3.jsonl"
    with jsonl_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[{time.time()-t_start:6.1f}s] wrote {jsonl_path}  "
          f"({len(rows)} rows)", flush=True)

    head_counts_by_domain = defaultdict(lambda: Counter())
    head_counts_total = Counter()
    probe_pass_counts = Counter()
    for r in rows:
        if r["passed"]:
            key = (r["L"], r["H"])
            head_counts_by_domain[key][r["domain"]] += 1
            head_counts_total[key] += 1
            probe_pass_counts[(r["domain"], r["stereo"], r["anti"])] += 1

    domains = ("gender", "race", "profession", "religion")
    sum_path = OUT_DIR / "stereoset_scan_phi3_summary.txt"
    with sum_path.open("w") as f:
        f.write(f"StereoSet multi-probe scan summary ({MODEL_NAME})\n")
        f.write(f"Probes total: {P}\n")
        f.write(f"Planes scanned (causal): {n_planes_cand} / "
                f"{n_planes_total}\n")
        f.write(f"(plane,probe) tests: {n_pairs_cand}\n")
        f.write(f"Scan rows: {len(rows)}\n")
        f.write(f"Thresholds: R2>={LIN_THRESH}  "
                f"dphi<={PHASE_THRESH}  slope>={AMP_THRESH}\n")
        f.write(f"Alphas: {ALPHAS}  angles: {N_ANGLES}  "
                f"decode_topk: {DECODE_TOPK}\n\n")
        f.write("Top 30 (L,H) by total passes across probes:\n")
        f.write(f"  {'L':>3} {'H':>3}  " + "  ".join(
            f"{d[:6]:>6}" for d in domains) + f"  {'TOTAL':>6}\n")
        for (L, H), n in head_counts_total.most_common(30):
            per = head_counts_by_domain[(L, H)]
            cols = "  ".join(f"{per.get(d, 0):>6}" for d in domains)
            f.write(f"  {L:>3} {H:>3}  {cols}  {n:>6}\n")
        f.write("\n")
        f.write(f"Total distinct heads with >=1 pass: "
                f"{len(head_counts_total)}\n")
        f.write(f"Total distinct probes with >=1 pass: "
                f"{len(probe_pass_counts)}/{P}\n\n")
        f.write("Top 60 probes by how many planes they pass in:\n")
        for (domain, s, a), n in probe_pass_counts.most_common(60):
            f.write(f"  {domain:12s}  {s:>18s} / {a:<18s}  {n}\n")
    print(f"[{time.time()-t_start:6.1f}s] wrote {sum_path}", flush=True)


if __name__ == "__main__":
    main()
