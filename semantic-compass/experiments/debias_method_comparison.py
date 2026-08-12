"""Head-to-head comparison: baseline vs. INLP vs. SentenceDebias vs.
routed+calib compass intervention, on CrowS-Pairs SS per (model, domain).

Reads existing CSVs under helix_usage_validated/ — no forward passes.

INLP and SentenceDebias runs are per-domain (one hook per (domain, layer));
routed+calib is the cross-domain ensemble from crowspairs_routed_eval.py.

Output:
  helix_usage_validated/debias_method_comparison.csv
  helix_usage_validated/debias_method_comparison.md
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

OUT_DIR = Path("helix_usage_validated")
MODELS = ["gpt2", "phi3", "gemma", "llama"]
DOMAINS = ["gender", "race", "profession", "religion", "overall"]

LAYER = {"gpt2": 10, "phi3": 24, "gemma": 21, "llama": 22}

ROUTED_CSV = {
    "gpt2":  "crowspairs_routed_gpt2_calib.csv",
    "phi3":  "crowspairs_routed_phi3_snr0.08.csv",
    "gemma": "crowspairs_routed_gemma_calib.csv",
    "llama": "crowspairs_routed_llama_snr0.10.csv",
}


def read_csv(path):
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def add_rows(store, model, method, rows):
    for r in rows:
        cond = r.get("condition", "")
        dom = r.get("domain", "")
        try:
            ss = float(r["ss"])
            n = int(r["n"])
        except (KeyError, ValueError, TypeError):
            continue
        store[(model, method, cond, dom)] = dict(ss=ss, n=n)


def load_matched(model):
    """Return {domain: (ss, alpha, mult)} for the best-|Δ50| multiplier."""
    path = OUT_DIR / f"target_ss50_{model}.csv"
    if not path.exists():
        return {}
    best = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            d = r["domain"]
            ss = float(r["ss"])
            m = float(r["multiplier"])
            a = float(r["alpha"])
            dist = abs(ss - 50.0)
            if (d not in best) or (dist < best[d]["dist"]):
                best[d] = dict(ss=ss, alpha=a, mult=m, dist=dist)
    return best


def main():
    store = {}
    for m in MODELS:
        layer = LAYER[m]
        for d in ("gender", "race"):
            add_rows(store, m, "inlp",
                     read_csv(OUT_DIR / f"crowspairs_inlp_{m}_{d}_L{layer}.csv"))
            add_rows(store, m, "sentdebias",
                     read_csv(OUT_DIR /
                              f"crowspairs_sentdebias_{m}_{d}_L{layer}.csv"))
        add_rows(store, m, "routed",
                 read_csv(OUT_DIR / ROUTED_CSV[m]))
    matched = {m: load_matched(m) for m in MODELS}

    out = []
    for m in MODELS:
        for d in DOMAINS:
            row = {"model": m, "domain": d}
            base = store.get((m, "routed", "baseline", d))
            row["baseline"] = base["ss"] if base else float("nan")
            row["n"] = base["n"] if base else 0

            row["inlp"] = (store.get((m, "inlp", "inlp", d))
                           or {}).get("ss", float("nan"))
            row["sentdebias"] = (store.get((m, "sentdebias",
                                            "sentdebias", d))
                                 or {}).get("ss", float("nan"))
            routed = (store.get((m, "routed", "routed_calib", d))
                      or store.get((m, "routed", "routed_K4", d))
                      or {})
            row["routed_calib"] = routed.get("ss", float("nan"))

            matched_cell = matched.get(m, {}).get(d, {})
            row["routed_matched"] = matched_cell.get("ss", float("nan"))
            row["matched_mult"] = matched_cell.get("mult", float("nan"))

            def delta(x):
                if math.isnan(x) or math.isnan(row["baseline"]):
                    return float("nan")
                return x - row["baseline"]

            def dist50(x):
                return abs(x - 50.0) if not math.isnan(x) else float("nan")

            row["inlp_delta"] = delta(row["inlp"])
            row["sent_delta"] = delta(row["sentdebias"])
            row["routed_delta"] = delta(row["routed_calib"])
            row["matched_delta"] = delta(row["routed_matched"])
            row["base_dist50"] = dist50(row["baseline"])
            row["inlp_dist50"] = dist50(row["inlp"])
            row["sent_dist50"] = dist50(row["sentdebias"])
            row["routed_dist50"] = dist50(row["routed_calib"])
            row["matched_dist50"] = dist50(row["routed_matched"])
            out.append(row)

    csv_cols = ["model", "domain", "n",
                "baseline", "inlp", "sentdebias",
                "routed_calib", "routed_matched", "matched_mult",
                "inlp_delta", "sent_delta", "routed_delta",
                "matched_delta",
                "base_dist50", "inlp_dist50", "sent_dist50",
                "routed_dist50", "matched_dist50"]
    csv_path = OUT_DIR / "debias_method_comparison.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        for r in out:
            w.writerow({c: r[c] for c in csv_cols})
    print(f"wrote {csv_path}  ({len(out)} rows)")

    def fmt(x, spec="6.2f"):
        return ("--" if isinstance(x, float) and math.isnan(x)
                else format(x, spec))

    def fmt_delta(x):
        if isinstance(x, float) and math.isnan(x):
            return "--"
        return f"{x:+.2f}"

    md = [
        "# Debias-method head-to-head: INLP vs SentenceDebias vs routed+calib",
        "",
        "CrowS-Pairs Stereotype Score (SS) per (model, domain). Ideal SS = 50 "
        "(no stereotypical preference). `*_delta` is the SS change relative "
        "to baseline (negative = debiasing in the intended direction when "
        "baseline > 50). INLP and SentenceDebias are single-domain "
        "interventions trained on `gender` or `race` and evaluated at "
        "layer L[model]; the same `gender` hook is also applied to "
        "`profession` rows (following Liang 2020). `routed+calib` is our "
        "per-domain SVD-plane ensemble at calibrated α.",
        "",
    ]
    for m in MODELS:
        md.append(f"## {m}  (layer L{LAYER[m]} for INLP / SentenceDebias)")
        md.append("")
        md.append("| domain | n | baseline | INLP | SentDebias | "
                  "routed+calib | routed+matched (M*) | "
                  "Δ INLP | Δ Sent | Δ calib | Δ matched |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in out:
            if r["model"] != m:
                continue
            matched_cell = (
                f"{fmt(r['routed_matched'])} "
                f"(M={r['matched_mult']:.2f})"
                if not (isinstance(r['routed_matched'], float)
                        and math.isnan(r['routed_matched']))
                else "--")
            md.append(
                f"| {r['domain']} | {r['n']} | "
                f"{fmt(r['baseline'])} | "
                f"{fmt(r['inlp'])} | "
                f"{fmt(r['sentdebias'])} | "
                f"{fmt(r['routed_calib'])} | "
                f"{matched_cell} | "
                f"{fmt_delta(r['inlp_delta'])} | "
                f"{fmt_delta(r['sent_delta'])} | "
                f"{fmt_delta(r['routed_delta'])} | "
                f"{fmt_delta(r['matched_delta'])} |"
            )
        md.append("")
    md += [
        "## Per-cell |Δ50| (distance from neutral; lower is better)",
        "",
        "Only gender and race cells shown — these are the domains where "
        "INLP / SentenceDebias are trained. `matched` = best-multiplier "
        "compass run from target_ss50_sweep.py; `--` where the sweep was "
        "not run.",
        "",
        "| model | domain | baseline | INLP | SentDebias | routed+calib "
        "| routed+matched |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in MODELS:
        for d in ("gender", "race"):
            r = next((x for x in out
                      if x["model"] == m and x["domain"] == d), None)
            if r is None:
                continue
            md.append(
                f"| {m} | {d} | {fmt(r['base_dist50'])} | "
                f"{fmt(r['inlp_dist50'])} | "
                f"{fmt(r['sent_dist50'])} | "
                f"{fmt(r['routed_dist50'])} | "
                f"{fmt(r['matched_dist50'])} |"
            )
    md.append("")
    md_path = OUT_DIR / "debias_method_comparison.md"
    md_path.write_text("\n".join(md))
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
