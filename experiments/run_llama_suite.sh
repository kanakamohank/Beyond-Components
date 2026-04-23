#!/usr/bin/env bash
# End-to-end compass pipeline for Llama-3.2-3B.
# Stages: scan (top4 null) -> loose null (full_ov) -> pick head
#         -> WinoGender sweep -> StereoSet -> compass dict
# Each stage is resume-safe: skipped if the output file already exists.
# Override with FORCE=1.
#
# Usage:
#   HF_TOKEN=hf_xxx bash experiments/run_llama_suite.sh
#   HF_TOKEN=hf_xxx FORCE=1 bash experiments/run_llama_suite.sh
#
# Per-stage logs: logs/llama_suite_<stage>.log
set -e

MODEL="meta-llama/Llama-3.2-3B"
TAG="llama32_3b"
OUT="helix_usage_validated"
LOGS="logs"
PY=".venv/bin/python"

mkdir -p "$LOGS" "$OUT"
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN not set}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

stage() {
  local name=$1
  local probe=$2
  shift 2
  if [ -z "$FORCE" ] && [ -f "$probe" ]; then
    echo "[SKIP] $name (found $probe)"
    return
  fi
  echo "[RUN ] $name -> $probe"
  local start=$(date +%s)
  "$@" 2>&1 | tee "$LOGS/llama_suite_${name}.log"
  local end=$(date +%s)
  echo "[DONE] $name ($((end - start))s)"
}

# ---------- Stage 1: scan with strict (top-4) null -------------------------
stage "scan" "$OUT/${TAG}_scan_gender_scan.txt" \
  $PY experiments/compass_scan.py \
    --model "$MODEL" \
    --tok_plus " he" --tok_minus " she" \
    --top_svs 4 --alphas 3 10 --n_angles 12 \
    --null_mode top4 --null_seeds 3 \
    --out_prefix "${TAG}_scan_gender"

# ---------- Stage 2: scan with loose (full_ov) null ------------------------
# DISABLED: Stage 1 strict-null already gives 9/4032 (0.22%) vs 3/2016 (0.15%).
# Loose null ≤ strict null by construction, so the base rate is established
# without running this ~5hr stage for Llama-3.2-3B.
# Re-enable by removing the `if false; then` guard.
if false; then
stage "scan_fullov" "$OUT/${TAG}_scan_gender_nullfullov_scan.txt" \
  $PY experiments/compass_scan.py \
    --model "$MODEL" \
    --tok_plus " he" --tok_minus " she" \
    --top_svs 4 --alphas 3 10 --n_angles 12 \
    --null_mode full_ov --null_seeds 3 \
    --out_prefix "${TAG}_scan_gender_nullfullov"
fi

# ---------- Stage 3: pick top passing head ---------------------------------
TOP_HEAD_JSON="$OUT/${TAG}_top_head.json"
export TAG OUT
if [ -z "$FORCE" ] && [ -f "$TOP_HEAD_JSON" ]; then
  echo "[SKIP] pick_top_head (found $TOP_HEAD_JSON)"
else
  echo "[RUN ] pick_top_head"
  $PY - <<'PYEOF'
import json, os, re
from pathlib import Path
tag = os.environ["TAG"]
out = os.environ["OUT"]
scan_txt = Path(f"{out}/{tag}_scan_gender_scan.txt")
if not scan_txt.exists():
    raise SystemExit(f"scan output missing: {scan_txt}")
lines = scan_txt.read_text().splitlines()
# Table rows look like:
#   1  10   9 (0,3)   5.11  14.17  1.417  0.997    0.3    +4.7
row_re = re.compile(
    r"^\s*\d+\s+(\d+)\s+(\d+)\s+\((\d+),(\d+)\)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
)
best = None
for line in lines:
    m = row_re.match(line)
    if not m:
        continue
    L, H, d1, d2 = int(m[1]), int(m[2]), int(m[3]), int(m[4])
    slope = float(m[7])
    if best is None or slope > best["slope"]:
        best = dict(layer=L, head=H, svd=d1, svd2=d2, slope=slope)
if best is None:
    raise SystemExit("no passing heads found; suite cannot continue")
Path(f"{out}/{tag}_top_head.json").write_text(json.dumps(best, indent=2))
print("picked:", best)
PYEOF
fi

LAYER=$($PY -c "import json; print(json.load(open('$TOP_HEAD_JSON'))['layer'])")
HEAD=$($PY -c  "import json; print(json.load(open('$TOP_HEAD_JSON'))['head'])")
SVD=$($PY -c   "import json; print(json.load(open('$TOP_HEAD_JSON'))['svd'])")
echo "Using compass head: L${LAYER} H${HEAD} SV${SVD}"

# ---------- Stage 4: WinoGender sweep --------------------------------------
stage "winogender" "$OUT/winogender_${TAG}_sweep.csv" \
  $PY experiments/winogender_sweep.py \
    --model "$MODEL" \
    --primary_layer "$LAYER" \
    --primary_head "$HEAD" \
    --primary_svd "$SVD" \
    --alphas "0.25,0.5,1.0,1.5,2.0" \
    --out_prefix "winogender_${TAG}"

# ---------- Stage 5: StereoSet ---------------------------------------------
stage "stereoset" "$OUT/stereoset_${TAG}_l${LAYER}h${HEAD}.csv" \
  $PY experiments/stereoset_eval.py \
    --model "$MODEL" \
    --layer "$LAYER" --head "$HEAD" --svd_dim "$SVD" \
    --alphas "1.0,1.5" \
    --domains "gender,race,profession" \
    --out_prefix "stereoset_${TAG}_l${LAYER}h${HEAD}"

# ---------- Stage 6: compass dictionary (single model) ---------------------
stage "dict" "$OUT/compass_dict_${TAG}.txt" \
  $PY experiments/compass_dictionary_single.py \
    --model "$MODEL" --tag "$TAG" \
    --heads_json "$TOP_HEAD_JSON"

echo ""
echo "============================================================"
echo "  Llama-3.2-3B suite complete."
echo "  Outputs in $OUT/:"
ls -1 "$OUT" | grep -i "$TAG" | sed 's/^/    /'
echo "============================================================"
