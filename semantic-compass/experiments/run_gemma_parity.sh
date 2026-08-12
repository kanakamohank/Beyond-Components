#!/bin/bash
# Gemma parity queue: re-scan with AMP_THRESH=0.08, then mirror the bias suite
# already run for gpt2/phi3/llama (StereoSet ensemble, CrowS-Pairs, TruthfulQA,
# per-domain alpha calibration, routed CrowS-Pairs, SNR sweep).
set -u
cd "$(dirname "$0")/.."

PY=.venv/bin/python
OUT=helix_usage_validated
LOGDIR=/tmp
M=gemma

stage() {
    local name="$1"; shift
    local log="${LOGDIR}/gemma_parity_${name}.log"
    echo "=== ${name} ===" > "${log}"
    date >> "${log}"
    echo "cmd: $*" >> "${log}"
    "$@" >> "${log}" 2>&1 || echo "!! ${name} exited $?" >> "${log}"
    date >> "${log}"
}

# 1. Re-run plane scan (AMP_THRESH already lowered to 0.08 in-script).
stage scan "${PY}" -u experiments/stereoset_probe_scan_gemma.py

# 2. StereoSet K=4 ensemble.
stage stereoset_ensemble "${PY}" -u experiments/stereoset_ensemble_eval_gemma.py \
    --alphas 5.0,10.0,20.0

# 3. Baseline CrowS-Pairs ensemble.
stage crowspairs "${PY}" -u experiments/crowspairs_eval.py --model ${M}

# 4. Baseline TruthfulQA ensemble.
stage truthfulqa "${PY}" -u experiments/truthfulqa_eval.py --model ${M}

# 5. Per-domain alpha calibration (default target_snr=0.20 inside script).
stage calibrate "${PY}" -u experiments/calibrate_per_domain_alpha.py --model ${M}

# 6. Routed CrowS-Pairs using the default calibrated alphas file.
stage crowspairs_routed "${PY}" -u experiments/crowspairs_routed_eval.py \
    --model ${M} --alpha_json "${OUT}/per_domain_alpha_${M}.json"

# 7. SNR sweep skipped for Gemma: target_snr 0.05/0.08/0.10 produces alphas too
# small to move CrowS-Pairs once the logit soft-cap + RMSNorm compress the
# residual perturbation. Revisit with 0.20/0.30/0.40 if needed.

echo "=== GEMMA PARITY DONE ===" > ${LOGDIR}/gemma_parity_done.log
date >> ${LOGDIR}/gemma_parity_done.log
