#!/bin/bash
# Per-domain calibrated alpha routed-ensemble CrowS-Pairs queue.
# Target SNR=0.10; alphas pre-computed by calibrate_per_domain_alpha.py.
set -u
cd "$(dirname "$0")/.."

PY=.venv/bin/python

stage() {
    local tag="$1"; shift
    local log="/tmp/calib_${tag}.log"
    echo "=== ${tag} ===" > "${log}"
    date >> "${log}"
    echo "CMD: $*" >> "${log}"
    "$@" >> "${log}" 2>&1 || echo "!! ${tag} exited $?" >> "${log}"
    date >> "${log}"
}

stage crows_gpt2  "${PY}" -u experiments/crowspairs_routed_eval.py \
    --model gpt2  \
    --alpha_json helix_usage_validated/per_domain_alpha_gpt2.json
stage crows_phi3  "${PY}" -u experiments/crowspairs_routed_eval.py \
    --model phi3  \
    --alpha_json helix_usage_validated/per_domain_alpha_phi3.json
stage crows_llama "${PY}" -u experiments/crowspairs_routed_eval.py \
    --model llama \
    --alpha_json helix_usage_validated/per_domain_alpha_llama.json

echo "=== CALIB QUEUE DONE ===" > /tmp/calib_done.log
date >> /tmp/calib_done.log
