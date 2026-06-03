#!/bin/bash
# Per-domain calibrated alpha routed-ensemble CrowS-Pairs queue.
# Canonical SNR per model: GPT-2=0.20, Phi-3=0.08, Gemma=0.08, Llama=0.10.
# Calibrated alphas pre-computed by experiments/calibrate_per_domain_alpha.py.
#
# JSON file naming convention:
#   GPT-2 / Gemma : per_domain_alpha_<model>.json (single SNR target)
#   Phi-3 / Llama : per_domain_alpha_<model>_snr<TARGET>.json
#                   (SNR-tagged because run_phi3_snr_sweep.sh /
#                    run_llama_snr_sweep.sh produce a family at
#                    different SNR targets; pick the canonical one)
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
    --alpha_json helix_usage_validated/per_domain_alpha_phi3_snr0.08.json
stage crows_llama "${PY}" -u experiments/crowspairs_routed_eval.py \
    --model llama \
    --alpha_json helix_usage_validated/per_domain_alpha_llama_snr0.10.json

echo "=== CALIB QUEUE DONE ===" > /tmp/calib_done.log
date >> /tmp/calib_done.log
