#!/bin/bash
# Llama SNR sweep: 0.10, 0.15, 0.20 under calibrated routed ensemble.
# Output CSV/txt suffixes handled by the eval script's _calib naming, so
# we rename after each run to keep all 3 distinct.
set -u
cd "$(dirname "$0")/.."

PY=.venv/bin/python
OUT=helix_usage_validated

run_one() {
    local snr="$1"
    local log="/tmp/llama_snr_${snr}.log"
    echo "=== llama SNR=${snr} ===" > "${log}"
    date >> "${log}"
    "${PY}" -u experiments/crowspairs_routed_eval.py \
        --model llama \
        --alpha_json ${OUT}/per_domain_alpha_llama_snr${snr}.json \
        >> "${log}" 2>&1 || echo "!! llama snr=${snr} exited $?" >> "${log}"
    # rename default _calib outputs to include snr tag
    mv "${OUT}/crowspairs_routed_llama_calib.csv" \
       "${OUT}/crowspairs_routed_llama_snr${snr}.csv"
    mv "${OUT}/crowspairs_routed_llama_calib.txt" \
       "${OUT}/crowspairs_routed_llama_snr${snr}.txt"
    date >> "${log}"
}

run_one 0.10
run_one 0.15
run_one 0.20

echo "=== LLAMA SNR SWEEP DONE ===" > /tmp/llama_snr_done.log
date >> /tmp/llama_snr_done.log
