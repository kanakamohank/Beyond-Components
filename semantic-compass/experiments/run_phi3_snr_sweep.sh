#!/bin/bash
# Phi-3 SNR sweep: 0.07, 0.08, 0.10 under calibrated routed ensemble.
set -u
cd "$(dirname "$0")/.."

PY=.venv/bin/python
OUT=helix_usage_validated

run_one() {
    local snr="$1"
    local log="/tmp/phi3_snr_${snr}.log"
    echo "=== phi3 SNR=${snr} ===" > "${log}"
    date >> "${log}"
    "${PY}" -u experiments/crowspairs_routed_eval.py \
        --model phi3 \
        --alpha_json ${OUT}/per_domain_alpha_phi3_snr${snr}.json \
        >> "${log}" 2>&1 || echo "!! phi3 snr=${snr} exited $?" >> "${log}"
    mv "${OUT}/crowspairs_routed_phi3_calib.csv" \
       "${OUT}/crowspairs_routed_phi3_snr${snr}.csv"
    mv "${OUT}/crowspairs_routed_phi3_calib.txt" \
       "${OUT}/crowspairs_routed_phi3_snr${snr}.txt"
    date >> "${log}"
}

run_one 0.07
run_one 0.08
run_one 0.10

echo "=== PHI-3 SNR SWEEP DONE ===" > /tmp/phi3_snr_done.log
date >> /tmp/phi3_snr_done.log
