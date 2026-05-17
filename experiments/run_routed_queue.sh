#!/bin/bash
# Routed-ensemble CrowS-Pairs queue.
# Gemma skipped: only 1 head ever passed, nothing to route.
set -u
cd "$(dirname "$0")/.."

PY=.venv/bin/python

stage() {
    local tag="$1"; shift
    local log="/tmp/routed_${tag}.log"
    echo "=== ${tag} ===" > "${log}"
    date >> "${log}"
    echo "CMD: $*" >> "${log}"
    "$@" >> "${log}" 2>&1 || echo "!! ${tag} exited $?" >> "${log}"
    date >> "${log}"
}

stage crows_gpt2  "${PY}" -u experiments/crowspairs_routed_eval.py \
    --model gpt2  --alphas "0.75,1.5"
stage crows_phi3  "${PY}" -u experiments/crowspairs_routed_eval.py \
    --model phi3  --alphas "5.0,10.0"
stage crows_llama "${PY}" -u experiments/crowspairs_routed_eval.py \
    --model llama --alphas "10.0,20.0"

echo "=== ROUTED QUEUE DONE ===" > /tmp/routed_done.log
date >> /tmp/routed_done.log
