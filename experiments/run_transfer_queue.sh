#!/bin/bash
# Transfer + capability eval queue.
#
# Stage A: CrowS-Pairs at two alphas per model  (~3 hrs)
# Stage B: TruthfulQA MC1 at best-alpha per model (~40 min)
#
# Run sequentially so only one model is in MPS memory at a time.
# Logs under /tmp. Each step runs even if prior step crashed (so a
# single-model failure doesn't sink the whole queue).
set -u
cd "$(dirname "$0")/.."

PY=.venv/bin/python

stage() {
    local tag="$1"; shift
    local log="/tmp/transfer_${tag}.log"
    echo "=== ${tag} ===" > "${log}"
    date >> "${log}"
    echo "CMD: $*" >> "${log}"
    "$@" >> "${log}" 2>&1 || echo "!! ${tag} exited $?" >> "${log}"
    date >> "${log}"
}

# ---------- Stage A: CrowS-Pairs transfer ----------
stage crows_gpt2  "${PY}" -u experiments/crowspairs_eval.py \
    --model gpt2  --alphas "0.75,1.5"
stage crows_gemma "${PY}" -u experiments/crowspairs_eval.py \
    --model gemma --alphas "5.0,10.0"
stage crows_phi3  "${PY}" -u experiments/crowspairs_eval.py \
    --model phi3  --alphas "5.0,10.0"
stage crows_llama "${PY}" -u experiments/crowspairs_eval.py \
    --model llama --alphas "10.0,20.0"

# ---------- Stage B: TruthfulQA capability ----------
stage tqa_gpt2  "${PY}" -u experiments/truthfulqa_eval.py \
    --model gpt2  --alphas "1.5"
stage tqa_gemma "${PY}" -u experiments/truthfulqa_eval.py \
    --model gemma --alphas "20.0"
stage tqa_phi3  "${PY}" -u experiments/truthfulqa_eval.py \
    --model phi3  --alphas "10.0"
stage tqa_llama "${PY}" -u experiments/truthfulqa_eval.py \
    --model llama --alphas "20.0"

echo "=== TRANSFER QUEUE DONE ===" > /tmp/transfer_done.log
date >> /tmp/transfer_done.log
