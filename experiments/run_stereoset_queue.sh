#!/bin/bash
# Sequential queue: Phi-3 pipeline, then Llama pipeline. Each step's
# output is logged under /tmp. Each sub-step must complete before the
# next one starts so we don't have two models in GPU/MPS memory at once.
set -e
cd "$(dirname "$0")/.."

echo "=== PHI-3: extract ===" > /tmp/queue_phi3_extract.log
.venv/bin/python -u experiments/stereoset_probe_extract_phi3.py \
    >> /tmp/queue_phi3_extract.log 2>&1

echo "=== PHI-3: scan ===" > /tmp/queue_phi3_scan.log
.venv/bin/python -u experiments/stereoset_probe_scan_phi3.py \
    >> /tmp/queue_phi3_scan.log 2>&1

echo "=== PHI-3: ensemble ===" > /tmp/queue_phi3_ensemble.log
.venv/bin/python -u experiments/stereoset_ensemble_eval_phi3.py \
    --alphas "0.5,1.0,1.5" \
    >> /tmp/queue_phi3_ensemble.log 2>&1

echo "=== LLAMA: extract ===" > /tmp/queue_llama_extract.log
.venv/bin/python -u experiments/stereoset_probe_extract_llama.py \
    >> /tmp/queue_llama_extract.log 2>&1

echo "=== LLAMA: scan ===" > /tmp/queue_llama_scan.log
.venv/bin/python -u experiments/stereoset_probe_scan_llama.py \
    >> /tmp/queue_llama_scan.log 2>&1

echo "=== LLAMA: ensemble ===" > /tmp/queue_llama_ensemble.log
.venv/bin/python -u experiments/stereoset_ensemble_eval_llama.py \
    --alphas "5.0,10.0,20.0" \
    >> /tmp/queue_llama_ensemble.log 2>&1

echo "=== QUEUE DONE ===" > /tmp/queue_done.log
date >> /tmp/queue_done.log
