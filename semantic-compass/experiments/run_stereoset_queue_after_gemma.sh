#!/bin/bash
# Wait for the Gemma ensemble to finish (marker file appears) then
# run the Phi-3 and Llama pipelines sequentially.
set -e
cd "$(dirname "$0")/.."

GEMMA_MARK="helix_usage_validated/stereoset_ensemble_gemma.txt"
echo "supervisor: waiting for $GEMMA_MARK to exist and be stable..."
# Wait until the file exists
while [ ! -f "$GEMMA_MARK" ]; do
    sleep 30
done
# Wait until the file has been untouched for >=60s (so we know the
# Gemma process isn't still writing it and holding MPS memory).
prev_mtime=0
stable=0
while [ $stable -lt 2 ]; do
    cur_mtime=$(stat -f %m "$GEMMA_MARK" 2>/dev/null || stat -c %Y "$GEMMA_MARK")
    if [ "$cur_mtime" = "$prev_mtime" ]; then
        stable=$((stable + 1))
    else
        stable=0
    fi
    prev_mtime=$cur_mtime
    sleep 30
done
echo "supervisor: gemma done, starting phi3/llama queue"

bash experiments/run_stereoset_queue.sh
