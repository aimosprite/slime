#!/bin/bash
# Wrapper to run SFT training detached from terminal
# Output goes to /root/slime/models/sft-run.log
set -e
cd /root/slime
export DATASET_PATH=/root/slime/models/am-qwen3-distilled-train.parquet
export TEST_DATA_PATH=/root/slime/models/test-eval-sample.jsonl
export EVAL_INTERVAL=50
export APPLY_CHAT_TEMPLATE=0
exec bash gimran/emb-surgery-sft/qwen3-8b/train.sh
