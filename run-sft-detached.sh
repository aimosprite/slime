#!/bin/bash
# Wrapper to run SFT training detached from terminal
# Output goes to /root/slime/models/sft-run.log
set -e
cd /root/slime
export DATASET_PATH=/root/slime/models/am-qwen3-distilled-200k.parquet
export APPLY_CHAT_TEMPLATE=0
exec bash scripts/sft-qwen3-8b-AM-embedding-swap.sh
