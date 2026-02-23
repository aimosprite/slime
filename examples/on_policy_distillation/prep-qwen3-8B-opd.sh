#!/bin/bash

# =============================================================================
# Prep script for Qwen3-32B -> Qwen3-8B On-Policy Distillation (FSDP backend)
#
# MANUAL STEPS TO DO BEFORE RUNNING THIS SCRIPT (in order):
#
# 1. Download training data (dapo-math-17k or your own):
#    huggingface-cli download BytedTsinghua/DAPO-Math-17k \
#        --repo-type dataset --local-dir ${POOL_DIR}/dapo-math-17k
#
# 2. Install dependencies (sglang, ray, etc.) — see slime README
#
# 3. Set up .env with WANDB_API_KEY if using wandb
#
# After running this script, you can run:
#    bash examples/on_policy_distillation/run-qwen3-8B-opd-fsdp.sh
# =============================================================================

set -ex

ROOT_DIR="/home/rohin"
POOL_DIR="${ROOT_DIR}/orcd/pool"

# ---- Download models if not already present ----

if [ ! -d "${POOL_DIR}/Qwen3-32B" ]; then
    echo "Downloading Qwen3-32B (teacher)..."
    huggingface-cli download Qwen/Qwen3-32B --local-dir "${POOL_DIR}/Qwen3-32B"
else
    echo "Qwen3-32B already exists, skipping download."
fi

if [ ! -d "${POOL_DIR}/Qwen3-8B" ]; then
    echo "Downloading Qwen3-8B (student)..."
    huggingface-cli download Qwen/Qwen3-8B --local-dir "${POOL_DIR}/Qwen3-8B"
else
    echo "Qwen3-8B already exists, skipping download."
fi

# ---- Convert dataset from parquet to JSONL ----

if [ ! -f "${POOL_DIR}/dapo-math-17k/dapo-math-17k.jsonl" ]; then
    echo "Converting dapo-math-17k from parquet to JSONL..."
    python3 -c "
import pandas as pd
df = pd.read_parquet('${POOL_DIR}/dapo-math-17k/data/dapo-math-17k.parquet')
df.to_json('${POOL_DIR}/dapo-math-17k/dapo-math-17k.jsonl', orient='records', lines=True)
print(f'Wrote {len(df)} rows to dapo-math-17k.jsonl')
"
else
    echo "dapo-math-17k.jsonl already exists, skipping conversion."
fi

echo ""
echo "========================================="
echo "Prep complete. Ready to run:"
echo "  bash examples/on_policy_distillation/run-qwen3-8B-opd-fsdp.sh"
echo "========================================="
