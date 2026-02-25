#!/bin/bash
#SBATCH --job-name=slime-prep
#SBATCH --output=slurm-prep-%j.out
#SBATCH --error=slurm-prep-%j.out

# =============================================================================
# Prep script for Qwen3-32B -> Qwen3-8B On-Policy Distillation
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
#    bash examples/on_policy_distillation/run-qwen3-8B-opd.sh
# =============================================================================

set -ex

# Use env from config (e.g. sfcompute/config-8xh100.env) or sensible defaults for local VM
ROOT_DIR="${ROOT_DIR:-$HOME}"
POOL_DIR="${POOL_DIR:-${ROOT_DIR}/pool}"
MEGATRON_PATH="${MEGATRON_PATH:-${ROOT_DIR}/Megatron-LM}"
if [ ! -d "${MEGATRON_PATH}" ] && [ -d "${ROOT_DIR}/Megatron-LM" ]; then
    MEGATRON_PATH="${ROOT_DIR}/Megatron-LM"
fi
if [ ! -d "${MEGATRON_PATH}" ]; then
    echo "Megatron-LM path not found: ${MEGATRON_PATH}"
    exit 1
fi

REPO_DIR="${REPO_DIR:-}"
for candidate in "${REPO_DIR}" "/root/slime" "${ROOT_DIR}/slime"; do
    [ -n "${candidate}" ] || continue
    if [ -f "${candidate}/tools/convert_hf_to_torch_dist.py" ] && [ -f "${candidate}/scripts/models/qwen3-8B.sh" ]; then
        REPO_DIR="${candidate}"
        break
    fi
done
if [ -z "${REPO_DIR}" ]; then
    echo "Could not locate slime repo root with required scripts."
    exit 1
fi

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

# ---- Convert HF checkpoint to Megatron torch_dist for OPD run script ----

if [ ! -d "${POOL_DIR}/Qwen3-8B_torch_dist" ]; then
    echo "Converting Qwen3-8B HF checkpoint to Megatron torch_dist..."
    source "${REPO_DIR}/scripts/models/qwen3-8B.sh"
    PYTHONPATH="${MEGATRON_PATH}" python3 "${REPO_DIR}/tools/convert_hf_to_torch_dist.py" \
        "${MODEL_ARGS[@]}" \
        --hf-checkpoint "${POOL_DIR}/Qwen3-8B" \
        --save "${POOL_DIR}/Qwen3-8B_torch_dist"
else
    echo "Qwen3-8B_torch_dist already exists, skipping conversion."
fi

echo ""
echo "========================================="
echo "Prep complete. Ready to run:"
echo "  bash examples/on_policy_distillation/run-qwen3-8B-opd.sh"
echo "========================================="
