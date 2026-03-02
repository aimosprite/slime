#!/bin/bash
# =============================================================================
# prep-embedding-surgery.sh — Download, randomize embeddings, convert, get dataset
#
# Idempotent: skips steps whose outputs already exist.
# Exits 0 on success, 1 on missing artifacts.
#
# Usage:
#   bash scripts/prep-embedding-surgery.sh                            # uses default config
#   TRAIN_CONFIG=my.yaml bash scripts/prep-embedding-surgery.sh       # custom config
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ======================== LOAD CONFIG ========================
source "${SCRIPT_DIR}/lib/config.sh"

TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_DIR}/configs/sft-qwen3-8b-embedding-surgery.yaml}"
load_config "${TRAIN_CONFIG}"
load_env "${REPO_DIR}"

# ======================== DEFAULTS ========================
POOL_DIR="${POOL_DIR:-/root/slime/models}"
MEGATRON_PATH="${MEGATRON_PATH:-${REPO_DIR}/Megatron-LM}"

HF_MODEL="${HF_MODEL:-Qwen/Qwen3-8B}"
HF_DATASET="${HF_DATASET:-a-m-team/AM-Qwen3-Distilled}"
MODEL_NAME="${HF_MODEL##*/}"

MODEL_DIR="${POOL_DIR}/${MODEL_NAME}"
RANDOM_EMB_DIR="${POOL_DIR}/${MODEL_NAME}-random-emb"
MEGATRON_REF_DIR="${POOL_DIR}/${MODEL_NAME}-random-emb_torch_dist"
DATASET_PATH="${POOL_DIR}/am-qwen3-distilled.parquet"

INIT_STD="${INIT_STD:-0.02}"
INIT_SEED="${INIT_SEED:-42}"
CONVERT_NPROC_PER_NODE="${CONVERT_NPROC_PER_NODE:-8}"
TRANSFORMER_IMPL="${TRANSFORMER_IMPL:-local}"

# ======================== PREP ========================
echo "=========================================="
echo "  PREP: Download, randomize, convert"
echo "=========================================="

# 1. Download base model
if [ ! -d "${MODEL_DIR}" ]; then
    echo "--- Downloading ${HF_MODEL} ---"
    huggingface-cli download "${HF_MODEL}" --local-dir "${MODEL_DIR}"
else
    echo "--- Model already exists at ${MODEL_DIR}, skipping ---"
fi

# 2. Randomize embed_tokens + lm_head
if [ ! -d "${RANDOM_EMB_DIR}" ]; then
    echo "--- Randomizing embeddings (std=${INIT_STD}, seed=${INIT_SEED}) ---"
    python3 "${REPO_DIR}/tools/randomize_embeddings.py" \
        --input-dir  "${MODEL_DIR}" \
        --output-dir "${RANDOM_EMB_DIR}" \
        --init-std   "${INIT_STD}" \
        --seed       "${INIT_SEED}"
else
    echo "--- Random-emb model already exists at ${RANDOM_EMB_DIR}, skipping ---"
fi

# 3. Convert HF -> Megatron torch_dist format
if [ ! -d "${MEGATRON_REF_DIR}" ]; then
    echo "--- Converting HF -> Megatron (nproc=${CONVERT_NPROC_PER_NODE}) ---"
    source "${SCRIPT_DIR}/models/qwen3-8B.sh"
    PYTHONPATH="${MEGATRON_PATH}" \
    torchrun --nproc_per_node="${CONVERT_NPROC_PER_NODE}" \
        "${REPO_DIR}/tools/convert_hf_to_torch_dist.py" \
        "${MODEL_ARGS[@]}" \
        --hf-checkpoint                "${RANDOM_EMB_DIR}" \
        --save                         "${MEGATRON_REF_DIR}" \
        --tensor-model-parallel-size   1 \
        --pipeline-model-parallel-size 1 \
        --context-parallel-size        1 \
        --expert-model-parallel-size   1 \
        --expert-tensor-parallel-size  1 \
        --untie-embeddings-and-output-weights \
        --no-gradient-accumulation-fusion
else
    echo "--- Megatron checkpoint already exists at ${MEGATRON_REF_DIR}, skipping ---"
fi

# 4. Download & convert dataset
if [ ! -f "${DATASET_PATH}" ]; then
    echo "--- Downloading & converting dataset ---"
    python3 "${REPO_DIR}/tools/prep_am_dataset.py" \
        --dataset "${HF_DATASET}" \
        --output  "${DATASET_PATH}"
else
    echo "--- Dataset already exists at ${DATASET_PATH}, skipping ---"
fi

echo "=========================================="
echo "  PREP DONE"
echo "=========================================="

# ======================== VERIFY ARTIFACTS ========================
echo "--- Verifying artifacts ---"
MISSING=0
for required in "${RANDOM_EMB_DIR}" "${MEGATRON_REF_DIR}"; do
    if [ ! -d "${required}" ]; then
        echo "MISSING DIR:  ${required}"
        MISSING=1
    fi
done
if [ ! -f "${DATASET_PATH}" ]; then
    echo "MISSING FILE: ${DATASET_PATH}"
    MISSING=1
fi
if [ "${MISSING}" = "1" ]; then
    echo "ERROR: Prep completed but artifacts are missing."
    exit 1
fi
echo "All artifacts present."

# ======================== ENV CHECKS ========================
WANDB_KEY="${WANDB_KEY:-${WANDB_API_KEY:-}}"
if [ -z "${WANDB_KEY}" ]; then
    echo "ERROR: WANDB_KEY (or WANDB_API_KEY) not set. Add it to .env."
    exit 1
fi
if [ ! -d "${MEGATRON_PATH}" ]; then
    echo "ERROR: Megatron-LM not found at ${MEGATRON_PATH}."
    echo "       Run: git clone https://github.com/NVIDIA/Megatron-LM.git ${MEGATRON_PATH}"
    exit 1
fi

echo "Environment OK."
