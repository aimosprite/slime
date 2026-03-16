#!/bin/bash
# =============================================================================
# prep-gpt-oss.sh — Download, tokenizer swap, dequant, convert, dataset
#
# Idempotent: skips steps whose outputs already exist.
#
# Usage:
#   bash gimran/emb-surgery-sft/scripts/prep-gpt-oss.sh
#   TRAIN_CONFIG=my.yaml bash gimran/emb-surgery-sft/scripts/prep-gpt-oss.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
EMB_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${EMB_DIR}/../.." && pwd)"

# ======================== LOAD CONFIG ========================
source "${EMB_DIR}/lib/config.sh"

TRAIN_CONFIG="${TRAIN_CONFIG:-${SCRIPT_DIR}/stage1.yaml}"
load_config "${TRAIN_CONFIG}"
load_env "${REPO_DIR}"

# ======================== DEFAULTS ========================
POOL_DIR="${POOL_DIR:-/root/slime/models}"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"

HF_MODEL="${HF_MODEL:-openai/gpt-oss-20b}"
DONOR_TOKENIZER="${DONOR_TOKENIZER:-Qwen/Qwen3.5-35B-A3B}"
NEW_VOCAB_SIZE="${NEW_VOCAB_SIZE:-248320}"
MODEL_NAME="${HF_MODEL##*/}"

MODEL_DIR="${POOL_DIR}/${MODEL_NAME}"
DONOR_TOKENIZER_DIR="${POOL_DIR}/Qwen3.5-35B-A3B-tokenizer"
SWAPPED_DIR="${POOL_DIR}/${MODEL_NAME}-qwen3.5-tokenizer"
MEGATRON_REF_DIR="${POOL_DIR}/${MODEL_NAME}-qwen3.5-tokenizer_torch_dist"
# Canonical dataset paths — must match training script defaults
DATASET_PATH="${DATASET_PATH:-${POOL_DIR}/am-qwen3-distilled-train.parquet}"
TEST_DATA_PATH="${TEST_DATA_PATH:-${POOL_DIR}/am-qwen3-distilled-test.parquet}"
# Use ~200k samples (sample-fraction=0.11 of 1.89M)
AM_SAMPLE_FRACTION="${AM_SAMPLE_FRACTION:-0.11}"

INIT_STD="${INIT_STD:-0.02}"
INIT_SEED="${INIT_SEED:-42}"
CONVERT_NPROC_PER_NODE="${CONVERT_NPROC_PER_NODE:-1}"
TRANSFORMER_IMPL="${TRANSFORMER_IMPL:-transformer_engine}"

# ======================== PREP ========================
echo "=========================================="
echo "  PREP: GPT-OSS-20b Embedding Surgery"
echo "=========================================="

# 1. Download base model
if [ ! -d "${MODEL_DIR}" ] || [ ! -f "${MODEL_DIR}/config.json" ]; then
    echo "--- Downloading ${HF_MODEL} ---"
    huggingface-cli download "${HF_MODEL}" --local-dir "${MODEL_DIR}"
else
    echo "--- Model already exists at ${MODEL_DIR}, skipping ---"
fi

# 2. Download donor tokenizer
if [ ! -d "${DONOR_TOKENIZER_DIR}" ] || [ ! -f "${DONOR_TOKENIZER_DIR}/tokenizer.json" ]; then
    echo "--- Downloading donor tokenizer from ${DONOR_TOKENIZER} ---"
    huggingface-cli download "${DONOR_TOKENIZER}" \
        --include "tokenizer*" "special_tokens*" "chat_template*" \
        --local-dir "${DONOR_TOKENIZER_DIR}"
else
    echo "--- Donor tokenizer already exists at ${DONOR_TOKENIZER_DIR}, skipping ---"
fi

# 3. Tokenizer swap + MXFP4 dequant + embedding resize
if [ ! -d "${SWAPPED_DIR}" ] || [ ! -f "${SWAPPED_DIR}/config.json" ]; then
    echo "--- Tokenizer swap + dequant (std=${INIT_STD}, seed=${INIT_SEED}) ---"
    python3 "${SCRIPT_DIR}/tokenizer_swap.py" \
        --input-dir          "${MODEL_DIR}" \
        --donor-tokenizer-dir "${DONOR_TOKENIZER_DIR}" \
        --output-dir         "${SWAPPED_DIR}" \
        --new-vocab-size     "${NEW_VOCAB_SIZE}" \
        --init-std           "${INIT_STD}" \
        --seed               "${INIT_SEED}"
else
    echo "--- Swapped model already exists at ${SWAPPED_DIR}, skipping ---"
fi

# 4. Convert HF -> Megatron torch_dist format
# Use python3 directly (single process, no torchrun) to avoid port conflicts.
# Check both existence AND size (a valid 20B checkpoint is multiple GB).
MEGATRON_REF_SIZE=$(du -sm "${MEGATRON_REF_DIR}" 2>/dev/null | awk '{print $1}')
if [ ! -d "${MEGATRON_REF_DIR}" ] || [ "${MEGATRON_REF_SIZE:-0}" -lt 1000 ]; then
    [ -d "${MEGATRON_REF_DIR}" ] && echo "--- Removing incomplete checkpoint (${MEGATRON_REF_SIZE:-0}MB) ---" && rm -rf "${MEGATRON_REF_DIR}"
    echo "--- Converting HF -> Megatron (single process) ---"
    source "${REPO_DIR}/scripts/models/gpt-oss-20b.sh"
    PYTHONPATH="${MEGATRON_PATH}" \
    WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
    MASTER_ADDR=127.0.0.1 MASTER_PORT=29600 \
    python3 "${REPO_DIR}/tools/convert_hf_to_torch_dist.py" \
        "${MODEL_ARGS[@]}" \
        --hf-checkpoint                "${SWAPPED_DIR}" \
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

# 5. Download & convert dataset (AM-Qwen3-Distilled + Qwen3.5 rollouts)
# Uses prep_datasets_with_splits.py which handles both datasets correctly.
# AM sample fraction ~0.11 = ~200k of 1.89M rows (enough to learn embeddings).
if [ ! -f "${DATASET_PATH}" ]; then
    echo "--- Downloading & converting datasets (AM fraction=${AM_SAMPLE_FRACTION}) ---"
    python3 "${SCRIPT_DIR}/prep_datasets_with_splits.py" \
        --output-dir   "${POOL_DIR}" \
        --test-fraction 0.05 \
        --seed          42 \
        --sample-fraction "${AM_SAMPLE_FRACTION}" \
        --skip-qwen35
else
    echo "--- Dataset already exists at ${DATASET_PATH}, skipping ---"
fi

echo "=========================================="
echo "  PREP DONE"
echo "=========================================="

# ======================== VERIFY ARTIFACTS ========================
echo "--- Verifying artifacts ---"
MISSING=0
for required in "${SWAPPED_DIR}" "${MEGATRON_REF_DIR}"; do
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
    echo "WARNING: WANDB_KEY not set. WandB logging will be disabled at train time."
elif [ "${#WANDB_KEY}" -lt 40 ]; then
    echo "WARNING: WANDB_KEY is only ${#WANDB_KEY} chars (need 40+)."
fi
if [ ! -d "${MEGATRON_PATH}" ]; then
    echo "ERROR: Megatron-LM not found at ${MEGATRON_PATH}."
    exit 1
fi

echo "Environment OK."
