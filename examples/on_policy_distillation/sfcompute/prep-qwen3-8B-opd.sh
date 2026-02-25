#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR_DEFAULT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/config-8xh100.env}"

if [ -f "${CONFIG_FILE}" ]; then
    set -a
    # shellcheck disable=SC1090
    source "${CONFIG_FILE}"
    set +a
fi

ROOT_DIR="${ROOT_DIR:-$HOME}"
POOL_DIR="${POOL_DIR:-${ROOT_DIR}/pool}"
MEGATRON_PATH="${MEGATRON_PATH:-${ROOT_DIR}/Megatron-LM}"
REPO_DIR="${REPO_DIR:-${REPO_DIR_DEFAULT}}"

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1"
        exit 1
    fi
}

echo "Running prep with:"
echo "  REPO_DIR=${REPO_DIR}"
echo "  ROOT_DIR=${ROOT_DIR}"
echo "  POOL_DIR=${POOL_DIR}"
echo "  MEGATRON_PATH=${MEGATRON_PATH}"
echo "  CONFIG_FILE=${CONFIG_FILE}"

require_cmd python3
require_cmd huggingface-cli

if [ ! -d "${REPO_DIR}" ]; then
    echo "Repo directory not found: ${REPO_DIR}"
    exit 1
fi
if [ ! -f "${REPO_DIR}/tools/convert_hf_to_torch_dist.py" ]; then
    echo "Missing conversion script: ${REPO_DIR}/tools/convert_hf_to_torch_dist.py"
    exit 1
fi
if [ ! -f "${REPO_DIR}/scripts/models/qwen3-8B.sh" ]; then
    echo "Missing model args script: ${REPO_DIR}/scripts/models/qwen3-8B.sh"
    exit 1
fi
if [ ! -d "${MEGATRON_PATH}" ]; then
    echo "Megatron-LM path not found: ${MEGATRON_PATH}"
    exit 1
fi
if ! huggingface-cli whoami >/dev/null 2>&1; then
    echo "Hugging Face auth is required for model downloads."
    echo "Run: huggingface-cli login"
    exit 1
fi

mkdir -p "${POOL_DIR}"

DATA_DIR="${POOL_DIR}/dapo-math-17k"
DATA_JSONL="${DATA_DIR}/dapo-math-17k.jsonl"
DATA_PARQUET="${DATA_DIR}/data/dapo-math-17k.parquet"
DATA_PARQUET_ALT="${DATA_DIR}/dapo-math-17k.parquet"

if [ ! -f "${DATA_PARQUET}" ] && [ ! -f "${DATA_PARQUET_ALT}" ]; then
    echo "Downloading DAPO-Math-17k dataset..."
    mkdir -p "${DATA_DIR}"
    huggingface-cli download BytedTsinghua-SIA/DAPO-Math-17k \
        --repo-type dataset \
        --local-dir "${DATA_DIR}"
fi

PARQUET_INPUT=""
if [ -f "${DATA_PARQUET}" ]; then
    PARQUET_INPUT="${DATA_PARQUET}"
elif [ -f "${DATA_PARQUET_ALT}" ]; then
    PARQUET_INPUT="${DATA_PARQUET_ALT}"
else
    echo "Dataset parquet not found after download."
    echo "Expected one of:"
    echo "  ${DATA_PARQUET}"
    echo "  ${DATA_PARQUET_ALT}"
    exit 1
fi

if [ ! -f "${DATA_JSONL}" ]; then
    echo "Converting dataset parquet to JSONL..."
    PARQUET_INPUT="${PARQUET_INPUT}" DATA_JSONL="${DATA_JSONL}" python3 - <<'PY'
import os
import pandas as pd

parquet_input = os.environ["PARQUET_INPUT"]
data_jsonl = os.environ["DATA_JSONL"]

df = pd.read_parquet(parquet_input)
df.to_json(data_jsonl, orient="records", lines=True)
print(f"Wrote {len(df)} rows to {data_jsonl}")
PY
else
    echo "Dataset JSONL already exists, skipping conversion: ${DATA_JSONL}"
fi

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

if [ ! -d "${POOL_DIR}/Qwen3-8B_torch_dist" ]; then
    echo "Converting Qwen3-8B HF checkpoint to Megatron torch_dist..."
    # shellcheck disable=SC1091
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
echo "Prep complete. Ready to run training:"
echo "  bash examples/on_policy_distillation/sfcompute/run-qwen3-8B-opd.sh"
echo "========================================="
