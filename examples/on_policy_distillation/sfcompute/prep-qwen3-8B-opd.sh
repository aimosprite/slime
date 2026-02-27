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
HF_CLI=""

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1"
        exit 1
    fi
}

load_repo_env() {
    if [ -f "${REPO_DIR}/.env" ]; then
        set -a
        # shellcheck disable=SC1090
        source "${REPO_DIR}/.env"
        set +a
    fi
}

set_hf_cli() {
    if command -v hf >/dev/null 2>&1; then
        HF_CLI="hf"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        HF_CLI="huggingface-cli"
    else
        HF_CLI=""
    fi
}

install_hf_cli_if_missing() {
    set_hf_cli
    if [ -n "${HF_CLI}" ]; then
        return 0
    fi

    echo "Installing Hugging Face CLI..."
    python3 -m pip install -U "huggingface_hub[cli]"
    set_hf_cli
    if [ -z "${HF_CLI}" ]; then
        echo "Failed to install Hugging Face CLI."
        exit 1
    fi
}

hf_auth_whoami() {
    if [ "${HF_CLI}" = "hf" ]; then
        hf auth whoami
    else
        huggingface-cli whoami
    fi
}

hf_auth_login_token() {
    local token="$1"
    if [ -z "${token}" ]; then
        return 1
    fi

    if [ "${HF_CLI}" = "hf" ]; then
        hf auth login --token "${token}" --add-to-git-credential >/dev/null 2>&1 || \
            hf auth login --token "${token}" >/dev/null 2>&1
    else
        huggingface-cli login --token "${token}" --add-to-git-credential >/dev/null 2>&1 || \
            huggingface-cli login --token "${token}" >/dev/null 2>&1
    fi
}

hf_download() {
    if [ "${HF_CLI}" = "hf" ]; then
        hf download "$@"
    else
        huggingface-cli download "$@"
    fi
}

echo "Running prep with:"
echo "  REPO_DIR=${REPO_DIR}"
echo "  ROOT_DIR=${ROOT_DIR}"
echo "  POOL_DIR=${POOL_DIR}"
echo "  MEGATRON_PATH=${MEGATRON_PATH}"
echo "  CONFIG_FILE=${CONFIG_FILE}"

require_cmd python3

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

load_repo_env
install_hf_cli_if_missing
HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}}"
if [ -n "${HF_TOKEN}" ]; then
    export HF_TOKEN
fi

if ! hf_auth_whoami >/dev/null 2>&1; then
    if [ -n "${HF_TOKEN}" ]; then
        xtrace_was_on=0
        if [[ "$-" == *x* ]]; then
            xtrace_was_on=1
            set +x
        fi
        hf_auth_login_token "${HF_TOKEN}" || true
        if [ "${xtrace_was_on}" -eq 1 ]; then
            set -x
        fi
    fi
fi

if ! hf_auth_whoami >/dev/null 2>&1; then
    echo "Hugging Face auth is required for model downloads."
    echo "Set HF_TOKEN in ${REPO_DIR}/.env or run 'hf auth login'."
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
    hf_download BytedTsinghua-SIA/DAPO-Math-17k \
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

TRAIN_JSONL="${DATA_DIR}/dapo-math-17k-train.jsonl"
EVAL_JSONL="${DATA_DIR}/dapo-math-17k-eval.jsonl"

if [ ! -f "${DATA_JSONL}" ] && [ ! -f "${TRAIN_JSONL}" ]; then
    echo "Converting dataset parquet to JSONL with train/eval split..."
    PARQUET_INPUT="${PARQUET_INPUT}" DATA_JSONL="${DATA_JSONL}" \
    TRAIN_JSONL="${TRAIN_JSONL}" EVAL_JSONL="${EVAL_JSONL}" python3 - <<'PY'
import os
import pandas as pd

parquet_input = os.environ["PARQUET_INPUT"]
data_jsonl = os.environ["DATA_JSONL"]
train_jsonl = os.environ["TRAIN_JSONL"]
eval_jsonl = os.environ["EVAL_JSONL"]

df = pd.read_parquet(parquet_input)
df.to_json(data_jsonl, orient="records", lines=True)
print(f"Wrote {len(df)} rows to {data_jsonl}")

# Create train/eval split (last 500 rows for eval)
eval_size = min(500, len(df) // 10)
train_df = df.iloc[:-eval_size]
eval_df = df.iloc[-eval_size:]
train_df.to_json(train_jsonl, orient="records", lines=True)
eval_df.to_json(eval_jsonl, orient="records", lines=True)
print(f"Split: {len(train_df)} train, {len(eval_df)} eval")
PY
else
    echo "Dataset JSONL already exists, skipping conversion."
fi

if [ ! -d "${POOL_DIR}/Qwen3-32B" ]; then
    echo "Downloading Qwen3-32B (teacher)..."
    hf_download Qwen/Qwen3-32B --local-dir "${POOL_DIR}/Qwen3-32B"
else
    echo "Qwen3-32B already exists, skipping download."
fi

if [ ! -d "${POOL_DIR}/Qwen3-8B" ]; then
    echo "Downloading Qwen3-8B (student)..."
    hf_download Qwen/Qwen3-8B --local-dir "${POOL_DIR}/Qwen3-8B"
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
