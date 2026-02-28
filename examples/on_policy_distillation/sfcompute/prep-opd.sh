#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR_DEFAULT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Load train-config.yaml first (same logic as run-opd.sh)
TRAIN_CONFIG="${TRAIN_CONFIG:-${SCRIPT_DIR}/train-config.yaml}"
if [ -f "${TRAIN_CONFIG}" ]; then
    eval "$(python3 - "${TRAIN_CONFIG}" <<'PYEOF'
import sys, re, os
for line in open(sys.argv[1]):
    line = line.split('#')[0].strip()
    m = re.match(r'^([a-z_]+):\s*(\S.*)', line)
    if m:
        k, v = m.group(1).upper(), m.group(2).strip()
        if k not in os.environ:
            print(f"export {k}='{v}'")
PYEOF)"
fi

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

# Model config
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3.5-122B-A10B}"
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3.5-35B-A3B}"
STUDENT_MODEL_ARGS="${STUDENT_MODEL_ARGS:-qwen3.5-35B-A3B.sh}"
TEACHER_SHORT="${TEACHER_MODEL##*/}"
STUDENT_SHORT="${STUDENT_MODEL##*/}"

# Dataset config
DATASET="${DATASET:-BytedTsinghua-SIA/DAPO-Math-17k}"
DATASET_SHORT="$(echo "${DATASET##*/}" | tr '[:upper:]' '[:lower:]')"

# Resolve model paths: local absolute paths are used directly, HF IDs resolve to POOL_DIR.
if [[ "${TEACHER_MODEL}" == /* ]]; then
    TEACHER_PATH="${TEACHER_MODEL}"
else
    TEACHER_PATH="${POOL_DIR}/${TEACHER_SHORT}"
fi
if [[ "${STUDENT_MODEL}" == /* ]]; then
    STUDENT_PATH="${STUDENT_MODEL}"
else
    STUDENT_PATH="${POOL_DIR}/${STUDENT_SHORT}"
fi

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

ensure_transformers_qwen35_support() {
    if python3 - <<'PY'
try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if "qwen3_5_moe" in CONFIG_MAPPING else 1)
PY
    then
        echo "Transformers has qwen3_5_moe support."
        return 0
    fi

    echo "Transformers is missing qwen3_5_moe. Installing a newer build..."
    local transformers_src="${TRANSFORMERS_QWEN35_SOURCE:-git+https://github.com/huggingface/transformers.git}"
    python3 -m pip install -U "${transformers_src}"

    if ! python3 - <<'PY'
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
raise SystemExit(0 if "qwen3_5_moe" in CONFIG_MAPPING else 1)
PY
    then
        echo "Failed to install a Transformers version with qwen3_5_moe support."
        echo "Set TRANSFORMERS_QWEN35_SOURCE to a compatible package and rerun."
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

detect_convert_nproc_per_node() {
    # Respect explicit override first.
    if [ -n "${CONVERT_NPROC_PER_NODE:-}" ]; then
        echo "${CONVERT_NPROC_PER_NODE}"
        return 0
    fi

    # If CUDA_VISIBLE_DEVICES is set to a list, use its length.
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${CUDA_VISIBLE_DEVICES}" != "all" ]; then
        IFS=',' read -r -a cvd_arr <<< "${CUDA_VISIBLE_DEVICES}"
        if [ "${#cvd_arr[@]}" -gt 0 ]; then
            echo "${#cvd_arr[@]}"
            return 0
        fi
    fi

    # Fallback to nvidia-smi device count.
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_count
        gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d '[:space:]')"
        if [ -n "${gpu_count}" ] && [ "${gpu_count}" -gt 0 ]; then
            echo "${gpu_count}"
            return 0
        fi
    fi

    echo "1"
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
if [ ! -f "${REPO_DIR}/scripts/models/${STUDENT_MODEL_ARGS}" ]; then
    echo "Missing model args script: ${REPO_DIR}/scripts/models/${STUDENT_MODEL_ARGS}"
    exit 1
fi
if [ ! -d "${MEGATRON_PATH}" ]; then
    echo "Megatron-LM path not found: ${MEGATRON_PATH}"
    exit 1
fi

load_repo_env
install_hf_cli_if_missing
ensure_transformers_qwen35_support
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

DATA_DIR="${POOL_DIR}/${DATASET_SHORT}"
DATA_JSONL="${DATA_DIR}/${DATASET_SHORT}.jsonl"
TRAIN_JSONL="${DATA_DIR}/${DATASET_SHORT}-train.jsonl"
EVAL_JSONL="${DATA_DIR}/${DATASET_SHORT}-eval.jsonl"

# Download the dataset from HuggingFace if no parquet files exist locally.
FOUND_PARQUET="$(find "${DATA_DIR}" -name '*.parquet' 2>/dev/null | head -1 || true)"
if [ -z "${FOUND_PARQUET}" ]; then
    echo "Downloading dataset ${DATASET}..."
    mkdir -p "${DATA_DIR}"
    hf_download "${DATASET}" \
        --repo-type dataset \
        --local-dir "${DATA_DIR}"
fi

# Locate the first parquet file in the downloaded directory.
PARQUET_INPUT="$(find "${DATA_DIR}" -name '*.parquet' 2>/dev/null | head -1 || true)"
if [ -z "${PARQUET_INPUT}" ]; then
    echo "No .parquet file found in ${DATA_DIR} after download."
    echo "Check that dataset '${DATASET}' exists on HuggingFace."
    exit 1
fi
echo "Using parquet: ${PARQUET_INPUT}"

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

# Create train/eval split (last 10% for eval, at least 1 row)
eval_size = max(1, min(500, len(df) // 10))
if len(df) <= eval_size:
    # Dataset too small to split; use all rows for both train and eval.
    df.to_json(train_jsonl, orient="records", lines=True)
    df.to_json(eval_jsonl, orient="records", lines=True)
    print(f"Dataset has only {len(df)} rows; using all for both train and eval.")
else:
    train_df = df.iloc[:-eval_size]
    eval_df = df.iloc[-eval_size:]
    train_df.to_json(train_jsonl, orient="records", lines=True)
    eval_df.to_json(eval_jsonl, orient="records", lines=True)
    print(f"Split: {len(train_df)} train, {len(eval_df)} eval")
PY
else
    echo "Dataset JSONL already exists, skipping conversion."
fi

if [ ! -d "${TEACHER_PATH}" ]; then
    if [[ "${TEACHER_MODEL}" == /* ]]; then
        echo "Teacher model path not found: ${TEACHER_PATH}"
        exit 1
    fi
    echo "Downloading ${TEACHER_SHORT} (teacher)..."
    hf_download "${TEACHER_MODEL}" --local-dir "${TEACHER_PATH}"
else
    echo "Teacher model (${TEACHER_PATH}) already exists, skipping download."
fi

if [ ! -d "${STUDENT_PATH}" ]; then
    if [[ "${STUDENT_MODEL}" == /* ]]; then
        echo "Student model path not found: ${STUDENT_PATH}"
        exit 1
    fi
    echo "Downloading ${STUDENT_SHORT} (student)..."
    hf_download "${STUDENT_MODEL}" --local-dir "${STUDENT_PATH}"
else
    echo "Student model (${STUDENT_PATH}) already exists, skipping download."
fi

STUDENT_TORCH_DIST="${POOL_DIR}/${STUDENT_SHORT}_torch_dist"
if [ ! -d "${STUDENT_TORCH_DIST}" ]; then
    echo "Converting ${STUDENT_SHORT} HF checkpoint to Megatron torch_dist..."
    # shellcheck disable=SC1091
    source "${REPO_DIR}/scripts/models/${STUDENT_MODEL_ARGS}"
    CONVERT_NPROC="$(detect_convert_nproc_per_node)"
    if [ "${CONVERT_NPROC}" -gt 1 ] && command -v torchrun >/dev/null 2>&1; then
        echo "Using torchrun for conversion with nproc-per-node=${CONVERT_NPROC}."
        PYTHONPATH="${MEGATRON_PATH}" torchrun --nproc-per-node "${CONVERT_NPROC}" \
            "${REPO_DIR}/tools/convert_hf_to_torch_dist.py" \
            "${MODEL_ARGS[@]}" \
            --hf-checkpoint "${STUDENT_PATH}" \
            --save "${STUDENT_TORCH_DIST}"
    else
        echo "Using single-process conversion (set CONVERT_NPROC_PER_NODE>1 to parallelize)."
        PYTHONPATH="${MEGATRON_PATH}" python3 "${REPO_DIR}/tools/convert_hf_to_torch_dist.py" \
            "${MODEL_ARGS[@]}" \
            --hf-checkpoint "${STUDENT_PATH}" \
            --save "${STUDENT_TORCH_DIST}"
    fi
else
    echo "${STUDENT_SHORT}_torch_dist already exists, skipping conversion."
fi

echo ""
echo "========================================="
echo "Prep complete. Ready to run training:"
echo "  bash examples/on_policy_distillation/sfcompute/run-opd.sh"
echo "========================================="
