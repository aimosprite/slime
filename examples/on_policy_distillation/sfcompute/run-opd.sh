#!/bin/bash
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${BASE_DIR}/../.." && pwd)"
HF_CLI=""
# Load order: env vars > YAML > .env defaults
# YAML is loaded first so its values are visible when .env does ${VAR:-default} checks.
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

hf_get_username() {
    hf_auth_whoami 2>/dev/null | awk '
        NR==1 {
            if ($1 == "name:" && NF >= 2) {
                print $2
            } else if (NF >= 1) {
                print $1
            }
            exit
        }'
}

hf_download() {
    if [ "${HF_CLI}" = "hf" ]; then
        hf download "$@"
    else
        huggingface-cli download "$@"
    fi
}

hf_repo_create() {
    local repo_id="$1"
    local repo_type="$2"
    local visibility_flag="${3:-}"
    if [ "${HF_CLI}" = "hf" ]; then
        if [ -n "${visibility_flag}" ]; then
            hf repo create "${repo_id}" --repo-type "${repo_type}" "${visibility_flag}"
        else
            hf repo create "${repo_id}" --repo-type "${repo_type}"
        fi
    else
        if [ -n "${visibility_flag}" ]; then
            huggingface-cli repo create "${repo_id}" --type "${repo_type}" "${visibility_flag}"
        else
            huggingface-cli repo create "${repo_id}" --type "${repo_type}"
        fi
    fi
}

hf_upload() {
    if [ "${HF_CLI}" = "hf" ]; then
        hf upload "$@"
    else
        huggingface-cli upload "$@"
    fi
}

# Model config (fallbacks if train-config.yaml is absent)
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3.5-122B-A10B}"
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3.5-35B-A3B}"
STUDENT_MODEL_ARGS="${STUDENT_MODEL_ARGS:-qwen3.5-35B-A3B.sh}"

# Derive short names from HF model IDs (e.g. "Qwen/Qwen3.5-122B-A10B" -> "Qwen3.5-122B-A10B")
TEACHER_SHORT="${TEACHER_MODEL##*/}"
STUDENT_SHORT="${STUDENT_MODEL##*/}"

# Dataset config (fallbacks if train-config.yaml is absent)
DATASET="${DATASET:-BytedTsinghua-SIA/DAPO-Math-17k}"
DATASET_SHORT="$(echo "${DATASET##*/}" | tr '[:upper:]' '[:lower:]')"
INPUT_KEY="${INPUT_KEY:-prompt}"
LABEL_KEY="${LABEL_KEY:-reward_model}"
EVAL_RM_TYPE="${EVAL_RM_TYPE:-dapo}"

# Training hyperparameters (fallbacks if train-config.yaml is absent)
NUM_STEPS="${NUM_STEPS:-300}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-16384}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.98}"
OPD_KL_COEF="${OPD_KL_COEF:-1.0}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.00}"
ENTROPY_COEF="${ENTROPY_COEF:-0.00}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-4}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-16384}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"
CHECKPOINT_SHIP_ENABLED="${CHECKPOINT_SHIP_ENABLED:-0}"
CHECKPOINT_SHIP_EVERY="${CHECKPOINT_SHIP_EVERY:-${SAVE_INTERVAL}}"
CHECKPOINT_SHIP_POLL_SEC="${CHECKPOINT_SHIP_POLL_SEC:-15}"
CHECKPOINT_SHIP_CMD="${CHECKPOINT_SHIP_CMD:-}"
CHECKPOINT_SHIP_BACKEND="${CHECKPOINT_SHIP_BACKEND:-huggingface}"
CHECKPOINT_SHIP_LOG="${CHECKPOINT_SHIP_LOG:-/tmp/slime_checkpoint_shipper.log}"
CHECKPOINT_HF_REPO_ID="${CHECKPOINT_HF_REPO_ID:-}"
# Auto-derive basename from model names if not explicitly set
CHECKPOINT_HF_REPO_BASENAME="${CHECKPOINT_HF_REPO_BASENAME:-$(echo "${STUDENT_SHORT}-from-${TEACHER_SHORT}-opd" | tr '[:upper:]' '[:lower:]')}"
CHECKPOINT_HF_REPO_TYPE="${CHECKPOINT_HF_REPO_TYPE:-model}"
CHECKPOINT_HF_PRIVATE="${CHECKPOINT_HF_PRIVATE:-1}"
CHECKPOINT_HF_CREATE_REPO="${CHECKPOINT_HF_CREATE_REPO:-1}"
CHECKPOINT_HF_UPLOAD_TRACKER="${CHECKPOINT_HF_UPLOAD_TRACKER:-1}"

# Derive num_rollout from num_steps:
#   train_iters = num_rollout * rollout_batch_size * n_samples_per_prompt / global_batch_size
#   => num_rollout = num_steps * global_batch_size / (rollout_batch_size * n_samples_per_prompt)
NUM_ROLLOUT_DENOM=$(( ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT ))
if [ "${NUM_ROLLOUT_DENOM}" -le 0 ]; then
    echo "Invalid rollout settings: rollout_batch_size * n_samples_per_prompt must be > 0."
    exit 1
fi
if [ $((NUM_STEPS * GLOBAL_BATCH_SIZE % NUM_ROLLOUT_DENOM)) -ne 0 ]; then
    echo "Invalid batch geometry:"
    echo "  NUM_STEPS*GLOBAL_BATCH_SIZE must be divisible by (ROLLOUT_BATCH_SIZE*N_SAMPLES_PER_PROMPT)"
    echo "  NUM_STEPS=${NUM_STEPS}, GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}, ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE}, N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT}"
    exit 1
fi
NUM_ROLLOUT=$(( NUM_STEPS * GLOBAL_BATCH_SIZE / NUM_ROLLOUT_DENOM ))

ROOT_DIR="${ROOT_DIR:-$HOME}"
POOL_DIR="${POOL_DIR:-${ROOT_DIR}/pool}"

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

# Tee all output to a persistent log on the mounted volume
RUN_LOG="${RUN_LOG:-${POOL_DIR}/run-opd.log}"
mkdir -p "$(dirname "${RUN_LOG}")"
exec > >(tee -a "${RUN_LOG}") 2>&1
echo "=== Run started at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "Log file: ${RUN_LOG}"
MEGATRON_PATH="${MEGATRON_PATH:-${ROOT_DIR}/Megatron-LM}"
count_csv_items() {
    local csv="$1"
    if [ -z "${csv// }" ]; then
        echo 0
        return 0
    fi
    awk -F',' '{print NF}' <<< "${csv}"
}
build_default_gpu_csv() {
    local n="$1"
    if [ "${n}" -le 0 ]; then
        echo ""
        return 0
    fi
    local csv=""
    local i=0
    while [ "${i}" -lt "${n}" ]; do
        if [ -z "${csv}" ]; then
            csv="${i}"
        else
            csv="${csv},${i}"
        fi
        i=$((i + 1))
    done
    echo "${csv}"
}
build_gpu_csv_range() {
    local start="$1"
    local end="$2"
    if [ "${end}" -lt "${start}" ]; then
        echo ""
        return 0
    fi
    local csv=""
    local i="${start}"
    while [ "${i}" -le "${end}" ]; do
        if [ -z "${csv}" ]; then
            csv="${i}"
        else
            csv="${csv},${i}"
        fi
        i=$((i + 1))
    done
    echo "${csv}"
}
detect_primary_ip() {
    python3 - <<'PY'
import socket
ip = "127.0.0.1"
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
except Exception:
    pass
print(ip)
PY
}
CLUSTER_NUM_NODES="${CLUSTER_NUM_NODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
if ! [[ "${CLUSTER_NUM_NODES}" =~ ^[0-9]+$ ]] || [ "${CLUSTER_NUM_NODES}" -le 0 ]; then
    echo "CLUSTER_NUM_NODES must be a positive integer; got '${CLUSTER_NUM_NODES}'."
    exit 1
fi
if ! [[ "${GPUS_PER_NODE}" =~ ^[0-9]+$ ]] || [ "${GPUS_PER_NODE}" -le 0 ]; then
    echo "GPUS_PER_NODE must be a positive integer; got '${GPUS_PER_NODE}'."
    exit 1
fi
if [ -z "${TEACHER_VISIBLE_GPUS+x}" ] || [ -z "${TEACHER_VISIBLE_GPUS}" ]; then
    if [ -n "${TEACHER_GPU:-}" ]; then
        TEACHER_VISIBLE_GPUS="${TEACHER_GPU}"
    elif [ "${CLUSTER_NUM_NODES}" -gt 1 ]; then
        TEACHER_VISIBLE_GPUS="$(build_default_gpu_csv "${GPUS_PER_NODE}")"
    else
        SINGLE_NODE_TEACHER_START=$(( GPUS_PER_NODE / 2 ))
        if [ "${SINGLE_NODE_TEACHER_START}" -le 0 ]; then
            SINGLE_NODE_TEACHER_START=0
        fi
        TEACHER_VISIBLE_GPUS="$(build_gpu_csv_range "${SINGLE_NODE_TEACHER_START}" "$((GPUS_PER_NODE - 1))")"
    fi
fi
if [ -z "${NUM_GPUS+x}" ] || [ -z "${NUM_GPUS}" ]; then
    if [ "${CLUSTER_NUM_NODES}" -gt 1 ]; then
        NUM_GPUS="${GPUS_PER_NODE}"
    else
        NUM_GPUS=$(( GPUS_PER_NODE / 2 ))
        if [ "${NUM_GPUS}" -le 0 ]; then
            NUM_GPUS=1
        fi
    fi
fi
if ! [[ "${NUM_GPUS}" =~ ^[0-9]+$ ]] || [ "${NUM_GPUS}" -le 0 ]; then
    echo "NUM_GPUS must be a positive integer; got '${NUM_GPUS}'."
    exit 1
fi
DEFAULT_RAY_VISIBLE_GPUS="$(build_default_gpu_csv "${NUM_GPUS}")"
TEACHER_NUM_GPUS="$(count_csv_items "${TEACHER_VISIBLE_GPUS}")"
TEACHER_EP="${TEACHER_EP:-${TEACHER_NUM_GPUS}}"
TEACHER_TP="${TEACHER_TP:-${TEACHER_NUM_GPUS}}"
# RAY_VISIBLE_GPUS controls local GPUs visible to the Ray head process.
# Values:
#   - unset / "auto" -> cluster-aware default
#   - "none"         -> no local GPUs (teacher-only head node)
#   - explicit CSV   -> use that list
if [ -z "${RAY_VISIBLE_GPUS+x}" ] || [ -z "${RAY_VISIBLE_GPUS}" ] || [ "${RAY_VISIBLE_GPUS}" = "auto" ]; then
    if [ "${CLUSTER_NUM_NODES}" -gt 1 ]; then
        RAY_VISIBLE_GPUS=""
    else
        RAY_VISIBLE_GPUS="${DEFAULT_RAY_VISIBLE_GPUS}"
    fi
elif [ "${RAY_VISIBLE_GPUS}" = "none" ]; then
    RAY_VISIBLE_GPUS=""
fi
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-${NUM_GPUS}}"
if [ -z "${ACTOR_NUM_NODES+x}" ] || [ -z "${ACTOR_NUM_NODES}" ]; then
    if [ "${CLUSTER_NUM_NODES}" -gt 1 ]; then
        ACTOR_NUM_NODES=$(( CLUSTER_NUM_NODES - 1 ))
    else
        ACTOR_NUM_NODES=1
    fi
fi
DEFAULT_STUDENT_PARALLEL_SIZE=2
if [ "${NUM_GPUS}" -lt 2 ]; then
    DEFAULT_STUDENT_PARALLEL_SIZE=1
fi
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-${DEFAULT_STUDENT_PARALLEL_SIZE}}"
EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-${DEFAULT_STUDENT_PARALLEL_SIZE}}"
EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-${DEFAULT_STUDENT_PARALLEL_SIZE}}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-${ACTOR_NUM_GPUS_PER_NODE}}"
SGLANG_EXPERT_PARALLEL_SIZE="${SGLANG_EXPERT_PARALLEL_SIZE:-${ACTOR_NUM_GPUS_PER_NODE}}"
USE_COLOCATE="${USE_COLOCATE:-}"
if [ -z "${USE_CPU_OPTIMIZER_OFFLOAD+x}" ] || [ -z "${USE_CPU_OPTIMIZER_OFFLOAD}" ]; then
    if [ "${CLUSTER_NUM_NODES}" -gt 1 ]; then
        USE_CPU_OPTIMIZER_OFFLOAD=0
    else
        USE_CPU_OPTIMIZER_OFFLOAD=1
    fi
fi
if [ -z "${TRAIN_MEMORY_MARGIN_BYTES+x}" ] || [ -z "${TRAIN_MEMORY_MARGIN_BYTES}" ]; then
    if [ "${CLUSTER_NUM_NODES}" -gt 1 ]; then
        TRAIN_MEMORY_MARGIN_BYTES=$((1024 * 1024 * 1024))
    else
        TRAIN_MEMORY_MARGIN_BYTES=$((256 * 1024 * 1024))
    fi
fi
TEACHER_PORT="${TEACHER_PORT:-13141}"
TEACHER_MEM_FRACTION="${TEACHER_MEM_FRACTION:-0.75}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.6}"
if [ -z "${RAY_HEAD_IP+x}" ] || [ -z "${RAY_HEAD_IP}" ]; then
    if [ "${CLUSTER_NUM_NODES}" -gt 1 ]; then
        RAY_HEAD_IP="$(detect_primary_ip)"
    else
        RAY_HEAD_IP="${MASTER_ADDR:-127.0.0.1}"
    fi
fi
MASTER_ADDR="${MASTER_ADDR:-${RAY_HEAD_IP}}"
TEACHER_IP="${TEACHER_IP:-${RAY_HEAD_IP}}"
MAX_TEACHER_WAIT_SEC="${MAX_TEACHER_WAIT_SEC:-300}"
ENABLE_EVAL="${ENABLE_EVAL:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-slime-dev}"
WANDB_GROUP="${WANDB_GROUP:-${TEACHER_SHORT}-to-${STUDENT_SHORT}-opd}"

RAY_NUM_GPUS="$(count_csv_items "${RAY_VISIBLE_GPUS}")"
if [ -z "${USE_COLOCATE}" ]; then
    if [ "${CLUSTER_NUM_NODES}" -gt 1 ] || [ "${RAY_NUM_GPUS}" -eq 0 ]; then
        USE_COLOCATE=0
    else
        USE_COLOCATE=1
    fi
fi
if [ "${ACTOR_NUM_NODES}" -eq 1 ] && [ "${RAY_NUM_GPUS}" -gt 0 ] && [ "${ACTOR_NUM_GPUS_PER_NODE}" -gt "${RAY_NUM_GPUS}" ]; then
    echo "Invalid GPU layout:"
    echo "  RAY_VISIBLE_GPUS=${RAY_VISIBLE_GPUS} (${RAY_NUM_GPUS} GPUs)"
    echo "  ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE}"
    exit 1
fi
if [ "${TEACHER_NUM_GPUS}" -le 0 ]; then
    echo "Invalid teacher GPU layout: TEACHER_VISIBLE_GPUS='${TEACHER_VISIBLE_GPUS}'"
    exit 1
fi
if [ "${TEACHER_TP}" -le 0 ] || [ "${TEACHER_EP}" -le 0 ]; then
    echo "Invalid teacher parallelism: TEACHER_TP=${TEACHER_TP}, TEACHER_EP=${TEACHER_EP}"
    exit 1
fi
if [ "${TEACHER_TP}" -ne "${TEACHER_NUM_GPUS}" ]; then
    echo "Teacher parallelism mismatch:"
    echo "  TEACHER_VISIBLE_GPUS=${TEACHER_VISIBLE_GPUS} (${TEACHER_NUM_GPUS} GPUs)"
    echo "  TEACHER_TP=${TEACHER_TP}, TEACHER_EP=${TEACHER_EP}"
    echo "  Expected TEACHER_TP == TEACHER_NUM_GPUS"
    exit 1
fi
if [ "${TEACHER_EP}" -gt "${TEACHER_TP}" ]; then
    echo "Invalid teacher parallelism for sglang:"
    echo "  TEACHER_EP=${TEACHER_EP} must be <= TEACHER_TP=${TEACHER_TP}"
    exit 1
fi

if [ ! -d "${POOL_DIR}" ]; then
    echo "Pool directory not found: ${POOL_DIR}"
    exit 1
fi
if [ ! -d "${MEGATRON_PATH}" ]; then
    echo "Megatron-LM path not found: ${MEGATRON_PATH}"
    exit 1
fi

TEACHER_PID=""
CHECKPOINT_SHIPPER_PID=""
CKPT_SAVE_DIR="${POOL_DIR}/${STUDENT_SHORT}_slime"
STUDENT_TORCH_DIST="${POOL_DIR}/${STUDENT_SHORT}_torch_dist"
HF_REPO_CREATED=0

CHECKPOINT_HF_ORG="${CHECKPOINT_HF_ORG:-aimosprite}"

resolve_hf_checkpoint_repo_id() {
    if [ -n "${CHECKPOINT_HF_REPO_ID}" ]; then
        return 0
    fi

    CHECKPOINT_HF_REPO_ID="${CHECKPOINT_HF_ORG}/${CHECKPOINT_HF_REPO_BASENAME}"
    echo "Auto-selected Hugging Face checkpoint repo: ${CHECKPOINT_HF_REPO_ID}"
}

ship_checkpoint_to_hf() {
    local step="$1"
    local iter_dir="$2"
    local remote_iter_dir=""
    local visibility_flag=""
    printf -v remote_iter_dir "iter_%07d" "${step}"

    resolve_hf_checkpoint_repo_id || return 1

    if [ "${CHECKPOINT_HF_CREATE_REPO}" = "1" ] && [ "${HF_REPO_CREATED}" != "1" ]; then
        if [ "${CHECKPOINT_HF_PRIVATE}" = "1" ]; then
            visibility_flag="--private"
        else
            visibility_flag="--public"
        fi
        local create_out=""
        if create_out="$(hf_repo_create "${CHECKPOINT_HF_REPO_ID}" "${CHECKPOINT_HF_REPO_TYPE}" "${visibility_flag}" 2>&1)"; then
            echo "Created HF repo: ${CHECKPOINT_HF_REPO_ID}"
        else
            if echo "${create_out}" | grep -qi "already"; then
                echo "HF repo already exists: ${CHECKPOINT_HF_REPO_ID}"
            else
                echo "ERROR: hf repo create failed for ${CHECKPOINT_HF_REPO_ID}:"
                echo "${create_out}"
                return 1
            fi
        fi
        HF_REPO_CREATED=1
    fi

    hf_upload \
        "${CHECKPOINT_HF_REPO_ID}" \
        "${iter_dir}" \
        "${remote_iter_dir}" \
        --repo-type "${CHECKPOINT_HF_REPO_TYPE}"

    if [ "${CHECKPOINT_HF_UPLOAD_TRACKER}" = "1" ] && [ -f "${CKPT_SAVE_DIR}/latest_checkpointed_iteration.txt" ]; then
        hf_upload \
            "${CHECKPOINT_HF_REPO_ID}" \
            "${CKPT_SAVE_DIR}/latest_checkpointed_iteration.txt" \
            "latest_checkpointed_iteration.txt" \
            --repo-type "${CHECKPOINT_HF_REPO_TYPE}"
    fi
}

ship_checkpoint_step() {
    local step="$1"
    local iter_dir=""
    printf -v iter_dir "%s/iter_%07d" "${CKPT_SAVE_DIR}" "${step}"

    if [ ! -d "${iter_dir}" ]; then
        echo "Checkpoint directory not found yet for step ${step}: ${iter_dir}"
        return 1
    fi

    echo "Shipping checkpoint step ${step} from ${iter_dir}"
    if [ -n "${CHECKPOINT_SHIP_CMD}" ]; then
        CHECKPOINT_STEP="${step}" CHECKPOINT_ITER_DIR="${iter_dir}" CHECKPOINT_SAVE_DIR="${CKPT_SAVE_DIR}" \
            bash -lc "${CHECKPOINT_SHIP_CMD}"
        return $?
    fi

    if [ "${CHECKPOINT_SHIP_BACKEND}" = "huggingface" ]; then
        ship_checkpoint_to_hf "${step}" "${iter_dir}"
        return $?
    fi

    echo "CHECKPOINT_SHIP_ENABLED=1 but no shipping route configured."
    echo "Set CHECKPOINT_SHIP_CMD, or set CHECKPOINT_SHIP_BACKEND=huggingface."
    return 1
}

start_checkpoint_shipper() {
    local tracker="${CKPT_SAVE_DIR}/latest_checkpointed_iteration.txt"
    echo "Starting checkpoint shipper loop (every ${CHECKPOINT_SHIP_EVERY} steps, poll ${CHECKPOINT_SHIP_POLL_SEC}s)."
    (
        last_synced_step=""
        while true; do
            if [ -f "${tracker}" ]; then
                step="$(tr -d '[:space:]' < "${tracker}" || true)"
                if [[ "${step}" =~ ^[0-9]+$ ]] && [ "${step}" -gt 0 ] && [ "${step}" != "${last_synced_step}" ]; then
                    if [ $((step % CHECKPOINT_SHIP_EVERY)) -eq 0 ]; then
                        if ship_checkpoint_step "${step}"; then
                            last_synced_step="${step}"
                        else
                            echo "Checkpoint ship failed for step ${step}; will retry."
                        fi
                    fi
                fi
            fi
            sleep "${CHECKPOINT_SHIP_POLL_SEC}"
        done
    ) 2>&1 | tee -a "${CHECKPOINT_SHIP_LOG}" &
    CHECKPOINT_SHIPPER_PID=$!
    echo "Checkpoint shipper pid=${CHECKPOINT_SHIPPER_PID}, log=${CHECKPOINT_SHIP_LOG}"
}

preflight_hf_checkpoint_shipping() {
    echo "=== HF checkpoint shipping preflight ==="

    echo "1. Checking HF auth..."
    local whoami=""
    if ! whoami="$(hf_auth_whoami 2>&1)"; then
        echo "FAIL: HF auth failed. Output:"
        echo "${whoami}"
        return 1
    fi
    echo "   OK: authenticated as $(hf_get_username)"

    echo "2. Resolving repo ID..."
    if ! resolve_hf_checkpoint_repo_id; then
        echo "FAIL: could not resolve HF repo ID."
        return 1
    fi
    echo "   OK: repo = ${CHECKPOINT_HF_REPO_ID}"

    echo "3. Creating repo (if needed)..."
    local visibility_flag=""
    if [ "${CHECKPOINT_HF_PRIVATE}" = "1" ]; then
        visibility_flag="--private"
    else
        visibility_flag="--public"
    fi
    local create_out=""
    if create_out="$(hf_repo_create "${CHECKPOINT_HF_REPO_ID}" "${CHECKPOINT_HF_REPO_TYPE}" "${visibility_flag}" 2>&1)"; then
        echo "   OK: created repo ${CHECKPOINT_HF_REPO_ID}"
    else
        if echo "${create_out}" | grep -qi "already"; then
            echo "   OK: repo already exists"
        else
            echo "FAIL: hf repo create error:"
            echo "${create_out}"
            return 1
        fi
    fi

    echo "4. Test upload..."
    local test_file=""
    test_file="$(mktemp /tmp/slime_preflight_XXXXXX.txt)"
    echo "preflight-test $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${test_file}"
    local upload_out=""
    if upload_out="$(hf_upload "${CHECKPOINT_HF_REPO_ID}" "${test_file}" ".preflight_test.txt" --repo-type "${CHECKPOINT_HF_REPO_TYPE}" 2>&1)"; then
        echo "   OK: test file uploaded successfully"
    else
        echo "FAIL: hf upload error:"
        echo "${upload_out}"
        rm -f "${test_file}"
        return 1
    fi
    rm -f "${test_file}"

    echo "=== Preflight PASSED: checkpoint shipping to ${CHECKPOINT_HF_REPO_ID} is working ==="
}

cleanup() {
    echo "Received signal, cleaning up..."
    [ -n "${CHECKPOINT_SHIPPER_PID}" ] && kill -TERM "${CHECKPOINT_SHIPPER_PID}" 2>/dev/null || true
    [ -n "${TEACHER_PID}" ] && kill -TERM "${TEACHER_PID}" 2>/dev/null || true
    pkill -9 -f "python3 -m sglang.launch_server" 2>/dev/null || true
    pkill -9 sglang 2>/dev/null || true
    sleep 1
    ray stop --force 2>/dev/null || true
    pkill -9 ray 2>/dev/null || true
}
trap 'cleanup; exit 0' TERM INT USR1

if [ -f "${REPO_DIR}/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "${REPO_DIR}/.env"
    set +a
fi

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
    echo "Hugging Face auth failed. Set HF_TOKEN in ${REPO_DIR}/.env or run 'hf auth login'."
    exit 1
fi

# --preflight: test HF checkpoint shipping and exit (no training)
if [ "${1:-}" = "--preflight" ]; then
    preflight_hf_checkpoint_shipping
    exit $?
fi

AUTO_PREP="${AUTO_PREP:-1}"
if [ "${AUTO_PREP}" = "1" ]; then
    echo "Ensuring prep artifacts are ready..."
    CONFIG_FILE="${CONFIG_FILE}" \
    TRAIN_CONFIG="${TRAIN_CONFIG}" \
    ROOT_DIR="${ROOT_DIR}" \
    POOL_DIR="${POOL_DIR}" \
    MEGATRON_PATH="${MEGATRON_PATH}" \
    REPO_DIR="${REPO_DIR}" \
        bash "${SCRIPT_DIR}/prep-opd.sh"
elif [ ! -d "${STUDENT_TORCH_DIST}" ]; then
    echo "Missing student Megatron checkpoint: ${STUDENT_TORCH_DIST}"
    echo "Set AUTO_PREP=1 (default) or run prep manually."
    exit 1
fi

if [ ! -d "${TEACHER_PATH}" ]; then
    if [[ "${TEACHER_MODEL}" == /* ]]; then
        echo "Teacher model path not found: ${TEACHER_PATH}"
        exit 1
    fi
    echo "Downloading ${TEACHER_SHORT} (teacher)..."
    hf_download "${TEACHER_MODEL}" --local-dir "${TEACHER_PATH}"
fi
if [ ! -d "${STUDENT_PATH}" ]; then
    if [[ "${STUDENT_MODEL}" == /* ]]; then
        echo "Student model path not found: ${STUDENT_PATH}"
        exit 1
    fi
    echo "Downloading ${STUDENT_SHORT} (student)..."
    hf_download "${STUDENT_MODEL}" --local-dir "${STUDENT_PATH}"
fi

WANDB_KEY="${WANDB_API_KEY:-${WANDB_KEY:-}}"
if [ -z "${WANDB_KEY}" ]; then
    echo "WANDB is enabled but no API key found in environment (.env)."
    echo "Set WANDB_API_KEY (or WANDB_KEY) before running."
    exit 1
fi

if [ "${CHECKPOINT_SHIP_ENABLED}" = "1" ]; then
    if ! [[ "${CHECKPOINT_SHIP_EVERY}" =~ ^[0-9]+$ ]] || [ "${CHECKPOINT_SHIP_EVERY}" -le 0 ]; then
        echo "CHECKPOINT_SHIP_EVERY must be a positive integer; got '${CHECKPOINT_SHIP_EVERY}'."
        exit 1
    fi
    if [ -z "${CHECKPOINT_SHIP_CMD}" ] && [ "${CHECKPOINT_SHIP_BACKEND}" != "huggingface" ]; then
        echo "CHECKPOINT_SHIP_ENABLED=1 requires either:"
        echo "  - CHECKPOINT_SHIP_CMD, or"
        echo "  - CHECKPOINT_SHIP_BACKEND=huggingface"
        exit 1
    fi
    if [ "${CHECKPOINT_SHIP_BACKEND}" = "huggingface" ]; then
        preflight_hf_checkpoint_shipping || exit 1
    fi
    mkdir -p "${CKPT_SAVE_DIR}"
    start_checkpoint_shipper
fi

LOG_FILE="/tmp/sglang_$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 6).log"
TEACHER_SGLANG_ARGS=(
    --model-path "${TEACHER_PATH}"
    --host 0.0.0.0
    --port "${TEACHER_PORT}"
    --tp "${TEACHER_TP}"
    --chunked-prefill-size 4096
    --mem-fraction-static "${TEACHER_MEM_FRACTION}"
)
if [ "${TEACHER_EP}" -gt 1 ]; then
    TEACHER_SGLANG_ARGS+=(--expert-parallel-size "${TEACHER_EP}")
fi
CUDA_VISIBLE_DEVICES="${TEACHER_VISIBLE_GPUS}" python3 -m sglang.launch_server \
    "${TEACHER_SGLANG_ARGS[@]}" \
    > "${LOG_FILE}" 2>&1 &
TEACHER_PID=$!

teacher_waited_sec=0
until curl -sf "http://${TEACHER_IP}:${TEACHER_PORT}/health_generate" > /dev/null; do
    echo "Waiting for teacher server to start..."
    tail -n 10 "${LOG_FILE}"
    teacher_waited_sec=$((teacher_waited_sec + 5))
    if [ "${teacher_waited_sec}" -ge "${MAX_TEACHER_WAIT_SEC}" ]; then
        echo "Teacher server did not become healthy within ${MAX_TEACHER_WAIT_SEC}s."
        exit 1
    fi
    sleep 5
done
curl "http://${TEACHER_IP}:${TEACHER_PORT}/get_model_info"
sleep 10

export PYTHONUNBUFFERED=1
source "${REPO_DIR}/scripts/models/${STUDENT_MODEL_ARGS}"

DATA_DIR="${POOL_DIR}/${DATASET_SHORT}"
TRAIN_DATA="${DATA_DIR}/${DATASET_SHORT}-train.jsonl"
EVAL_DATA="${DATA_DIR}/${DATASET_SHORT}-eval.jsonl"
FALLBACK_DATA="${DATA_DIR}/${DATASET_SHORT}.jsonl"
if [ ! -f "${TRAIN_DATA}" ] && [ -f "${FALLBACK_DATA}" ]; then
    TRAIN_DATA="${FALLBACK_DATA}"
fi
if [ ! -f "${EVAL_DATA}" ] && [ -f "${FALLBACK_DATA}" ]; then
    EVAL_DATA="${FALLBACK_DATA}"
fi
if [ ! -f "${TRAIN_DATA}" ]; then
    echo "Missing training data file: ${TRAIN_DATA}"
    exit 1
fi
if [ ! -f "${EVAL_DATA}" ]; then
    echo "Missing eval data file: ${EVAL_DATA}"
    exit 1
fi

CKPT_ARGS=(
   --hf-checkpoint "${STUDENT_PATH}"
   --ref-load "${STUDENT_TORCH_DIST}"
   --load "${CKPT_SAVE_DIR}/"
   --save "${CKPT_SAVE_DIR}/"
   --save-interval "${SAVE_INTERVAL}"
)

USE_TOOLS="${USE_TOOLS:-1}"

ROLLOUT_ARGS=(
   --prompt-data "${TRAIN_DATA}"
   --input-key "${INPUT_KEY}"
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-temperature "${ROLLOUT_TEMPERATURE}"
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --balance-data
)

if [ "${USE_TOOLS}" = "1" ]; then
   RM_ARGS=(
      --custom-generate-function-path examples.on_policy_distillation.generate_with_tools.generate
      --custom-rm-path examples.on_policy_distillation.generate_with_tools.reward_func
      --custom-reward-post-process-path examples.on_policy_distillation.generate_with_tools.post_process_rewards
      --rm-url "http://${TEACHER_IP}:${TEACHER_PORT}/generate"
   )
else
   ROLLOUT_ARGS+=(--apply-chat-template)
   RM_ARGS=(
      --custom-rm-path examples.on_policy_distillation.on_policy_distillation.reward_func
      --custom-reward-post-process-path examples.on_policy_distillation.on_policy_distillation.post_process_rewards
      --rm-url "http://${TEACHER_IP}:${TEACHER_PORT}/generate"
   )
fi

EVAL_ARGS=()
if [ "${ENABLE_EVAL}" = "1" ]; then
   # Generate eval config YAML that uses rule-based grading (not the teacher RM)
   EVAL_CONFIG_FILE="/tmp/slime_eval_config.yaml"
   cat > "${EVAL_CONFIG_FILE}" <<EVALEOF
defaults:
  n_samples_per_eval_prompt: ${N_SAMPLES_PER_EVAL_PROMPT}
  temperature: ${EVAL_TEMPERATURE}
  max_response_len: ${EVAL_MAX_RESPONSE_LEN}

datasets:
  eval:
    path: ${EVAL_DATA}
    input_key: ${INPUT_KEY}
    label_key: ${LABEL_KEY}
    rm_type: ${EVAL_RM_TYPE}
EVALEOF
   EVAL_ARGS=(
      --eval-interval "${EVAL_INTERVAL}"
      --eval-config "${EVAL_CONFIG_FILE}"
      --eval-reward-key acc
   )
fi

PERF_ARGS=(
   --bf16
   --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE:-1}"
   --expert-tensor-parallel-size "${EXPERT_TENSOR_PARALLEL_SIZE:-1}"
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type sglang
   --opd-kl-coef "${OPD_KL_COEF}"
   --kl-loss-coef "${KL_LOSS_COEF}"
   --kl-loss-type low_var_kl
   --entropy-coef "${ENTROPY_COEF}"
)
# Only load the reference model when kl_loss_coef > 0 (saves ~32 GiB GPU memory).
if [ "$(echo "${KL_LOSS_COEF} > 0" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
   GRPO_ARGS+=(--use-kl-loss)
fi

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LEARNING_RATE}"
   --lr-decay-style constant
   --weight-decay "${WEIGHT_DECAY}"
   --adam-beta1 "${ADAM_BETA1}"
   --adam-beta2 "${ADAM_BETA2}"
)
if [ "${USE_CPU_OPTIMIZER_OFFLOAD}" = "1" ]; then
   OPTIMIZER_ARGS+=(
      --optimizer-cpu-offload
      --overlap-cpu-optimizer-d2h-h2d
      --use-precision-aware-optimizer
   )
fi

WANDB_ARGS=(
   --use-wandb
   --wandb-project "${WANDB_PROJECT}"
   --wandb-group "${WANDB_GROUP}"
   --wandb-key "${WANDB_KEY}"
   --wandb-config-file "${TRAIN_CONFIG}"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
   --sglang-expert-parallel-size "${SGLANG_EXPERT_PARALLEL_SIZE}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --attention-softmax-in-fp32
   --attention-backend flash
   --train-memory-margin-bytes "${TRAIN_MEMORY_MARGIN_BYTES}"
)

ray stop --force 2>/dev/null || true
CUDA_VISIBLE_DEVICES="${RAY_VISIBLE_GPUS}" ray start --head --node-ip-address "${RAY_HEAD_IP}" --num-gpus "${RAY_NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RAY_JOB_ARGS=(
   --actor-num-nodes "${ACTOR_NUM_NODES}"
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}"
)
if [ "${USE_COLOCATE}" = "1" ]; then
   RAY_JOB_ARGS+=(--colocate)
fi

set +e
ray job submit --address="http://${RAY_HEAD_IP}:8265" \
   --runtime-env-json="{
     \"env_vars\": {
        \"PYTHONPATH\": \"${MEGATRON_PATH}\",
        \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
     }
   }" \
   -- python3 "${REPO_DIR}/train.py" \
   "${RAY_JOB_ARGS[@]}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${RM_ARGS[@]}"
RAY_EXIT_CODE=$?
set -e

if [ "${CHECKPOINT_SHIP_ENABLED}" = "1" ] && [ -f "${CKPT_SAVE_DIR}/latest_checkpointed_iteration.txt" ]; then
    FINAL_STEP="$(tr -d '[:space:]' < "${CKPT_SAVE_DIR}/latest_checkpointed_iteration.txt" || true)"
    if [[ "${FINAL_STEP}" =~ ^[0-9]+$ ]] && [ "${FINAL_STEP}" -gt 0 ]; then
        ship_checkpoint_step "${FINAL_STEP}" || true
    fi
fi

cleanup
exit "${RAY_EXIT_CODE}"
