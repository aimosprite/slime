#!/bin/bash
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${BASE_DIR}/../.." && pwd)"
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
CHECKPOINT_HF_REPO_BASENAME="${CHECKPOINT_HF_REPO_BASENAME:-qwen3-8b-opd-checkpoints}"
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
MEGATRON_PATH="${MEGATRON_PATH:-${ROOT_DIR}/Megatron-LM}"
TEACHER_VISIBLE_GPUS="${TEACHER_VISIBLE_GPUS:-${TEACHER_GPU:-6,7}}"
TEACHER_TP="${TEACHER_TP:-$(awk -F',' '{print NF}' <<< "${TEACHER_VISIBLE_GPUS}")}"
RAY_VISIBLE_GPUS="${RAY_VISIBLE_GPUS:-0,1,2,3,4,5}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-4}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-1}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-4}"
TEACHER_PORT="${TEACHER_PORT:-13141}"
TEACHER_MEM_FRACTION="${TEACHER_MEM_FRACTION:-0.6}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.4}"
MAX_TEACHER_WAIT_SEC="${MAX_TEACHER_WAIT_SEC:-300}"
ENABLE_EVAL="${ENABLE_EVAL:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-slime-dev}"
WANDB_GROUP="${WANDB_GROUP:-qwen3-32B-to-8B-opd}"

NUM_GPUS="$(awk -F',' '{print NF}' <<< "${RAY_VISIBLE_GPUS}")"
if [ $((ACTOR_NUM_GPUS_PER_NODE + ROLLOUT_NUM_GPUS)) -gt "${NUM_GPUS}" ]; then
    echo "Invalid GPU layout:"
    echo "  RAY_VISIBLE_GPUS=${RAY_VISIBLE_GPUS} (${NUM_GPUS} GPUs)"
    echo "  ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE}"
    echo "  ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS}"
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
if [ ! -d "${POOL_DIR}/Qwen3-8B_torch_dist" ]; then
    echo "Missing student Megatron checkpoint: ${POOL_DIR}/Qwen3-8B_torch_dist"
    echo "Run prep first:"
    echo "  bash examples/on_policy_distillation/sfcompute/prep-qwen3-8B-opd.sh"
    exit 1
fi

TEACHER_PID=""
CHECKPOINT_SHIPPER_PID=""
CKPT_SAVE_DIR="${POOL_DIR}/Qwen3-8B_slime"
HF_REPO_CREATED=0

resolve_hf_checkpoint_repo_id() {
    if [ -n "${CHECKPOINT_HF_REPO_ID}" ]; then
        return 0
    fi

    local hf_user=""
    hf_user="$(huggingface-cli whoami 2>/dev/null | awk -F': ' '/^name:/{print $2; exit}')"
    if [ -z "${hf_user}" ]; then
        echo "Failed to infer Hugging Face username. Run 'huggingface-cli login' or set CHECKPOINT_HF_REPO_ID."
        return 1
    fi

    CHECKPOINT_HF_REPO_ID="${hf_user}/${CHECKPOINT_HF_REPO_BASENAME}"
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
        huggingface-cli repo create "${CHECKPOINT_HF_REPO_ID}" --type "${CHECKPOINT_HF_REPO_TYPE}" "${visibility_flag}" >/dev/null 2>&1 || true
        HF_REPO_CREATED=1
    fi

    huggingface-cli upload \
        "${CHECKPOINT_HF_REPO_ID}" \
        "${iter_dir}" \
        "${remote_iter_dir}" \
        --repo-type "${CHECKPOINT_HF_REPO_TYPE}"

    if [ "${CHECKPOINT_HF_UPLOAD_TRACKER}" = "1" ] && [ -f "${CKPT_SAVE_DIR}/latest_checkpointed_iteration.txt" ]; then
        huggingface-cli upload \
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
    ) >> "${CHECKPOINT_SHIP_LOG}" 2>&1 &
    CHECKPOINT_SHIPPER_PID=$!
    echo "Checkpoint shipper pid=${CHECKPOINT_SHIPPER_PID}, log=${CHECKPOINT_SHIP_LOG}"
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

if [ ! -d "${POOL_DIR}/Qwen3-32B" ]; then
    echo "Downloading Qwen3-32B (teacher)..."
    huggingface-cli download Qwen/Qwen3-32B --local-dir "${POOL_DIR}/Qwen3-32B"
fi
if [ ! -d "${POOL_DIR}/Qwen3-8B" ]; then
    echo "Downloading Qwen3-8B (student)..."
    huggingface-cli download Qwen/Qwen3-8B --local-dir "${POOL_DIR}/Qwen3-8B"
fi

if [ -f "${REPO_DIR}/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "${REPO_DIR}/.env"
    set +a
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
        if ! command -v huggingface-cli >/dev/null 2>&1; then
            echo "huggingface-cli not found but CHECKPOINT_SHIP_BACKEND=huggingface."
            exit 1
        fi
        resolve_hf_checkpoint_repo_id || exit 1
    fi
    mkdir -p "${CKPT_SAVE_DIR}"
    start_checkpoint_shipper
fi

TEACHER_IP="127.0.0.1"
LOG_FILE="/tmp/sglang_$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 6).log"
CUDA_VISIBLE_DEVICES="${TEACHER_VISIBLE_GPUS}" python3 -m sglang.launch_server \
    --model-path "${POOL_DIR}/Qwen3-32B" \
    --host 0.0.0.0 \
    --port "${TEACHER_PORT}" \
    --tp "${TEACHER_TP}" \
    --chunked-prefill-size 4096 \
    --mem-fraction-static "${TEACHER_MEM_FRACTION}" \
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
source "${REPO_DIR}/scripts/models/qwen3-8B.sh"

DATA_DIR="${POOL_DIR}/dapo-math-17k"
TRAIN_DATA="${DATA_DIR}/dapo-math-17k-train.jsonl"
EVAL_DATA="${DATA_DIR}/dapo-math-17k-eval.jsonl"
FALLBACK_DATA="${DATA_DIR}/dapo-math-17k.jsonl"
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
   --hf-checkpoint "${POOL_DIR}/Qwen3-8B"
   --ref-load "${POOL_DIR}/Qwen3-8B_torch_dist"
   --load "${CKPT_SAVE_DIR}/"
   --save "${CKPT_SAVE_DIR}/"
   --save-interval "${SAVE_INTERVAL}"
)

ROLLOUT_ARGS=(
   --prompt-data "${TRAIN_DATA}"
   --input-key prompt
   --apply-chat-template
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-temperature "${ROLLOUT_TEMPERATURE}"
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --balance-data
)

RM_ARGS=(
   --custom-rm-path examples.on_policy_distillation.on_policy_distillation.reward_func
   --custom-reward-post-process-path examples.on_policy_distillation.on_policy_distillation.post_process_rewards
   --rm-url "http://${TEACHER_IP}:${TEACHER_PORT}/generate"
)

EVAL_ARGS=()
if [ "${ENABLE_EVAL}" = "1" ]; then
   EVAL_ARGS=(
      --eval-interval "${EVAL_INTERVAL}"
      --eval-prompt-data eval "${EVAL_DATA}"
      --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
      --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
      --eval-temperature "${EVAL_TEMPERATURE}"
   )
fi

PERF_ARGS=(
   --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
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
   --use-kl-loss
   --kl-loss-coef "${KL_LOSS_COEF}"
   --kl-loss-type low_var_kl
   --entropy-coef "${ENTROPY_COEF}"
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LEARNING_RATE}"
   --lr-decay-style constant
   --weight-decay "${WEIGHT_DECAY}"
   --adam-beta1 "${ADAM_BETA1}"
   --adam-beta2 "${ADAM_BETA2}"
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project "${WANDB_PROJECT}"
   --wandb-group "${WANDB_GROUP}"
   --wandb-key "${WANDB_KEY}"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export CUDA_VISIBLE_DEVICES="${RAY_VISIBLE_GPUS}"
ray stop --force 2>/dev/null || true
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

set +e
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="{
     \"env_vars\": {
        \"PYTHONPATH\": \"${MEGATRON_PATH}\",
        \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
     }
   }" \
   -- python3 "${REPO_DIR}/train.py" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
   --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
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
