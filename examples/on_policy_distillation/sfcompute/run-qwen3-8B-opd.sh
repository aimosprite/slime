#!/bin/bash
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${BASE_DIR}/../.." && pwd)"
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
cleanup() {
    echo "Received signal, cleaning up..."
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

export PYTHONUNBUFFERED=16
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
   --load "${POOL_DIR}/Qwen3-8B_slime/"
   --save "${POOL_DIR}/Qwen3-8B_slime/"
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data "${TRAIN_DATA}"
   --input-key prompt
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 300
   --rollout-batch-size 16
   --n-samples-per-prompt 4
   --rollout-max-response-len 16384
   --rollout-temperature 1
   --global-batch-size 64
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
      --eval-interval 20
      --eval-prompt-data eval "${EVAL_DATA}"
      --n-samples-per-eval-prompt 4
      --eval-max-response-len 16384
      --eval-temperature 1
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
   --max-tokens-per-gpu 16384
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type sglang
   --opd-kl-coef 1.0
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
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

cleanup
exit "${RAY_EXIT_CODE}"
