#!/bin/bash

# On-Policy Distillation with Megatron training backend
# Qwen3-32B (teacher, SGLang server) -> Qwen3-8B (student, Megatron)
#
# Requires 4 GPUs:
#   GPU 0,1 - Megatron training actor (student)
#   GPU 2   - SGLang rollout engine (student)
#   GPU 3   - SGLang teacher server
#
# usage: bash examples/on_policy_distillation/run-qwen3-8B-opd.sh

set -euxo pipefail

ROOT_DIR="${ROOT_DIR:-/home/rohin}"
NUM_GPUS=${NUM_GPUS:-3}   # GPUs visible to Ray (0, 1, 2); teacher uses GPU 3 separately
if [ -z "${POOL_DIR:-}" ]; then
    if [ -d "${ROOT_DIR}/orcd/pool" ]; then
        POOL_DIR="${ROOT_DIR}/orcd/pool"
    else
        USER_POOL_DIR="$(ls -d /orcd/pool/*/"$(whoami)" 2>/dev/null | head -n 1 || true)"
        POOL_DIR="${USER_POOL_DIR:-/orcd/pool}"
    fi
fi
if [ ! -d "${POOL_DIR}" ]; then
    echo "Pool directory not found: ${POOL_DIR}"
    echo "Set POOL_DIR explicitly, e.g. export POOL_DIR=/orcd/pool/<bucket>/$(whoami)"
    exit 1
fi
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"
if [ ! -d "${MEGATRON_PATH}" ] && [ -d "${ROOT_DIR}/Megatron-LM" ]; then
    MEGATRON_PATH="${ROOT_DIR}/Megatron-LM"
fi
if [ ! -d "${MEGATRON_PATH}" ]; then
    echo "Megatron-LM path not found: ${MEGATRON_PATH}"
    exit 1
fi
if [ ! -d "${POOL_DIR}/Qwen3-8B_torch_dist" ]; then
    echo "Missing student Megatron checkpoint: ${POOL_DIR}/Qwen3-8B_torch_dist"
    echo "Run the prep/conversion step before starting training."
    exit 1
fi

# ---- Preemption-friendly cleanup ----
TEACHER_PID=""
cleanup() {
    echo "Received signal, cleaning up..."
    [ -n "$TEACHER_PID" ] && kill -TERM "$TEACHER_PID" 2>/dev/null
    pkill -9 -f "python3 -m sglang.launch_server" 2>/dev/null || true
    pkill -9 sglang 2>/dev/null || true
    sleep 1
    ray stop --force 2>/dev/null || true
    pkill -9 ray 2>/dev/null || true
}
trap 'cleanup; exit 0' TERM INT USR1

# Download models if not already present
if [ ! -d "${POOL_DIR}/Qwen3-32B" ]; then
    echo "Downloading Qwen3-32B (teacher)..."
    huggingface-cli download Qwen/Qwen3-32B --local-dir "${POOL_DIR}/Qwen3-32B"
fi

if [ ! -d "${POOL_DIR}/Qwen3-8B" ]; then
    echo "Downloading Qwen3-8B (student)..."
    huggingface-cli download Qwen/Qwen3-8B --local-dir "${POOL_DIR}/Qwen3-8B"
fi

# Load environment variables (e.g., WANDB_API_KEY)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [ -f "${SCRIPT_DIR}/../../.env" ]; then
    set -a
    source "${SCRIPT_DIR}/../../.env"
    set +a
fi
WANDB_KEY="${WANDB_API_KEY:-${WANDB_KEY:-}}"
if [ -z "${WANDB_KEY}" ]; then
    echo "WANDB is enabled but no API key found in environment (.env)."
    echo "Set WANDB_API_KEY (or WANDB_KEY) before running."
    exit 1
fi

# Start the teacher model server
TEACHER_IP="127.0.0.1" # Use localhost here, you can change it to your IP
TEACHER_PORT=13141
LOG_FILE="/tmp/sglang_$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 6).log"

## Launch the teacher model server in the background (outside Ray's GPUs)
CUDA_VISIBLE_DEVICES=3 python3 -m sglang.launch_server \
    --model-path ${POOL_DIR}/Qwen3-32B \
    --host 0.0.0.0 \
    --port $TEACHER_PORT \
    --tp 1 \
    --chunked-prefill-size 4096 \
    --mem-fraction-static 0.6 \
    > "$LOG_FILE" 2>&1 &
TEACHER_PID=$!

echo "Starting teacher model server..."

## Wait for the teacher model server to be ready
MAX_TEACHER_WAIT_SEC=${MAX_TEACHER_WAIT_SEC:-300}
teacher_waited_sec=0
until curl -sf http://$TEACHER_IP:$TEACHER_PORT/health_generate > /dev/null; do
    echo "Waiting for the teacher model server to start..."
    tail -n 10 "$LOG_FILE"
    teacher_waited_sec=$((teacher_waited_sec + 5))
    if [ "${teacher_waited_sec}" -ge "${MAX_TEACHER_WAIT_SEC}" ]; then
        echo "Teacher server did not become healthy within ${MAX_TEACHER_WAIT_SEC}s."
        exit 1
    fi
    sleep 5
done

curl http://$TEACHER_IP:$TEACHER_PORT/get_model_info
echo "Teacher model server is up and running at $TEACHER_IP:$TEACHER_PORT."
sleep 10


export PYTHONUNBUFFERED=16

TOPOLOGY_OUTPUT="$(nvidia-smi topo -m 2>/dev/null || true)"
NVLINK_COUNT=$(printf "%s\n" "${TOPOLOGY_OUTPUT}" | awk '{for (i = 1; i <= NF; i++) if ($i ~ /^NV[0-9]+$/) c++} END {print c + 0}')
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "${REPO_DIR}/scripts/models/qwen3-8B.sh"

DATA_DIR="${POOL_DIR}/dapo-math-17k"
TRAIN_DATA="${DATA_DIR}/dapo-math-17k-train.jsonl"
EVAL_DATA="${DATA_DIR}/dapo-math-17k-eval.jsonl"
FALLBACK_DATA="${DATA_DIR}/dapo-math-17k.jsonl"

if [ ! -f "${TRAIN_DATA}" ] && [ -f "${FALLBACK_DATA}" ]; then
    echo "Training split not found; falling back to ${FALLBACK_DATA}"
    TRAIN_DATA="${FALLBACK_DATA}"
fi
if [ ! -f "${EVAL_DATA}" ] && [ -f "${FALLBACK_DATA}" ]; then
    echo "Eval split not found; falling back to ${FALLBACK_DATA}"
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
   --hf-checkpoint ${POOL_DIR}/Qwen3-8B
   --ref-load ${POOL_DIR}/Qwen3-8B_torch_dist
   --load ${POOL_DIR}/Qwen3-8B_slime/
   --save ${POOL_DIR}/Qwen3-8B_slime/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data ${TRAIN_DATA}
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
   --rm-url http://$TEACHER_IP:$TEACHER_PORT/generate
)

# Eval is disabled by default for OPD custom RM because teacher responses are dict-shaped
# and the default eval logger expects scalar rewards.
# Set ENABLE_EVAL=1 to turn eval back on once eval reward extraction is configured.
ENABLE_EVAL="${ENABLE_EVAL:-0}"
EVAL_ARGS=()
if [ "${ENABLE_EVAL}" = "1" ]; then
   EVAL_ARGS=(
      --eval-interval 20
      --eval-prompt-data eval ${EVAL_DATA}
      --n-samples-per-eval-prompt 4
      --eval-max-response-len 16384
      --eval-temperature 1
   )
fi

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
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
   --wandb-project slime-dev
   --wandb-group qwen3-8B-test
   --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.4
)


MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)




# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# Restrict Ray to GPUs 0-2 only; GPU 3 is reserved for the teacher.
export CUDA_VISIBLE_DEVICES=0,1,2
ray stop --force 2>/dev/null || true
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265


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
   --actor-num-gpus-per-node 2 \
   --rollout-num-gpus 1 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]}
RAY_EXIT_CODE=$?
set -e



####clear after training
cleanup
exit $RAY_EXIT_CODE
