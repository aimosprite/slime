#!/bin/bash

# On-Policy Distillation with FSDP backend
# Qwen3-32B (teacher, SGLang server) -> Qwen3-8B (student, FSDP training)
#
# Requires 4 GPUs:
#   GPU 0,1 - FSDP training actors (model sharded across 2 GPUs)
#   GPU 2   - SGLang rollout engine (student)
#   GPU 3   - SGLang teacher server
#
# usage: bash examples/on_policy_distillation/run-qwen3-8B-opd-fsdp.sh

set -ex

ROOT_DIR="/home/rohin"
NUM_GPUS=${NUM_GPUS:-3}   # GPUs visible to Ray (0, 1, 2); teacher uses GPU 3 separately
POOL_DIR="${ROOT_DIR}/orcd/pool"

SAVE_DIR="${POOL_DIR}/Qwen3-8B_fsdp_opd"

# ---- Preemption-friendly cleanup ----
TEACHER_PID=""
cleanup() {
    echo "Received signal, cleaning up..."
    [ -n "$TEACHER_PID" ] && kill -TERM "$TEACHER_PID" 2>/dev/null
    pkill -9 sglang 2>/dev/null || true
    sleep 1
    ray stop --force 2>/dev/null || true
    pkill -9 ray 2>/dev/null || true
    pkill -9 python 2>/dev/null || true
    exit 0
}
trap cleanup TERM INT USR1

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
if [ -f "${SCRIPT_DIR}/../../.env" ]; then
    set -a
    source "${SCRIPT_DIR}/../../.env"
    set +a
fi

# Start the teacher model server on GPU 2 (outside Ray's control)
TEACHER_IP="127.0.0.1"
TEACHER_PORT=13141
LOG_FILE="/tmp/sglang_$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 6).log"

## Launch the teacher model server in the background
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
until curl -sf http://$TEACHER_IP:$TEACHER_PORT/health_generate > /dev/null; do
    echo "Waiting for the teacher model server to start..."
    tail -n 10 "$LOG_FILE"
    sleep 5
done

curl http://$TEACHER_IP:$TEACHER_PORT/get_model_info
echo "Teacher model server is up and running at $TEACHER_IP:$TEACHER_PORT."
sleep 10


export PYTHONUNBUFFERED=16

# No MODEL_ARGS needed — FSDP loads the HF model directly

CKPT_ARGS=(
   --hf-checkpoint ${POOL_DIR}/Qwen3-8B
   --load ${SAVE_DIR}
   --ref-load ${POOL_DIR}/Qwen3-8B
   --save ${SAVE_DIR}
   --save-interval 10
)

ROLLOUT_ARGS=(
   --prompt-data ${POOL_DIR}/dapo-math-17k/dapo-math-17k.jsonl
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

EVAL_ARGS=(
   # --eval-interval 20
   # --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
   # --n-samples-per-eval-prompt 16
   # --eval-max-response-len 16384
   # --eval-top-p 1
)

TRAIN_BACKEND_ARGS=(
   --train-backend fsdp
   --update-weight-buffer-size 536870912
   --gradient-checkpointing
   --attn-implementation flash_attention_2
   --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
)

PERF_ARGS=(
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
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
   #--use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3-8B-fsdp-opd
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.4
)

MISC_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node 2
   --rollout-num-gpus 1
)


# Restrict Ray to GPUs 0-2 only; GPU 3 is reserved for the teacher
export CUDA_VISIBLE_DEVICES=0,1,2

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265


ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="{
     \"env_vars\": {
        \"PYTHONPATH\": \"${MEGATRON_PATH:-/root/Megatron-LM}\",
        \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
     }
   }" \
   -- python3 train.py \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${TRAIN_BACKEND_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]} &
RAY_JOB_PID=$!
wait $RAY_JOB_PID
RAY_EXIT_CODE=$?

#### cleanup after training
cleanup
exit $RAY_EXIT_CODE
