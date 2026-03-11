#!/bin/bash
# =============================================================================
# Embedding Surgery SFT: GPT-OSS-20b with Qwen3.5 tokenizer swap
#
# Experiment: Take GPT-OSS-20b, swap tokenizer to Qwen3.5 (vocab 201088→248320),
# dequantize MXFP4→bf16, randomize embed_tokens + lm_head, then SFT on
# Qwen3.5-35B-A3B distilled rollouts. Only train embedding + output layers.
#
# Usage:
#   bash scripts/sft-gpt-oss-20b-embedding-swap.sh             # train (prep must be done)
#   DO_PREP=1 bash scripts/sft-gpt-oss-20b-embedding-swap.sh   # prep + train
#
# Config:  configs/sft-gpt-oss-20b-embedding-surgery.yaml
# Secrets: .env  (WANDB_KEY, HF_TOKEN)
# =============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ======================== CONFIG + SECRETS ========================
source "${SCRIPT_DIR}/lib/config.sh"

TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_DIR}/configs/sft-gpt-oss-20b-embedding-surgery-stage1.yaml}"
load_config "${TRAIN_CONFIG}"
load_env "${REPO_DIR}"

# ======================== REQUIRED CONFIG (no silent defaults) ========================
# Infrastructure — these have safe defaults tied to repo layout
POOL_DIR="${POOL_DIR:-/root/slime/models}"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"
RUN_LOG="${RUN_LOG:-${POOL_DIR}/run-sft-gpt-oss.log}"
HF_MODEL="${HF_MODEL:-openai/gpt-oss-20b}"
MODEL_NAME="${HF_MODEL##*/}"

# Derived paths
SWAPPED_DIR="${POOL_DIR}/${MODEL_NAME}-qwen3.5-tokenizer"
MEGATRON_REF_DIR="${POOL_DIR}/${MODEL_NAME}-qwen3.5-tokenizer_torch_dist"
SLIME_DIR="${POOL_DIR}/${MODEL_NAME}-qwen3.5-tokenizer_slime"

# All training params MUST come from the config yaml — fail loudly if missing.
require_var() { if [ -z "${!1:-}" ]; then echo "ERROR: $1 not set. Add it to the config yaml." >&2; exit 1; fi; }

REQUIRED_VARS=(
    DATASET_PATH TEST_DATA_PATH
    INIT_STD INIT_SEED TRAIN_PARAMS
    NUM_EPOCH GLOBAL_BATCH_SIZE ROLLOUT_BATCH_SIZE SAVE_INTERVAL MICRO_BATCH_SIZE
    INPUT_KEY LOSS_TYPE LOSS_MASK_TYPE APPLY_CHAT_TEMPLATE TOOL_KEY ROLLOUT_SEED
    SEQ_LENGTH ROLLOUT_MAX_CONTEXT_LEN
    LR MIN_LR LR_DECAY_STYLE LR_WARMUP_FRACTION WEIGHT_DECAY
    ADAM_BETA1 ADAM_BETA2 CLIP_GRAD OPTIMIZER
    TENSOR_MODEL_PARALLEL_SIZE PIPELINE_MODEL_PARALLEL_SIZE CONTEXT_PARALLEL_SIZE
    EXPERT_MODEL_PARALLEL_SIZE EXPERT_TENSOR_PARALLEL_SIZE
    ACTOR_NUM_NODES ACTOR_NUM_GPUS_PER_NODE
    RECOMPUTE_GRANULARITY RECOMPUTE_METHOD RECOMPUTE_NUM_LAYERS
    ATTENTION_DROPOUT HIDDEN_DROPOUT TRANSFORMER_IMPL ATTENTION_BACKEND
)
for var in "${REQUIRED_VARS[@]}"; do require_var "$var"; done

CHECKPOINT_SHIP_ENABLED="${CHECKPOINT_SHIP_ENABLED:-1}"
CHECKPOINT_SHIP_POLL_SEC="${CHECKPOINT_SHIP_POLL_SEC:-30}"
CHECKPOINT_HF_REPO_ID="${CHECKPOINT_HF_REPO_ID:-aimosprite/gpt-oss-20b-embedding-surgery}"
CHECKPOINT_HF_PRIVATE="${CHECKPOINT_HF_PRIVATE:-0}"
CHECKPOINT_HF_CREATE_REPO="${CHECKPOINT_HF_CREATE_REPO:-1}"

WANDB_PROJECT="${WANDB_PROJECT:-slime-dev}"
WANDB_GROUP="${WANDB_GROUP:-gpt-oss-20b-embedding-surgery}"
WANDB_KEY="${WANDB_KEY:-${WANDB_API_KEY:-${WANDB_TOKEN:-}}}"
USE_WANDB="${USE_WANDB:-1}"
if [ -n "${WANDB_KEY}" ] && [ "${#WANDB_KEY}" -lt 40 ]; then
    echo "WARNING: WANDB_KEY is only ${#WANDB_KEY} chars (need 40+). WandB logging disabled."
    USE_WANDB=0
fi

# ======================== PERSISTENT LOG ========================
mkdir -p "$(dirname "${RUN_LOG}")"
exec > >(tee -a "${RUN_LOG}") 2>&1
print_config

# ======================== PREP (idempotent) ========================
DO_PREP="${DO_PREP:-0}"
if [ "${DO_PREP}" = "1" ]; then
    bash "${SCRIPT_DIR}/prep-gpt-oss-embedding-surgery.sh" || exit 1
else
    echo "--- Verifying artifacts (DO_PREP=0) ---"
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
        echo "Run with DO_PREP=1 (or set do_prep: 1 in config) to create missing artifacts."
        exit 1
    fi
    echo "All artifacts present."
fi

# ======================== KILL LEFTOVERS ========================
pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 3

set -ex
export PYTHONUNBUFFERED=1

# ======================== NVLINK DETECTION ========================
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: ${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"

# ======================== HF CHECKPOINT SHIPPING ========================
source "${SCRIPT_DIR}/lib/hf_shipper.sh"
if [ "${CHECKPOINT_SHIP_ENABLED}" = "1" ]; then
    hf_login || exit 1
    preflight_hf || exit 1
    start_checkpoint_shipper
fi

# ======================== MODEL ARGS ========================
source "${SCRIPT_DIR}/models/gpt-oss-20b.sh"

# ======================== TRAINING ARGS ========================

CKPT_ARGS=(
    --hf-checkpoint "${SWAPPED_DIR}"
    --ref-load      "${MEGATRON_REF_DIR}"
    --load          "${SLIME_DIR}/"
    --save          "${SLIME_DIR}/"
    --save-interval "${SAVE_INTERVAL}"
)

SFT_ARGS=(
    --rollout-function-path slime.rollout.sft_rollout.generate_rollout
    --prompt-data           "${DATASET_PATH}"
    --input-key             "${INPUT_KEY}"
    --loss-mask-type        "${LOSS_MASK_TYPE}"
    --tool-key              "${TOOL_KEY}"
    --rollout-shuffle
    --rollout-seed          "${ROLLOUT_SEED}"
    --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
    --num-epoch             "${NUM_EPOCH}"
    --rollout-batch-size    "${ROLLOUT_BATCH_SIZE}"
    --global-batch-size     "${GLOBAL_BATCH_SIZE}"

    --loss-type                            "${LOSS_TYPE}"
    --calculate-per-token-loss
    --disable-compute-advantages-and-returns
    --debug-train-only
)
[ "${APPLY_CHAT_TEMPLATE}" = "1" ] && SFT_ARGS+=(--apply-chat-template)

FREEZE_ARGS=(
    --only-train-params-name-list ${TRAIN_PARAMS}
)

PERF_ARGS=(
    --bf16
    --tensor-model-parallel-size   "${TENSOR_MODEL_PARALLEL_SIZE}"
    --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE}"
    --context-parallel-size        "${CONTEXT_PARALLEL_SIZE}"
    --expert-model-parallel-size   "${EXPERT_MODEL_PARALLEL_SIZE}"
    --expert-tensor-parallel-size  "${EXPERT_TENSOR_PARALLEL_SIZE}"

    --recompute-granularity  "${RECOMPUTE_GRANULARITY}"
    --recompute-method       "${RECOMPUTE_METHOD}"
    --recompute-num-layers   "${RECOMPUTE_NUM_LAYERS}"

    --micro-batch-size "${MICRO_BATCH_SIZE}"
)

OPTIMIZER_ARGS=(
    --optimizer           "${OPTIMIZER}"
    --lr                  "${LR}"
    --lr-decay-style      "${LR_DECAY_STYLE}"
    --min-lr              "${MIN_LR}"
    --lr-warmup-fraction  "${LR_WARMUP_FRACTION}"
    --weight-decay        "${WEIGHT_DECAY}"
    --clip-grad           "${CLIP_GRAD}"
    --adam-beta1          "${ADAM_BETA1}"
    --adam-beta2          "${ADAM_BETA2}"
)

WANDB_ARGS=()
if [ "${USE_WANDB}" = "1" ]; then
    WANDB_ARGS=(
        --use-wandb
        --wandb-project "${WANDB_PROJECT}"
        --wandb-group   "${WANDB_GROUP}"
        --wandb-key     "${WANDB_KEY}"
    )
else
    echo "=== WandB DISABLED (invalid or missing key) ==="
fi

EVAL_HOOK_ARGS=()
if [ -n "${TEST_DATA_PATH:-}" ] && [ -f "${TEST_DATA_PATH}" ]; then
    EVAL_HOOK_ARGS=(
        --custom-megatron-before-train-step-hook-path slime.hooks.eval_hook.eval_before_step
    )
    echo "=== Test-loss eval hook ENABLED ==="
else
    echo "=== Test-loss eval hook DISABLED ==="
fi

MISC_ARGS=(
    --attention-dropout             "${ATTENTION_DROPOUT}"
    --hidden-dropout                "${HIDDEN_DROPOUT}"
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend             "${ATTENTION_BACKEND}"
    --no-gradient-accumulation-fusion
    --seq-length                    "${SEQ_LENGTH}"
)
# --no-persist-layer-norm required when transformer_impl=local (mistake #21)
if [ "${TRANSFORMER_IMPL}" = "local" ]; then
    MISC_ARGS+=(--no-persist-layer-norm)
fi

# ======================== LAUNCH ========================

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export no_proxy="127.0.0.1,${MASTER_ADDR}"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\":                \"${MEGATRON_PATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\":          \"${HAS_NVLINK}\",
    \"NCCL_CUMEM_ENABLE\":         \"0\",
    \"PYTORCH_CUDA_ALLOC_CONF\":   \"expandable_segments:True\",
    \"TEST_DATA_PATH\":            \"${TEST_DATA_PATH:-}\",
    \"EVAL_INTERVAL\":             \"${EVAL_INTERVAL:-50}\",
    \"EVAL_BATCH_SIZE\":           \"${EVAL_BATCH_SIZE:-32}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 train_async.py \
    --actor-num-nodes          "${ACTOR_NUM_NODES}" \
    --actor-num-gpus-per-node  "${ACTOR_NUM_GPUS_PER_NODE}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${SFT_ARGS[@]}" \
    "${FREEZE_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "${EVAL_HOOK_ARGS[@]}"
