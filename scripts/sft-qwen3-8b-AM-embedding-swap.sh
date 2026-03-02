#!/bin/bash
# =============================================================================
# Embedding Surgery SFT: Qwen3-8B with randomized embeddings
#
# Experiment: Take Qwen3-8B, randomize embed_tokens + lm_head weights,
# then SFT on AM-Qwen3-Distilled (1.89M reasoning traces from Qwen3-235B-A22B).
# Only train embedding + output layers; transformer is frozen.
#
# Usage:
#   bash scripts/sft-qwen3-8b-AM-embedding-swap.sh             # train (prep must be done)
#   DO_PREP=1 bash scripts/sft-qwen3-8b-AM-embedding-swap.sh   # prep + train
#
# Config:  configs/sft-qwen3-8b-embedding-surgery.yaml  (all params, documented)
# Secrets: .env  (WANDB_KEY, HF_TOKEN — never commit these)
# =============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ======================== CONFIG + SECRETS ========================
source "${SCRIPT_DIR}/lib/config.sh"

TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_DIR}/configs/sft-qwen3-8b-embedding-surgery.yaml}"
load_config "${TRAIN_CONFIG}"
load_env "${REPO_DIR}"

# ======================== DEFAULTS ========================
POOL_DIR="${POOL_DIR:-/root/slime/models}"
MEGATRON_PATH="${MEGATRON_PATH:-${REPO_DIR}/Megatron-LM}"
RUN_LOG="${RUN_LOG:-${POOL_DIR}/run-sft.log}"

HF_MODEL="${HF_MODEL:-Qwen/Qwen3-8B}"
HF_DATASET="${HF_DATASET:-a-m-team/AM-Qwen3-Distilled}"
MODEL_NAME="${HF_MODEL##*/}"

MODEL_DIR="${POOL_DIR}/${MODEL_NAME}"
RANDOM_EMB_DIR="${POOL_DIR}/${MODEL_NAME}-random-emb"
MEGATRON_REF_DIR="${POOL_DIR}/${MODEL_NAME}-random-emb_torch_dist"
SLIME_DIR="${POOL_DIR}/${MODEL_NAME}-random-emb_slime"
DATASET_PATH="${DATASET_PATH:-${POOL_DIR}/am-qwen3-distilled.parquet}"

INIT_STD="${INIT_STD:-0.02}"
INIT_SEED="${INIT_SEED:-42}"
TRAIN_PARAMS="${TRAIN_PARAMS:-embedding output_layer}"

NUM_EPOCH="${NUM_EPOCH:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
INPUT_KEY="${INPUT_KEY:-messages}"
LOSS_TYPE="${LOSS_TYPE:-sft_loss}"
LOSS_MASK_TYPE="${LOSS_MASK_TYPE:-qwen3}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-0}"
TOOL_KEY="${TOOL_KEY:-tools}"
ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-8192}"

LR="${LR:-5e-4}"
MIN_LR="${MIN_LR:-1e-5}"
LR_DECAY_STYLE="${LR_DECAY_STYLE:-cosine}"
LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.95}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"
OPTIMIZER="${OPTIMIZER:-adam}"

TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-1}"
EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-1}"
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"

RECOMPUTE_GRANULARITY="${RECOMPUTE_GRANULARITY:-full}"
RECOMPUTE_METHOD="${RECOMPUTE_METHOD:-uniform}"
RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-1}"

ATTENTION_DROPOUT="${ATTENTION_DROPOUT:-0.0}"
HIDDEN_DROPOUT="${HIDDEN_DROPOUT:-0.0}"
TRANSFORMER_IMPL="${TRANSFORMER_IMPL:-local}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash}"

CHECKPOINT_SHIP_ENABLED="${CHECKPOINT_SHIP_ENABLED:-1}"
CHECKPOINT_SHIP_POLL_SEC="${CHECKPOINT_SHIP_POLL_SEC:-30}"
CHECKPOINT_HF_REPO_ID="${CHECKPOINT_HF_REPO_ID:-lerchen3/qwen3-32b-to-8b-embedding-surgery}"
CHECKPOINT_HF_PRIVATE="${CHECKPOINT_HF_PRIVATE:-0}"
CHECKPOINT_HF_CREATE_REPO="${CHECKPOINT_HF_CREATE_REPO:-1}"

WANDB_PROJECT="${WANDB_PROJECT:-slime-dev}"
WANDB_GROUP="${WANDB_GROUP:-qwen3-8b-embedding-surgery}"
WANDB_KEY="${WANDB_KEY:-${WANDB_API_KEY:-}}"
USE_WANDB="${USE_WANDB:-1}"
# Validate wandb key length (must be 40+ chars)
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
    bash "${SCRIPT_DIR}/prep-embedding-surgery.sh" || exit 1
else
    # Verify artifacts exist even when skipping prep
    echo "--- Verifying artifacts ---"
    MISSING=0
    for required in "${RANDOM_EMB_DIR}" "${MEGATRON_REF_DIR}"; do
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
        echo "Run with DO_PREP=1 to create missing artifacts."
        exit 1
    fi
    echo "All artifacts present."

    if [ -z "${WANDB_KEY}" ]; then
        echo "WARNING: WANDB_KEY (or WANDB_API_KEY) not set. WandB logging will be disabled."
        USE_WANDB=0
    fi
    if [ ! -d "${MEGATRON_PATH}" ]; then
        echo "ERROR: Megatron-LM not found at ${MEGATRON_PATH}."
        echo "       Run: git clone https://github.com/NVIDIA/Megatron-LM.git ${MEGATRON_PATH}"
        exit 1
    fi
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
source "${SCRIPT_DIR}/models/qwen3-8B.sh"

# ======================== TRAINING ARGS ========================

CKPT_ARGS=(
    --hf-checkpoint "${RANDOM_EMB_DIR}"
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

MISC_ARGS=(
    --attention-dropout             "${ATTENTION_DROPOUT}"
    --hidden-dropout                "${HIDDEN_DROPOUT}"
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend             "${ATTENTION_BACKEND}"
    --no-gradient-accumulation-fusion
    --seq-length                    "${SEQ_LENGTH}"
)

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
    \"PYTORCH_CUDA_ALLOC_CONF\":   \"expandable_segments:True\"
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
    "${MISC_ARGS[@]}"
