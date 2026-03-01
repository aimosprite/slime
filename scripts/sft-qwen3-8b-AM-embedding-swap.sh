#!/bin/bash
# =============================================================================
# Embedding Surgery SFT: Qwen3-8B with randomized embeddings
#
# Experiment: Take Qwen3-8B, randomize embed_tokens + lm_head weights,
# then SFT on AM-Qwen3-Distilled (1.89M reasoning traces from Qwen3-235B-A22B).
# Only train embedding + output layers; transformer is frozen.
#
# Usage:
#   DO_PREP=1 bash scripts/sft-qwen3-8b-AM-embedding-swap.sh  # first time
#   bash scripts/sft-qwen3-8b-AM-embedding-swap.sh             # resume / retrain
#
# Required env vars:
#   WANDB_KEY    — Weights & Biases API key
#   HF_TOKEN     — HuggingFace token (for model download + checkpoint shipping)
# =============================================================================

# ======================== CONFIGURATION ========================

POOL_DIR="${POOL_DIR:-/root}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- Model ---
HF_MODEL="Qwen/Qwen3-8B"
MODEL_DIR="${POOL_DIR}/Qwen3-8B"
RANDOM_EMB_DIR="${POOL_DIR}/Qwen3-8B-random-emb"
MEGATRON_REF_DIR="${POOL_DIR}/Qwen3-8B-random-emb_torch_dist"
SLIME_DIR="${POOL_DIR}/Qwen3-8B-random-emb_slime"

# --- Dataset ---
HF_DATASET="a-m-team/AM-Qwen3-Distilled"
DATASET_PATH="${POOL_DIR}/am-qwen3-distilled.parquet"

# --- Embedding init ---
INIT_STD=0.02
INIT_SEED=42

# --- Training ---
NUM_EPOCH=1
GLOBAL_BATCH_SIZE=128
ROLLOUT_BATCH_SIZE=128
SAVE_INTERVAL=500
MAX_TOKENS_PER_GPU=8192

# --- Optimizer (higher LR — embeddings start from random) ---
LR=5e-4
MIN_LR=1e-5
LR_WARMUP_FRACTION=0.05
WEIGHT_DECAY=0.01

# --- Checkpoint shipping to HF Hub ---
CHECKPOINT_SHIP_ENABLED="${CHECKPOINT_SHIP_ENABLED:-1}"
CHECKPOINT_SHIP_POLL_SEC=30
CHECKPOINT_HF_REPO_ID="${CHECKPOINT_HF_REPO_ID:-qwen3-32b-to-8b-embedding-surgery}"

# ======================== PREP ========================

DO_PREP="${DO_PREP:-0}"

if [ "${DO_PREP}" = "1" ]; then
    echo "=========================================="
    echo "  PREP: Download, randomize, convert"
    echo "=========================================="

    # 1. Download model
    if [ ! -d "${MODEL_DIR}" ]; then
        echo "--- Downloading ${HF_MODEL} ---"
        huggingface-cli download ${HF_MODEL} --local-dir ${MODEL_DIR}
    else
        echo "--- Model already exists at ${MODEL_DIR}, skipping download ---"
    fi

    # 2. Randomize embeddings
    if [ ! -d "${RANDOM_EMB_DIR}" ]; then
        echo "--- Randomizing embeddings (std=${INIT_STD}, seed=${INIT_SEED}) ---"
        python3 ${REPO_DIR}/tools/randomize_embeddings.py \
            --input-dir ${MODEL_DIR} \
            --output-dir ${RANDOM_EMB_DIR} \
            --init-std ${INIT_STD} \
            --seed ${INIT_SEED}
    else
        echo "--- Random-emb model already exists at ${RANDOM_EMB_DIR}, skipping ---"
    fi

    # 3. Convert HF -> Megatron torch_dist format
    if [ ! -d "${MEGATRON_REF_DIR}" ]; then
        echo "--- Converting HF -> Megatron ---"
        source "${SCRIPT_DIR}/models/qwen3-8B.sh"

        torchrun --nproc_per_node=1 ${REPO_DIR}/tools/convert_hf_to_torch_dist.py \
            "${MODEL_ARGS[@]}" \
            --hf-checkpoint ${RANDOM_EMB_DIR} \
            --save ${MEGATRON_REF_DIR} \
            --tensor-model-parallel-size 1 \
            --pipeline-model-parallel-size 1 \
            --context-parallel-size 1 \
            --expert-model-parallel-size 1 \
            --expert-tensor-parallel-size 1 \
            --untie-embeddings-and-output-weights
    else
        echo "--- Megatron checkpoint already exists at ${MEGATRON_REF_DIR}, skipping ---"
    fi

    # 4. Download & convert dataset
    if [ ! -f "${DATASET_PATH}" ]; then
        echo "--- Downloading & converting dataset ---"
        python3 ${REPO_DIR}/tools/prep_am_dataset.py \
            --dataset ${HF_DATASET} \
            --output ${DATASET_PATH}
    else
        echo "--- Dataset already exists at ${DATASET_PATH}, skipping ---"
    fi

    echo "=========================================="
    echo "  PREP DONE"
    echo "=========================================="
fi

# ======================== VERIFY ARTIFACTS ========================

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
    echo "Run with DO_PREP=1 first."
    exit 1
fi
echo "All artifacts present."

# ======================== ENV CHECKS ========================

if [ -z "${WANDB_KEY:-}" ]; then
    echo "ERROR: WANDB_KEY not set."
    exit 1
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

# ======================== CHECKPOINT SHIPPER ========================

CHECKPOINT_SHIPPER_PID=""

start_checkpoint_shipper() {
    local tracker="${SLIME_DIR}/latest_checkpointed_iteration.txt"
    echo "Starting checkpoint shipper (poll every ${CHECKPOINT_SHIP_POLL_SEC}s) -> ${CHECKPOINT_HF_REPO_ID}"

    # Create repo if needed
    huggingface-cli repo create "${CHECKPOINT_HF_REPO_ID}" --type model 2>/dev/null || true

    (
        last_synced_step=""
        while true; do
            if [ -f "${tracker}" ]; then
                step="$(tr -d '[:space:]' < "${tracker}" || true)"
                if [ -n "${step}" ] && [ "${step}" != "${last_synced_step}" ]; then
                    iter_dir="${SLIME_DIR}/iter_$(printf '%07d' "${step}")"
                    if [ -d "${iter_dir}" ]; then
                        echo "[shipper] Uploading step ${step}..."
                        huggingface-cli upload "${CHECKPOINT_HF_REPO_ID}" \
                            "${iter_dir}" "iter_$(printf '%07d' "${step}")" \
                            --repo-type model 2>&1 || echo "[shipper] Upload failed for step ${step}"
                        huggingface-cli upload "${CHECKPOINT_HF_REPO_ID}" \
                            "${tracker}" "latest_checkpointed_iteration.txt" \
                            --repo-type model 2>&1 || true
                        last_synced_step="${step}"
                        echo "[shipper] Step ${step} uploaded."
                    fi
                fi
            fi
            sleep "${CHECKPOINT_SHIP_POLL_SEC}"
        done
    ) 2>&1 | tee -a /tmp/slime_checkpoint_shipper.log &
    CHECKPOINT_SHIPPER_PID=$!
    echo "Checkpoint shipper PID=${CHECKPOINT_SHIPPER_PID}"
}

cleanup() {
    echo "Cleaning up..."
    [ -n "${CHECKPOINT_SHIPPER_PID}" ] && kill -TERM "${CHECKPOINT_SHIPPER_PID}" 2>/dev/null || true
    ray stop --force 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [ "${CHECKPOINT_SHIP_ENABLED}" = "1" ]; then
    start_checkpoint_shipper
fi

# ======================== MODEL ARGS ========================

source "${SCRIPT_DIR}/models/qwen3-8B.sh"

# ======================== TRAINING ARGS ========================

CKPT_ARGS=(
    --hf-checkpoint ${RANDOM_EMB_DIR}
    --ref-load ${MEGATRON_REF_DIR}
    --load ${SLIME_DIR}/
    --save ${SLIME_DIR}/
    --save-interval ${SAVE_INTERVAL}
)

SFT_ARGS=(
    --rollout-function-path slime.rollout.sft_rollout.generate_rollout
    --prompt-data ${DATASET_PATH}
    --input-key messages
    --rollout-shuffle
    --num-epoch ${NUM_EPOCH}
    --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
    --global-batch-size ${GLOBAL_BATCH_SIZE}

    --loss-type sft_loss
    --calculate-per-token-loss
    --disable-compute-advantages-and-returns
    --debug-train-only
)

FREEZE_ARGS=(
    # Only train embedding + output projection; freeze all transformer layers
    --only-train-params-name-list embedding output_layer
)

PERF_ARGS=(
    --tensor-model-parallel-size 1
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1

    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1

    --use-dynamic-batch-size
    --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr ${LR}
    --lr-decay-style cosine
    --min-lr ${MIN_LR}
    --lr-warmup-fraction ${LR_WARMUP_FRACTION}
    --weight-decay ${WEIGHT_DECAY}
    --adam-beta1 0.9
    --adam-beta2 0.95
)

WANDB_ARGS=(
    --use-wandb
    --wandb-project slime-dev
    --wandb-group qwen3-8b-embedding-surgery
    --wandb-key ${WANDB_KEY}
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

# ======================== LAUNCH ========================

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats \
    --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${REPO_DIR}/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 train_async.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 8 \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${SFT_ARGS[@]}" \
    "${FREEZE_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${MISC_ARGS[@]}"
