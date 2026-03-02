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
# Config:  configs/sft-qwen3-8b-embedding-surgery.yaml  (all params, documented)
# Secrets: .env  (WANDB_KEY, HF_TOKEN — never commit these)
# =============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ======================== LOAD CONFIG ========================
# YAML keys become uppercase env vars. Already-set env vars take precedence.
# Uses a temp file to avoid bash heredoc-inside-$() parsing issues.
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_DIR}/configs/sft-qwen3-8b-embedding-surgery.yaml}"
if [ -f "${TRAIN_CONFIG}" ]; then
    _cfg_py="$(mktemp /tmp/slime_cfg_XXXXXX.py)"
    cat > "${_cfg_py}" << 'PYEOF'
import sys, re, os
for line in open(sys.argv[1]):
    line = line.split('#')[0].strip()
    m = re.match(r'^([a-z_]+):\s*(\S.*)', line)
    if m:
        k, v = m.group(1).upper(), m.group(2).strip()
        if k not in os.environ:
            print(f"export {k}='{v}'")
PYEOF
    eval "$(python3 "${_cfg_py}" "${TRAIN_CONFIG}")"
    rm -f "${_cfg_py}"
    echo "Loaded config: ${TRAIN_CONFIG}"
else
    echo "WARNING: config not found at ${TRAIN_CONFIG}, using script defaults"
fi

# Load .env for secrets (WANDB_KEY, HF_TOKEN) — these take precedence if already set
if [ -f "${REPO_DIR}/.env" ]; then
    set -a
    source "${REPO_DIR}/.env"
    set +a
fi

# ======================== PATHS ========================
POOL_DIR="${POOL_DIR:-/root/slime/models}"
MEGATRON_PATH="${MEGATRON_PATH:-${REPO_DIR}/Megatron-LM}"
RUN_LOG="${RUN_LOG:-${POOL_DIR}/run-sft.log}"

HF_MODEL="${HF_MODEL:-Qwen/Qwen3-8B}"
HF_DATASET="${HF_DATASET:-a-m-team/AM-Qwen3-Distilled}"
MODEL_NAME="${HF_MODEL##*/}"                       # e.g. "Qwen3-8B"

MODEL_DIR="${POOL_DIR}/${MODEL_NAME}"
RANDOM_EMB_DIR="${POOL_DIR}/${MODEL_NAME}-random-emb"
MEGATRON_REF_DIR="${POOL_DIR}/${MODEL_NAME}-random-emb_torch_dist"
SLIME_DIR="${POOL_DIR}/${MODEL_NAME}-random-emb_slime"
DATASET_PATH="${POOL_DIR}/am-qwen3-distilled.parquet"

# Embedding surgery
INIT_STD="${INIT_STD:-0.02}"
INIT_SEED="${INIT_SEED:-42}"
TRAIN_PARAMS="${TRAIN_PARAMS:-embedding output_layer}"
CONVERT_NPROC_PER_NODE="${CONVERT_NPROC_PER_NODE:-8}"

# Training
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

# Optimizer
LR="${LR:-5e-4}"
MIN_LR="${MIN_LR:-1e-5}"
LR_DECAY_STYLE="${LR_DECAY_STYLE:-cosine}"
LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.95}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"
OPTIMIZER="${OPTIMIZER:-adam}"

# Parallelism
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-1}"
EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-1}"
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"

# Recompute
RECOMPUTE_GRANULARITY="${RECOMPUTE_GRANULARITY:-full}"
RECOMPUTE_METHOD="${RECOMPUTE_METHOD:-uniform}"
RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-1}"

# Regularization
ATTENTION_DROPOUT="${ATTENTION_DROPOUT:-0.0}"
HIDDEN_DROPOUT="${HIDDEN_DROPOUT:-0.0}"

# Precision
TRANSFORMER_IMPL="${TRANSFORMER_IMPL:-local}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash}"

# Checkpoint shipping
CHECKPOINT_SHIP_ENABLED="${CHECKPOINT_SHIP_ENABLED:-1}"
CHECKPOINT_SHIP_POLL_SEC="${CHECKPOINT_SHIP_POLL_SEC:-30}"
CHECKPOINT_HF_REPO_ID="${CHECKPOINT_HF_REPO_ID:-lerchen3/qwen3-32b-to-8b-embedding-surgery}"
CHECKPOINT_HF_PRIVATE="${CHECKPOINT_HF_PRIVATE:-0}"
CHECKPOINT_HF_CREATE_REPO="${CHECKPOINT_HF_CREATE_REPO:-1}"

# WandB
WANDB_PROJECT="${WANDB_PROJECT:-slime-dev}"
WANDB_GROUP="${WANDB_GROUP:-qwen3-8b-embedding-surgery}"
WANDB_KEY="${WANDB_KEY:-${WANDB_API_KEY:-}}"

# ======================== PERSISTENT LOG ========================
mkdir -p "$(dirname "${RUN_LOG}")"
exec > >(tee -a "${RUN_LOG}") 2>&1
echo "=== Run started at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo ""
echo "  config:               ${TRAIN_CONFIG}"
echo "  pool_dir:             ${POOL_DIR}"
echo "  megatron:             ${MEGATRON_PATH}"
echo "  log:                  ${RUN_LOG}"
echo ""
echo "  model:                ${HF_MODEL}"
echo "  dataset:              ${HF_DATASET}"
echo "  dataset_path:         ${DATASET_PATH}"
echo ""
echo "  train_params:         ${TRAIN_PARAMS}"
echo "  init_std:             ${INIT_STD}  seed=${INIT_SEED}"
echo ""
echo "  loss_mask_type:       ${LOSS_MASK_TYPE}"
echo "  apply_chat_template:  ${APPLY_CHAT_TEMPLATE}"
echo "  tool_key:             ${TOOL_KEY}"
echo "  seq_length:           ${SEQ_LENGTH}"
echo "  rollout_max_ctx:      ${ROLLOUT_MAX_CONTEXT_LEN}"
echo ""
echo "  num_epoch:            ${NUM_EPOCH}"
echo "  global_batch_size:    ${GLOBAL_BATCH_SIZE}"
echo "  rollout_batch_size:   ${ROLLOUT_BATCH_SIZE}"
echo "  micro_batch_size:     ${MICRO_BATCH_SIZE}"
echo "  save_interval:        ${SAVE_INTERVAL}"
echo ""
echo "  lr:                   ${LR}  min=${MIN_LR}  decay=${LR_DECAY_STYLE}  warmup=${LR_WARMUP_FRACTION}"
echo "  weight_decay:         ${WEIGHT_DECAY}  clip_grad=${CLIP_GRAD}"
echo "  adam:                 b1=${ADAM_BETA1}  b2=${ADAM_BETA2}"
echo ""
echo "  tp=${TENSOR_MODEL_PARALLEL_SIZE}  pp=${PIPELINE_MODEL_PARALLEL_SIZE}  cp=${CONTEXT_PARALLEL_SIZE}  gpus=${ACTOR_NUM_GPUS_PER_NODE}x${ACTOR_NUM_NODES}"
echo "  transformer_impl:     ${TRANSFORMER_IMPL}  attention_backend=${ATTENTION_BACKEND}"
echo "  recompute:            ${RECOMPUTE_GRANULARITY}/${RECOMPUTE_METHOD}  layers=${RECOMPUTE_NUM_LAYERS}"
echo ""
echo "  wandb:                ${WANDB_PROJECT}/${WANDB_GROUP}"
echo "  hf_repo:              ${CHECKPOINT_HF_REPO_ID}"
echo ""

# ======================== HF LOGIN ========================
export HF_TOKEN="${HF_TOKEN:-}"
if [ -n "${HF_TOKEN}" ]; then
    huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential >/dev/null 2>&1 \
        || huggingface-cli login --token "${HF_TOKEN}" >/dev/null 2>&1 \
        || true
fi
if ! huggingface-cli whoami >/dev/null 2>&1; then
    echo "ERROR: HF auth failed. Set HF_TOKEN in .env or run 'huggingface-cli login'."
    exit 1
fi
echo "HF auth OK: $(huggingface-cli whoami 2>/dev/null | head -1)"

# ======================== PREP ========================
DO_PREP="${DO_PREP:-0}"

if [ "${DO_PREP}" = "1" ]; then
    echo "=========================================="
    echo "  PREP: Download, randomize, convert"
    echo "=========================================="

    # 1. Download base model
    if [ ! -d "${MODEL_DIR}" ]; then
        echo "--- Downloading ${HF_MODEL} ---"
        huggingface-cli download "${HF_MODEL}" --local-dir "${MODEL_DIR}"
    else
        echo "--- Model already exists at ${MODEL_DIR}, skipping ---"
    fi

    # 2. Randomize embed_tokens + lm_head
    if [ ! -d "${RANDOM_EMB_DIR}" ]; then
        echo "--- Randomizing embeddings (std=${INIT_STD}, seed=${INIT_SEED}) ---"
        python3 "${REPO_DIR}/tools/randomize_embeddings.py" \
            --input-dir  "${MODEL_DIR}" \
            --output-dir "${RANDOM_EMB_DIR}" \
            --init-std   "${INIT_STD}" \
            --seed       "${INIT_SEED}"
    else
        echo "--- Random-emb model already exists at ${RANDOM_EMB_DIR}, skipping ---"
    fi

    # 3. Convert HF → Megatron torch_dist format (parallelized over all GPUs)
    if [ ! -d "${MEGATRON_REF_DIR}" ]; then
        echo "--- Converting HF -> Megatron (nproc=${CONVERT_NPROC_PER_NODE}) ---"
        source "${SCRIPT_DIR}/models/qwen3-8B.sh"
        PYTHONPATH="${MEGATRON_PATH}" \
        torchrun --nproc_per_node="${CONVERT_NPROC_PER_NODE}" \
            "${REPO_DIR}/tools/convert_hf_to_torch_dist.py" \
            "${MODEL_ARGS[@]}" \
            --hf-checkpoint                "${RANDOM_EMB_DIR}" \
            --save                         "${MEGATRON_REF_DIR}" \
            --tensor-model-parallel-size   1 \
            --pipeline-model-parallel-size 1 \
            --context-parallel-size        1 \
            --expert-model-parallel-size   1 \
            --expert-tensor-parallel-size  1 \
            --untie-embeddings-and-output-weights \
            --no-gradient-accumulation-fusion
    else
        echo "--- Megatron checkpoint already exists at ${MEGATRON_REF_DIR}, skipping ---"
    fi

    # 4. Download & convert dataset
    if [ ! -f "${DATASET_PATH}" ]; then
        echo "--- Downloading & converting dataset ---"
        python3 "${REPO_DIR}/tools/prep_am_dataset.py" \
            --dataset "${HF_DATASET}" \
            --output  "${DATASET_PATH}"
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
    echo "Run with DO_PREP=1 to create missing artifacts."
    exit 1
fi
echo "All artifacts present."

# ======================== ENV CHECKS ========================
if [ -z "${WANDB_KEY}" ]; then
    echo "ERROR: WANDB_KEY (or WANDB_API_KEY) not set. Add it to .env."
    exit 1
fi
if [ ! -d "${MEGATRON_PATH}" ]; then
    echo "ERROR: Megatron-LM not found at ${MEGATRON_PATH}."
    echo "       Run: git clone https://github.com/NVIDIA/Megatron-LM.git ${MEGATRON_PATH}"
    exit 1
fi

# ======================== HF PREFLIGHT ========================
preflight_hf() {
    echo "=== HF checkpoint shipping preflight ==="

    echo "1. Auth check..."
    if ! huggingface-cli whoami >/dev/null 2>&1; then
        echo "FAIL: not authenticated to HuggingFace."
        return 1
    fi
    echo "   OK: $(huggingface-cli whoami 2>/dev/null | head -1)"

    if [ "${CHECKPOINT_HF_CREATE_REPO}" = "1" ]; then
        echo "2. Creating HF repo (if needed): ${CHECKPOINT_HF_REPO_ID}"
        local private_flag=""
        [ "${CHECKPOINT_HF_PRIVATE}" = "1" ] && private_flag="--private"
        local create_out=""
        if create_out="$(huggingface-cli repo create "${CHECKPOINT_HF_REPO_ID}" \
                --repo-type model --exist-ok ${private_flag} -y 2>&1)"; then
            echo "   Repo ready: ${CHECKPOINT_HF_REPO_ID}"
        else
            echo "FAIL: repo create error:"
            echo "${create_out}"
            return 1
        fi
    fi

    echo "3. Test upload..."
    local test_file=""
    test_file="$(mktemp /tmp/slime_preflight_XXXXXX.txt)"
    echo "preflight $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${test_file}"
    local upload_out=""
    if upload_out="$(huggingface-cli upload "${CHECKPOINT_HF_REPO_ID}" \
            "${test_file}" ".preflight_test.txt" \
            --repo-type model 2>&1)"; then
        echo "   Upload OK"
    else
        echo "FAIL: test upload failed:"
        echo "${upload_out}"
        rm -f "${test_file}"
        return 1
    fi
    rm -f "${test_file}"
    echo "=== Preflight PASSED: shipping to ${CHECKPOINT_HF_REPO_ID} is working ==="
}

if [ "${CHECKPOINT_SHIP_ENABLED}" = "1" ]; then
    preflight_hf || exit 1
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

    (
        last_synced_step=""
        while true; do
            if [ -f "${tracker}" ]; then
                step="$(tr -d '[:space:]' < "${tracker}" || true)"
                # Only ship positive steps that are newly completed
                if [[ "${step}" =~ ^[0-9]+$ ]] && [ "${step}" -gt 0 ] && \
                   [ "${step}" != "${last_synced_step}" ]; then
                    iter_dir="${SLIME_DIR}/iter_$(printf '%07d' "${step}")"
                    if [ -d "${iter_dir}" ]; then
                        echo "[shipper] Uploading step ${step} -> checkpoint/ ..."
                        if huggingface-cli upload "${CHECKPOINT_HF_REPO_ID}" \
                                "${iter_dir}" \
                                "checkpoint" \
                                --repo-type model 2>&1; then
                            # Also upload the tracker file so HF repo shows latest step
                            huggingface-cli upload "${CHECKPOINT_HF_REPO_ID}" \
                                "${tracker}" "latest_checkpointed_iteration.txt" \
                                --repo-type model >/dev/null 2>&1 || true
                            last_synced_step="${step}"
                            echo "[shipper] Step ${step} uploaded (overwrote checkpoint/)."
                            # Delete old checkpoint dirs — keep only the latest on disk
                            for old_dir in "${SLIME_DIR}/iter_"*/; do
                                [ "${old_dir%/}" != "${iter_dir}" ] && rm -rf "${old_dir}" \
                                    && echo "[shipper] Deleted old checkpoint: ${old_dir}"
                            done
                        else
                            echo "[shipper] Upload FAILED for step ${step} — will retry next poll."
                        fi
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
    # Only train embedding + output projection; freeze all transformer layers.
    # TRAIN_PARAMS="embedding output_layer" — set in config to change.
    --only-train-params-name-list ${TRAIN_PARAMS}
)

PERF_ARGS=(
    --bf16
    --tensor-model-parallel-size   "${TENSOR_MODEL_PARALLEL_SIZE}"
    --sequence-parallel
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

WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group   "${WANDB_GROUP}"
    --wandb-key     "${WANDB_KEY}"
)

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
