#!/bin/bash
# Embedding surgery SFT: train Qwen3-8B with randomized embeddings + output layer
# on math-filtered AM-Qwen3-Distilled.

# ── Kill stale processes ─────────────────────────────────────────────
pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 3
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true

set -ex

export PYTHONUNBUFFERED=1

# ── Detect NVLink ────────────────────────────────────────────────────
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# ── Resolve paths ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Load optional config.yaml overrides ──────────────────────────────
TRAIN_CONFIG="${SCRIPT_DIR}/config.yaml"
if [ -f "${TRAIN_CONFIG}" ]; then
    _config_exports=$(python3 - "${TRAIN_CONFIG}" <<'PYEOF'
import sys, re, os
for line in open(sys.argv[1]):
    line = line.split('#')[0].strip()
    m = re.match(r'^([a-z_]+):\s*(\S.*)', line)
    if m:
        k, v = m.group(1).upper(), m.group(2).strip()
        if k not in os.environ:
            print(f"export {k}='{v}'")
PYEOF
)
    eval "${_config_exports}"
fi

# Load .env for secrets
if [ -f "${REPO_DIR}/.env" ]; then
    set -a; source "${REPO_DIR}/.env"; set +a
fi

# ── All params from config (every value explicit, nothing hidden) ────

# Run name (single source of truth for all output naming)
RUN_NAME="${RUN_NAME:-qwen3-8b-emb-surgery-v1}"

# Model
MODEL="${MODEL:-Qwen/Qwen3-8B}"
MODEL_SHORT="${MODEL##*/}"
MODEL_ARGS_SCRIPT="${MODEL_ARGS:-qwen3-8B.sh}"
MODELS_DIR="${REPO_DIR}/models"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"
MBRIDGE_PATH="${MBRIDGE_PATH:-/tmp/mbridge}"

# Derived paths (model prep — fixed per model, independent of run)
HF_MODEL_DIR="${MODELS_DIR}/${MODEL_SHORT}-random-emb"
TORCH_DIST_DIR="${MODELS_DIR}/${MODEL_SHORT}-random-emb_torch_dist"
DATA_DIR="${MODELS_DIR}/am-qwen3-distilled-math"
TRAIN_JSONL="${DATA_DIR}/train.jsonl"

# Derived paths (run outputs — all under RUN_NAME)
SLIME_SAVE_DIR="${MODELS_DIR}/${RUN_NAME}_slime"

# SFT
ROLLOUT_FUNCTION_PATH="${ROLLOUT_FUNCTION_PATH:-slime.rollout.sft_rollout.generate_rollout}"
LOSS_TYPE="${LOSS_TYPE:-sft_loss}"
CALCULATE_PER_TOKEN_LOSS="${CALCULATE_PER_TOKEN_LOSS:-1}"
DISABLE_COMPUTE_ADVANTAGES_AND_RETURNS="${DISABLE_COMPUTE_ADVANTAGES_AND_RETURNS:-1}"
DEBUG_TRAIN_ONLY="${DEBUG_TRAIN_ONLY:-0}"
ROLLOUT_SHUFFLE="${ROLLOUT_SHUFFLE:-1}"
INPUT_KEY="${INPUT_KEY:-messages}"
NUM_EPOCH="${NUM_EPOCH:-3}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"

# HF-format checkpoint saving (model weights only, for shipping)
SAVE_HF="${SAVE_HF:-${MODELS_DIR}/${RUN_NAME}-hf/iter_{rollout_id}}"

# Optimizer
OPTIMIZER="${OPTIMIZER:-adam}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
LR_DECAY_STYLE="${LR_DECAY_STYLE:-cosine}"
MIN_LR="${MIN_LR:-1e-6}"
LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.95}"

# Performance
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-1}"
EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-1}"
SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-1}"
RECOMPUTE_GRANULARITY="${RECOMPUTE_GRANULARITY:-full}"
RECOMPUTE_METHOD="${RECOMPUTE_METHOD:-uniform}"
RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-1}"
USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-1}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-9216}"

# Numerical stability
ATTENTION_DROPOUT="${ATTENTION_DROPOUT:-0.0}"
HIDDEN_DROPOUT="${HIDDEN_DROPOUT:-0.0}"
ACCUMULATE_ALLREDUCE_GRADS_IN_FP32="${ACCUMULATE_ALLREDUCE_GRADS_IN_FP32:-1}"
ATTENTION_SOFTMAX_IN_FP32="${ATTENTION_SOFTMAX_IN_FP32:-1}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash}"

# Checkpointing
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
CHECKPOINT_SHIP_ENABLED="${CHECKPOINT_SHIP_ENABLED:-1}"
CHECKPOINT_SHIP_EVERY="${CHECKPOINT_SHIP_EVERY:-${SAVE_INTERVAL}}"
CHECKPOINT_SHIP_POLL_SEC="${CHECKPOINT_SHIP_POLL_SEC:-15}"
CHECKPOINT_HF_REPO_ID="${CHECKPOINT_HF_REPO_ID:-aimosprite/${RUN_NAME}}"
CHECKPOINT_HF_PRIVATE="${CHECKPOINT_HF_PRIVATE:-1}"

# Wandb
WANDB_PROJECT="${WANDB_PROJECT:-slime-dev}"
WANDB_GROUP="${WANDB_GROUP:-${RUN_NAME}}"
WANDB_KEY="${WANDB_API_KEY:-${WANDB_KEY:-}}"

# ── Persistent logging ──────────────────────────────────────────────
RUN_LOG="${MODELS_DIR}/${RUN_NAME}.log"
SHIP_LOG="${MODELS_DIR}/${RUN_NAME}_ship.log"
mkdir -p "$(dirname "${RUN_LOG}")"
exec > >(tee -a "${RUN_LOG}") 2>&1
echo "== Running idempotent prep =="
bash "${SCRIPT_DIR}/prep.sh"

# ── Source model args ────────────────────────────────────────────────
source "${REPO_DIR}/scripts/models/${MODEL_ARGS_SCRIPT}"

# ── HF CLI helpers (for checkpoint shipping) ─────────────────────────
hf_auth_whoami() {
    if command -v hf >/dev/null 2>&1; then
        hf auth whoami
    else
        huggingface-cli whoami
    fi
}

hf_upload() {
    if command -v hf >/dev/null 2>&1; then
        hf upload "$@"
    else
        huggingface-cli upload "$@"
    fi
}

hf_repo_create() {
    local repo_id="$1" visibility="${2:-}"
    if command -v hf >/dev/null 2>&1; then
        hf repo create "${repo_id}" --repo-type model ${visibility}
    else
        huggingface-cli repo create "${repo_id}" --repo-type model ${visibility}
    fi
}

ship_checkpoint_step() {
    local step="$1"
    # Ship HF-format checkpoint (model weights only, ~16GB) instead of
    # full Megatron checkpoint (~48GB with optimizer states).
    local hf_dir="${SAVE_HF/\{rollout_id\}/${step}}"

    if [ ! -d "${hf_dir}" ]; then
        echo "HF checkpoint dir not found for step ${step}: ${hf_dir}"
        return 1
    fi

    if [ "${HF_REPO_CREATED}" != "1" ]; then
        local vis_flag=""
        [ "${CHECKPOINT_HF_PRIVATE}" = "1" ] && vis_flag="--private" || vis_flag="--public"
        hf_repo_create "${CHECKPOINT_HF_REPO_ID}" "${vis_flag}" 2>&1 || true
        HF_REPO_CREATED=1
    fi

    echo "Shipping HF checkpoint step ${step} from ${hf_dir}..."
    # Always overwrite a single 'latest/' directory on HF (~16GB total, not accumulating)
    hf_upload "${CHECKPOINT_HF_REPO_ID}" "${hf_dir}" "latest" --repo-type model

    # Also upload a step marker so we know which step 'latest' corresponds to
    local step_file
    step_file="$(mktemp /tmp/slime_step_XXXXXX.txt)"
    echo "${step}" > "${step_file}"
    hf_upload "${CHECKPOINT_HF_REPO_ID}" "${step_file}" "latest_step.txt" --repo-type model
    rm -f "${step_file}"
}

CHECKPOINT_SHIPPER_PID=""

start_checkpoint_shipper() {
    local tracker="${SLIME_SAVE_DIR}/latest_checkpointed_iteration.txt"
    echo "Starting checkpoint shipper (every ${CHECKPOINT_SHIP_EVERY} steps, poll ${CHECKPOINT_SHIP_POLL_SEC}s)."
    (
        last_synced=""
        while true; do
            if [ -f "${tracker}" ]; then
                step="$(tr -d '[:space:]' < "${tracker}" || true)"
                if [[ "${step}" =~ ^[0-9]+$ ]] && [ "${step}" -gt 0 ] && [ "${step}" != "${last_synced}" ]; then
                    if [ $((step % CHECKPOINT_SHIP_EVERY)) -eq 0 ]; then
                        if ship_checkpoint_step "${step}"; then
                            last_synced="${step}"
                        else
                            echo "Ship failed for step ${step}; will retry."
                        fi
                    fi
                fi
            fi
            sleep "${CHECKPOINT_SHIP_POLL_SEC}"
        done
    ) 2>&1 | tee -a "${SHIP_LOG}" &
    CHECKPOINT_SHIPPER_PID=$!
    echo "Shipper pid=${CHECKPOINT_SHIPPER_PID}, log=${SHIP_LOG}"
}

# ── Cleanup trap ─────────────────────────────────────────────────────
cleanup() {
    echo "Cleaning up..."
    [ -n "${CHECKPOINT_SHIPPER_PID}" ] && kill -TERM "${CHECKPOINT_SHIPPER_PID}" 2>/dev/null || true
    ray stop --force 2>/dev/null || true
    pkill -9 ray 2>/dev/null || true
}
trap 'cleanup; exit 0' TERM INT USR1

# ── HF preflight check ──────────────────────────────────────────────
preflight_hf_checkpoint_shipping() {
    echo "=== HF checkpoint shipping preflight ==="

    echo "1. Checking HF auth..."
    local whoami=""
    if ! whoami="$(hf_auth_whoami 2>&1)"; then
        echo "FAIL: HF auth failed. Output:"
        echo "${whoami}"
        return 1
    fi
    echo "   OK: logged in as ${whoami}"

    echo "2. Creating/verifying repo..."
    local vis_flag=""
    [ "${CHECKPOINT_HF_PRIVATE}" = "1" ] && vis_flag="--private" || vis_flag="--public"
    local create_out=""
    if create_out="$(hf_repo_create "${CHECKPOINT_HF_REPO_ID}" "${vis_flag}" 2>&1)"; then
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
    HF_REPO_CREATED=1

    echo "3. Test upload..."
    local test_file=""
    test_file="$(mktemp /tmp/slime_preflight_XXXXXX.txt)"
    echo "preflight-test $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${test_file}"
    local upload_out=""
    if upload_out="$(hf_upload "${CHECKPOINT_HF_REPO_ID}" "${test_file}" ".preflight_test.txt" --repo-type model 2>&1)"; then
        echo "   OK: test file uploaded"
    else
        echo "FAIL: hf upload error:"
        echo "${upload_out}"
        rm -f "${test_file}"
        return 1
    fi
    rm -f "${test_file}"

    echo "=== Preflight PASSED ==="
}

# ── Start checkpoint shipper ─────────────────────────────────────────
if [ "${CHECKPOINT_SHIP_ENABLED}" = "1" ]; then
    preflight_hf_checkpoint_shipping || exit 1
    mkdir -p "${SLIME_SAVE_DIR}"
    start_checkpoint_shipper
fi

# ── Build arg arrays (every value from config, nothing hidden) ───────
CKPT_ARGS=(
    --hf-checkpoint "${HF_MODEL_DIR}"
    --ref-load "${TORCH_DIST_DIR}"
    --load "${SLIME_SAVE_DIR}/"
    --save "${SLIME_SAVE_DIR}/"
    --save-interval "${SAVE_INTERVAL}"
    --save-hf "${SAVE_HF}"
)

SFT_ARGS=(
    --rollout-function-path "${ROLLOUT_FUNCTION_PATH}"
    --prompt-data "${TRAIN_JSONL}"
    --input-key "${INPUT_KEY}"
    --num-epoch "${NUM_EPOCH}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --loss-type "${LOSS_TYPE}"
)
[ "${ROLLOUT_SHUFFLE}" = "1" ] && SFT_ARGS+=(--rollout-shuffle)
[ "${CALCULATE_PER_TOKEN_LOSS}" = "1" ] && SFT_ARGS+=(--calculate-per-token-loss)
[ "${DISABLE_COMPUTE_ADVANTAGES_AND_RETURNS}" = "1" ] && SFT_ARGS+=(--disable-compute-advantages-and-returns)
[ "${DEBUG_TRAIN_ONLY}" = "1" ] && SFT_ARGS+=(--debug-train-only)

PERF_ARGS=(
    --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
    --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE}"
    --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
    --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE}"
    --expert-tensor-parallel-size "${EXPERT_TENSOR_PARALLEL_SIZE}"
    --recompute-granularity "${RECOMPUTE_GRANULARITY}"
    --recompute-method "${RECOMPUTE_METHOD}"
    --recompute-num-layers "${RECOMPUTE_NUM_LAYERS}"
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)
[ "${SEQUENCE_PARALLEL}" = "1" ] && PERF_ARGS+=(--sequence-parallel)
[ "${USE_DYNAMIC_BATCH_SIZE}" = "1" ] && PERF_ARGS+=(--use-dynamic-batch-size)

OPTIMIZER_ARGS=(
    --optimizer "${OPTIMIZER}"
    --lr "${LEARNING_RATE}"
    --lr-decay-style "${LR_DECAY_STYLE}"
    --min-lr "${MIN_LR}"
    --lr-warmup-fraction "${LR_WARMUP_FRACTION}"
    --weight-decay "${WEIGHT_DECAY}"
    --adam-beta1 "${ADAM_BETA1}"
    --adam-beta2 "${ADAM_BETA2}"
)

WANDB_ARGS=()
if [ -n "${WANDB_KEY}" ]; then
    WANDB_ARGS=(
        --use-wandb
        --wandb-project "${WANDB_PROJECT}"
        --wandb-group "${WANDB_GROUP}"
        --wandb-key "${WANDB_KEY}"
    )
else
    echo "WARNING: No WANDB_API_KEY found. Training without wandb."
fi

MISC_ARGS=(
    --attention-dropout "${ATTENTION_DROPOUT}"
    --hidden-dropout "${HIDDEN_DROPOUT}"
    --attention-backend "${ATTENTION_BACKEND}"
    --no-gradient-accumulation-fusion
)
[ "${ACCUMULATE_ALLREDUCE_GRADS_IN_FP32}" = "1" ] && MISC_ARGS+=(--accumulate-allreduce-grads-in-fp32)
[ "${ATTENTION_SOFTMAX_IN_FP32}" = "1" ] && MISC_ARGS+=(--attention-softmax-in-fp32)

EVAL_ARGS=()
# Eval disabled (debug_train_only=1). SFT eval loss not yet supported in slime.
# Eval offline on shipped HF checkpoints instead.

# ── Ray launch ───────────────────────────────────────────────────────
RAY_HEAD_IP=$(python3 -c "
import socket
ip = '127.0.0.1'
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 80))
    ip = s.getsockname()[0]
    s.close()
except Exception:
    pass
print(ip)
")
export MASTER_ADDR=${MASTER_ADDR:-"${RAY_HEAD_IP}"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
RAY_NUM_GPUS="${ACTOR_NUM_GPUS_PER_NODE}"
ray start --head --node-ip-address "${RAY_HEAD_IP}" --num-gpus "${RAY_NUM_GPUS}" \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Wait for Ray dashboard to be ready
echo "Waiting for Ray dashboard on ${RAY_HEAD_IP}:8265..."
for i in $(seq 1 60); do
    if curl -s "http://${RAY_HEAD_IP}:8265/api/version" >/dev/null 2>&1; then
        echo "Ray dashboard ready."
        break
    fi
    sleep 1
done

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PATH}:${REPO_DIR}:${MBRIDGE_PATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

set +e
ray job submit --address="http://${RAY_HEAD_IP}:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 "${REPO_DIR}/train_async.py" \
    --actor-num-nodes "${ACTOR_NUM_NODES}" \
    --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${SFT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "${EVAL_ARGS[@]}"
RAY_EXIT_CODE=$?
set -e

# Ship final checkpoint (HF format)
if [ "${CHECKPOINT_SHIP_ENABLED}" = "1" ] && [ -f "${SLIME_SAVE_DIR}/latest_checkpointed_iteration.txt" ]; then
    FINAL_STEP="$(tr -d '[:space:]' < "${SLIME_SAVE_DIR}/latest_checkpointed_iteration.txt" || true)"
    if [[ "${FINAL_STEP}" =~ ^[0-9]+$ ]] && [ "${FINAL_STEP}" -gt 0 ]; then
        echo "Shipping final HF checkpoint (step ${FINAL_STEP})..."
        ship_checkpoint_step "${FINAL_STEP}" || true
    fi
fi

cleanup
exit "${RAY_EXIT_CODE}"
