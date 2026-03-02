#!/bin/bash
# =============================================================================
# config.sh — Config + env loader (source this, then call the functions)
#
# Usage:
#   source scripts/lib/config.sh
#   load_config "configs/sft-qwen3-8b-embedding-surgery.yaml"
#   load_env
#   print_config
# =============================================================================

# load_config <yaml_path>
# Reads a flat YAML file and exports uppercase env vars.
# Already-set env vars take precedence (so CLI overrides work).
load_config() {
    local yaml_file="${1:?Usage: load_config <yaml_path>}"
    if [ ! -f "${yaml_file}" ]; then
        echo "WARNING: config not found at ${yaml_file}, using script defaults"
        return 0
    fi
    local _cfg_py
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
    eval "$(python3 "${_cfg_py}" "${yaml_file}")"
    rm -f "${_cfg_py}"
    echo "Loaded config: ${yaml_file}"
}

# load_env [repo_dir]
# Sources .env for secrets (WANDB_KEY, HF_TOKEN).
# Already-set vars take precedence via set -a / source / set +a.
load_env() {
    local repo_dir="${1:-.}"
    if [ -f "${repo_dir}/.env" ]; then
        set -a
        source "${repo_dir}/.env"
        set +a
    fi
}

# print_config
# Pretty-prints all the training config vars. Call after load_config + defaults.
print_config() {
    echo "=== Run started at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    echo ""
    echo "  config:               ${TRAIN_CONFIG:-<not set>}"
    echo "  pool_dir:             ${POOL_DIR:-<not set>}"
    echo "  megatron:             ${MEGATRON_PATH:-<not set>}"
    echo "  log:                  ${RUN_LOG:-<not set>}"
    echo ""
    echo "  model:                ${HF_MODEL:-<not set>}"
    echo "  dataset:              ${HF_DATASET:-<not set>}"
    echo "  dataset_path:         ${DATASET_PATH:-<not set>}"
    echo ""
    echo "  train_params:         ${TRAIN_PARAMS:-<not set>}"
    echo "  init_std:             ${INIT_STD:-<not set>}  seed=${INIT_SEED:-<not set>}"
    echo ""
    echo "  loss_mask_type:       ${LOSS_MASK_TYPE:-<not set>}"
    echo "  apply_chat_template:  ${APPLY_CHAT_TEMPLATE:-<not set>}"
    echo "  tool_key:             ${TOOL_KEY:-<not set>}"
    echo "  seq_length:           ${SEQ_LENGTH:-<not set>}"
    echo "  rollout_max_ctx:      ${ROLLOUT_MAX_CONTEXT_LEN:-<not set>}"
    echo ""
    echo "  num_epoch:            ${NUM_EPOCH:-<not set>}"
    echo "  global_batch_size:    ${GLOBAL_BATCH_SIZE:-<not set>}"
    echo "  rollout_batch_size:   ${ROLLOUT_BATCH_SIZE:-<not set>}"
    echo "  micro_batch_size:     ${MICRO_BATCH_SIZE:-<not set>}"
    echo "  save_interval:        ${SAVE_INTERVAL:-<not set>}"
    echo ""
    echo "  lr:                   ${LR:-<not set>}  min=${MIN_LR:-<not set>}  decay=${LR_DECAY_STYLE:-<not set>}  warmup=${LR_WARMUP_FRACTION:-<not set>}"
    echo "  weight_decay:         ${WEIGHT_DECAY:-<not set>}  clip_grad=${CLIP_GRAD:-<not set>}"
    echo "  adam:                 b1=${ADAM_BETA1:-<not set>}  b2=${ADAM_BETA2:-<not set>}"
    echo ""
    echo "  tp=${TENSOR_MODEL_PARALLEL_SIZE:-?}  pp=${PIPELINE_MODEL_PARALLEL_SIZE:-?}  cp=${CONTEXT_PARALLEL_SIZE:-?}  gpus=${ACTOR_NUM_GPUS_PER_NODE:-?}x${ACTOR_NUM_NODES:-?}"
    echo "  transformer_impl:     ${TRANSFORMER_IMPL:-<not set>}  attention_backend=${ATTENTION_BACKEND:-<not set>}"
    echo "  recompute:            ${RECOMPUTE_GRANULARITY:-<not set>}/${RECOMPUTE_METHOD:-<not set>}  layers=${RECOMPUTE_NUM_LAYERS:-<not set>}"
    echo ""
    echo "  wandb:                ${WANDB_PROJECT:-<not set>}/${WANDB_GROUP:-<not set>}"
    echo "  hf_repo:              ${CHECKPOINT_HF_REPO_ID:-<not set>}"
    echo ""
}
