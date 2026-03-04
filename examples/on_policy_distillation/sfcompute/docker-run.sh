#!/bin/bash
set -euo pipefail

if [ "${DEBUG:-0}" = "1" ]; then
    export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
    set -x
fi

IMAGE="${IMAGE:-slimerl/slime:latest}"
REPO_DIR="${REPO_DIR:-/root/slime}"
POOL_DIR="${POOL_DIR:-/root/pool}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/root/.cache/huggingface}"

usage() {
    cat <<'EOF'
Usage:
  bash examples/on_policy_distillation/sfcompute/docker-run.sh prep
  bash examples/on_policy_distillation/sfcompute/docker-run.sh train
  bash examples/on_policy_distillation/sfcompute/docker-run.sh worker      # start ray worker and block
  bash examples/on_policy_distillation/sfcompute/docker-run.sh preflight   # test HF checkpoint upload

Optional environment overrides:
  IMAGE    Docker image tag (default: slimerl/slime:latest)
  REPO_DIR Host path to slime repo (default: /root/slime)
  POOL_DIR Host path to pool dir (default: /root/pool)
  HF_CACHE_DIR Host HF cache/token path (default: /root/.cache/huggingface)
  RAY_HEAD_IP Reachable IP of ray head node (required for worker mode)
  GPUS_PER_NODE Number of GPUs to register on this worker (default: 8)
  RAY_PORT Ray GCS port (default: 6379)
EOF
}

if [ $# -lt 1 ]; then
    usage
    exit 1
fi

COMMAND="$1"
shift || true

if [ ! -d "${REPO_DIR}" ]; then
    echo "REPO_DIR not found: ${REPO_DIR}"
    exit 1
fi
mkdir -p "${POOL_DIR}"
mkdir -p "${HF_CACHE_DIR}"

if [ -f "${REPO_DIR}/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "${REPO_DIR}/.env"
    set +a
fi

HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}}"
WANDB_API_KEY="${WANDB_API_KEY:-${WANDB_KEY:-}}"

case "${COMMAND}" in
    prep)
        docker run --rm --gpus all --ipc=host \
            --ulimit memlock=-1 --ulimit stack=67108864 \
            -v "${REPO_DIR}:/root/slime" \
            -v "${POOL_DIR}:/root/pool" \
            -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
            -e HF_TOKEN="${HF_TOKEN}" \
            -e WANDB_API_KEY="${WANDB_API_KEY}" \
            -e CHECKPOINT_HF_REPO_ID="${CHECKPOINT_HF_REPO_ID:-}" \
            -w /root/slime \
            "${IMAGE}" \
            bash examples/on_policy_distillation/sfcompute/prep-opd.sh "$@"
        ;;
    train)
        # SLIME_HOST_IP tells slime's router/engines to use this IP instead of auto-detect.
        # Critical for Tailscale setups where the auto-detected LAN IP isn't reachable cross-node.
        SLIME_HOST_IP="${SLIME_HOST_IP:-${RAY_HEAD_IP:-}}"
        docker run --rm --gpus all --network host --ipc=host --shm-size=64g \
            --ulimit memlock=-1 --ulimit stack=67108864 \
            -v "${REPO_DIR}:/root/slime" \
            -v "${POOL_DIR}:/root/pool" \
            -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
            -e HF_TOKEN="${HF_TOKEN}" \
            -e WANDB_API_KEY="${WANDB_API_KEY}" \
            -e CHECKPOINT_HF_REPO_ID="${CHECKPOINT_HF_REPO_ID:-}" \
            -e SLIME_HOST_IP="${SLIME_HOST_IP}" \
            -e DEBUG="${DEBUG:-0}" \
            -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
            -w /root/slime \
            "${IMAGE}" \
            bash examples/on_policy_distillation/sfcompute/run-opd.sh "$@"
        ;;
    worker)
        docker run --rm --gpus all --network host --ipc=host --shm-size=64g \
            --ulimit memlock=-1 --ulimit stack=67108864 \
            -v "${REPO_DIR}:/root/slime" \
            -v "${POOL_DIR}:/root/pool" \
            -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
            -e RAY_HEAD_IP="${RAY_HEAD_IP:-}" \
            -e GPUS_PER_NODE="${GPUS_PER_NODE:-8}" \
            -e RAY_PORT="${RAY_PORT:-6379}" \
            -e HF_TOKEN="${HF_TOKEN}" \
            -e PREP_WORKER_ONLY=1 \
            -e DEBUG="${DEBUG:-0}" \
            -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
            -w /root/slime \
            "${IMAGE}" \
            bash -lc 'set -euo pipefail
if [ -z "${RAY_HEAD_IP:-}" ] || [ "${RAY_HEAD_IP}" = "127.0.0.1" ]; then
    echo "Set RAY_HEAD_IP to the reachable head-node IP before starting worker."
    exit 1
fi

# Download student model + convert to Megatron format (skips dataset + teacher)
echo "=== Worker prep: ensuring student model is ready ==="
PREP_WORKER_ONLY=1 bash examples/on_policy_distillation/sfcompute/prep-opd.sh

# Join Ray cluster with retries
ray stop --force || true
echo "Connecting to Ray head at ${RAY_HEAD_IP}:${RAY_PORT} (will retry up to 10 min)..."
for attempt in $(seq 1 40); do
    if ray start --address "${RAY_HEAD_IP}:${RAY_PORT}" --num-gpus "${GPUS_PER_NODE}" --disable-usage-stats --block; then
        exit 0
    fi
    echo "  Attempt ${attempt}/40 failed, retrying in 15s..."
    ray stop --force 2>/dev/null || true
    sleep 15
done
echo "Failed to connect to Ray head."
exit 1'
        ;;
    preflight)
        docker run --rm --gpus all --ipc=host \
            --ulimit memlock=-1 --ulimit stack=67108864 \
            -v "${REPO_DIR}:/root/slime" \
            -v "${POOL_DIR}:/root/pool" \
            -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
            -e HF_TOKEN="${HF_TOKEN}" \
            -e WANDB_API_KEY="${WANDB_API_KEY}" \
            -e CHECKPOINT_HF_REPO_ID="${CHECKPOINT_HF_REPO_ID:-}" \
            -w /root/slime \
            "${IMAGE}" \
            bash examples/on_policy_distillation/sfcompute/run-opd.sh --preflight
        ;;
    *)
        echo "Unknown command: ${COMMAND}"
        usage
        exit 1
        ;;
esac
