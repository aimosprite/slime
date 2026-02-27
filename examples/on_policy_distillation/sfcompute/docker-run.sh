#!/bin/bash
set -euo pipefail

IMAGE="${IMAGE:-slimerl/slime:latest}"
REPO_DIR="${REPO_DIR:-/root/slime}"
POOL_DIR="${POOL_DIR:-/root/pool}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/root/.cache/huggingface}"

usage() {
    cat <<'EOF'
Usage:
  bash examples/on_policy_distillation/sfcompute/docker-run.sh prep
  bash examples/on_policy_distillation/sfcompute/docker-run.sh train
  bash examples/on_policy_distillation/sfcompute/docker-run.sh preflight   # test HF checkpoint upload

Optional environment overrides:
  IMAGE    Docker image tag (default: slimerl/slime:latest)
  REPO_DIR Host path to slime repo (default: /root/slime)
  POOL_DIR Host path to pool dir (default: /root/pool)
  HF_CACHE_DIR Host HF cache/token path (default: /root/.cache/huggingface)
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
        docker run --rm --gpus all --network host --ipc=host --shm-size=64g \
            --ulimit memlock=-1 --ulimit stack=67108864 \
            -v "${REPO_DIR}:/root/slime" \
            -v "${POOL_DIR}:/root/pool" \
            -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
            -e HF_TOKEN="${HF_TOKEN}" \
            -e WANDB_API_KEY="${WANDB_API_KEY}" \
            -e CHECKPOINT_HF_REPO_ID="${CHECKPOINT_HF_REPO_ID:-}" \
            -w /root/slime \
            "${IMAGE}" \
            bash examples/on_policy_distillation/sfcompute/run-opd.sh "$@"
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
