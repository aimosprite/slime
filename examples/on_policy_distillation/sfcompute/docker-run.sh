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

case "${COMMAND}" in
    prep)
        docker run --rm --gpus all --ipc=host \
            --ulimit memlock=-1 --ulimit stack=67108864 \
            -v "${REPO_DIR}:/root/slime" \
            -v "${POOL_DIR}:/root/pool" \
            -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
            -w /root/slime \
            "${IMAGE}" \
            bash examples/on_policy_distillation/sfcompute/prep-qwen3-8B-opd.sh "$@"
        ;;
    train)
        docker run --rm --gpus all --network host --ipc=host --shm-size=64g \
            --ulimit memlock=-1 --ulimit stack=67108864 \
            -v "${REPO_DIR}:/root/slime" \
            -v "${POOL_DIR}:/root/pool" \
            -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
            -w /root/slime \
            "${IMAGE}" \
            bash examples/on_policy_distillation/sfcompute/run-qwen3-8B-opd.sh "$@"
        ;;
    *)
        echo "Unknown command: ${COMMAND}"
        usage
        exit 1
        ;;
esac
