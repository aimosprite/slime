#!/bin/bash
set -euo pipefail

# docker-run.sh — launches the gimran container with correct flags.
# Usage:
#   bash gimran/docker-run.sh              # interactive shell
#   bash gimran/docker-run.sh init         # run first-boot setup, then shell
#   bash gimran/docker-run.sh <command>    # run a command inside the container

IMAGE="${IMAGE:-slimerl/gimran:latest}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HOME}/.cache/huggingface}"

COMMAND="${1:-}"

mkdir -p "${HF_CACHE_DIR}"

# Load .env if it exists
HF_TOKEN=""
WANDB_API_KEY=""
ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"
if [ -f "${REPO_DIR}/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "${REPO_DIR}/.env"
    set +a
fi

DOCKER_ARGS=(
    --gpus all
    --network host
    --ipc=host
    --shm-size=64g
    --ulimit memlock=-1
    --ulimit stack=67108864
    -v "${REPO_DIR}:/root/slime"
    -v "${HF_CACHE_DIR}:/root/.cache/huggingface"
    -e HF_TOKEN="${HF_TOKEN:-}"
    -e WANDB_API_KEY="${WANDB_API_KEY:-${WANDB_TOKEN:-}}"
    -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}"
    -w /root/slime
)

case "${COMMAND}" in
    init)
        # First-boot: run init then drop to interactive shell
        exec docker run -it --rm "${DOCKER_ARGS[@]}" "${IMAGE}" \
            bash -c 'gimran-init && exec bash -l'
        ;;
    "")
        # Interactive shell
        exec docker run -it --rm "${DOCKER_ARGS[@]}" "${IMAGE}" bash -l
        ;;
    *)
        # Run a specific command
        exec docker run --rm "${DOCKER_ARGS[@]}" "${IMAGE}" "$@"
        ;;
esac
