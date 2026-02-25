#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/config-8xh100.env}"

if [ -f "${CONFIG_FILE}" ]; then
    set -a
    # shellcheck disable=SC1090
    source "${CONFIG_FILE}"
    set +a
fi

echo "Running prep with:"
echo "  REPO_DIR=${REPO_DIR:-unset}"
echo "  ROOT_DIR=${ROOT_DIR:-unset}"
echo "  POOL_DIR=${POOL_DIR:-unset}"
echo "  MEGATRON_PATH=${MEGATRON_PATH:-unset}"

bash "${BASE_DIR}/prep-qwen3-8B-opd.sh"
