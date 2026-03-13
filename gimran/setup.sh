#!/bin/bash
set -euo pipefail

# setup.sh — one-command setup for a fresh GPU node.
#
# What it does:
#   1. Installs Docker + NVIDIA container toolkit (host-level, ~2 min)
#   2. Builds the gimran image from slimerl/slime:latest (~3 min)
#   3. Launches the container and runs first-boot init (secrets, git, models)
#
# After this, you're inside a container with everything working.
# No compilation, no version conflicts, no dependency hell.
#
# Usage:
#   bash gimran/setup.sh          # full setup + interactive shell
#   bash gimran/setup.sh --shell  # skip setup, just enter container
#   bash gimran/setup.sh --build  # just rebuild the image

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="${IMAGE:-slimerl/gimran:latest}"

export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a

# ── 1. Install Docker ────────────────────────────────────────────
install_docker() {
    if command -v docker >/dev/null 2>&1; then
        echo "[1/4] Docker already installed, skipping."
        return 0
    fi

    echo "[1/4] Installing Docker..."
    apt-get update -qq
    apt-get install -y -qq docker.io curl ca-certificates gnupg
}

# ── 2. NVIDIA container toolkit ──────────────────────────────────
install_nvidia_toolkit() {
    if dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
        echo "[2/4] NVIDIA container toolkit already installed, skipping."
        return 0
    fi

    echo "[2/4] Installing NVIDIA container toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    apt-get update -qq
    apt-get install -y -qq nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
}

# ── 3. Verify GPU passthrough ────────────────────────────────────
verify_gpu() {
    echo "[3/4] Verifying Docker GPU passthrough..."
    if ! docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi; then
        echo "GPU passthrough failed. Check NVIDIA driver and container toolkit."
        exit 1
    fi
    echo "GPU passthrough OK."
}

# ── 4. Build gimran image ────────────────────────────────────────
build_image() {
    echo "[4/4] Building gimran image (on top of slimerl/slime:latest)..."
    echo "  This pulls the base image (~15 GB) and adds gimran tools (~3 min)."
    docker build -f "${SCRIPT_DIR}/Dockerfile" "${REPO_DIR}" -t "${IMAGE}"
    echo "Image built: ${IMAGE}"
}

# ── Parse args ───────────────────────────────────────────────────
case "${1:-}" in
    --shell)
        exec bash "${SCRIPT_DIR}/docker-run.sh"
        ;;
    --build)
        build_image
        exit 0
        ;;
    --help|-h)
        cat <<EOF
Usage:
  bash gimran/setup.sh          # full setup + interactive shell
  bash gimran/setup.sh --shell  # skip setup, just enter container
  bash gimran/setup.sh --build  # just rebuild the image
EOF
        exit 0
        ;;
esac

# ── Full setup ───────────────────────────────────────────────────
echo "=== gimran setup ==="
echo "  REPO_DIR=${REPO_DIR}"
echo ""

install_docker
install_nvidia_toolkit
verify_gpu
build_image

echo ""
echo "========================================="
echo "Host setup complete. Entering container..."
echo "========================================="
echo ""

# Launch container with first-boot init
exec bash "${SCRIPT_DIR}/docker-run.sh" init
