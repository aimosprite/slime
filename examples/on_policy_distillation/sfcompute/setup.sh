#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "=== sfcompute OPD setup ==="
echo "  REPO_DIR=${REPO_DIR}"

# ── 1. Docker engine ────────────────────────────────────────────────
install_docker() {
    if command -v docker >/dev/null 2>&1; then
        echo "Docker already installed, skipping."
        return 0
    fi

    echo "Installing Docker..."
    apt-get update
    apt-get install -y docker.io curl ca-certificates gnupg
}

# ── 2. NVIDIA container toolkit ─────────────────────────────────────
install_nvidia_toolkit() {
    if dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
        echo "NVIDIA container toolkit already installed, skipping."
        return 0
    fi

    echo "Installing NVIDIA container toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    apt-get update
    apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
}

# ── 3. Verify GPU passthrough ───────────────────────────────────────
verify_gpu() {
    echo "Verifying Docker GPU passthrough..."
    if ! docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi; then
        echo "GPU passthrough failed. Check your NVIDIA driver and container toolkit installation."
        exit 1
    fi
    echo "GPU passthrough OK."
}

# ── 4. Pull image ──────────────────────────────────────────────────
IMAGE="${IMAGE:-slimerl/slime:latest}"

pull_image() {
    echo "Pulling ${IMAGE}..."
    docker pull "${IMAGE}"
}

# ── 5. Create .env ──────────────────────────────────────────────────
create_dotenv() {
    local env_file="${REPO_DIR}/.env"

    if [ -f "${env_file}" ]; then
        echo ".env already exists at ${env_file}, skipping creation."
        echo "  (Delete it and re-run setup if you need to regenerate.)"
        return 0
    fi

    echo ""
    echo "Creating ${env_file} — paste your tokens below."
    umask 077
    read -rsp "WANDB_API_KEY: " WANDB_API_KEY; echo
    read -rsp "HF token (hf_...): " HF_TOKEN; echo
    read -rp  "CHECKPOINT_HF_REPO_ID (optional, auto-derived from models if empty): " CHECKPOINT_HF_REPO_ID

    cat > "${env_file}" <<EOF
WANDB_API_KEY=${WANDB_API_KEY}
HF_TOKEN=${HF_TOKEN}
CHECKPOINT_HF_REPO_ID=${CHECKPOINT_HF_REPO_ID}
EOF
    unset WANDB_API_KEY HF_TOKEN CHECKPOINT_HF_REPO_ID
    echo "Wrote ${env_file}"
}

# ── 6. Launch training ──────────────────────────────────────────────
launch_training() {
    echo ""
    echo "========================================="
    echo "Setup complete. Starting training..."
    echo "========================================="
    exec bash "${SCRIPT_DIR}/docker-run.sh" train
}

# ── Run ─────────────────────────────────────────────────────────────
install_docker
install_nvidia_toolkit
verify_gpu
pull_image
create_dotenv
launch_training
