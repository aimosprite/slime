#!/bin/bash
set -euo pipefail

# setup.sh — one-command setup for a fresh GPU node.
#
# What it does:
#   1. Installs Docker + NVIDIA container toolkit
#   2. Pulls slimerl/slime:latest (all ML deps pre-compiled)
#   3. Creates .env (HF_TOKEN, WANDB)
#   4. Installs Claude Code + clauded alias
#   5. Sets up gh auth + git identity
#
# Usage:
#   bash gimran/setup.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="${IMAGE:-slimerl/slime:latest}"

export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a

# ── 1. Install Docker ────────────────────────────────────────────
install_docker() {
    if command -v docker >/dev/null 2>&1; then
        echo "[1/5] Docker already installed."
        return 0
    fi
    echo "[1/5] Installing Docker..."
    apt-get update -qq
    apt-get install -y -qq docker.io curl ca-certificates gnupg
}

# ── 2. NVIDIA container toolkit ──────────────────────────────────
install_nvidia_toolkit() {
    if dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
        echo "[2/5] NVIDIA container toolkit already installed."
        return 0
    fi
    echo "[2/5] Installing NVIDIA container toolkit..."
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

# ── 3. Verify GPU + pull image ───────────────────────────────────
verify_and_pull() {
    echo "[3/5] Verifying Docker GPU passthrough..."
    if ! docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi; then
        echo "GPU passthrough failed. Check NVIDIA driver and container toolkit."
        exit 1
    fi
    echo "GPU passthrough OK."
    echo "Pulling ${IMAGE}..."
    docker pull "${IMAGE}"
}

# ── 4. Secrets (.env) ────────────────────────────────────────────
create_dotenv() {
    echo "[4/5] Secrets..."
    if [ -f "${REPO_DIR}/.env" ]; then
        echo ".env already exists, skipping."
        return 0
    fi
    read -rp "HF_TOKEN: " _hf_token
    read -rp "WANDB_TOKEN: " _wandb_token
    printf "HF_TOKEN=%s\nWANDB_TOKEN=%s\n" "${_hf_token}" "${_wandb_token}" > "${REPO_DIR}/.env"
    echo "Created .env"
}

# ── 5. Claude Code + gh + git ────────────────────────────────────
install_claude_and_tools() {
    echo "[5/5] Claude Code + tools..."

    # Node.js
    if ! command -v node >/dev/null 2>&1; then
        curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
        apt-get install -y -qq nodejs
    fi

    # Claude Code
    if ! command -v claude >/dev/null 2>&1; then
        npm install -g @anthropic-ai/claude-code
    fi

    # aimo user for sandboxed Claude execution
    if ! id aimo &>/dev/null; then
        useradd -m -s /bin/bash aimo
        echo "aimo ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
    fi

    # clauded command
    cat > /usr/local/bin/clauded <<'WRAPPER'
#!/bin/bash
if [ "$(id -u)" -eq 0 ]; then
    [ -d /root/.claude ] && cp -r /root/.claude /home/aimo/.claude && chown -R aimo:aimo /home/aimo/.claude
    chmod o+x /root
    chgrp -R aimo "${PWD}" 2>/dev/null || true; chmod -R g+rwX "${PWD}" 2>/dev/null || true
    SUDO_ENV=(IS_SANDBOX=1)
    [ -n "${ANTHROPIC_API_KEY:-}" ] && SUDO_ENV+=(ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY")
    exec sudo -u aimo "${SUDO_ENV[@]}" claude --dangerously-skip-permissions "$@"
else
    exec env IS_SANDBOX=1 claude --dangerously-skip-permissions "$@"
fi
WRAPPER
    chmod +x /usr/local/bin/clauded

    # GitHub CLI
    if ! command -v gh >/dev/null 2>&1; then
        curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
            | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
            | tee /etc/apt/sources.list.d/github-cli.list > /dev/null
        apt-get update -qq && apt-get install -y -qq gh
    fi

    # gh auth
    if ! gh auth status &>/dev/null; then
        read -rsp "GitHub token (leave empty to skip): " _github_token
        echo ""
        if [ -n "${_github_token}" ]; then
            echo "${_github_token}" | gh auth login --with-token
        fi
    fi

    # git identity
    if [ -z "$(git config --global user.name)" ]; then
        GH_NAME=$(gh api user --jq '.name' 2>/dev/null || true)
        GH_ID=$(gh api user --jq '.id' 2>/dev/null || true)
        GH_LOGIN=$(gh api user --jq '.login' 2>/dev/null || true)
        read -rp "Git name [${GH_NAME}]: " INPUT_NAME
        git config --global user.name "${INPUT_NAME:-$GH_NAME}"
        DEFAULT_EMAIL="${GH_ID}+${GH_LOGIN}@users.noreply.github.com"
        read -rp "Git email [${DEFAULT_EMAIL}]: " INPUT_EMAIL
        git config --global user.email "${INPUT_EMAIL:-$DEFAULT_EMAIL}"
    fi
}

# ── Run ──────────────────────────────────────────────────────────
echo "=== gimran setup ==="
echo "  REPO_DIR=${REPO_DIR}"
echo ""

install_docker
install_nvidia_toolkit
verify_and_pull
create_dotenv
install_claude_and_tools

echo ""
echo "========================================="
echo "Setup complete."
echo ""
echo "Run training:"
echo "  bash gimran/docker-run.sh bash gimran/emb-surgery-sft/scripts/prep-gpt-oss.sh"
echo "  bash gimran/docker-run.sh bash gimran/emb-surgery-sft/scripts/train-gpt-oss.sh"
echo ""
echo "Interactive shell inside container:"
echo "  bash gimran/docker-run.sh"
echo ""
echo "Claude Code:"
echo "  clauded"
echo "========================================="
