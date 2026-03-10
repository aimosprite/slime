#!/bin/bash
set -e

# Always run from repo root (where this script lives)
cd "$(dirname "$0")"
SLIME_DIR="$(pwd)"
ROOT_DIR="$(dirname "$SLIME_DIR")"

export MEGATRON_COMMIT="3714d81d418c9f1bca4594fc35f9e8289f652862"
export SGLANG_COMMIT="24c91001cf99ba642be791e099d358f4dfe955f5"

# Suppress interactive prompts from apt (needrestart, daemon restart dialogs)
export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a

echo "=== installing base dependencies ==="
apt-get update -qq && apt-get install -y -qq curl git wget sudo python3-dev python3-pip

echo "=== installing node.js ==="
if command -v node &>/dev/null; then
    echo "node already installed: $(node --version)"
else
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
    apt-get install -y -qq nodejs
    echo "installed: $(node --version)"
fi

echo "=== installing claude code ==="
if command -v claude &>/dev/null; then
    echo "claude already installed: $(claude --version)"
else
    npm install -g @anthropic-ai/claude-code
    echo "installed: $(claude --version)"
fi

echo "=== setting up clauded (non-root user for --dangerously-skip-permissions) ==="
if ! id aimo &>/dev/null; then
    useradd -m -s /bin/bash aimo
    echo "aimo ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
    echo "created user 'aimo' with passwordless sudo"
else
    echo "user 'aimo' already exists"
fi
# Copy config to aimo user
cp -r /root/.claude /home/aimo/.claude 2>/dev/null || true
mkdir -p /home/aimo/.local/bin
[ -f /root/.local/bin/env ] && cp /root/.local/bin/env /home/aimo/.local/bin/env
chown -R aimo:aimo /home/aimo/

# Make repo accessible to aimo user
# /root is 700 by default — aimo needs +x to traverse into /root/slime
chmod o+x /root
REPO_DIR="$(pwd)"
chgrp -R aimo "$REPO_DIR"
chmod -R g+rwX "$REPO_DIR"

# Install clauded as a script in PATH (preserves TTY via sudo -u)
cat > /usr/local/bin/clauded <<'SCRIPT'
#!/bin/bash
if [ "$(id -u)" -eq 0 ]; then
    # Sync latest claude config to aimo user
    if [ -d /root/.claude ]; then
        cp -r /root/.claude /home/aimo/.claude
        chown -R aimo:aimo /home/aimo/.claude
    fi

    # Build env passthrough
    SUDO_ENV=(IS_SANDBOX=1)
    [ -n "$ANTHROPIC_API_KEY" ] && SUDO_ENV+=(ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY")

    exec sudo -u aimo "${SUDO_ENV[@]}" claude --dangerously-skip-permissions "$@"
else
    exec env IS_SANDBOX=1 claude --dangerously-skip-permissions "$@"
fi
SCRIPT
chmod +x /usr/local/bin/clauded
echo "installed /usr/local/bin/clauded"

# Clean up old .bashrc function/alias if present
for rc in /root/.bashrc /home/aimo/.bashrc; do
    sed -i '/clauded()/,/^}/d' "$rc" 2>/dev/null || true
    sed -i '/alias clauded=/d' "$rc" 2>/dev/null || true
done

echo "=== installing uv ==="
if command -v uv &>/dev/null; then
    echo "uv already installed: $(uv --version)"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "installed uv"
fi
# Always source so uv is available for the rest of this script
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"
echo "uv: $(uv --version)"

echo "=== installing hf CLI (huggingface-hub) ==="
if command -v hf &>/dev/null; then
    echo "hf already installed"
else
    uv pip install --system "huggingface-hub[cli]"
    echo "installed hf"
fi

echo "=== upgrading CuDNN (fix PyTorch 2.9.1 compatibility) ==="
uv pip install --system "nvidia-cudnn-cu12==9.16.0.29"
echo "installed nvidia-cudnn-cu12==9.16.0.29"

echo "=== installing CUDA toolkit (nvcc for JIT compilation) ==="
if [ -f /usr/local/cuda-12.8/bin/nvcc ]; then
    echo "cuda-nvcc-12-8 already installed"
else
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
    dpkg -i /tmp/cuda-keyring.deb
    apt-get update -qq
    apt-get install -y -qq cuda-nvcc-12-8
    echo "installed cuda-nvcc-12-8"
fi

echo "=== installing gh (GitHub CLI) ==="
if command -v gh &>/dev/null; then
    echo "gh already installed: $(gh --version | head -1)"
else
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        | tee /etc/apt/sources.list.d/github-cli.list > /dev/null
    apt-get update -qq && apt-get install -y -qq gh
    echo "installed: $(gh --version | head -1)"
fi

echo "=== gh auth login ==="
if gh auth status &>/dev/null; then
    echo "gh already authenticated"
else
    gh auth login
fi

echo "=== configuring git identity ==="
if [ -z "$(git config --global user.name)" ]; then
    GH_NAME=$(gh api user --jq '.name' 2>/dev/null || true)
    GH_ID=$(gh api user --jq '.id' 2>/dev/null || true)
    GH_LOGIN=$(gh api user --jq '.login' 2>/dev/null || true)

    read -rp "Git name [${GH_NAME}]: " INPUT_NAME
    git config --global user.name "${INPUT_NAME:-$GH_NAME}"

    DEFAULT_EMAIL="${GH_ID}+${GH_LOGIN}@users.noreply.github.com"
    read -rp "Git email [${DEFAULT_EMAIL}]: " INPUT_EMAIL
    git config --global user.email "${INPUT_EMAIL:-$DEFAULT_EMAIL}"

    echo "git identity set to: $(git config --global user.name) <$(git config --global user.email)>"
else
    echo "git identity already configured: $(git config --global user.name) <$(git config --global user.email)>"
fi

echo "=== cloning slime ==="
if [ ! -d "${ROOT_DIR}/slime" ]; then
    git clone https://github.com/aimosprite/slime.git "${ROOT_DIR}/slime"
else
    echo "slime/ already exists, skipping"
fi

echo "=== installing Megatron-LM (pinned commit + patch) ==="
if [ ! -d "${ROOT_DIR}/Megatron-LM" ]; then
    git clone https://github.com/NVIDIA/Megatron-LM.git --recursive "${ROOT_DIR}/Megatron-LM"
fi
cd "${ROOT_DIR}/Megatron-LM"
git checkout ${MEGATRON_COMMIT}
git checkout -- .
git apply "${SLIME_DIR}/docker/patch/latest/megatron.patch"
uv pip install --system -e .
cd "${SLIME_DIR}"

echo "=== installing sglang (pinned commit + patch) ==="
if [ ! -d "${ROOT_DIR}/sglang" ]; then
    git clone https://github.com/sgl-project/sglang.git "${ROOT_DIR}/sglang"
fi
cd "${ROOT_DIR}/sglang"
git checkout ${SGLANG_COMMIT}
git checkout -- .
git apply "${SLIME_DIR}/docker/patch/latest/sglang.patch" || true
uv pip install --system -e "python[all]"
cd "${SLIME_DIR}"

echo "=== installing PyTorch ==="
python3 -c "import torch; print(torch.__version__)" 2>/dev/null || \
    uv pip install --system torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
        --index-url https://download.pytorch.org/whl/cu129

echo "=== installing flash-attn ==="
MAX_JOBS=$(nproc) uv pip install --system flash-attn==2.7.4.post1 --no-build-isolation

echo "=== installing ML training stack ==="
uv pip install --system \
    "mbridge @ git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c" \
    --no-deps
uv pip install --system --no-build-isolation "transformer_engine[pytorch]==2.10.0"
NVCC_APPEND_FLAGS="--threads 4" \
    uv pip install --system --no-build-isolation \
    --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
    "apex @ git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4"
uv pip install --system \
    "git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b" \
    --force-reinstall
uv pip install --system "git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl" --no-build-isolation
uv pip install --system "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation
uv pip install --system flash-linear-attention==0.4.0
uv pip install --system einops
uv pip install --system "numpy<2"
uv pip install --system cmake ninja

echo "=== installing slime ==="
uv pip install --system -e .

echo "=== downloading models ==="
mkdir -p models

MODELS_FILE="${SLIME_DIR}/models.txt"
if [ ! -f "$MODELS_FILE" ]; then
    echo "FATAL: models.txt not found at $MODELS_FILE" >&2
    exit 1
fi

# Read models from models.txt (skip comments and blank lines)
hf_ids=()
local_dirs=()
while IFS= read -r line; do
    line="${line%%#*}"            # strip comments
    line="$(echo "$line" | xargs)" # trim whitespace
    [[ -z "$line" ]] && continue
    hf_id="$(echo "$line" | awk '{print $1}')"
    local_dir="$(echo "$line" | awk '{print $2}')"
    hf_ids+=("$hf_id")
    local_dirs+=("$local_dir")
done < "$MODELS_FILE"

if [ ${#hf_ids[@]} -eq 0 ]; then
    echo "No models found in models.txt"
else
    echo "Available models (from models.txt):"
    missing=()
    for i in "${!hf_ids[@]}"; do
        if [ -d "models/${local_dirs[$i]}" ]; then
            printf "  [%d] %-35s -> models/%-25s \e[32mOK\e[0m\n" "$((i+1))" "${hf_ids[$i]}" "${local_dirs[$i]}"
        else
            printf "  [%d] %-35s -> models/%-25s \e[33mMISSING\e[0m\n" "$((i+1))" "${hf_ids[$i]}" "${local_dirs[$i]}"
            missing+=("$((i+1))")
        fi
    done
    echo ""

    if [ ${#missing[@]} -eq 0 ]; then
        echo "All models already downloaded."
    else
        echo "Enter model numbers to download (space-separated), 'all' missing, or 'none':"
        read -rp "> " model_choice

        download_model() {
            local idx=$1
            if [ -d "models/${local_dirs[$idx]}" ]; then
                echo "--- models/${local_dirs[$idx]} already exists, skipping ---"
                return
            fi
            echo "--- Downloading ${hf_ids[$idx]} -> models/${local_dirs[$idx]} ---"
            hf download "${hf_ids[$idx]}" --local-dir "models/${local_dirs[$idx]}"
        }

        if [[ "$model_choice" == "none" || -z "$model_choice" ]]; then
            echo "Skipping model downloads."
        elif [[ "$model_choice" == "all" ]]; then
            for i in "${!hf_ids[@]}"; do
                download_model "$i"
            done
        else
            for num in $model_choice; do
                idx=$((num - 1))
                if (( idx < 0 || idx >= ${#hf_ids[@]} )); then
                    echo "WARNING: invalid choice '$num', skipping"
                    continue
                fi
                download_model "$idx"
            done
        fi
    fi
fi

echo "=== done ==="
