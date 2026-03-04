#!/bin/bash
set -euo pipefail

# DEBUG=1 enables NCCL_DEBUG=INFO and verbose bash tracing.
if [ "${DEBUG:-0}" = "1" ]; then
    export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
    set -x
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TRAIN_CONFIG="${TRAIN_CONFIG:-${SCRIPT_DIR}/train-config.yaml}"
MODE="${1:-single}"
ROLE_IP_ARG="${2:-}"

usage() {
    cat <<EOF
Usage:
  bash examples/on_policy_distillation/sfcompute/setup.sh single   # one-node flow (default)
  bash examples/on_policy_distillation/sfcompute/setup.sh teacher  # two-node head/teacher
  bash examples/on_policy_distillation/sfcompute/setup.sh student <teacher_ip>  # two-node worker
EOF
}

echo "=== sfcompute OPD setup ==="
echo "  REPO_DIR=${REPO_DIR}"
echo "  MODE=${MODE}"

load_train_config_env() {
    if [ ! -f "${TRAIN_CONFIG}" ]; then
        return 0
    fi
    eval "$(python3 - "${TRAIN_CONFIG}" <<'PYEOF'
import os, re, sys
for line in open(sys.argv[1], encoding="utf-8"):
    line = line.split('#')[0].strip()
    m = re.match(r'^([a-z_]+):\s*(\S.*)$', line)
    if not m:
        continue
    k, v = m.group(1).upper(), m.group(2).strip()
    if k not in os.environ:
        print(f"export {k}='{v}'")
PYEOF
)"
}

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

# ── Helpers for role-based launch ────────────────────────────────────
detect_primary_ip() {
    local ip=""
    if command -v tailscale >/dev/null 2>&1; then
        ip="$(tailscale ip -4 2>/dev/null | awk 'NF {print; exit}' || true)"
        if [ -n "${ip}" ]; then
            echo "${ip}"
            return 0
        fi
    fi
    python3 - <<'PY'
import socket
ip = "127.0.0.1"
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
except Exception:
    pass
print(ip)
PY
}

detect_gpu_count() {
    nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
}

write_train_config_ips() {
    local ip="$1"
    if [ -z "${ip}" ] || [ "${ip}" = "127.0.0.1" ]; then
        return 1
    fi
    if [ ! -f "${TRAIN_CONFIG}" ]; then
        echo "train-config not found at ${TRAIN_CONFIG}; skipping IP write."
        return 0
    fi
    python3 - "${TRAIN_CONFIG}" "${ip}" <<'PY'
import re
import sys
path, ip = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as f:
    lines = f.readlines()

def upsert(key: str):
    pat = re.compile(rf"^\s*{re.escape(key)}\s*:")
    for i, line in enumerate(lines):
        if pat.match(line):
            lines[i] = f"{key}: {ip}\n"
            return True
    return False

have_ray = upsert("ray_head_ip")
have_teacher = upsert("teacher_ip")

if not have_ray or not have_teacher:
    insert_at = None
    for i, line in enumerate(lines):
        if line.strip().startswith("ray_visible_gpus:"):
            insert_at = i
            break
    add_lines = []
    if not have_ray:
        add_lines.append(f"ray_head_ip: {ip}\n")
    if not have_teacher:
        add_lines.append(f"teacher_ip: {ip}\n")
    if add_lines:
        if insert_at is None:
            lines.extend(["\n"] + add_lines)
        else:
            lines[insert_at:insert_at] = add_lines

with open(path, "w", encoding="utf-8") as f:
    f.writelines(lines)
PY
    echo "Updated ${TRAIN_CONFIG}: ray_head_ip=${ip}, teacher_ip=${ip}"
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

# ── 6. Launch roles ─────────────────────────────────────────────────
launch_training() {
    echo ""
    echo "========================================="
    echo "Setup complete. Starting training..."
    echo "========================================="
    exec bash "${SCRIPT_DIR}/docker-run.sh" train
}

launch_worker() {
    local default_gpus
    default_gpus="$(detect_gpu_count)"
    export GPUS_PER_NODE="${GPUS_PER_NODE:-${default_gpus:-8}}"
    export RAY_PORT="${RAY_PORT:-6379}"

    if [ -z "${RAY_HEAD_IP:-}" ] || [ "${RAY_HEAD_IP}" = "127.0.0.1" ]; then
        echo ""
        echo "RAY_HEAD_IP is not set (or still localhost)."
        read -rp "Enter head/teacher node IP: " RAY_HEAD_IP
        export RAY_HEAD_IP
    fi

    echo ""
    echo "========================================="
    echo "Setup complete. Starting Ray worker..."
    echo "  RAY_HEAD_IP=${RAY_HEAD_IP}"
    echo "  RAY_PORT=${RAY_PORT}"
    echo "  GPUS_PER_NODE=${GPUS_PER_NODE}"
    echo "========================================="
    exec bash "${SCRIPT_DIR}/docker-run.sh" worker
}

# ── Run ─────────────────────────────────────────────────────────────
case "${MODE}" in
    single|teacher|student) ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown mode: ${MODE}"
        usage
        exit 1
        ;;
esac

load_train_config_env
if [ "${MODE}" = "student" ] && [ -n "${ROLE_IP_ARG}" ]; then
    export RAY_HEAD_IP="${ROLE_IP_ARG}"
    export TEACHER_IP="${ROLE_IP_ARG}"
fi
install_docker
install_nvidia_toolkit
verify_gpu
pull_image
if [ "${MODE}" = "student" ]; then
    launch_worker
else
    create_dotenv
    export RAY_PORT="${RAY_PORT:-6379}"
    export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
    if [ "${MODE}" = "teacher" ] && { [ -z "${RAY_HEAD_IP:-}" ] || [ "${RAY_HEAD_IP}" = "127.0.0.1" ]; }; then
        export RAY_HEAD_IP="$(detect_primary_ip)"
        export TEACHER_IP="${TEACHER_IP:-${RAY_HEAD_IP}}"
        echo "Auto-detected RAY_HEAD_IP=${RAY_HEAD_IP}"
        write_train_config_ips "${RAY_HEAD_IP}" || true
        echo "Run this on the student node:"
        echo "  RAY_PORT=${RAY_PORT} bash examples/on_policy_distillation/sfcompute/setup.sh student ${RAY_HEAD_IP}"
    elif [ "${MODE}" = "teacher" ]; then
        write_train_config_ips "${RAY_HEAD_IP}" || true
        echo "Run this on the student node:"
        echo "  RAY_PORT=${RAY_PORT} bash examples/on_policy_distillation/sfcompute/setup.sh student ${RAY_HEAD_IP}"
    fi
    if [ "${MODE}" = "teacher" ] && [[ "${CLUSTER_NUM_NODES:-1}" =~ ^[0-9]+$ ]] && [ "${CLUSTER_NUM_NODES:-1}" -gt 1 ]; then
        echo ""
        echo "Start the student worker on the other node (now or while teacher sets up):"
        echo "  RAY_PORT=${RAY_PORT} bash examples/on_policy_distillation/sfcompute/setup.sh student ${RAY_HEAD_IP}"
        echo "Training will wait for workers after Ray head starts."
    fi
    launch_training
fi
