#!/bin/bash
set -e

# init.sh — runs INSIDE the container on first boot.
# Handles runtime config that can't be baked into the image:
#   - .env secrets (HF_TOKEN, WANDB)
#   - GitHub CLI auth
#   - git identity
#   - model downloads

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# When run from /usr/local/bin/gimran-init, SLIME_DIR is the workdir
SLIME_DIR="${SLIME_DIR:-/root/slime}"
cd "${SLIME_DIR}"

# Source uv if available
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"

# ── 1. Secrets (.env) ────────────────────────────────────────────
echo "=== secrets (.env) ==="
if [ -f "${SLIME_DIR}/.env" ]; then
    echo ".env already exists, skipping"
else
    read -rp "HF_TOKEN: " _hf_token
    read -rp "WANDB_TOKEN: " _wandb_token
    printf "HF_TOKEN=%s\nWANDB_TOKEN=%s\n" "${_hf_token}" "${_wandb_token}" > "${SLIME_DIR}/.env"
    echo "Created .env"
fi
set -a; source "${SLIME_DIR}/.env"; set +a

# ── 2. GitHub CLI auth ───────────────────────────────────────────
echo "=== gh auth ==="
if gh auth status &>/dev/null; then
    echo "gh already authenticated"
else
    read -rsp "GitHub token (leave empty to skip): " _github_token
    echo ""
    if [ -n "${_github_token}" ]; then
        echo "${_github_token}" | gh auth login --with-token
        echo "gh authenticated via token"
    else
        echo "Skipping gh auth (no token provided)"
    fi
fi

# ── 3. Git identity ──────────────────────────────────────────────
echo "=== git identity ==="
if [ -z "$(git config --global user.name)" ]; then
    GH_NAME=$(gh api user --jq '.name' 2>/dev/null || true)
    GH_ID=$(gh api user --jq '.id' 2>/dev/null || true)
    GH_LOGIN=$(gh api user --jq '.login' 2>/dev/null || true)

    read -rp "Git name [${GH_NAME}]: " INPUT_NAME
    git config --global user.name "${INPUT_NAME:-$GH_NAME}"

    DEFAULT_EMAIL="${GH_ID}+${GH_LOGIN}@users.noreply.github.com"
    read -rp "Git email [${DEFAULT_EMAIL}]: " INPUT_EMAIL
    git config --global user.email "${INPUT_EMAIL:-$DEFAULT_EMAIL}"

    echo "git identity: $(git config --global user.name) <$(git config --global user.email)>"
else
    echo "already configured: $(git config --global user.name) <$(git config --global user.email)>"
fi

# ── 4. Model downloads ───────────────────────────────────────────
echo "=== models ==="
MODELS_FILE="${SLIME_DIR}/models.txt"
if [ ! -f "$MODELS_FILE" ]; then
    echo "No models.txt found, skipping downloads"
else
    mkdir -p models
    hf_ids=()
    local_dirs=()
    while IFS= read -r line; do
        line="${line%%#*}"
        line="$(echo "$line" | xargs)"
        [[ -z "$line" ]] && continue
        hf_id="$(echo "$line" | awk '{print $1}')"
        local_dir="$(echo "$line" | awk '{print $2}')"
        hf_ids+=("$hf_id")
        local_dirs+=("$local_dir")
    done < "$MODELS_FILE"

    if [ ${#hf_ids[@]} -eq 0 ]; then
        echo "No models in models.txt"
    else
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
            echo "Enter model numbers to download (space-separated), 'all', or 'none':"
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
fi

echo ""
echo "========================================="
echo "Init complete. You're inside the container."
echo "  - Run 'clauded' to start Claude Code"
echo "  - Run 'gimran-init' again to re-run this setup"
echo "  - Models are in ./models/"
echo "  - Your repo is mounted from the host"
echo "========================================="
