#!/bin/bash
# =============================================================================
# hf_shipper.sh — HF checkpoint shipping (source this, then call the functions)
#
# Expects these env vars to be set before sourcing:
#   HF_TOKEN, CHECKPOINT_HF_REPO_ID, CHECKPOINT_HF_PRIVATE,
#   CHECKPOINT_HF_CREATE_REPO, CHECKPOINT_SHIP_POLL_SEC, SLIME_DIR
#
# Usage:
#   source scripts/lib/hf_shipper.sh
#   hf_login
#   preflight_hf || exit 1
#   start_checkpoint_shipper
#   # ... training runs ...
#   # cleanup_shipper is called automatically via trap
# =============================================================================

CHECKPOINT_SHIPPER_PID=""

# hf_login — authenticate to HuggingFace
hf_login() {
    export HF_TOKEN="${HF_TOKEN:-}"
    if [ -n "${HF_TOKEN}" ]; then
        huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential >/dev/null 2>&1 \
            || huggingface-cli login --token "${HF_TOKEN}" >/dev/null 2>&1 \
            || true
    fi
    if ! huggingface-cli whoami >/dev/null 2>&1; then
        echo "ERROR: HF auth failed. Set HF_TOKEN in .env or run 'huggingface-cli login'."
        return 1
    fi
    echo "HF auth OK: $(huggingface-cli whoami 2>/dev/null | head -1)"
}

# preflight_hf — verify auth, create repo, test upload
preflight_hf() {
    echo "=== HF checkpoint shipping preflight ==="

    echo "1. Auth check..."
    if ! huggingface-cli whoami >/dev/null 2>&1; then
        echo "FAIL: not authenticated to HuggingFace."
        return 1
    fi
    echo "   OK: $(huggingface-cli whoami 2>/dev/null | head -1)"

    if [ "${CHECKPOINT_HF_CREATE_REPO}" = "1" ]; then
        echo "2. Creating HF repo (if needed): ${CHECKPOINT_HF_REPO_ID}"
        local private_flag=""
        [ "${CHECKPOINT_HF_PRIVATE}" = "1" ] && private_flag="--private"
        local create_out=""
        if create_out="$(huggingface-cli repo create "${CHECKPOINT_HF_REPO_ID}" \
                --repo-type model --exist-ok ${private_flag} -y 2>&1)"; then
            echo "   Repo ready: ${CHECKPOINT_HF_REPO_ID}"
        else
            echo "FAIL: repo create error:"
            echo "${create_out}"
            return 1
        fi
    fi

    echo "3. Test upload..."
    local test_file=""
    test_file="$(mktemp /tmp/slime_preflight_XXXXXX.txt)"
    echo "preflight $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${test_file}"
    local upload_out=""
    if upload_out="$(huggingface-cli upload "${CHECKPOINT_HF_REPO_ID}" \
            "${test_file}" ".preflight_test.txt" \
            --repo-type model 2>&1)"; then
        echo "   Upload OK"
    else
        echo "FAIL: test upload failed:"
        echo "${upload_out}"
        rm -f "${test_file}"
        return 1
    fi
    rm -f "${test_file}"
    echo "=== Preflight PASSED: shipping to ${CHECKPOINT_HF_REPO_ID} is working ==="
}

# start_checkpoint_shipper — background daemon that polls for new checkpoints
start_checkpoint_shipper() {
    local tracker="${SLIME_DIR}/latest_checkpointed_iteration.txt"
    echo "Starting checkpoint shipper (poll every ${CHECKPOINT_SHIP_POLL_SEC}s) -> ${CHECKPOINT_HF_REPO_ID}"

    (
        last_synced_step=""
        while true; do
            if [ -f "${tracker}" ]; then
                step="$(tr -d '[:space:]' < "${tracker}" || true)"
                if [[ "${step}" =~ ^[0-9]+$ ]] && [ "${step}" -gt 0 ] && \
                   [ "${step}" != "${last_synced_step}" ]; then
                    iter_dir="${SLIME_DIR}/iter_$(printf '%07d' "${step}")"
                    if [ -d "${iter_dir}" ]; then
                        echo "[shipper] Uploading step ${step} -> checkpoint/ ..."
                        if huggingface-cli upload "${CHECKPOINT_HF_REPO_ID}" \
                                "${iter_dir}" \
                                "checkpoint" \
                                --repo-type model 2>&1; then
                            huggingface-cli upload "${CHECKPOINT_HF_REPO_ID}" \
                                "${tracker}" "latest_checkpointed_iteration.txt" \
                                --repo-type model >/dev/null 2>&1 || true
                            last_synced_step="${step}"
                            echo "[shipper] Step ${step} uploaded (overwrote checkpoint/)."
                            for old_dir in "${SLIME_DIR}/iter_"*/; do
                                [ "${old_dir%/}" != "${iter_dir}" ] && rm -rf "${old_dir}" \
                                    && echo "[shipper] Deleted old checkpoint: ${old_dir}"
                            done
                        else
                            echo "[shipper] Upload FAILED for step ${step} — will retry next poll."
                        fi
                    fi
                fi
            fi
            sleep "${CHECKPOINT_SHIP_POLL_SEC}"
        done
    ) 2>&1 | tee -a /tmp/slime_checkpoint_shipper.log &
    CHECKPOINT_SHIPPER_PID=$!
    echo "Checkpoint shipper PID=${CHECKPOINT_SHIPPER_PID}"
}

# cleanup_shipper — kill shipper + ray stop (set as trap automatically)
cleanup_shipper() {
    echo "Cleaning up..."
    [ -n "${CHECKPOINT_SHIPPER_PID}" ] && kill -TERM "${CHECKPOINT_SHIPPER_PID}" 2>/dev/null || true
    ray stop --force 2>/dev/null || true
}
trap cleanup_shipper EXIT INT TERM
