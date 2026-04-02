#!/usr/bin/env bash
# One-time / repeat-safe Modal prep for GPT-OSS scaffolding runs:
#   - Local venv at repo-root .venv-modal/ with the Modal CLI
#   - Persistent volume "slime-data" (mounted at /root/data in the training image)
#
# Run from anywhere; paths are resolved from this file's location.
#
# Prerequisites: `modal token new` or `modal setup` already done in this workspace.
#
# Optional flag:
#   --upload-sample-data   copy .hf-dataset-inspect/train_data_filtered.jsonl to the volume
#
# You still must:
#   - Put HF_TOKEN in a Modal secret named slime-training-secrets (gpt-oss is gated), e.g.:
#       modal secret create slime-training-secrets HF_TOKEN=... WANDB_API_KEY=...
#   - Upload training JSONL to the volume (weights load from Hub inside the GPU worker; cache persists
#     under /root/data/hf-cache on slime-data).

set -euo pipefail

UPLOAD_SAMPLE=0
for arg in "$@"; do
  case "$arg" in
    --upload-sample-data) UPLOAD_SAMPLE=1 ;;
  esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

VENV="${REPO_ROOT}/.venv-modal"
MODAL="${VENV}/bin/modal"

if [ ! -x "${MODAL}" ]; then
  python3 -m venv "${VENV}"
  "${VENV}/bin/pip" install -U pip
  "${VENV}/bin/pip" install 'modal>=0.64'
fi

if "${MODAL}" volume list 2>/dev/null | grep -q 'slime-data'; then
  echo "Modal volume 'slime-data' already exists."
else
  "${MODAL}" volume create slime-data
fi

SAMPLE_JSONL="${REPO_ROOT}/.hf-dataset-inspect/train_data_filtered.jsonl"
if [[ "${UPLOAD_SAMPLE}" -eq 1 ]]; then
  if [[ ! -f "${SAMPLE_JSONL}" ]]; then
    echo "ERROR: missing ${SAMPLE_JSONL}" >&2
    exit 1
  fi
  echo "Uploading sample train JSONL to volume slime-data -> /train_data_filtered.jsonl ..."
  "${MODAL}" volume put slime-data "${SAMPLE_JSONL}" /train_data_filtered.jsonl
fi

echo
echo "Prep done. Use (from repo root):"
echo "  examples/scaffolding/scripts/prep-modal.sh --upload-sample-data   # optional: put inspect JSONL on slime-data"
echo "  examples/scaffolding/scripts/run-gpt-oss-20b-scaffolding-modal.sh"
echo "  (volume/secret names are fixed in run_gpt_oss_scaffolding_modal.py)"
echo
echo "Upload JSONL only (from repo root), e.g.:"
echo "  ${MODAL} volume put slime-data ./train_data_filtered.jsonl /train_data_filtered.jsonl"
echo "Default --hf-checkpoint is openai/gpt-oss-20b (downloaded on the worker; HF_TOKEN required)."
