#!/usr/bin/env bash
# Launch GPT-OSS 20B scaffolding RL on Modal (2×H200, delegates to
# examples/scaffolding/run_gpt_oss_scaffolding_modal.py).
#
# Local setup: examples/scaffolding/scripts/prep-modal.sh (venv + volume slime-data), then modal token/setup.
#
# Volume + secret names are fixed in run_gpt_oss_scaffolding_modal.py (slime-data,
# slime-training-secrets). Exports below are optional reminders only; edit the .py to change names.
#
# Weights: Hub openai/gpt-oss-20b inside the worker. Training JSONL: on the volume or override --data-jsonl.
#
# Usage (from repo root is easiest):
#   examples/scaffolding/scripts/run-gpt-oss-20b-scaffolding-modal.sh
#   examples/scaffolding/scripts/run-gpt-oss-20b-scaffolding-modal.sh \
#     --data-jsonl /root/data/train_data_filtered.jsonl
#   # optional: --hf-checkpoint <hub-id-or-path>
#   examples/scaffolding/scripts/run-gpt-oss-20b-scaffolding-modal.sh --num-rollout 4 --attempts 4
#
# Extra arguments are forwarded to `modal run` → the Python local entrypoint.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

MODAL_BIN="modal"
if [ -x "${REPO_ROOT}/.venv-modal/bin/modal" ]; then
  MODAL_BIN="${REPO_ROOT}/.venv-modal/bin/modal"
fi

exec "${MODAL_BIN}" run examples/scaffolding/run_gpt_oss_scaffolding_modal.py \
  --model-size 20b \
  "$@"
