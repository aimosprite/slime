#!/bin/bash
# gimran/run.sh — fix issues and run stage1 training
# Logs to /root/slime/models/gimran-run.log

set -euo pipefail
cd "$(dirname "$0")/.."
LOG=/root/slime/models/gimran-run.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "========================================"
echo "  gimran/run.sh started $(date)"
echo "========================================"

# ── Fix #12: install slime_plugins + mbridge ──────────────────────────────────
# ── Run prep (idempotent — skips steps already done) ─────────────────────────
echo "--- Running prep ---"
TRAIN_CONFIG=/root/slime/configs/sft-gpt-oss-20b-embedding-surgery-stage1.yaml \
    bash /root/slime/scripts/prep-gpt-oss-embedding-surgery.sh

# ── Run training ──────────────────────────────────────────────────────────────
echo "--- Starting training ---"
TRAIN_CONFIG=/root/slime/configs/sft-gpt-oss-20b-embedding-surgery-stage1.yaml \
    bash /root/slime/scripts/sft-gpt-oss-20b-embedding-swap.sh

echo "========================================"
echo "  gimran/run.sh finished $(date)"
echo "========================================"
