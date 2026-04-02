#!/bin/bash -l
#SBATCH --job-name=slime-scaff-20b-16k-4gpu
#SBATCH --output=/home/rohin/slime/examples/scaffolding/slime-scaffolding-full-20b-16k-preempt-4gpu-%j.log
#SBATCH --error=/home/rohin/slime/examples/scaffolding/slime-scaffolding-full-20b-16k-preempt-4gpu-%j.log
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=5:59:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --requeue

# Colocated GPT-OSS 20B scaffolding on 4x H200 (classic Megatron path: TP=4, full recompute, dynamic batch).
# Partition may preempt; --requeue lets Slurm rerun the job when policy allows.
#
# Memory: see examples/scaffolding/README.md "Colocation memory tuning". Empirically on 4×H200 140GiB
# class GPUs, GPT-OSS 20B TP=4 needed ~73GiB for Megatron grad buffer; f=0.52 left ~60GiB free → OOM.
# Default f=0.38 gives ~(1-0.38)*M ≈ 87GiB headroom before overhead (adjust via SLIME_SCRIPT_SGLANG_MEM_FRACTION).
# GBS forced to rollout_batch_size * 2 * attempts. Do not set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# here: it disables TorchMemorySaver in SGLang and crashes startup (see README "Fragmentation").

set -euo pipefail

ENV_FILE="/home/rohin/slime/examples/scaffolding/.env"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
  set +a
fi
# TorchMemorySaver (colocated SGLang) cannot run with expandable_segments (see job log / README).
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}${PYTORCH_ALLOC_CONF:-}" == *expandable_segments* ]]; then
  echo "WARN: Clearing PYTORCH_CUDA_ALLOC_CONF / PYTORCH_ALLOC_CONF (expandable_segments breaks colocated SGLang)." >&2
  unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
fi

module load cuda
module load apptainer/1.4.2

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: WANDB_API_KEY must be set (e.g. in ${ENV_FILE})." >&2
  exit 1
fi

export SLIME_SCRIPT_HF_CHECKPOINT="${SLIME_SCRIPT_HF_CHECKPOINT:-/orcd/scratch/orcd/013/rohin/models/gpt-oss-20b}"
export SLIME_SCRIPT_DATA_JSONL="${SLIME_SCRIPT_DATA_JSONL:-/home/rohin/slime/.hf-dataset-inspect/train_data_filtered.jsonl}"

if [[ ! -d "${SLIME_SCRIPT_HF_CHECKPOINT}" ]]; then
  echo "ERROR: SLIME_SCRIPT_HF_CHECKPOINT is not a directory: ${SLIME_SCRIPT_HF_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${SLIME_SCRIPT_DATA_JSONL}" ]]; then
  echo "ERROR: SLIME_SCRIPT_DATA_JSONL not found: ${SLIME_SCRIPT_DATA_JSONL}" >&2
  exit 1
fi

unset SLIME_SCRIPT_SKIP_WANDB || true

export SLIME_SCRIPT_MODEL_SIZE="${SLIME_SCRIPT_MODEL_SIZE:-20b}"
export SLIME_SCRIPT_NUM_GPUS="${SLIME_SCRIPT_NUM_GPUS:-4}"
export SLIME_SCRIPT_TP="${SLIME_SCRIPT_TP:-4}"
export SLIME_SCRIPT_EP="${SLIME_SCRIPT_EP:-1}"
export SLIME_SCRIPT_ETP="${SLIME_SCRIPT_ETP:-1}"
export SLIME_SCRIPT_ROLLOUT_TP="${SLIME_SCRIPT_ROLLOUT_TP:-4}"
export SLIME_SCRIPT_TRAIN_BACKEND="${SLIME_SCRIPT_TRAIN_BACKEND:-megatron}"

export SLIME_SCAFFOLDING_ATTEMPTS="${SLIME_SCAFFOLDING_ATTEMPTS:-8}"

export SLIME_SCRIPT_NUM_ROLLOUT="${SLIME_SCRIPT_NUM_ROLLOUT:-16}"
export SLIME_SCRIPT_ROLLOUT_BATCH_SIZE="${SLIME_SCRIPT_ROLLOUT_BATCH_SIZE:-1}"
# Must match dual-group scaffolding: n_samples_per_prompt = 2 * attempts
export SLIME_SCRIPT_GLOBAL_BATCH_SIZE="$((${SLIME_SCRIPT_ROLLOUT_BATCH_SIZE:-1} * 2 * ${SLIME_SCAFFOLDING_ATTEMPTS:-8}))"

export SLIME_SCRIPT_SGLANG_MEM_FRACTION="${SLIME_SCRIPT_SGLANG_MEM_FRACTION:-0.38}"
export SLIME_SCRIPT_MAX_TOKENS_PER_GPU="${SLIME_SCRIPT_MAX_TOKENS_PER_GPU:-2048}"

export SLIME_SCRIPT_ROLLOUT_MAX_RESPONSE_LEN="${SLIME_SCRIPT_ROLLOUT_MAX_RESPONSE_LEN:-8192}"
export SLIME_SCRIPT_ROLLOUT_MAX_CONTEXT_LEN="${SLIME_SCRIPT_ROLLOUT_MAX_CONTEXT_LEN:-16384}"

cd /home/rohin/slime
export PYTHONUNBUFFERED=1

if [[ -f "/home/rohin/slime/slime.sif" ]]; then
  APPTAINER_BIND_ARGS=(--bind /home/rohin:/home/rohin)
  [[ -d /orcd ]] && APPTAINER_BIND_ARGS+=(--bind /orcd:/orcd)
  apptainer exec --nv \
    "${APPTAINER_BIND_ARGS[@]}" \
    /home/rohin/slime/slime.sif \
    bash -lc '
      set -euo pipefail
      cd /home/rohin/slime
      export PYTHONNOUSERSITE=1
      export PATH="/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
      export CC=/usr/bin/gcc
      export CXX=/usr/bin/g++
      export PYTHONPATH="/home/rohin/slime:${PYTHONPATH:-}"
      export PYTHONUNBUFFERED=1
      export WANDB_API_KEY="'"${WANDB_API_KEY}"'"
      export HF_TOKEN="'"${HF_TOKEN:-}"'"
      export HUGGING_FACE_HUB_TOKEN="'"${HF_TOKEN:-}"'"
      export SLIME_SCRIPT_HF_CHECKPOINT="'"${SLIME_SCRIPT_HF_CHECKPOINT}"'"
      export SLIME_SCRIPT_DATA_JSONL="'"${SLIME_SCRIPT_DATA_JSONL}"'"
      export SLIME_SCRIPT_MODEL_SIZE="'"${SLIME_SCRIPT_MODEL_SIZE}"'"
      export SLIME_SCRIPT_NUM_GPUS="'"${SLIME_SCRIPT_NUM_GPUS}"'"
      export SLIME_SCRIPT_TP="'"${SLIME_SCRIPT_TP}"'"
      export SLIME_SCRIPT_EP="'"${SLIME_SCRIPT_EP}"'"
      export SLIME_SCRIPT_ETP="'"${SLIME_SCRIPT_ETP}"'"
      export SLIME_SCRIPT_ROLLOUT_TP="'"${SLIME_SCRIPT_ROLLOUT_TP}"'"
      export SLIME_SCRIPT_TRAIN_BACKEND="'"${SLIME_SCRIPT_TRAIN_BACKEND}"'"
      export SLIME_SCAFFOLDING_ATTEMPTS="'"${SLIME_SCAFFOLDING_ATTEMPTS}"'"
      export SLIME_SCRIPT_NUM_ROLLOUT="'"${SLIME_SCRIPT_NUM_ROLLOUT}"'"
      export SLIME_SCRIPT_ROLLOUT_BATCH_SIZE="'"${SLIME_SCRIPT_ROLLOUT_BATCH_SIZE}"'"
      export SLIME_SCRIPT_GLOBAL_BATCH_SIZE="'"${SLIME_SCRIPT_GLOBAL_BATCH_SIZE}"'"
      export SLIME_SCRIPT_SGLANG_MEM_FRACTION="'"${SLIME_SCRIPT_SGLANG_MEM_FRACTION}"'"
      export SLIME_SCRIPT_MAX_TOKENS_PER_GPU="'"${SLIME_SCRIPT_MAX_TOKENS_PER_GPU}"'"
      export SLIME_SCRIPT_ROLLOUT_MAX_RESPONSE_LEN="'"${SLIME_SCRIPT_ROLLOUT_MAX_RESPONSE_LEN}"'"
      export SLIME_SCRIPT_ROLLOUT_MAX_CONTEXT_LEN="'"${SLIME_SCRIPT_ROLLOUT_MAX_CONTEXT_LEN}"'"
      python3 examples/scaffolding/run_gpt_oss_scaffolding_rl.py
    '
  exit $?
fi

module load miniforge
export PYTHONPATH="/home/rohin/slime:${PYTHONPATH:-}"
python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' || {
  echo "ERROR: Need Python >= 3.10." >&2
  exit 1
}
python3 examples/scaffolding/run_gpt_oss_scaffolding_rl.py
