#!/bin/bash -l
#SBATCH --job-name=slime-scaff-full-20b-16k
#SBATCH --output=/home/rohin/slime/examples/scaffolding/slime-scaffolding-full-20b-16k-%j.log
#SBATCH --error=/home/rohin/slime/examples/scaffolding/slime-scaffolding-full-20b-16k-%j.log
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=5:59:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --requeue

# Same as sbatch_train_small_full_mit_orcd.sh but rollout context cap 16384 (~16 Ki) if 32k OOMs.

set -euo pipefail

ENV_FILE="/home/rohin/slime/examples/scaffolding/.env"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
  set +a
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
export SLIME_SCRIPT_NUM_GPUS="${SLIME_SCRIPT_NUM_GPUS:-2}"
export SLIME_SCRIPT_TP="${SLIME_SCRIPT_TP:-2}"
export SLIME_SCRIPT_EP="${SLIME_SCRIPT_EP:-1}"
export SLIME_SCRIPT_ROLLOUT_TP="${SLIME_SCRIPT_ROLLOUT_TP:-2}"
export SLIME_SCRIPT_TRAIN_BACKEND="${SLIME_SCRIPT_TRAIN_BACKEND:-megatron}"

export SLIME_SCAFFOLDING_ATTEMPTS="${SLIME_SCAFFOLDING_ATTEMPTS:-8}"

export SLIME_SCRIPT_NUM_ROLLOUT="${SLIME_SCRIPT_NUM_ROLLOUT:-16}"
export SLIME_SCRIPT_ROLLOUT_BATCH_SIZE="${SLIME_SCRIPT_ROLLOUT_BATCH_SIZE:-1}"
export SLIME_SCRIPT_GLOBAL_BATCH_SIZE="${SLIME_SCRIPT_GLOBAL_BATCH_SIZE:-9}"

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
      export SLIME_SCRIPT_ROLLOUT_TP="'"${SLIME_SCRIPT_ROLLOUT_TP}"'"
      export SLIME_SCRIPT_TRAIN_BACKEND="'"${SLIME_SCRIPT_TRAIN_BACKEND}"'"
      export SLIME_SCAFFOLDING_ATTEMPTS="'"${SLIME_SCAFFOLDING_ATTEMPTS}"'"
      export SLIME_SCRIPT_NUM_ROLLOUT="'"${SLIME_SCRIPT_NUM_ROLLOUT}"'"
      export SLIME_SCRIPT_ROLLOUT_BATCH_SIZE="'"${SLIME_SCRIPT_ROLLOUT_BATCH_SIZE}"'"
      export SLIME_SCRIPT_GLOBAL_BATCH_SIZE="'"${SLIME_SCRIPT_GLOBAL_BATCH_SIZE}"'"
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
