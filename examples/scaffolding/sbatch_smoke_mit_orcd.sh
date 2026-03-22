#!/bin/bash -l
#SBATCH --job-name=slime-scaffolding-smoke
#SBATCH --output=/home/rohin/slime/examples/scaffolding/slime-scaffolding-smoke-%j.log
#SBATCH --error=/home/rohin/slime/examples/scaffolding/slime-scaffolding-smoke-%j.log
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5:59:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --requeue

set -euo pipefail

module load cuda
module load apptainer/1.4.2

# --- set before submit (sbatch exports your shell env) or uncomment below ---
# export SLIME_SCRIPT_HF_CHECKPOINT=/path/to/gpt-oss-20b
# export SLIME_SCRIPT_DATA_JSONL=/path/to/train_data_filtered.jsonl

if [[ -z "${SLIME_SCRIPT_HF_CHECKPOINT:-}" || -z "${SLIME_SCRIPT_DATA_JSONL:-}" ]]; then
  echo "ERROR: SLIME_SCRIPT_HF_CHECKPOINT and SLIME_SCRIPT_DATA_JSONL must be set." >&2
  echo "  export SLIME_SCRIPT_HF_CHECKPOINT=/path/to/gpt-oss-20b" >&2
  echo "  export SLIME_SCRIPT_DATA_JSONL=/path/to/train_data_filtered.jsonl" >&2
  echo "  sbatch examples/scaffolding/sbatch_smoke_mit_orcd.sh" >&2
  exit 1
fi

cd /home/rohin/slime
export PYTHONUNBUFFERED=1

# Prefer container runtime (all slime deps preinstalled); fallback to miniforge python only if missing.
if [[ -f "/home/rohin/slime/slime.sif" ]]; then
  # Default bind is only $HOME; checkpoints on ORCD scratch (/orcd/...) are invisible unless bound.
  APPTAINER_BIND_ARGS=(--bind /home/rohin:/home/rohin)
  [[ -d /orcd ]] && APPTAINER_BIND_ARGS+=(--bind /orcd:/orcd)
  apptainer exec --nv \
    "${APPTAINER_BIND_ARGS[@]}" \
    /home/rohin/slime/slime.sif \
    bash -lc '
      set -euo pipefail
      cd /home/rohin/slime
      # Avoid loading host ~/.local packages inside container (can break sglang/transformers compatibility).
      export PYTHONNOUSERSITE=1
      # Login-node PATH often prepends Spack gcc; Triton then JIT-compiles with host gcc against
      # container glibc headers and fails (e.g. bits/libc-header-start.h). Force container toolchain.
      export PATH="/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
      export CC=/usr/bin/gcc
      export CXX=/usr/bin/g++
      export PYTHONPATH="/home/rohin/slime:${PYTHONPATH:-}"
      export PYTHONUNBUFFERED=1
      python3 examples/scaffolding/run_gpt_oss_scaffolding_rl.py --smoke
    '
  exit $?
fi

module load miniforge
export PYTHONPATH="/home/rohin/slime:${PYTHONPATH:-}"

python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' || {
  echo "ERROR: Need Python >= 3.10 (repo target). Got: $(python3 -V 2>&1) ($(command -v python3))" >&2
  echo "Load miniforge before python3: module load miniforge, or provide /home/rohin/slime/slime.sif" >&2
  exit 1
}

python3 examples/scaffolding/run_gpt_oss_scaffolding_rl.py --smoke
