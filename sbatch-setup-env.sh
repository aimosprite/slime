#!/bin/bash -l
#SBATCH --job-name=slime-setup
#SBATCH --output=slime-setup-%j.log
#SBATCH --error=slime-setup-%j.log
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=05:50:00
#SBATCH --ntasks=1
#SBATCH --nodes=1

set -euo pipefail

# If 'module' still isn't defined for some reason, this usually fixes it:
if ! command -v module >/dev/null 2>&1; then
  # common locations; harmless if missing
  [[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh
  [[ -f /usr/share/Modules/init/bash ]] && source /usr/share/Modules/init/bash
fi

module load cuda
module list

source /home/rohin/slime/.venv/bin/activate

echo "CUDA-related sanity:"
command -v nvidia-smi && nvidia-smi
command -v nvcc && nvcc --version || true
echo "CUDA_HOME=${CUDA_HOME-}"
echo "PATH=$PATH"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH-}"

source /home/rohin/slime/setup-env.sh