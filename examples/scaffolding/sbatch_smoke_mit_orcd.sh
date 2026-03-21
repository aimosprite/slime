#!/bin/bash -l
#SBATCH --job-name=slime-scaffolding-smoke
#SBATCH --output=/home/rohin/slime/examples/scaffolding/slime-scaffolding-smoke-%j.log
#SBATCH --error=/home/rohin/slime/examples/scaffolding/slime-scaffolding-smoke-%j.log
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=5:59:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --requeue

set -euo pipefail

module load apptainer/1.4.2
module load cuda

# --- set before submit or edit here ---
# export SLIME_SCRIPT_HF_CHECKPOINT=/path/to/gpt-oss-20b
# export SLIME_SCRIPT_DATA_JSONL=/path/to/train_data_filtered.jsonl

cd /home/rohin/slime
export PYTHONPATH="/home/rohin/slime:${PYTHONPATH:-}"

python3 examples/scaffolding/run_gpt_oss_scaffolding_rl.py --smoke
