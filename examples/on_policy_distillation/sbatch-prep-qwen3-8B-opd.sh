#!/bin/bash -l
#SBATCH --job-name=slime-prep
#SBATCH --output=slime-prep-%j.log
#SBATCH --error=slime-prep-%j.log
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --requeue

set -euo pipefail

module load apptainer/1.4.2

apptainer exec \
    --bind /home/rohin/slime:/root/slime \
    --bind /orcd:/orcd \
    /home/rohin/slime/slime.sif \
    bash /root/slime/examples/on_policy_distillation/prep-qwen3-8B-opd.sh
