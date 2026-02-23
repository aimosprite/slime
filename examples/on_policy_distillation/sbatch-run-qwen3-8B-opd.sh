#!/bin/bash -l
#SBATCH --job-name=slime-run
#SBATCH --output=slime-run-%j.log
#SBATCH --error=slime-run-%j.log
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --requeue
#SBATCH --signal=B:TERM@120

set -euo pipefail

module load apptainer/1.4.2

# Forward SIGTERM into the apptainer child so the inner script can clean up
trap 'echo "SLURM PREEMPTION: forwarding SIGTERM to child"; kill -TERM $CHILD_PID 2>/dev/null; wait $CHILD_PID 2>/dev/null' TERM

apptainer exec --nv --cleanenv \
    --env "HOME=/root" \
    --bind /home/rohin/slime:/root/slime \
    --bind /orcd:/orcd \
    /home/rohin/slime/slime.sif \
    bash /root/slime/examples/on_policy_distillation/run-qwen3-8B-opd.sh &
CHILD_PID=$!
wait $CHILD_PID
