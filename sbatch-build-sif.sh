#!/bin/bash -l
#SBATCH --job-name=slime-sif-build
#SBATCH --output=slime-sif-build-%j.log
#SBATCH --error=slime-sif-build-%j.log
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1

set -euo pipefail

module load apptainer/1.4.2

apptainer build /home/rohin/slime/slime.sif docker://slimerl/slime:latest

echo "Done. SIF file at /home/rohin/slime/slime.sif"
