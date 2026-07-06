#!/bin/bash
#SBATCH --job-name=phase38_xai
#SBATCH --output=logs/phase38_xai_%A_%a.out
#SBATCH --error=logs/phase38_xai_%A_%a.err
#SBATCH --array=0-29
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate multiomics22

cd /home/taha.mahir/Multiomicd-data-integration-

python 10_analysis/scripts/38_xai_interpretation.py \
    --project-root . \
    --metabolite-index ${SLURM_ARRAY_TASK_ID}
