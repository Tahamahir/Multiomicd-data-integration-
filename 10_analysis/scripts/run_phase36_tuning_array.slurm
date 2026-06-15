#!/bin/bash
#SBATCH --job-name=phase36_tune_mb
#SBATCH --output=logs/phase36_tune_%A_%a.out
#SBATCH --error=logs/phase36_tune_%A_%a.err
#SBATCH --array=0-29
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

mkdir -p logs

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"

export QT_XCB_GL_INTEGRATION=""
export MPLBACKEND=Agg

eval "$(conda shell.bash hook)"
conda activate multiomics22

cd /home/taha.mahir/Multiomicd-data-integration-

python 10_analysis/scripts/36_tune_best_model_per_mb_slurm.py \
    --project-root . \
    --task-id ${SLURM_ARRAY_TASK_ID}

echo "End: $(date)"
