#!/bin/bash
#SBATCH --job-name=phase42_articleval
#SBATCH --output=logs/phase42_%A_%a.out
#SBATCH --error=logs/phase42_%A_%a.err
#SBATCH --array=0-29
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail

mkdir -p logs

echo "Job ID: ${SLURM_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MPLBACKEND=Agg

eval "$(conda shell.bash hook)"
conda activate multiomics22

cd /home/taha.mahir/Multiomicd-data-integration-

python 10_analysis/scripts/42_article_grade_validation.py \
  --project-root . \
  --task-id ${SLURM_ARRAY_TASK_ID}

echo "End: $(date)"

