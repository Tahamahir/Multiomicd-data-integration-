#!/bin/bash
#SBATCH --job-name=phase37_pro_tune
#SBATCH --output=logs/phase37_pro_%A_%a.out
#SBATCH --error=logs/phase37_pro_%A_%a.err
#SBATCH --array=0-63
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail

mkdir -p logs

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"

export QT_XCB_GL_INTEGRATION=""
export MPLBACKEND=Agg
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

eval "$(conda shell.bash hook)"
conda activate multiomics22

cd /home/taha.mahir/Multiomicd-data-integration-

python 10_analysis/scripts/37_pro_full_pipeline_tuning_array.py \
    --project-root . \
    --task-id ${SLURM_ARRAY_TASK_ID}

echo "End: $(date)"
