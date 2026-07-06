#!/bin/bash
#SBATCH --job-name=phase22_opt
#SBATCH --output=logs/phase22_%A_%a.out
#SBATCH --error=logs/phase22_%A_%a.err
#SBATCH --array=1-12
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Adapt these lines to your lab server environment
# module load python/3.10
# source ~/envs/multiomics/bin/activate

PROJECT_ROOT=/path/to/Multiomicd-data-integration-
SCRIPT=$PROJECT_ROOT/10_analysis/scripts/22_pipeline_optimization/22_pipeline_optimization_basic.py
CONFIG_FILE=$PROJECT_ROOT/10_analysis/scripts/22_pipeline_optimization/experiments_list.txt
EXP_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CONFIG_FILE")

mkdir -p logs
python "$SCRIPT" --project-root "$PROJECT_ROOT" --only "$EXP_ID"
