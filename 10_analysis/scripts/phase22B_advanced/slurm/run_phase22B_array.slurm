#!/bin/bash
#SBATCH --job-name=phase22B
#SBATCH --output=10_analysis/outputs/phase22B_pipeline_optimization/logs/%x_%A_%a.out
#SBATCH --error=10_analysis/outputs/phase22B_pipeline_optimization/logs/%x_%A_%a.err
#SBATCH --array=1-16
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

cd ~/Multiomicd-data-integration-
mkdir -p 10_analysis/outputs/phase22B_pipeline_optimization/logs
source ~/anaconda3/etc/profile.d/conda.sh
conda activate multiomics22

EXP_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" 10_analysis/scripts/22_pipeline_optimization/phase22B_experiments_list.txt)
echo "Running ${EXP_ID}"
python 10_analysis/scripts/22_pipeline_optimization/22B_pipeline_optimization_advanced.py --project-root . --only "${EXP_ID}"

# Rebuild aggregate table after each job. This is safe if multiple jobs do it.
python 10_analysis/scripts/22_pipeline_optimization/22B_pipeline_optimization_advanced.py --project-root . --aggregate-only
