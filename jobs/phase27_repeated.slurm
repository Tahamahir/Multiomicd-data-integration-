#!/bin/bash
#SBATCH --job-name=P27_REP
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/P27_REP_%j.out
#SBATCH --error=logs/P27_REP_%j.err

module load Anaconda3
source /srv/software/easybuild/software/Anaconda3/2024.06-1/etc/profile.d/conda.sh
conda activate multiomics22

cd ~/Multiomicd-data-integration-

python 10_analysis/scripts/27_repeated_nested_cv_champion.py \
--project-root . \
--mode repeated
