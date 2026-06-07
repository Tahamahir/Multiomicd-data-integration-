#!/bin/bash
#SBATCH --job-name=P31_BEST
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=logs/P31_%j.out
#SBATCH --error=logs/P31_%j.err

module load Anaconda3
source /srv/software/easybuild/software/Anaconda3/2024.06-1/etc/profile.d/conda.sh

conda activate multiomics22

cd ~/Multiomicd-data-integration-

python \
10_analysis/scripts/31_best_model_per_metabolite.py
