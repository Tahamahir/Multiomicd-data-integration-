#!/bin/bash
#SBATCH --job-name=P29_TOP30
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=logs/P29_TOP30_%j.out
#SBATCH --error=logs/P29_TOP30_%j.err

module load Anaconda3
source /srv/software/easybuild/software/Anaconda3/2024.06-1/etc/profile.d/conda.sh
conda activate multiomics22

cd ~/Multiomicd-data-integration-

python 10_analysis/scripts/29_top30_metabolites_champion.py
