#!/bin/bash
#SBATCH --job-name=P28_NEST
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/P28_%j.out
#SBATCH --error=logs/P28_%j.err

module load Anaconda3
source /srv/software/easybuild/software/Anaconda3/2024.06-1/etc/profile.d/conda.sh
conda activate multiomics22

cd ~/Multiomicd-data-integration-

python 10_analysis/scripts/28_nested_cv_light.py
