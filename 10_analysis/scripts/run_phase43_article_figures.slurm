#!/bin/bash
#SBATCH --job-name=phase43_figures
#SBATCH --output=logs/phase43_%j.out
#SBATCH --error=logs/phase43_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate multiomics22

cd /home/taha.mahir/Multiomicd-data-integration-

export MPLBACKEND=Agg

python 10_analysis/scripts/43_article_validation_figures.py
