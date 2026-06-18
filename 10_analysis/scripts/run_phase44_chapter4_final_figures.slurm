#!/bin/bash
#SBATCH --job-name=phase44_ch4figs
#SBATCH --output=logs/phase44_ch4figs_%j.out
#SBATCH --error=logs/phase44_ch4figs_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate multiomics22

cd /home/taha.mahir/Multiomicd-data-integration-

export MPLBACKEND=Agg

python 10_analysis/scripts/44_chapter4_final_figures.py
