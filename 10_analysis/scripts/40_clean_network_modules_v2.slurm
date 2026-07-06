#!/bin/bash
#SBATCH --job-name=phase40_clean
#SBATCH --output=logs/phase40_%j.out
#SBATCH --error=logs/phase40_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate multiomics22

cd /home/taha.mahir/Multiomicd-data-integration-

python 10_analysis/scripts/40_clean_network_modules_v2.py
