#!/bin/bash
#SBATCH --job-name=nul_local
#SBATCH --output=nul_local_%j.log
#SBATCH --time=03:00:00
#SBATCH --qos=default-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate multiomics22
cd ~/Multiomicd-data-integration-
python nul_local.py
