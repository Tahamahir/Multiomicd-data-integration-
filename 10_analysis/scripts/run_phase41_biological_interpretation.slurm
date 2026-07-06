#!/bin/bash
#SBATCH --job-name=phase41_biointerp
#SBATCH --output=logs/phase41_%j.out
#SBATCH --error=logs/phase41_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate multiomics22

cd /home/taha.mahir/Multiomicd-data-integration-

python 10_analysis/scripts/41_biological_interpretation.py
