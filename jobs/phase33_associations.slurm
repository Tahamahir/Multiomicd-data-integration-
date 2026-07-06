#!/bin/bash
#SBATCH --job-name=P33_ASSOC
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/P33_%j.out
#SBATCH --error=logs/P33_%j.err

cd /home/taha.mahir/Multiomicd-data-integration-

/home/taha.mahir/.conda/envs/multiomics22/bin/python \
10_analysis/scripts/33_mg_mb_associations.py
