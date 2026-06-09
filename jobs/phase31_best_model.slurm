#!/bin/bash
#SBATCH --job-name=P31_BEST
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=logs/P31_%j.out
#SBATCH --error=logs/P31_%j.err

cd /home/taha.mahir/Multiomicd-data-integration-

echo "HOSTNAME:"
hostname

echo "PYTHON USED:"
/home/taha.mahir/.conda/envs/multiomics22/bin/python -c "import sys; print(sys.executable)"

echo "CHECK XGBOOST:"
/home/taha.mahir/.conda/envs/multiomics22/bin/python -c "from xgboost import XGBRegressor; print('xgboost OK')"

echo "START PHASE31"

/home/taha.mahir/.conda/envs/multiomics22/bin/python \
10_analysis/scripts/31_best_model_per_metabolite.py
