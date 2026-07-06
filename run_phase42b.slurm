#!/bin/bash
#SBATCH --job-name=phase42b_reval
#SBATCH --output=logs/phase42b_%A_%a.out
#SBATCH --error=logs/phase42b_%A_%a.err
#SBATCH --array=0-46
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail
mkdir -p logs

# Parametres surchargeables :
#   sbatch --export=ALL,NPERM=50,NREP=1 run_phase42b.slurm
NPERM="${NPERM:-100}"
NREP="${NREP:-2}"

echo "Job ${SLURM_JOB_ID} | Task ${SLURM_ARRAY_TASK_ID} | Node $(hostname)"
echo "NPERM=${NPERM} NREP=${NREP} | Start: $(date)"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MPLBACKEND=Agg

eval "$(conda shell.bash hook)"
conda activate multiomics22

# >>> ADAPTE CE CHEMIN SI BESOIN <<<
cd /home/taha.mahir/Multiomicd-data-integration-

python 10_analysis/scripts/42b_revalidate_full_fdr.py \
  --project-root . \
  --task-id ${SLURM_ARRAY_TASK_ID} \
  --n-permutations ${NPERM} \
  --n-repeats ${NREP}

echo "End: $(date)"

