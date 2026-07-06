# Phase 22B - Advanced pipeline optimization

This folder extends Phase 22A. It adds ExtraTrees, lightweight tuned XGBoost, LassoCV, ElasticNetCV, PLS with several components, MiniBatchSparsePCA, late integration, and a Boruta-like RF shadow feature selection.

## Install/check environment

```bash
conda activate multiomics22
python -c "import numpy, scipy, sklearn, xgboost; print('ok')"
```

## Copy files into the project

From the project root:

```bash
mkdir -p 10_analysis/scripts/22_pipeline_optimization
cp phase22B_advanced/scripts/22B_pipeline_optimization_advanced.py 10_analysis/scripts/22_pipeline_optimization/
cp phase22B_advanced/scripts/phase22B_experiments_list.txt 10_analysis/scripts/22_pipeline_optimization/
```

## List available experiments

```bash
python 10_analysis/scripts/22_pipeline_optimization/22B_pipeline_optimization_advanced.py --project-root . --list
```

## Run one experiment first

```bash
python 10_analysis/scripts/22_pipeline_optimization/22B_pipeline_optimization_advanced.py --project-root . --only B01_extratrees_baseline
```

## Run all experiments locally

```bash
python 10_analysis/scripts/22_pipeline_optimization/22B_pipeline_optimization_advanced.py --project-root .
```

## Aggregate existing outputs

```bash
python 10_analysis/scripts/22_pipeline_optimization/22B_pipeline_optimization_advanced.py --project-root . --aggregate-only
```

Outputs are written to:

```text
10_analysis/outputs/phase22B_pipeline_optimization/
```

## SLURM array

Copy the SLURM file:

```bash
cp phase22B_advanced/slurm/run_phase22B_array.slurm 10_analysis/scripts/22_pipeline_optimization/
```

Then edit the resource lines if needed and run:

```bash
sbatch 10_analysis/scripts/22_pipeline_optimization/run_phase22B_array.slurm
```

Important: B16 Boruta-like can be heavy. If needed, change `#SBATCH --array=1-15` to skip B16 first.
