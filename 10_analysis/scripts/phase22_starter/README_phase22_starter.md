# Phase 22 starter

Copy files into your repository:

```bash
mkdir -p 10_analysis/scripts/22_pipeline_optimization
cp scripts/22_pipeline_optimization_basic.py 10_analysis/scripts/22_pipeline_optimization/
cp scripts/experiments_list.txt 10_analysis/scripts/22_pipeline_optimization/
```

Run locally:

```bash
python 10_analysis/scripts/22_pipeline_optimization/22_pipeline_optimization_basic.py --project-root .
```

Run one experiment:

```bash
python 10_analysis/scripts/22_pipeline_optimization/22_pipeline_optimization_basic.py --project-root . --only E01_rare5_rf
```

Expected outputs:

```text
10_analysis/outputs/phase22_pipeline_optimization/experiment_summary.csv
10_analysis/outputs/phase22_pipeline_optimization/metrics_per_metabolite.csv
```

For HPC, adapt `slurm/run_phase22_array.slurm` with your project path and environment.
