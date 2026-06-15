#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor

try:
    import shap
    HAS_SHAP = True
except:
    HAS_SHAP = False


def load_best_config(project_root):
    path = Path(project_root) / "10_analysis/outputs/phase37_pro_full_pipeline_tuning/best_config_per_metabolite.csv"
    return pd.read_csv(path)


def load_data(project_root):
    X = pd.read_csv(Path(project_root) / "10_analysis/outputs/phase3_soil_dedup/X_deduplicated.csv")
    Y = pd.read_csv(Path(project_root) / "10_analysis/outputs/phase2_preprocessing_fixed/Y_ml_filtered_log1p.csv")
    return X, Y


def train_global_model(X, y):
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model


def run_xai(X, Y, metabolite, out_dir):
    y = Y[metabolite].values

    model = train_global_model(X, y)

    # =====================
    # 1. Permutation importance
    # =====================
    print("Computing permutation importance...")
    perm = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=-1)

    feat_imp = pd.DataFrame({
        "feature": X.columns,
        "importance": perm.importances_mean
    }).sort_values("importance", ascending=False)

    feat_imp.to_csv(out_dir / f"perm_importance_{metabolite}.csv", index=False)

    plt.figure(figsize=(10,6))
    plt.barh(feat_imp["feature"][:20][::-1], feat_imp["importance"][:20][::-1])
    plt.title("Top 20 Feature Importance (Permutation)")
    plt.tight_layout()
    plt.savefig(out_dir / f"perm_importance_{metabolite}.png")

    # =====================
    # 2. SHAP analysis
    # =====================
    if HAS_SHAP:
        print("Computing SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        shap.summary_plot(shap_values, X, show=False)
        plt.savefig(out_dir / f"shap_summary_{metabolite}.png")
        plt.close()

    print("Done:", metabolite)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--metabolite-index", type=int, default=0)
    args = parser.parse_args()

    project_root = Path(args.project_root)

    out_dir = project_root / "10_analysis/outputs/phase38_xai"
    out_dir.mkdir(parents=True, exist_ok=True)

    X, Y = load_data(project_root)

    metabolites = Y.columns.tolist()
    metabolite = metabolites[args.metabolite_index]

    print("Metabolite:", metabolite)
    print("X shape:", X.shape)

    run_xai(X, Y, metabolite, out_dir)


if __name__ == "__main__":
    main()
