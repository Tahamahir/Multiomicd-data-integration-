#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 42 - Article-grade validation

Goal:
- Compare MG-only, Soil-only, and MG+Soil integrated pipeline.
- Add null model by permuting metabolite values.
- Produce article-ready validation tables.

Inputs:
    X:
    10_analysis/outputs/phase3_soil_dedup/X_deduplicated.csv

    Y:
    10_analysis/outputs/phase2_preprocessing_fixed/Y_ml_filtered_log1p.csv

    Selected metabolites:
    10_analysis/outputs/phase37_pro_full_pipeline_tuning/best_config_per_metabolite.csv

Outputs:
    10_analysis/outputs/phase42_article_validation/
"""

from pathlib import Path
import argparse
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import SparsePCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(".")
OUT_DIR = PROJECT_ROOT / "10_analysis/outputs/phase42_article_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOIL_PREFIXES = ("soil_", "chem__", "psize__", "moist__", "nitrif__", "denit__")

MI_K = 500
N_COMPONENTS = 75
W_MG = 0.7
W_SOIL = 0.3

N_SPLITS = 5
N_REPEATS = 2
N_PERMUTATIONS = 20


# ============================================================
# DATA LOADING
# ============================================================

def split_blocks(X):
    soil_cols = []
    for c in X.columns:
        cl = str(c).lower().strip()
        if any(cl.startswith(p) for p in SOIL_PREFIXES):
            soil_cols.append(c)

    mg_cols = [c for c in X.columns if c not in soil_cols]
    return mg_cols, soil_cols


def load_data(project_root):
    project_root = Path(project_root)

    x_path = project_root / "10_analysis/outputs/phase3_soil_dedup/X_deduplicated.csv"
    y_path = project_root / "10_analysis/outputs/phase2_preprocessing_fixed/Y_ml_filtered_log1p.csv"
    best_path = project_root / "10_analysis/outputs/phase37_pro_full_pipeline_tuning/best_config_per_metabolite.csv"

    if not x_path.exists():
        raise FileNotFoundError(x_path)
    if not y_path.exists():
        raise FileNotFoundError(y_path)
    if not best_path.exists():
        raise FileNotFoundError(best_path)

    X = pd.read_csv(x_path, low_memory=False)
    Y_all = pd.read_csv(y_path, low_memory=False)
    best = pd.read_csv(best_path)

    metabolites = [m for m in best["metabolite"].tolist() if m in Y_all.columns]
    Y = Y_all[metabolites].copy()

    mg_cols, soil_cols = split_blocks(X)

    print("[INFO] X:", X.shape)
    print("[INFO] Y selected:", Y.shape)
    print("[INFO] MG features:", len(mg_cols))
    print("[INFO] Soil features:", len(soil_cols))

    return X, Y, metabolites, mg_cols, soil_cols


# ============================================================
# PIPELINE HELPERS
# ============================================================

def preprocess_block(X_train_df, X_test_df, strategy="constant"):
    if strategy == "median":
        imputer = SimpleImputer(strategy="median")
    else:
        imputer = SimpleImputer(strategy="constant", fill_value=0)

    scaler = StandardScaler()

    X_train = imputer.fit_transform(X_train_df)
    X_test = imputer.transform(X_test_df)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def select_mi(X_train, y_train, X_test, k):
    if X_train.shape[1] <= k:
        return X_train, X_test

    mi = mutual_info_regression(
        X_train,
        y_train,
        random_state=42,
        discrete_features=False
    )

    mi = np.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0)
    idx = np.argsort(mi)[::-1][:k]

    return X_train[:, idx], X_test[:, idx]


def sparsepca_reduce(X_train, X_test, n_components):
    n_components = min(n_components, X_train.shape[0] - 1, X_train.shape[1])

    if n_components < 1:
        return X_train, X_test

    reducer = SparsePCA(
        n_components=n_components,
        alpha=1,
        random_state=42,
        n_jobs=-1,
        max_iter=500
    )

    X_train_r = reducer.fit_transform(X_train)
    X_test_r = reducer.transform(X_test)

    return X_train_r, X_test_r


def build_rf(seed):
    return RandomForestRegressor(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_metabolite(X, y, mg_cols, soil_cols, metabolite, shuffle=False, seed=42):
    cv = RepeatedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=seed
    )

    if shuffle:
        rng = np.random.default_rng(seed)
        y_eval = rng.permutation(y)
    else:
        y_eval = y.copy()

    predictions = {
        "MG_only": np.zeros(len(y_eval)),
        "Soil_only": np.zeros(len(y_eval)),
        "MG_Soil_late": np.zeros(len(y_eval)),
    }

    for fold_id, (train_idx, test_idx) in enumerate(cv.split(X)):
        y_train = y_eval[train_idx]

        # -----------------------------
        # MG-only branch
        # -----------------------------
        X_train_mg_df = X.iloc[train_idx][mg_cols]
        X_test_mg_df = X.iloc[test_idx][mg_cols]

        X_train_mg, X_test_mg = preprocess_block(
            X_train_mg_df,
            X_test_mg_df,
            strategy="constant"
        )

        X_train_mg, X_test_mg = select_mi(
            X_train_mg,
            y_train,
            X_test_mg,
            MI_K
        )

        X_train_mg, X_test_mg = sparsepca_reduce(
            X_train_mg,
            X_test_mg,
            N_COMPONENTS
        )

        model_mg = build_rf(seed + fold_id)
        model_mg.fit(X_train_mg, y_train)
        pred_mg = model_mg.predict(X_test_mg)

        predictions["MG_only"][test_idx] = pred_mg

        # -----------------------------
        # Soil-only branch
        # -----------------------------
        if len(soil_cols) > 0:
            X_train_soil_df = X.iloc[train_idx][soil_cols]
            X_test_soil_df = X.iloc[test_idx][soil_cols]

            X_train_soil, X_test_soil = preprocess_block(
                X_train_soil_df,
                X_test_soil_df,
                strategy="median"
            )

            model_soil = build_rf(1000 + seed + fold_id)
            model_soil.fit(X_train_soil, y_train)
            pred_soil = model_soil.predict(X_test_soil)

            predictions["Soil_only"][test_idx] = pred_soil
        else:
            pred_soil = np.zeros_like(pred_mg)
            predictions["Soil_only"][test_idx] = pred_soil

        # -----------------------------
        # Late integration MG + Soil
        # -----------------------------
        pred_integrated = W_MG * pred_mg + W_SOIL * pred_soil
        predictions["MG_Soil_late"][test_idx] = pred_integrated

    rows = []

    for model_name, pred in predictions.items():
        r2 = r2_score(y_eval, pred)
        rmse = np.sqrt(mean_squared_error(y_eval, pred))
        mae = mean_absolute_error(y_eval, pred)

        rows.append({
            "metabolite": metabolite,
            "model": model_name,
            "shuffle": shuffle,
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "n_samples": len(y_eval),
            "n_features": X.shape[1],
            "mi_k": MI_K,
            "n_components": N_COMPONENTS,
            "w_mg": W_MG,
            "w_soil": W_SOIL,
        })

    return pd.DataFrame(rows)


# ============================================================
# AGGREGATION
# ============================================================

def summarize(metrics):
    summary = (
        metrics.groupby(["model", "shuffle"])
        .agg(
            mean_r2=("r2", "mean"),
            median_r2=("r2", "median"),
            std_r2=("r2", "std"),
            min_r2=("r2", "min"),
            max_r2=("r2", "max"),
            mean_rmse=("rmse", "mean"),
            mean_mae=("mae", "mean"),
            n_metabolites=("metabolite", "nunique"),
        )
        .reset_index()
        .sort_values(["shuffle", "mean_r2"], ascending=[True, False])
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--null-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_dir = project_root / "10_analysis/outputs/phase42_article_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        files = sorted(out_dir.glob("metrics_task_*.csv"))
        if not files:
            raise FileNotFoundError("No metrics_task_*.csv files found.")

        all_metrics = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        all_metrics.to_csv(out_dir / "baseline_validation_metrics.csv", index=False)

        summary = summarize(all_metrics)
        summary.to_csv(out_dir / "baseline_validation_summary.csv", index=False)

        with pd.ExcelWriter(out_dir / "article_validation_report.xlsx", engine="openpyxl") as writer:
            all_metrics.to_excel(writer, sheet_name="all_metrics", index=False)
            summary.to_excel(writer, sheet_name="summary", index=False)

        print("\n[AGGREGATION DONE]")
        print(summary.to_string(index=False))
        print("\nSaved:", out_dir)
        return

    X, Y, metabolites, mg_cols, soil_cols = load_data(project_root)

    if args.task_id is None:
        print("Number of metabolites:", len(metabolites))
        print("Use --task-id 0 to", len(metabolites) - 1)
        return

    if args.task_id < 0 or args.task_id >= len(metabolites):
        raise ValueError(f"Invalid task-id {args.task_id}")

    metabolite = metabolites[args.task_id]
    y = Y[metabolite].values

    print("[TASK]", args.task_id)
    print("[METABOLITE]", metabolite)

    all_rows = []

    # Real validation
    if not args.null_only:
        print("[RUN] Real baseline validation")
        real_df = evaluate_metabolite(
            X=X,
            y=y,
            mg_cols=mg_cols,
            soil_cols=soil_cols,
            metabolite=metabolite,
            shuffle=False,
            seed=42
        )
        all_rows.append(real_df)

    # Null model validation
    print("[RUN] Null model permutations")
    null_rows = []
    for p in range(N_PERMUTATIONS):
        print(f"  permutation {p+1}/{N_PERMUTATIONS}")
        perm_df = evaluate_metabolite(
            X=X,
            y=y,
            mg_cols=mg_cols,
            soil_cols=soil_cols,
            metabolite=metabolite,
            shuffle=True,
            seed=1000 + p
        )
        perm_df["permutation"] = p
        null_rows.append(perm_df)

    null_df = pd.concat(null_rows, ignore_index=True)
    all_rows.append(null_df)

    result = pd.concat(all_rows, ignore_index=True)

    out_file = out_dir / f"metrics_task_{args.task_id:03d}.csv"
    result.to_csv(out_file, index=False)

    print("[DONE] Saved:", out_file)


if __name__ == "__main__":
    main()
