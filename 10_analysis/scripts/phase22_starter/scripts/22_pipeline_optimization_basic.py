#!/usr/bin/env python3
"""
Phase 22 - Basic comparative pipeline optimization

Goal:
Compare a first wave of pipeline variants:
- rare feature filtering: current / 5% / 10%
- feature selection: none / mutual information / Spearman / Lasso-selected features
- dimensionality reduction: none / PCA / SparsePCA
- integration: early / intermediate
- models: RF / XGBoost / ElasticNet / Lasso / PLS

Inputs expected from the current project:
- 10_analysis/outputs/phase3_soil_dedup/X_deduplicated.csv
- 10_analysis/outputs/phase2_preprocessing_fixed/Y_ml_filtered_log1p.csv
- 10_analysis/outputs/phase17_final_best_model_pipeline/best_model_per_metabolite_final.csv

Outputs:
- 10_analysis/outputs/phase22_pipeline_optimization/experiment_summary.csv
- 10_analysis/outputs/phase22_pipeline_optimization/metrics_per_metabolite.csv
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, SparsePCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False


@dataclass
class Experiment:
    experiment_id: str
    rare_filter: float
    feature_selection: str
    fs_top_k: Optional[int]
    dim_reduction: str
    n_components: Optional[int]
    integration: str
    model: str


def load_data(project_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    x_path = project_root / "10_analysis/outputs/phase3_soil_dedup/X_deduplicated.csv"
    y_path = project_root / "10_analysis/outputs/phase2_preprocessing_fixed/Y_ml_filtered_log1p.csv"
    mb_path = project_root / "10_analysis/outputs/phase17_final_best_model_pipeline/best_model_per_metabolite_final.csv"

    X = pd.read_csv(x_path)
    Y = pd.read_csv(y_path)
    best = pd.read_csv(mb_path)
    metabolites = [m for m in best["metabolite"].tolist() if m in Y.columns]

    # same rows/order already expected from previous pipeline
    assert len(X) == len(Y), f"X and Y do not have same number of rows: {X.shape}, {Y.shape}"
    return X, Y[metabolites], metabolites


def split_blocks(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    # soil variables in this project usually contain soil/chem/psize/nitrif or units in names.
    soil_keywords = ["soil", "chem__", "psize__", "nitrif", "no3", "nh4", "ph", "mg_kg", "water", "clay", "sand", "silt"]
    soil_cols = []
    for c in X.columns:
        cl = c.lower()
        if any(k in cl for k in soil_keywords):
            soil_cols.append(c)
    mg_cols = [c for c in X.columns if c not in soil_cols]
    return mg_cols, soil_cols


def rare_feature_filter(X: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if threshold <= 0:
        return X.copy()
    # presence = non-zero and not NaN
    presence = (X.fillna(0) != 0).mean(axis=0)
    keep = presence >= threshold
    return X.loc[:, keep].copy()


def select_features(X_train: np.ndarray, y_train: np.ndarray, X_all: np.ndarray, method: str, top_k: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    n_features = X_train.shape[1]
    if method == "none" or top_k is None or top_k >= n_features:
        return X_train, X_all

    top_k = min(top_k, n_features)

    if method == "mutual_info":
        scores = mutual_info_regression(X_train, y_train, random_state=42)
        idx = np.argsort(np.nan_to_num(scores))[-top_k:]

    elif method == "spearman":
        scores = []
        for j in range(n_features):
            try:
                r, _ = spearmanr(X_train[:, j], y_train)
                scores.append(abs(r) if np.isfinite(r) else 0.0)
            except Exception:
                scores.append(0.0)
        idx = np.argsort(scores)[-top_k:]

    elif method == "lasso_fs":
        # Use Lasso only to select features inside each training fold.
        selector = make_pipeline(StandardScaler(), Lasso(alpha=0.001, max_iter=10000, random_state=42))
        selector.fit(X_train, y_train)
        coef = np.abs(selector.named_steps["lasso"].coef_)
        if np.sum(coef > 0) == 0:
            idx = np.argsort(coef)[-top_k:]
        else:
            idx = np.argsort(coef)[-min(top_k, int(np.sum(coef > 0))):]

    else:
        raise ValueError(f"Unknown feature selection method: {method}")

    return X_train[:, idx], X_all[:, idx]


def build_model(name: str):
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
    if name == "xgboost":
        if not HAS_XGB:
            raise RuntimeError("xgboost is not installed")
        return XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=4,
        )
    if name == "elasticnet":
        return make_pipeline(StandardScaler(), ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=20000, random_state=42))
    if name == "lasso":
        return make_pipeline(StandardScaler(), Lasso(alpha=0.01, max_iter=20000, random_state=42))
    if name == "pls":
        return make_pipeline(StandardScaler(), PLSRegression(n_components=5))
    raise ValueError(f"Unknown model: {name}")


def apply_dim_reduction(X_train: np.ndarray, X_test: np.ndarray, method: str, n_components: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    if method == "none":
        return X_train, X_test
    n_components = int(n_components or 50)
    n_components = max(2, min(n_components, X_train.shape[0] - 1, X_train.shape[1]))
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
    elif method == "sparse_pca":
        reducer = SparsePCA(n_components=n_components, alpha=1.0, random_state=42, n_jobs=-1, max_iter=1000)
    else:
        raise ValueError(f"Unknown dim reduction: {method}")
    return reducer.fit_transform(X_train_s), reducer.transform(X_test_s)


def evaluate_experiment(X_df: pd.DataFrame, Y: pd.DataFrame, exp: Experiment, cv: KFold, mg_cols: List[str], soil_cols: List[str]) -> Tuple[pd.DataFrame, Dict]:
    start = time.time()
    X_f = rare_feature_filter(X_df, exp.rare_filter)
    mg_cols_f = [c for c in mg_cols if c in X_f.columns]
    soil_cols_f = [c for c in soil_cols if c in X_f.columns]

    rows = []
    for metabolite in Y.columns:
        y = Y[metabolite].values.astype(float)
        preds = np.zeros_like(y, dtype=float)

        for train_idx, test_idx in cv.split(X_f):
            if exp.integration == "early":
                X_train_df = X_f.iloc[train_idx]
                X_test_df = X_f.iloc[test_idx]

                imp = SimpleImputer(strategy="median")
                X_train = imp.fit_transform(X_train_df)
                X_test = imp.transform(X_test_df)

                X_train_fs, X_all_fs = select_features(
                    X_train, y[train_idx], np.vstack([X_train, X_test]), exp.feature_selection, exp.fs_top_k
                )
                X_train_fs = X_all_fs[: len(train_idx)]
                X_test_fs = X_all_fs[len(train_idx):]
                X_train_final, X_test_final = apply_dim_reduction(X_train_fs, X_test_fs, exp.dim_reduction, exp.n_components)

            elif exp.integration == "intermediate":
                # Dimensionality reduction only on MG block; soil remains explicit.
                X_train_mg = X_f.loc[X_f.index[train_idx], mg_cols_f]
                X_test_mg = X_f.loc[X_f.index[test_idx], mg_cols_f]
                X_train_soil = X_f.loc[X_f.index[train_idx], soil_cols_f] if soil_cols_f else pd.DataFrame(index=train_idx)
                X_test_soil = X_f.loc[X_f.index[test_idx], soil_cols_f] if soil_cols_f else pd.DataFrame(index=test_idx)

                imp_mg = SimpleImputer(strategy="median")
                mg_train = imp_mg.fit_transform(X_train_mg)
                mg_test = imp_mg.transform(X_test_mg)

                mg_train_fs, mg_all_fs = select_features(
                    mg_train, y[train_idx], np.vstack([mg_train, mg_test]), exp.feature_selection, exp.fs_top_k
                )
                mg_train_fs = mg_all_fs[: len(train_idx)]
                mg_test_fs = mg_all_fs[len(train_idx):]
                mg_train_red, mg_test_red = apply_dim_reduction(mg_train_fs, mg_test_fs, exp.dim_reduction, exp.n_components)

                if soil_cols_f:
                    imp_soil = SimpleImputer(strategy="median")
                    soil_train = imp_soil.fit_transform(X_train_soil)
                    soil_test = imp_soil.transform(X_test_soil)
                    X_train_final = np.hstack([mg_train_red, soil_train])
                    X_test_final = np.hstack([mg_test_red, soil_test])
                else:
                    X_train_final, X_test_final = mg_train_red, mg_test_red
            else:
                raise ValueError(f"Unknown integration: {exp.integration}")

            model = build_model(exp.model)
            model.fit(X_train_final, y[train_idx])
            preds[test_idx] = model.predict(X_test_final).ravel()

        r2 = r2_score(y, preds)
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        mae = float(mean_absolute_error(y, preds))
        try:
            sp, sp_p = spearmanr(y, preds)
        except Exception:
            sp, sp_p = np.nan, np.nan
        rows.append({
            "experiment_id": exp.experiment_id,
            "metabolite": metabolite,
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "spearman": sp,
            "spearman_p": sp_p,
            "n_features_after_filter": X_f.shape[1],
        })

    per_mb = pd.DataFrame(rows)
    summary = asdict(exp)
    summary.update({
        "n_samples": X_f.shape[0],
        "n_features_after_filter": X_f.shape[1],
        "n_mg_features_after_filter": len(mg_cols_f),
        "n_soil_features_after_filter": len(soil_cols_f),
        "n_metabolites": Y.shape[1],
        "mean_r2": per_mb["r2"].mean(),
        "median_r2": per_mb["r2"].median(),
        "max_r2": per_mb["r2"].max(),
        "std_r2": per_mb["r2"].std(),
        "n_r2_gt_0": int((per_mb["r2"] > 0).sum()),
        "n_r2_gt_02": int((per_mb["r2"] > 0.2).sum()),
        "n_r2_gt_04": int((per_mb["r2"] > 0.4).sum()),
        "n_r2_gt_06": int((per_mb["r2"] > 0.6).sum()),
        "runtime_sec": round(time.time() - start, 2),
    })
    return per_mb, summary


def build_first_wave() -> List[Experiment]:
    return [
        Experiment("E00_baseline_rf", 0.00, "none", None, "none", None, "early", "rf"),
        Experiment("E01_rare5_rf", 0.05, "none", None, "none", None, "early", "rf"),
        Experiment("E02_rare10_rf", 0.10, "none", None, "none", None, "early", "rf"),
        Experiment("E03_rare5_mi500_rf", 0.05, "mutual_info", 500, "none", None, "early", "rf"),
        Experiment("E04_rare5_spearman500_rf", 0.05, "spearman", 500, "none", None, "early", "rf"),
        Experiment("E05_rare5_lassoFS500_elasticnet", 0.05, "lasso_fs", 500, "none", None, "early", "elasticnet"),
        Experiment("E06_rare5_pca50_pls_intermediate", 0.05, "none", None, "pca", 50, "intermediate", "pls"),
        Experiment("E07_rare5_sparsepca30_pls_intermediate", 0.05, "none", None, "sparse_pca", 30, "intermediate", "pls"),
        Experiment("E08_rare5_mi500_xgboost", 0.05, "mutual_info", 500, "none", None, "early", "xgboost"),
        Experiment("E09_rare5_lasso", 0.05, "none", None, "none", None, "early", "lasso"),
        Experiment("E10_rare5_elasticnet", 0.05, "none", None, "none", None, "early", "elasticnet"),
        Experiment("E11_rare5_pls", 0.05, "none", None, "none", None, "early", "pls"),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--only", default=None, help="Optional experiment_id to run one experiment only")
    parser.add_argument("--n-jobs", default=1, type=int, help="Reserved for future use")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_dir = project_root / "10_analysis/outputs/phase22_pipeline_optimization"
    out_dir.mkdir(parents=True, exist_ok=True)

    X, Y, metabolites = load_data(project_root)
    mg_cols, soil_cols = split_blocks(X)

    experiments = build_first_wave()
    if args.only:
        experiments = [e for e in experiments if e.experiment_id == args.only]
        if not experiments:
            raise ValueError(f"Experiment not found: {args.only}")

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    all_per_mb = []
    summaries = []

    print(f"Loaded X={X.shape}, Y={Y.shape}")
    print(f"Detected MG features={len(mg_cols)}, soil features={len(soil_cols)}")
    print(f"Running {len(experiments)} experiments")

    for exp in experiments:
        print(f"\n[RUN] {exp.experiment_id}")
        try:
            per_mb, summary = evaluate_experiment(X, Y, exp, cv, mg_cols, soil_cols)
            all_per_mb.append(per_mb)
            summaries.append(summary)
            print(json.dumps({k: summary[k] for k in ["experiment_id", "mean_r2", "median_r2", "n_r2_gt_02", "runtime_sec"]}, indent=2))
            per_mb.to_csv(out_dir / f"{exp.experiment_id}_metrics_per_metabolite.csv", index=False)
        except Exception as e:
            print(f"[ERROR] {exp.experiment_id}: {e}")
            fail = asdict(exp)
            fail.update({"error": str(e)})
            summaries.append(fail)

        pd.DataFrame(summaries).to_csv(out_dir / "experiment_summary.csv", index=False)
        if all_per_mb:
            pd.concat(all_per_mb, ignore_index=True).to_csv(out_dir / "metrics_per_metabolite.csv", index=False)

    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
