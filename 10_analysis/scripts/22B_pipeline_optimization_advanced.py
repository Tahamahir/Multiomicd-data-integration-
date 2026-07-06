#!/usr/bin/env python3
"""
Phase 22B - Advanced comparative pipeline optimization

This script extends Phase 22A without overwriting previous results.
It tests a second wave of pipeline variants:
- ExtraTrees baseline / rare filtering
- lightweight tuned XGBoost
- LassoCV / ElasticNetCV with scaling
- PLS with different n_components
- MiniBatchSparsePCA + RF
- late integration: MG-only model + soil-only model averaged
- optional Boruta-like feature selection using RF shadow features

Inputs:
- 10_analysis/outputs/phase3_soil_dedup/X_deduplicated.csv
- 10_analysis/outputs/phase2_preprocessing_fixed/Y_ml_filtered_log1p.csv
- 10_analysis/outputs/phase17_final_best_model_pipeline/best_model_per_metabolite_final.csv

Outputs:
- 10_analysis/outputs/phase22B_pipeline_optimization/<experiment_id>_metrics_per_metabolite.csv
- 10_analysis/outputs/phase22B_pipeline_optimization/summary_<experiment_id>.csv
- 10_analysis/outputs/phase22B_pipeline_optimization/experiment_summary.csv
- 10_analysis/outputs/phase22B_pipeline_optimization/metrics_per_metabolite.csv
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, MiniBatchSparsePCA
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LassoCV, ElasticNet, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


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

    if not x_path.exists():
        raise FileNotFoundError(f"Missing X file: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing Y file: {y_path}")
    if not mb_path.exists():
        raise FileNotFoundError(f"Missing best metabolite file: {mb_path}")

    X = pd.read_csv(x_path)
    Y = pd.read_csv(y_path)
    best = pd.read_csv(mb_path)
    metabolites = [m for m in best["metabolite"].tolist() if m in Y.columns]
    if len(X) != len(Y):
        raise ValueError(f"X and Y do not have same number of rows: X={X.shape}, Y={Y.shape}")
    return X, Y[metabolites], metabolites

def split_blocks(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    soil_prefixes = [
        "soil_",
        "chem__",
        "psize__",
        "moist__",
        "nitrif__",
        "denit__",
    ]

    soil_cols = []
    for c in X.columns:
        cl = str(c).lower().strip()
        if any(cl.startswith(prefix) for prefix in soil_prefixes):
            soil_cols.append(c)

    mg_cols = [c for c in X.columns if c not in soil_cols]

    return mg_cols, soil_cols

def rare_feature_filter(X: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if threshold <= 0:
        return X.copy()
    presence = (X.fillna(0) != 0).mean(axis=0)
    keep = presence >= threshold
    return X.loc[:, keep].copy()


def _safe_top_idx(scores: np.ndarray, top_k: int) -> np.ndarray:
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    top_k = min(top_k, len(scores))
    return np.argsort(scores)[-top_k:]


def select_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_all: np.ndarray,
    method: str,
    top_k: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    n_features = X_train.shape[1]
    if method == "none" or top_k is None or top_k >= n_features:
        return X_train, X_all
    top_k = min(top_k, n_features)

    if method == "mutual_info":
        scores = mutual_info_regression(X_train, y_train, random_state=42)
        idx = _safe_top_idx(scores, top_k)

    elif method == "spearman":
        scores = np.zeros(n_features)
        for j in range(n_features):
            try:
                r, _ = spearmanr(X_train[:, j], y_train)
                scores[j] = abs(r) if np.isfinite(r) else 0.0
            except Exception:
                scores[j] = 0.0
        idx = _safe_top_idx(scores, top_k)

    elif method == "lasso_fs":
        selector = make_pipeline(
            StandardScaler(),
            LassoCV(alphas=np.logspace(-4, 0, 20), cv=3, max_iter=10000, random_state=42, n_jobs=-1),
        )
        selector.fit(X_train, y_train)
        coef = np.abs(selector.named_steps["lassocv"].coef_)
        nonzero = np.where(coef > 1e-10)[0]
        idx = nonzero if 0 < len(nonzero) <= top_k else _safe_top_idx(coef, top_k)

    elif method == "elasticnet_fs":
        selector = make_pipeline(
            StandardScaler(),
            ElasticNetCV(
                l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                alphas=np.logspace(-4, 0, 20),
                cv=3,
                max_iter=10000,
                random_state=42,
                n_jobs=-1,
            ),
        )
        selector.fit(X_train, y_train)
        coef = np.abs(selector.named_steps["elasticnetcv"].coef_)
        nonzero = np.where(coef > 1e-10)[0]
        idx = nonzero if 0 < len(nonzero) <= top_k else _safe_top_idx(coef, top_k)

    elif method == "boruta_light":
        # Fast Boruta-like feature screen: compare real feature importances with permuted shadow features.
        rng = np.random.default_rng(42)
        shadow = X_train.copy()
        for j in range(shadow.shape[1]):
            rng.shuffle(shadow[:, j])
        X_shadow = np.hstack([X_train, shadow])
        rf = RandomForestRegressor(
            n_estimators=150,
            max_depth=None,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_shadow, y_train)
        imp_real = rf.feature_importances_[:n_features]
        imp_shadow = rf.feature_importances_[n_features:]
        threshold = np.max(imp_shadow) if len(imp_shadow) else 0.0
        selected = np.where(imp_real > threshold)[0]
        if len(selected) == 0:
            idx = _safe_top_idx(imp_real, top_k)
        elif len(selected) > top_k:
            idx = selected[np.argsort(imp_real[selected])[-top_k:]]
        else:
            idx = selected

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
    if name == "extratrees":
        return ExtraTreesRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
    if name == "xgboost_light":
        if not HAS_XGB:
            raise RuntimeError("xgboost is not installed")
        return XGBRegressor(
            n_estimators=500,
            learning_rate=0.025,
            max_depth=2,
            min_child_weight=3,
            subsample=0.75,
            colsample_bytree=0.75,
            reg_alpha=0.5,
            reg_lambda=5.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=4,
        )
    if name == "xgboost_medium":
        if not HAS_XGB:
            raise RuntimeError("xgboost is not installed")
        return XGBRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=2.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=4,
        )
    if name == "elasticnet_cv":
        return make_pipeline(
            StandardScaler(),
            ElasticNetCV(
                l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                alphas=np.logspace(-4, 1, 30),
                cv=3,
                max_iter=20000,
                random_state=42,
                n_jobs=-1,
            ),
        )
    if name == "lasso_cv":
        return make_pipeline(
            StandardScaler(),
            LassoCV(alphas=np.logspace(-4, 1, 30), cv=3, max_iter=20000, random_state=42, n_jobs=-1),
        )
    if name.startswith("pls"):
        n_comp = int(name.replace("pls", ""))
        return make_pipeline(StandardScaler(), PLSRegression(n_components=n_comp))
    raise ValueError(f"Unknown model: {name}")


def apply_dim_reduction(
    X_train: np.ndarray,
    X_test: np.ndarray,
    method: str,
    n_components: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    if method == "none":
        return X_train, X_test
    n_components = int(n_components or 50)
    n_components = max(2, min(n_components, X_train.shape[0] - 1, X_train.shape[1]))

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
    elif method == "minibatch_sparse_pca":
        reducer = MiniBatchSparsePCA(
            n_components=n_components,
            alpha=1.0,
            ridge_alpha=0.01,
            batch_size=min(20, X_train_s.shape[0]),
            max_iter=100,
            tol=1e-2,
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown dim reduction: {method}")
    return reducer.fit_transform(X_train_s), reducer.transform(X_test_s)


def prepare_xy_for_fold(
    X_f: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    exp: Experiment,
    mg_cols_f: List[str],
    soil_cols_f: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
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
        return apply_dim_reduction(X_train_fs, X_test_fs, exp.dim_reduction, exp.n_components)

    if exp.integration == "intermediate":
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
            return np.hstack([mg_train_red, soil_train]), np.hstack([mg_test_red, soil_test])
        return mg_train_red, mg_test_red

    raise ValueError(f"Unknown integration for prepare_xy_for_fold: {exp.integration}")


def evaluate_experiment(
    X_df: pd.DataFrame,
    Y: pd.DataFrame,
    exp: Experiment,
    cv: KFold,
    mg_cols: List[str],
    soil_cols: List[str],
) -> Tuple[pd.DataFrame, Dict]:
    start = time.time()
    X_f = rare_feature_filter(X_df, exp.rare_filter)
    mg_cols_f = [c for c in mg_cols if c in X_f.columns]
    soil_cols_f = [c for c in soil_cols if c in X_f.columns]

    rows = []
    for metabolite in Y.columns:
        y = Y[metabolite].values.astype(float)
        preds = np.zeros_like(y, dtype=float)

        for train_idx, test_idx in cv.split(X_f):
            if exp.integration == "late":
                # Late integration = model on MG + model on soil, average predictions.
                X_train_mg = X_f.loc[X_f.index[train_idx], mg_cols_f]
                X_test_mg = X_f.loc[X_f.index[test_idx], mg_cols_f]
                X_train_soil = X_f.loc[X_f.index[train_idx], soil_cols_f]
                X_test_soil = X_f.loc[X_f.index[test_idx], soil_cols_f]

                imp_mg = SimpleImputer(strategy="median")
                mg_train = imp_mg.fit_transform(X_train_mg)
                mg_test = imp_mg.transform(X_test_mg)

                mg_train_fs, mg_all_fs = select_features(
                    mg_train, y[train_idx], np.vstack([mg_train, mg_test]), exp.feature_selection, exp.fs_top_k
                )
                mg_train_fs = mg_all_fs[: len(train_idx)]
                mg_test_fs = mg_all_fs[len(train_idx):]

                model_mg = build_model(exp.model)
                model_mg.fit(mg_train_fs, y[train_idx])
                pred_mg = model_mg.predict(mg_test_fs).ravel()

                if soil_cols_f:
                    imp_soil = SimpleImputer(strategy="median")
                    soil_train = imp_soil.fit_transform(X_train_soil)
                    soil_test = imp_soil.transform(X_test_soil)
                    model_soil = build_model(exp.model)
                    model_soil.fit(soil_train, y[train_idx])
                    pred_soil = model_soil.predict(soil_test).ravel()
                    preds[test_idx] = 0.7 * pred_mg + 0.3 * pred_soil
                else:
                    preds[test_idx] = pred_mg
            else:
                X_train_final, X_test_final = prepare_xy_for_fold(
                    X_f, y, train_idx, test_idx, exp, mg_cols_f, soil_cols_f
                )
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
        "mean_spearman": per_mb["spearman"].mean(),
        "runtime_sec": round(time.time() - start, 2),
    })
    return per_mb, summary


def build_second_wave() -> List[Experiment]:
    return [
        Experiment("B01_extratrees_baseline", 0.00, "none", None, "none", None, "early", "extratrees"),
        Experiment("B02_rare5_extratrees", 0.05, "none", None, "none", None, "early", "extratrees"),
        Experiment("B03_rare10_extratrees", 0.10, "none", None, "none", None, "early", "extratrees"),
        Experiment("B04_xgboost_light", 0.00, "none", None, "none", None, "early", "xgboost_light"),
        Experiment("B05_rare5_xgboost_light", 0.05, "none", None, "none", None, "early", "xgboost_light"),
        Experiment("B06_rare5_mi500_xgboost_light", 0.05, "mutual_info", 500, "none", None, "early", "xgboost_light"),
        Experiment("B07_elasticnet_cv", 0.05, "none", None, "none", None, "early", "elasticnet_cv"),
        Experiment("B08_lasso_cv", 0.05, "none", None, "none", None, "early", "lasso_cv"),
        Experiment("B09_pls2", 0.05, "none", None, "none", None, "early", "pls2"),
        Experiment("B10_pls5", 0.05, "none", None, "none", None, "early", "pls5"),
        Experiment("B11_pls10", 0.05, "none", None, "none", None, "early", "pls10"),
        Experiment("B12_mbspca10_rf_intermediate", 0.05, "none", None, "minibatch_sparse_pca", 10, "intermediate", "rf"),
        Experiment("B13_pca20_rf_intermediate", 0.05, "none", None, "pca", 20, "intermediate", "rf"),
        Experiment("B14_late_rf", 0.05, "none", None, "none", None, "late", "rf"),
        Experiment("B15_late_mi500_rf", 0.05, "mutual_info", 500, "none", None, "late", "rf"),
        Experiment("B16_borutaLight300_rf", 0.05, "boruta_light", 300, "none", None, "early", "rf"),
    ]


def aggregate_outputs(out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_files = sorted(out_dir.glob("summary_*.csv"))
    metric_files = sorted(out_dir.glob("*_metrics_per_metabolite.csv"))
    summaries = []
    metrics = []
    for f in summary_files:
        try:
            summaries.append(pd.read_csv(f))
        except Exception:
            pass
    for f in metric_files:
        try:
            metrics.append(pd.read_csv(f))
        except Exception:
            pass
    summary_df = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    metrics_df = pd.concat(metrics, ignore_index=True) if metrics else pd.DataFrame()
    if not summary_df.empty and "mean_r2" in summary_df.columns:
        summary_df = summary_df.sort_values("mean_r2", ascending=False)
    return summary_df, metrics_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--only", default=None, help="Optional experiment_id to run one experiment only")
    parser.add_argument("--list", action="store_true", help="List available experiments and exit")
    parser.add_argument("--aggregate-only", action="store_true", help="Only aggregate existing outputs and exit")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_dir = project_root / "10_analysis/outputs/phase22B_pipeline_optimization"
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = build_second_wave()
    if args.list:
        for e in experiments:
            print(e.experiment_id)
        return

    if args.aggregate_only:
        summary_df, metrics_df = aggregate_outputs(out_dir)
        summary_df.to_csv(out_dir / "experiment_summary.csv", index=False)
        metrics_df.to_csv(out_dir / "metrics_per_metabolite.csv", index=False)
        print(summary_df)
        print(f"Saved aggregate files in: {out_dir}")
        return

    if args.only:
        experiments = [e for e in experiments if e.experiment_id == args.only]
        if not experiments:
            raise ValueError(f"Experiment not found: {args.only}")

    X, Y, _ = load_data(project_root)
    mg_cols, soil_cols = split_blocks(X)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    print(f"Loaded X={X.shape}, Y={Y.shape}")
    print(f"Detected MG features={len(mg_cols)}, soil features={len(soil_cols)}")
    print(f"Running {len(experiments)} experiments")

    for exp in experiments:
        print(f"\n[RUN] {exp.experiment_id}", flush=True)
        try:
            per_mb, summary = evaluate_experiment(X, Y, exp, cv, mg_cols, soil_cols)
            per_mb.to_csv(out_dir / f"{exp.experiment_id}_metrics_per_metabolite.csv", index=False)
            pd.DataFrame([summary]).to_csv(out_dir / f"summary_{exp.experiment_id}.csv", index=False)
            print(json.dumps({
                "experiment_id": summary["experiment_id"],
                "mean_r2": summary["mean_r2"],
                "median_r2": summary["median_r2"],
                "n_r2_gt_02": summary["n_r2_gt_02"],
                "runtime_sec": summary["runtime_sec"],
            }, indent=2), flush=True)
        except Exception as e:
            fail = asdict(exp)
            fail.update({"error": str(e), "runtime_sec": np.nan})
            pd.DataFrame([fail]).to_csv(out_dir / f"summary_{exp.experiment_id}.csv", index=False)
            print(f"[ERROR] {exp.experiment_id}: {e}", flush=True)

        summary_df, metrics_df = aggregate_outputs(out_dir)
        summary_df.to_csv(out_dir / "experiment_summary.csv", index=False)
        metrics_df.to_csv(out_dir / "metrics_per_metabolite.csv", index=False)

    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
