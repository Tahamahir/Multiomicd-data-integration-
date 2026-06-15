#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 32 - Hyperparameter tuning for best model per metabolite

Objectif :
- Partir du fichier best_model_per_metabolite.csv
- Pour chaque métabolite, prendre le modèle gagnant actuel
- Tuner uniquement ce modèle avec GridSearchCV
- Sauvegarder un fichier résultat par métabolite
- Compatible avec SLURM array

Exemple :
python 32_tune_best_model_per_mb_slurm.py --task-id 0
"""

from pathlib import Path
import argparse
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG PATHS
# ============================================================

def get_paths(project_root: Path):
    x_path = project_root / "10_analysis/outputs/phase3_soil_dedup/X_deduplicated.csv"
    y_path = project_root / "10_analysis/outputs/phase2_preprocessing_fixed/Y_ml_filtered_log1p.csv"

    # Change ce chemin si ton fichier est dans un autre dossier
    best_path = project_root / "10_analysis/outputs/phase31_best_model_per_mb/best_model_per_metabolite.csv"

    # Fallback si tu utilises le fichier final de phase 17
    if not best_path.exists():
        best_path = project_root / "10_analysis/outputs/phase17_final_best_model_pipeline/best_model_per_metabolite_final.csv"

    out_dir = project_root / "10_analysis/outputs/phase32_tuned_best_model_per_mb"
    out_dir.mkdir(parents=True, exist_ok=True)

    return x_path, y_path, best_path, out_dir


# ============================================================
# DATA UTILS
# ============================================================

def split_blocks(X: pd.DataFrame):
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


def load_data(project_root: Path):
    x_path, y_path, best_path, out_dir = get_paths(project_root)

    if not x_path.exists():
        raise FileNotFoundError(f"Missing X file: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing Y file: {y_path}")
    if not best_path.exists():
        raise FileNotFoundError(f"Missing best model file: {best_path}")

    X = pd.read_csv(x_path, low_memory=False)
    Y = pd.read_csv(y_path, low_memory=False)
    best = pd.read_csv(best_path)

    if len(X) != len(Y):
        raise ValueError(f"X/Y mismatch: X={X.shape}, Y={Y.shape}")

    return X, Y, best, out_dir


# ============================================================
# MODELS AND GRIDS
# ============================================================

def build_model_and_grid(model_code: str):
    model_code = str(model_code).strip().upper()

    if model_code == "RF":
        model = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid = {
            "model__n_estimators": [300, 500, 800],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_leaf": [1, 2, 3, 5],
            "model__max_features": ["sqrt", 0.5],
        }

    elif model_code == "ET":
        model = ExtraTreesRegressor(random_state=42, n_jobs=-1)
        grid = {
            "model__n_estimators": [300, 500, 800],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_leaf": [1, 2, 3, 5],
            "model__max_features": ["sqrt", 0.5],
        }

    elif model_code == "XGB":
        if not HAS_XGB:
            raise RuntimeError("xgboost is not installed in this environment")

        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=4,
        )
        grid = {
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [2, 3, 4],
            "model__learning_rate": [0.01, 0.03, 0.05],
            "model__subsample": [0.7, 0.9],
            "model__colsample_bytree": [0.7, 0.9],
            "model__reg_alpha": [0.0, 0.1, 0.5],
            "model__reg_lambda": [1.0, 3.0, 5.0],
        }

    elif model_code == "ELASTIC":
        model = ElasticNet(max_iter=50000, random_state=42)
        grid = {
            "model__alpha": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
            "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        }

    elif model_code == "SVR":
        model = SVR()
        grid = {
            "model__C": [0.1, 1, 10, 100],
            "model__epsilon": [0.01, 0.05, 0.1],
            "model__gamma": ["scale", "auto"],
            "model__kernel": ["rbf"],
        }

    else:
        raise ValueError(f"Unknown model code: {model_code}")

    pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        model,
    )

    # Renommer automatiquement les steps sklearn
    # make_pipeline donne souvent : simpleimputer, standardscaler, randomforestregressor
    # Donc on reconstruit explicitement avec les bons noms.
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])

    return pipe, grid


# ============================================================
# TUNING ONE METABOLITE
# ============================================================

def tune_one_metabolite(X, Y, metabolite, model_code, out_dir):
    y = Y[metabolite].values.astype(float)

    model, grid = build_model_and_grid(model_code)

    cv = RepeatedKFold(
        n_splits=5,
        n_repeats=2,
        random_state=42,
    )

    search = GridSearchCV(
        estimator=model,
        param_grid=grid,
        scoring="r2",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    search.fit(X.values, y)

    preds = search.best_estimator_.predict(X.values)

    r2_train_refit = r2_score(y, preds)
    rmse_train_refit = float(np.sqrt(mean_squared_error(y, preds)))
    mae_train_refit = float(mean_absolute_error(y, preds))

    result = {
        "metabolite": metabolite,
        "model_code": model_code,
        "best_cv_r2": float(search.best_score_),
        "best_params": json.dumps(search.best_params_),
        "r2_train_refit": float(r2_train_refit),
        "rmse_train_refit": rmse_train_refit,
        "mae_train_refit": mae_train_refit,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
    }

    safe_name = (
        metabolite.replace("/", "_")
        .replace("|", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )

    out_file = out_dir / f"tuned_{safe_name}.csv"
    pd.DataFrame([result]).to_csv(out_file, index=False)

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    X, Y, best, out_dir = load_data(project_root)

    if args.aggregate_only:
        files = sorted(out_dir.glob("tuned_*.csv"))
        dfs = []
        for f in files:
            try:
                dfs.append(pd.read_csv(f))
            except Exception:
                pass

        if not dfs:
            print("No tuned files found.")
            return

        final = pd.concat(dfs, ignore_index=True)
        final = final.sort_values("best_cv_r2", ascending=False)
        final.to_csv(out_dir / "best_model_per_metabolite_tuned_summary.csv", index=False)

        dist = (
            final.groupby("model_code")
            .agg(
                n_metabolites=("metabolite", "count"),
                mean_cv_r2=("best_cv_r2", "mean"),
                median_cv_r2=("best_cv_r2", "median"),
                max_cv_r2=("best_cv_r2", "max"),
                min_cv_r2=("best_cv_r2", "min"),
            )
            .reset_index()
            .sort_values(["n_metabolites", "mean_cv_r2"], ascending=False)
        )
        dist.to_csv(out_dir / "tuned_model_distribution.csv", index=False)

        print(f"Saved final summary in: {out_dir}")
        print(final.head())
        print(dist)
        return

    if args.task_id is None:
        raise ValueError("You must provide --task-id when not using --aggregate-only")

    if args.task_id < 0 or args.task_id >= len(best):
        raise ValueError(f"task-id out of range: {args.task_id}, max={len(best)-1}")

    row = best.iloc[args.task_id]
    metabolite = row["metabolite"]
    model_code = row["model"]

    if metabolite not in Y.columns:
        raise ValueError(f"Metabolite not found in Y: {metabolite}")

    print("=" * 80)
    print(f"Task ID: {args.task_id}")
    print(f"Metabolite: {metabolite}")
    print(f"Model: {model_code}")
    print(f"X shape: {X.shape}")
    print("=" * 80)

    result = tune_one_metabolite(X, Y, metabolite, model_code, out_dir)

    print("Done:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
