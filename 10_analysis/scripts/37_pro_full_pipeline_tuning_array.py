#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import SparsePCA, TruncatedSVD
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def split_blocks(X):
    soil_prefixes = ["soil_", "chem__", "psize__", "moist__", "nitrif__", "denit__"]
    soil_cols = []
    for c in X.columns:
        cl = str(c).lower().strip()
        if any(cl.startswith(p) for p in soil_prefixes):
            soil_cols.append(c)
    mg_cols = [c for c in X.columns if c not in soil_cols]
    return mg_cols, soil_cols


def load_data(project_root):
    project_root = Path(project_root)

    x_path = project_root / "10_analysis/outputs/phase3_soil_dedup/X_deduplicated.csv"
    y_path = project_root / "10_analysis/outputs/phase2_preprocessing_fixed/Y_ml_filtered_log1p.csv"

    best_path = project_root / "10_analysis/outputs/phase31_best_model_per_mb/best_model_per_metabolite.csv"
    if not best_path.exists():
        best_path = project_root / "10_analysis/outputs/phase17_final_best_model_pipeline/best_model_per_metabolite_final.csv"

    X = pd.read_csv(x_path, low_memory=False)
    Y_all = pd.read_csv(y_path, low_memory=False)
    best = pd.read_csv(best_path)

    metabolites = [m for m in best["metabolite"].tolist() if m in Y_all.columns]
    Y = Y_all[metabolites].copy()

    if len(X) != len(Y):
        raise ValueError(f"X/Y mismatch: X={X.shape}, Y={Y.shape}")

    return X, Y, metabolites


def select_mi(X_train, y_train, X_test, k):
    if k is None or k <= 0 or k >= X_train.shape[1]:
        return X_train, X_test

    scores = mutual_info_regression(
        X_train,
        y_train,
        random_state=42,
        discrete_features=False,
    )
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    idx = np.argsort(scores)[::-1][:k]

    return X_train[:, idx], X_test[:, idx]


def reduce_dim(X_train, X_test, method, n_components):
    if method == "none":
        return X_train, X_test

    n_components = min(n_components, X_train.shape[0] - 1, X_train.shape[1])
    if n_components < 1:
        return X_train, X_test

    if method == "sparsepca":
        reducer = SparsePCA(
            n_components=n_components,
            alpha=1,
            random_state=42,
            n_jobs=-1,
            max_iter=500,
        )
        return reducer.fit_transform(X_train), reducer.transform(X_test)

    if method == "svd":
        reducer = TruncatedSVD(
            n_components=n_components,
            random_state=42,
        )
        return reducer.fit_transform(X_train), reducer.transform(X_test)

    raise ValueError(f"Unknown reduction method: {method}")


def build_model(model_name, seed):
    if model_name == "rf_strong":
        return RandomForestRegressor(
            n_estimators=800,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        )

    if model_name == "rf_regularized":
        return RandomForestRegressor(
            n_estimators=700,
            max_depth=20,
            min_samples_leaf=3,
            max_features=0.5,
            random_state=seed,
            n_jobs=-1,
        )

    if model_name == "et_strong":
        return ExtraTreesRegressor(
            n_estimators=800,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        )

    if model_name == "et_regularized":
        return ExtraTreesRegressor(
            n_estimators=700,
            max_depth=20,
            min_samples_leaf=3,
            max_features=0.5,
            random_state=seed,
            n_jobs=-1,
        )

    if model_name == "xgb_light":
        if not HAS_XGB:
            raise RuntimeError("xgboost not installed")
        return XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=2,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=5.0,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=4,
        )

    if model_name == "xgb_medium":
        if not HAS_XGB:
            raise RuntimeError("xgboost not installed")
        return XGBRegressor(
            n_estimators=700,
            learning_rate=0.02,
            max_depth=3,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=3.0,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=4,
        )

    raise ValueError(f"Unknown model: {model_name}")


def get_configs():
    configs = []

    config_id = 0

    # Autour du champion : MI500 + SparsePCA75 + late 0.7/0.3 + RF
    for mi_k in [300, 500, 800, 1000]:
        for n_comp in [50, 75, 100]:
            for w_mg in [0.6, 0.7, 0.8]:
                configs.append({
                    "config_id": f"C{config_id:03d}",
                    "mi_k": mi_k,
                    "reduction": "sparsepca",
                    "n_components": n_comp,
                    "w_mg": w_mg,
                    "model": "rf_strong",
                })
                config_id += 1

    # RF plus régularisé
    for mi_k in [500, 800]:
        for n_comp in [75, 100]:
            for w_mg in [0.7, 0.8]:
                configs.append({
                    "config_id": f"C{config_id:03d}",
                    "mi_k": mi_k,
                    "reduction": "sparsepca",
                    "n_components": n_comp,
                    "w_mg": w_mg,
                    "model": "rf_regularized",
                })
                config_id += 1

    # ExtraTrees
    for mi_k in [500, 800]:
        for n_comp in [75, 100]:
            for model in ["et_strong", "et_regularized"]:
                configs.append({
                    "config_id": f"C{config_id:03d}",
                    "mi_k": mi_k,
                    "reduction": "sparsepca",
                    "n_components": n_comp,
                    "w_mg": 0.7,
                    "model": model,
                })
                config_id += 1

    # XGBoost
    for mi_k in [500, 800]:
        for n_comp in [50, 75]:
            for model in ["xgb_light", "xgb_medium"]:
                configs.append({
                    "config_id": f"C{config_id:03d}",
                    "mi_k": mi_k,
                    "reduction": "sparsepca",
                    "n_components": n_comp,
                    "w_mg": 0.7,
                    "model": model,
                })
                config_id += 1

    # Comparaison SVD
    for mi_k in [500, 800]:
        for n_comp in [75, 100]:
            configs.append({
                "config_id": f"C{config_id:03d}",
                "mi_k": mi_k,
                "reduction": "svd",
                "n_components": n_comp,
                "w_mg": 0.7,
                "model": "rf_strong",
            })
            config_id += 1

    return configs


def evaluate_config(X, Y, mg_cols, soil_cols, config):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    results = []

    for metabolite in Y.columns:
        y = Y[metabolite].values
        preds = np.zeros(len(y))

        for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
            y_train = y[train_idx]

            # MG block
            X_train_mg = X.iloc[train_idx][mg_cols]
            X_test_mg = X.iloc[test_idx][mg_cols]

            imp_mg = SimpleImputer(strategy="constant", fill_value=0)
            mg_train = imp_mg.fit_transform(X_train_mg)
            mg_test = imp_mg.transform(X_test_mg)

            scaler_mg = StandardScaler()
            mg_train = scaler_mg.fit_transform(mg_train)
            mg_test = scaler_mg.transform(mg_test)

            mg_train, mg_test = select_mi(
                mg_train,
                y_train,
                mg_test,
                config["mi_k"],
            )

            mg_train, mg_test = reduce_dim(
                mg_train,
                mg_test,
                config["reduction"],
                config["n_components"],
            )

            model_mg = build_model(config["model"], seed=42 + fold)
            model_mg.fit(mg_train, y_train)
            pred_mg = model_mg.predict(mg_test)

            # Soil block
            if len(soil_cols) > 0:
                X_train_soil = X.iloc[train_idx][soil_cols]
                X_test_soil = X.iloc[test_idx][soil_cols]

                imp_soil = SimpleImputer(strategy="median")
                soil_train = imp_soil.fit_transform(X_train_soil)
                soil_test = imp_soil.transform(X_test_soil)

                scaler_soil = StandardScaler()
                soil_train = scaler_soil.fit_transform(soil_train)
                soil_test = scaler_soil.transform(soil_test)

                model_soil = build_model(config["model"], seed=100 + fold)
                model_soil.fit(soil_train, y_train)
                pred_soil = model_soil.predict(soil_test)

                w_mg = config["w_mg"]
                preds[test_idx] = w_mg * pred_mg + (1 - w_mg) * pred_soil
            else:
                preds[test_idx] = pred_mg

        r2 = r2_score(y, preds)
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        mae = float(mean_absolute_error(y, preds))

        row = dict(config)
        row.update({
            "metabolite": metabolite,
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "n_samples": len(y),
            "n_features": X.shape[1],
            "n_mg_features": len(mg_cols),
            "n_soil_features": len(soil_cols),
        })
        results.append(row)

    return pd.DataFrame(results)


def aggregate(out_dir):
    files = sorted(out_dir.glob("C*_metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No metrics files found in {out_dir}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(out_dir / "all_config_metrics.csv", index=False)

    summary = (
        df.groupby(["config_id", "model", "mi_k", "reduction", "n_components", "w_mg"])
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
        .sort_values("mean_r2", ascending=False)
    )

    summary.to_csv(out_dir / "config_summary_ranked.csv", index=False)

    best_per_metabolite = (
        df.sort_values("r2", ascending=False)
        .groupby("metabolite")
        .head(1)
        .sort_values("r2", ascending=False)
    )

    best_per_metabolite.to_csv(out_dir / "best_config_per_metabolite.csv", index=False)

    print("\nTop 15 configurations:")
    print(summary.head(15).to_string(index=False))

    print("\nBest-per-metabolite global:")
    print(best_per_metabolite["r2"].describe())

    print("\nOutputs:")
    print(out_dir / "all_config_metrics.csv")
    print(out_dir / "config_summary_ranked.csv")
    print(out_dir / "best_config_per_metabolite.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_dir = project_root / "10_analysis/outputs/phase37_pro_full_pipeline_tuning"
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = get_configs()

    with open(out_dir / "configs.json", "w") as f:
        json.dump(configs, f, indent=2)

    if args.aggregate_only:
        aggregate(out_dir)
        return

    if args.task_id is None:
        print("Number of configs:", len(configs))
        print("Use --task-id from 0 to", len(configs) - 1)
        return

    if args.task_id < 0 or args.task_id >= len(configs):
        raise ValueError(f"Invalid task-id {args.task_id}. Valid range: 0-{len(configs)-1}")

    config = configs[args.task_id]

    print("Running config:")
    print(json.dumps(config, indent=2))

    X, Y, metabolites = load_data(project_root)
    mg_cols, soil_cols = split_blocks(X)

    print("X:", X.shape)
    print("Y:", Y.shape)
    print("MG features:", len(mg_cols))
    print("Soil features:", len(soil_cols))

    df = evaluate_config(X, Y, mg_cols, soil_cols, config)

    out_file = out_dir / f"{config['config_id']}_metrics.csv"
    df.to_csv(out_file, index=False)

    print("Saved:", out_file)
    print("Mean R2:", df["r2"].mean())
    print("Median R2:", df["r2"].median())


if __name__ == "__main__":
    main()
