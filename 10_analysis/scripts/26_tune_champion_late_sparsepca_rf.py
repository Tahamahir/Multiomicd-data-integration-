from pathlib import Path
import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import SparsePCA

warnings.filterwarnings("ignore")


def split_blocks(X):
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


def load_data(project_root):
    project_root = Path(project_root)

    x_path = project_root / "10_analysis/outputs/phase3_soil_dedup/X_deduplicated.csv"
    y_path = project_root / "10_analysis/outputs/phase2_preprocessing_fixed/Y_ml_filtered_log1p.csv"
    best_path = project_root / "10_analysis/outputs/phase17_final_best_model_pipeline/best_model_per_metabolite_final.csv"

    X = pd.read_csv(x_path, low_memory=False)
    Y_all = pd.read_csv(y_path, low_memory=False)
    best = pd.read_csv(best_path)

    metabolites = [m for m in best["metabolite"].tolist() if m in Y_all.columns]
    Y = Y_all[metabolites].copy()

    if len(X) != len(Y):
        raise ValueError(f"X and Y mismatch: X={X.shape}, Y={Y.shape}")

    return X, Y, metabolites


def select_mi_topk(X_train, y_train, X_test, k):
    if k is None or k <= 0 or k >= X_train.shape[1]:
        return X_train, X_test

    scores = mutual_info_regression(
        X_train,
        y_train,
        random_state=42,
        discrete_features=False,
    )

    idx = np.argsort(scores)[::-1][:k]
    return X_train[:, idx], X_test[:, idx]


def build_rf(params):
    return RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_features=params["max_features"],
        min_samples_leaf=params["min_samples_leaf"],
        max_depth=params["max_depth"],
        random_state=42,
        n_jobs=-1,
    )


def get_configs():
    configs = []

    mi_values = [300, 500, 750, 1000]
    n_components_values = [50, 75, 100]
    alpha_values = [0.5, 1.0, 2.0]
    mg_weights = [0.6, 0.7, 0.8, 0.9]

    rf_params_list = [
        {
            "rf_name": "rf_a",
            "n_estimators": 500,
            "max_features": "sqrt",
            "min_samples_leaf": 2,
            "max_depth": None,
        },
        {
            "rf_name": "rf_b",
            "n_estimators": 800,
            "max_features": "sqrt",
            "min_samples_leaf": 2,
            "max_depth": None,
        },
        {
            "rf_name": "rf_c",
            "n_estimators": 500,
            "max_features": 0.3,
            "min_samples_leaf": 2,
            "max_depth": None,
        },
        {
            "rf_name": "rf_d",
            "n_estimators": 500,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "max_depth": None,
        },
        {
            "rf_name": "rf_e",
            "n_estimators": 500,
            "max_features": "sqrt",
            "min_samples_leaf": 3,
            "max_depth": None,
        },
    ]

    counter = 1

    for mi_k in mi_values:
        for n_comp in n_components_values:
            for alpha in alpha_values:
                for mg_w in mg_weights:
                    for rf_params in rf_params_list:
                        exp_id = (
                            f"T{counter:03d}_mi{mi_k}"
                            f"_spca{n_comp}"
                            f"_a{str(alpha).replace('.', '')}"
                            f"_w{int(mg_w * 10)}"
                            f"_{rf_params['rf_name']}"
                        )

                        cfg = {
                            "experiment_id": exp_id,
                            "mi_k": mi_k,
                            "n_components": n_comp,
                            "sparsepca_alpha": alpha,
                            "mg_weight": mg_w,
                            "soil_weight": 1.0 - mg_w,
                            **rf_params,
                        }

                        configs.append(cfg)
                        counter += 1

    return configs


def evaluate_config(X, Y, mg_cols, soil_cols, cfg, cv):
    rows = []

    for metabolite in Y.columns:
        y = Y[metabolite].values
        preds = np.zeros(len(y))

        for train_idx, test_idx in cv.split(X):
            X_train_mg = X.iloc[train_idx][mg_cols]
            X_test_mg = X.iloc[test_idx][mg_cols]

            X_train_soil = X.iloc[train_idx][soil_cols]
            X_test_soil = X.iloc[test_idx][soil_cols]

            y_train = y[train_idx]

            imp_mg = SimpleImputer(strategy="constant", fill_value=0)
            mg_train = imp_mg.fit_transform(X_train_mg)
            mg_test = imp_mg.transform(X_test_mg)

            scaler_mg = StandardScaler()
            mg_train = scaler_mg.fit_transform(mg_train)
            mg_test = scaler_mg.transform(mg_test)

            mg_train, mg_test = select_mi_topk(
                mg_train,
                y_train,
                mg_test,
                cfg["mi_k"],
            )

            n_components = min(
                cfg["n_components"],
                mg_train.shape[0] - 1,
                mg_train.shape[1],
            )

            reducer = SparsePCA(
                n_components=n_components,
                alpha=cfg["sparsepca_alpha"],
                random_state=42,
                n_jobs=-1,
                max_iter=500,
            )

            mg_train_red = reducer.fit_transform(mg_train)
            mg_test_red = reducer.transform(mg_test)

            model_mg = build_rf(cfg)
            model_mg.fit(mg_train_red, y_train)
            pred_mg = model_mg.predict(mg_test_red)

            imp_soil = SimpleImputer(strategy="median")
            soil_train = imp_soil.fit_transform(X_train_soil)
            soil_test = imp_soil.transform(X_test_soil)

            scaler_soil = StandardScaler()
            soil_train = scaler_soil.fit_transform(soil_train)
            soil_test = scaler_soil.transform(soil_test)

            model_soil = build_rf(cfg)
            model_soil.fit(soil_train, y_train)
            pred_soil = model_soil.predict(soil_test)

            preds[test_idx] = (
                cfg["mg_weight"] * pred_mg
                + cfg["soil_weight"] * pred_soil
            )

        r2 = r2_score(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))

        rows.append({
            "experiment_id": cfg["experiment_id"],
            "metabolite": metabolite,
            "r2": r2,
            "rmse": rmse,
            "mi_k": cfg["mi_k"],
            "n_components": cfg["n_components"],
            "sparsepca_alpha": cfg["sparsepca_alpha"],
            "mg_weight": cfg["mg_weight"],
            "soil_weight": cfg["soil_weight"],
            "rf_name": cfg["rf_name"],
            "n_estimators": cfg["n_estimators"],
            "max_features": cfg["max_features"],
            "min_samples_leaf": cfg["min_samples_leaf"],
            "max_depth": cfg["max_depth"],
        })

    return pd.DataFrame(rows)


def summarize(df, runtime):
    return {
        "experiment_id": df["experiment_id"].iloc[0],
        "mean_r2": float(df["r2"].mean()),
        "median_r2": float(df["r2"].median()),
        "max_r2": float(df["r2"].max()),
        "std_r2": float(df["r2"].std()),
        "n_r2_gt_0": int((df["r2"] > 0).sum()),
        "n_r2_gt_02": int((df["r2"] > 0.2).sum()),
        "n_r2_gt_04": int((df["r2"] > 0.4).sum()),
        "n_r2_gt_06": int((df["r2"] > 0.6).sum()),
        "runtime_sec": round(runtime, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--only", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--top", type=int, default=None)
    args = parser.parse_args()

    configs = get_configs()

    if args.list:
        for c in configs:
            print(c["experiment_id"])
        return

    if args.only:
        configs = [c for c in configs if c["experiment_id"] == args.only]
        if not configs:
            raise ValueError(f"Experiment not found: {args.only}")

    if args.top:
        configs = configs[:args.top]

    project_root = Path(args.project_root)
    output_dir = project_root / "10_analysis/outputs/phase26_tune_champion_late_sparsepca_rf"
    output_dir.mkdir(parents=True, exist_ok=True)

    X, Y, metabolites = load_data(project_root)
    mg_cols, soil_cols = split_blocks(X)

    print(f"Loaded X={X.shape}, Y={Y.shape}")
    print(f"Detected MG features={len(mg_cols)}, soil features={len(soil_cols)}")
    print(f"Running {len(configs)} configs")

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    summary_rows = []

    for cfg in configs:
        print("\n[RUN]", cfg["experiment_id"])
        print(json.dumps(cfg, indent=2))

        start = time.time()
        df = evaluate_config(X, Y, mg_cols, soil_cols, cfg, cv)
        runtime = time.time() - start

        df_path = output_dir / f"{cfg['experiment_id']}_metrics_per_metabolite.csv"
        df.to_csv(df_path, index=False)

        summary = summarize(df, runtime)
        summary.update({
            "mi_k": cfg["mi_k"],
            "n_components": cfg["n_components"],
            "sparsepca_alpha": cfg["sparsepca_alpha"],
            "mg_weight": cfg["mg_weight"],
            "soil_weight": cfg["soil_weight"],
            "rf_name": cfg["rf_name"],
            "n_estimators": cfg["n_estimators"],
            "max_features": str(cfg["max_features"]),
            "min_samples_leaf": cfg["min_samples_leaf"],
            "max_depth": cfg["max_depth"],
        })

        summary_rows.append(summary)

        with open(output_dir / f"{cfg['experiment_id']}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(json.dumps(summary, indent=2))

        all_summary = pd.DataFrame(summary_rows).sort_values(
            ["mean_r2", "median_r2"],
            ascending=False,
        )
        all_summary.to_csv(output_dir / "tuning_summary_partial.csv", index=False)

    final_summary = pd.DataFrame(summary_rows).sort_values(
        ["mean_r2", "median_r2"],
        ascending=False,
    )

    final_summary.to_csv(output_dir / "tuning_summary.csv", index=False)

    print("\nDone.")
    print("Outputs in:", output_dir)
    print(final_summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()