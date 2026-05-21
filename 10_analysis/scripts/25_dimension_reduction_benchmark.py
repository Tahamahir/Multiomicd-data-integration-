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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA, SparsePCA, TruncatedSVD, NMF, FastICA

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
    best_path = (
        project_root
        / "10_analysis/outputs/phase17_final_best_model_pipeline/best_model_per_metabolite_final.csv"
    )

    X = pd.read_csv(x_path, low_memory=False)
    Y_all = pd.read_csv(y_path, low_memory=False)
    best = pd.read_csv(best_path)

    metabolites = [m for m in best["metabolite"].tolist() if m in Y_all.columns]
    Y = Y_all[metabolites].copy()

    if len(X) != len(Y):
        raise ValueError(f"X and Y row mismatch: X={X.shape}, Y={Y.shape}")

    return X, Y, metabolites


def rare_filter(X, threshold):
    if threshold <= 0:
        return X.copy()

    presence = (X.fillna(0) != 0).mean(axis=0)
    keep_cols = presence[presence >= threshold].index.tolist()
    return X[keep_cols].copy()


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


def reduce_block(X_train, X_test, method, n_components):
    if method == "none":
        return X_train, X_test

    n_components = min(n_components, X_train.shape[0] - 1, X_train.shape[1])

    if n_components < 1:
        return X_train, X_test

    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
        return reducer.fit_transform(X_train), reducer.transform(X_test)

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

    if method == "nmf":
        scaler = MinMaxScaler()
        X_train_pos = scaler.fit_transform(X_train)
        X_test_pos = scaler.transform(X_test)

        reducer = NMF(
            n_components=n_components,
            random_state=42,
            init="nndsvda",
            max_iter=1000,
        )
        return reducer.fit_transform(X_train_pos), reducer.transform(X_test_pos)

    if method == "ica":
        reducer = FastICA(
            n_components=n_components,
            random_state=42,
            max_iter=1000,
            whiten="unit-variance",
        )
        return reducer.fit_transform(X_train), reducer.transform(X_test)

    raise ValueError(f"Unknown reduction method: {method}")


def build_rf():
    return RandomForestRegressor(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )


def evaluate_experiment(X, Y, mg_cols, soil_cols, exp, cv):
    results = []

    rare_threshold = exp["rare_threshold"]
    mi_k = exp["mi_k"]
    reduction = exp["reduction"]
    n_components = exp["n_components"]

    X_f = rare_filter(X, rare_threshold)

    mg_cols_f = [c for c in mg_cols if c in X_f.columns]
    soil_cols_f = [c for c in soil_cols if c in X_f.columns]

    for metabolite in Y.columns:
        y = Y[metabolite].values
        preds = np.zeros(len(y))

        for train_idx, test_idx in cv.split(X_f):
            X_train_mg = X_f.iloc[train_idx][mg_cols_f]
            X_test_mg = X_f.iloc[test_idx][mg_cols_f]

            X_train_soil = X_f.iloc[train_idx][soil_cols_f]
            X_test_soil = X_f.iloc[test_idx][soil_cols_f]

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
                mi_k,
            )

            mg_train_red, mg_test_red = reduce_block(
                mg_train,
                mg_test,
                reduction,
                n_components,
            )

            model_mg = build_rf()
            model_mg.fit(mg_train_red, y_train)
            pred_mg = model_mg.predict(mg_test_red)

            if len(soil_cols_f) > 0:
                imp_soil = SimpleImputer(strategy="median")
                soil_train = imp_soil.fit_transform(X_train_soil)
                soil_test = imp_soil.transform(X_test_soil)

                scaler_soil = StandardScaler()
                soil_train = scaler_soil.fit_transform(soil_train)
                soil_test = scaler_soil.transform(soil_test)

                model_soil = build_rf()
                model_soil.fit(soil_train, y_train)
                pred_soil = model_soil.predict(soil_test)

                preds[test_idx] = 0.7 * pred_mg + 0.3 * pred_soil
            else:
                preds[test_idx] = pred_mg

        r2 = r2_score(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))

        results.append({
            "experiment_id": exp["experiment_id"],
            "metabolite": metabolite,
            "r2": r2,
            "rmse": rmse,
            "rare_threshold": rare_threshold,
            "mi_k": mi_k,
            "reduction": reduction,
            "n_components": n_components,
            "n_mg_features_after_filter": len(mg_cols_f),
            "n_soil_features_after_filter": len(soil_cols_f),
        })

    return pd.DataFrame(results)


def get_experiments():
    experiments = []

    configs = [
        ("DR00_late_rf_none", 0.0, None, "none", None),
        ("DR01_late_mi500_rf_none", 0.0, 500, "none", None),

        ("DR02_late_mi500_pca10_rf", 0.0, 500, "pca", 10),
        ("DR03_late_mi500_pca20_rf", 0.0, 500, "pca", 20),
        ("DR04_late_mi500_pca50_rf", 0.0, 500, "pca", 50),

        ("DR05_late_mi500_sparsepca10_rf", 0.0, 500, "sparsepca", 10),
        ("DR06_late_mi500_sparsepca20_rf", 0.0, 500, "sparsepca", 20),
        ("DR07_late_mi500_sparsepca50_rf", 0.0, 500, "sparsepca", 50),

        ("DR08_late_mi500_svd10_rf", 0.0, 500, "svd", 10),
        ("DR09_late_mi500_svd20_rf", 0.0, 500, "svd", 20),
        ("DR10_late_mi500_svd50_rf", 0.0, 500, "svd", 50),

        ("DR11_late_mi500_nmf10_rf", 0.0, 500, "nmf", 10),
        ("DR12_late_mi500_nmf20_rf", 0.0, 500, "nmf", 20),
        ("DR13_late_mi500_nmf50_rf", 0.0, 500, "nmf", 50),

        ("DR14_late_mi500_ica10_rf", 0.0, 500, "ica", 10),
        ("DR15_late_mi500_ica20_rf", 0.0, 500, "ica", 20),

        ("DR16_rare5_late_mi500_sparsepca20_rf", 0.05, 500, "sparsepca", 20),
        ("DR17_rare5_late_mi500_svd20_rf", 0.05, 500, "svd", 20),
        ("DR18_rare5_late_mi500_nmf20_rf", 0.05, 500, "nmf", 20),
        ("DR19_late_mi500_sparsepca75_rf", 0.0, 500, "sparsepca", 75),
        ("DR20_late_mi500_sparsepca100_rf", 0.0, 500, "sparsepca", 100),
        ("DR21_late_mi500_sparsepca150_rf", 0.0, 500, "sparsepca", 150),
    ]

    for exp_id, rare, mi_k, red, n_comp in configs:
        experiments.append({
            "experiment_id": exp_id,
            "rare_threshold": rare,
            "mi_k": mi_k,
            "reduction": red,
            "n_components": n_comp,
        })

    return experiments


def summarize(df):
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
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--only", default=None)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    experiments = get_experiments()

    if args.list:
        for e in experiments:
            print(e["experiment_id"])
        return

    if args.only:
        experiments = [e for e in experiments if e["experiment_id"] == args.only]
        if not experiments:
            raise ValueError(f"Experiment not found: {args.only}")

    project_root = Path(args.project_root)
    output_dir = project_root / "10_analysis/outputs/phase25_dimension_reduction_late_rf"
    output_dir.mkdir(parents=True, exist_ok=True)

    X, Y, metabolites = load_data(project_root)
    mg_cols, soil_cols = split_blocks(X)

    print(f"Loaded X={X.shape}, Y={Y.shape}")
    print(f"Detected MG features={len(mg_cols)}, soil features={len(soil_cols)}")
    print(f"Running {len(experiments)} experiments")

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    summary_rows = []

    for exp in experiments:
        start = time.time()
        print("\n[RUN]", exp["experiment_id"])

        df = evaluate_experiment(X, Y, mg_cols, soil_cols, exp, cv)
        runtime = time.time() - start

        s = summarize(df)
        s["runtime_sec"] = round(runtime, 2)
        summary_rows.append(s)

        df.to_csv(output_dir / f"{exp['experiment_id']}_metrics_per_metabolite.csv", index=False)

        with open(output_dir / f"{exp['experiment_id']}_summary.json", "w") as f:
            json.dump(s, f, indent=2)

        print(json.dumps(s, indent=2))

    summary = pd.DataFrame(summary_rows).sort_values(
        ["mean_r2", "median_r2"],
        ascending=False,
    )

    summary.to_csv(output_dir / "dimension_reduction_summary.csv", index=False)

    print("\nDone.")
    print("Outputs in:", output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()