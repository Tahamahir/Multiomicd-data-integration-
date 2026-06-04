from pathlib import Path
import argparse, json, time, warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import SparsePCA

warnings.filterwarnings("ignore")


def split_blocks(X):
    prefixes = ["soil_", "chem__", "psize__", "moist__", "nitrif__", "denit__"]
    soil = [c for c in X.columns if any(str(c).lower().strip().startswith(p) for p in prefixes)]
    mg = [c for c in X.columns if c not in soil]
    return mg, soil


def load_data(root):
    root = Path(root)
    X = pd.read_csv(root / "10_analysis/outputs/phase3_soil_dedup/X_deduplicated.csv", low_memory=False)
    Y_all = pd.read_csv(root / "10_analysis/outputs/phase2_preprocessing_fixed/Y_ml_filtered_log1p.csv", low_memory=False)
    best = pd.read_csv(root / "10_analysis/outputs/phase17_final_best_model_pipeline/best_model_per_metabolite_final.csv")

    metabolites = [m for m in best["metabolite"].tolist() if m in Y_all.columns]
    Y = Y_all[metabolites].copy()

    return X, Y


def select_mi(X_train, y_train, X_test, k):
    if k is None or k >= X_train.shape[1]:
        return X_train, X_test

    scores = mutual_info_regression(X_train, y_train, random_state=42)
    idx = np.argsort(scores)[::-1][:k]
    return X_train[:, idx], X_test[:, idx]


def rf(params):
    return RandomForestRegressor(
        n_estimators=params.get("n_estimators", 500),
        max_features=params.get("max_features", "sqrt"),
        min_samples_leaf=params.get("min_samples_leaf", 2),
        max_depth=params.get("max_depth", None),
        random_state=42,
        n_jobs=-1,
    )


def predict_one_metabolite(X, y, mg_cols, soil_cols, cv, params):
    preds = np.zeros(len(y))

    splits = cv.split(X) if hasattr(cv, "split") else cv

    for train_idx, test_idx in splits:
        X_train_mg = X.iloc[train_idx][mg_cols]
        X_test_mg = X.iloc[test_idx][mg_cols]

        X_train_soil = X.iloc[train_idx][soil_cols]
        X_test_soil = X.iloc[test_idx][soil_cols]

        y_train = y[train_idx]

        imp_mg = SimpleImputer(strategy="constant", fill_value=0)
        mg_train = imp_mg.fit_transform(X_train_mg)
        mg_test = imp_mg.transform(X_test_mg)

        sc_mg = StandardScaler()
        mg_train = sc_mg.fit_transform(mg_train)
        mg_test = sc_mg.transform(mg_test)

        mg_train, mg_test = select_mi(mg_train, y_train, mg_test, params["mi_k"])

        n_comp = min(params["n_components"], mg_train.shape[0] - 1, mg_train.shape[1])

        reducer = SparsePCA(
            n_components=n_comp,
            alpha=params["alpha"],
            random_state=42,
            n_jobs=-1,
            max_iter=500,
        )

        mg_train = reducer.fit_transform(mg_train)
        mg_test = reducer.transform(mg_test)

        model_mg = rf(params)
        model_mg.fit(mg_train, y_train)
        pred_mg = model_mg.predict(mg_test)

        imp_soil = SimpleImputer(strategy="median")
        soil_train = imp_soil.fit_transform(X_train_soil)
        soil_test = imp_soil.transform(X_test_soil)

        sc_soil = StandardScaler()
        soil_train = sc_soil.fit_transform(soil_train)
        soil_test = sc_soil.transform(soil_test)

        model_soil = rf(params)
        model_soil.fit(soil_train, y_train)
        pred_soil = model_soil.predict(soil_test)

        preds[test_idx] = params["mg_weight"] * pred_mg + (1 - params["mg_weight"]) * pred_soil

    return preds


def evaluate(X, Y, mg_cols, soil_cols, cv, params, experiment_id):
    rows = []

    for metabolite in Y.columns:
        y = Y[metabolite].values
        pred = predict_one_metabolite(X, y, mg_cols, soil_cols, cv, params)

        rows.append({
            "experiment_id": experiment_id,
            "metabolite": metabolite,
            "r2": r2_score(y, pred),
            "rmse": np.sqrt(mean_squared_error(y, pred)),
            **params,
        })

    return pd.DataFrame(rows)


def repeated_cv(root):
    X, Y = load_data(root)
    mg_cols, soil_cols = split_blocks(X)

    params = {
        "mi_k": 500,
        "n_components": 75,
        "alpha": 1.0,
        "mg_weight": 0.7,
        "n_estimators": 500,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "max_depth": None,
    }

    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

    print(f"Loaded X={X.shape}, Y={Y.shape}")
    print(f"MG={len(mg_cols)}, Soil={len(soil_cols)}")
    print("Running Repeated CV: 5 folds × 10 repeats")

    start = time.time()
    df = evaluate(X, Y, mg_cols, soil_cols, cv, params, "T266_repeatedCV_5x10")
    runtime = time.time() - start

    return df, runtime


def nested_cv(root):
    X, Y = load_data(root)
    mg_cols, soil_cols = split_blocks(X)

    candidates = [
        {
            "name": "T266_champion",
            "mi_k": 500,
            "n_components": 75,
            "alpha": 1.0,
            "mg_weight": 0.7,
            "n_estimators": 500,
            "max_features": "sqrt",
            "min_samples_leaf": 2,
            "max_depth": None,
        },
        {
            "name": "T271_weight08",
            "mi_k": 500,
            "n_components": 75,
            "alpha": 1.0,
            "mg_weight": 0.8,
            "n_estimators": 500,
            "max_features": "sqrt",
            "min_samples_leaf": 2,
            "max_depth": None,
        },
        {
            "name": "T286_alpha2",
            "mi_k": 500,
            "n_components": 75,
            "alpha": 2.0,
            "mg_weight": 0.7,
            "n_estimators": 500,
            "max_features": "sqrt",
            "min_samples_leaf": 2,
            "max_depth": None,
        },
    ]

    outer = KFold(n_splits=5, shuffle=True, random_state=42)
    inner = KFold(n_splits=3, shuffle=True, random_state=123)

    outer_rows = []

    print(f"Loaded X={X.shape}, Y={Y.shape}")
    print(f"MG={len(mg_cols)}, Soil={len(soil_cols)}")
    print("Running light Nested CV: outer 5 folds, inner 3 folds")

    for outer_fold, (train_outer, test_outer) in enumerate(outer.split(X), start=1):
        X_train_outer = X.iloc[train_outer].reset_index(drop=True)
        X_test_outer = X.iloc[test_outer].reset_index(drop=True)
        Y_train_outer = Y.iloc[train_outer].reset_index(drop=True)
        Y_test_outer = Y.iloc[test_outer].reset_index(drop=True)

        inner_scores = []

        for cand in candidates:
            df_inner = evaluate(
                X_train_outer,
                Y_train_outer,
                mg_cols,
                soil_cols,
                inner,
                {k: v for k, v in cand.items() if k != "name"},
                cand["name"],
            )
            inner_scores.append({
                "candidate": cand["name"],
                "mean_inner_r2": df_inner["r2"].mean(),
            })

        best_name = sorted(inner_scores, key=lambda x: x["mean_inner_r2"], reverse=True)[0]["candidate"]
        best_params = [c for c in candidates if c["name"] == best_name][0]

        # final outer evaluation metabolite by metabolite
        for metabolite in Y.columns:
            y_train = Y_train_outer[metabolite].values
            y_test = Y_test_outer[metabolite].values

            # train/predict manually with one split
            single_cv = [(np.arange(len(X_train_outer)), np.arange(len(X_train_outer), len(X_train_outer) + len(X_test_outer)))]

            X_combined = pd.concat([X_train_outer, X_test_outer], axis=0).reset_index(drop=True)
            y_combined = np.concatenate([y_train, y_test])

            pred_all = predict_one_metabolite(
                X_combined,
                y_combined,
                mg_cols,
                soil_cols,
                single_cv,
                {k: v for k, v in best_params.items() if k != "name"},
            )

            pred_test = pred_all[len(X_train_outer):]

            outer_rows.append({
                "outer_fold": outer_fold,
                "metabolite": metabolite,
                "selected_pipeline": best_name,
                "r2": r2_score(y_test, pred_test),
                "rmse": np.sqrt(mean_squared_error(y_test, pred_test)),
            })

        print(f"Outer fold {outer_fold}: selected {best_name}")

    return pd.DataFrame(outer_rows)


def save_outputs(df, out_dir, name, runtime=None):
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / f"{name}_per_metabolite.csv", index=False)

    summary = {
        "experiment": name,
        "mean_r2": float(df["r2"].mean()),
        "median_r2": float(df["r2"].median()),
        "std_r2": float(df["r2"].std()),
        "max_r2": float(df["r2"].max()),
        "n_r2_gt_0": int((df["r2"] > 0).sum()),
        "n_r2_gt_02": int((df["r2"] > 0.2).sum()),
        "n_r2_gt_04": int((df["r2"] > 0.4).sum()),
        "n_r2_gt_06": int((df["r2"] > 0.6).sum()),
    }

    if runtime is not None:
        summary["runtime_sec"] = round(runtime, 2)

    pd.DataFrame([summary]).to_csv(out_dir / f"{name}_summary.csv", index=False)

    with open(out_dir / f"{name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--mode", choices=["repeated", "nested"], required=True)
    args = parser.parse_args()

    out_dir = Path(args.project_root) / "10_analysis/outputs/phase27_repeated_nested_cv"

    if args.mode == "repeated":
        df, runtime = repeated_cv(args.project_root)
        save_outputs(df, out_dir, "T266_repeatedCV_5x10", runtime)

    if args.mode == "nested":
        start = time.time()
        df = nested_cv(args.project_root)
        runtime = time.time() - start
        save_outputs(df, out_dir, "nestedCV_light_T266_T271_T286", runtime)


if __name__ == "__main__":
    main()
