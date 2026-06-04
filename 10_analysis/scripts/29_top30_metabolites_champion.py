from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import SparsePCA

warnings.filterwarnings("ignore")


ROOT = Path(".")
OUT = ROOT / "10_analysis/outputs/phase29_top30_metabolites_champion"
OUT.mkdir(parents=True, exist_ok=True)


def split_blocks(X):
    prefixes = ["soil_", "chem__", "psize__", "moist__", "nitrif__", "denit__"]
    soil_cols = [
        c for c in X.columns
        if any(str(c).lower().strip().startswith(p) for p in prefixes)
    ]
    mg_cols = [c for c in X.columns if c not in soil_cols]
    return mg_cols, soil_cols


def load_data():
    X = pd.read_csv(
        ROOT / "10_analysis/outputs/phase3_soil_dedup/X_deduplicated.csv",
        low_memory=False,
    )

    Y_all = pd.read_csv(
        ROOT / "10_analysis/outputs/phase2_preprocessing_fixed/Y_ml_filtered_log1p.csv",
        low_memory=False,
    )

    champion_metrics = pd.read_csv(
        ROOT / "10_analysis/outputs/phase26_tune_champion_late_sparsepca_rf/"
        "T266_mi500_spca75_a10_w7_rf_a_metrics_per_metabolite.csv"
    )

    top30 = (
        champion_metrics
        .sort_values("r2", ascending=False)
        .head(30)["metabolite"]
        .tolist()
    )

    top30 = [m for m in top30 if m in Y_all.columns]

    return X, Y_all[top30], top30


def select_mi(X_train, y_train, X_test, k=500):
    scores = mutual_info_regression(X_train, y_train, random_state=42)
    idx = np.argsort(scores)[::-1][:k]
    return X_train[:, idx], X_test[:, idx]


def predict_metabolite(X, y, mg_cols, soil_cols, cv):
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

        mg_train, mg_test = select_mi(mg_train, y_train, mg_test, k=500)

        reducer = SparsePCA(
            n_components=75,
            alpha=1.0,
            random_state=42,
            n_jobs=-1,
            max_iter=500,
        )

        mg_train = reducer.fit_transform(mg_train)
        mg_test = reducer.transform(mg_test)

        model_mg = RandomForestRegressor(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

        model_mg.fit(mg_train, y_train)
        pred_mg = model_mg.predict(mg_test)

        imp_soil = SimpleImputer(strategy="median")
        soil_train = imp_soil.fit_transform(X_train_soil)
        soil_test = imp_soil.transform(X_test_soil)

        scaler_soil = StandardScaler()
        soil_train = scaler_soil.fit_transform(soil_train)
        soil_test = scaler_soil.transform(soil_test)

        model_soil = RandomForestRegressor(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

        model_soil.fit(soil_train, y_train)
        pred_soil = model_soil.predict(soil_test)

        preds[test_idx] = 0.7 * pred_mg + 0.3 * pred_soil

    return preds


def main():
    start = time.time()

    X, Y, top30 = load_data()
    mg_cols, soil_cols = split_blocks(X)

    print(f"Loaded X={X.shape}, Y_top30={Y.shape}")
    print(f"MG={len(mg_cols)}, Soil={len(soil_cols)}")
    print("Running Top30 Champion with Repeated CV 5x10")

    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

    rows = []

    for i, metabolite in enumerate(Y.columns, start=1):
        print(f"[{i}/{len(Y.columns)}] {metabolite}")

        y = Y[metabolite].values
        pred = predict_metabolite(X, y, mg_cols, soil_cols, cv)

        rows.append({
            "metabolite": metabolite,
            "r2": r2_score(y, pred),
            "rmse": np.sqrt(mean_squared_error(y, pred)),
        })

    df = pd.DataFrame(rows).sort_values("r2", ascending=False)
    df.to_csv(OUT / "top30_repeatedCV_metrics.csv", index=False)

    summary = {
        "experiment": "phase29_top30_T266_repeatedCV_5x10",
        "n_metabolites": len(df),
        "mean_r2": float(df["r2"].mean()),
        "median_r2": float(df["r2"].median()),
        "std_r2": float(df["r2"].std()),
        "max_r2": float(df["r2"].max()),
        "min_r2": float(df["r2"].min()),
        "n_r2_gt_02": int((df["r2"] > 0.2).sum()),
        "n_r2_gt_04": int((df["r2"] > 0.4).sum()),
        "n_r2_gt_06": int((df["r2"] > 0.6).sum()),
        "runtime_sec": round(time.time() - start, 2),
    }

    pd.DataFrame([summary]).to_csv(OUT / "top30_repeatedCV_summary.csv", index=False)

    with open(OUT / "top30_repeatedCV_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
