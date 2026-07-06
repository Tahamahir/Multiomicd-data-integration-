from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


# ============================================================
# PHASE 10 - FINAL MB PREDICTION (LIVRABLE LABO)
# ============================================================


def main():
    repo_root = Path(__file__).resolve().parents[2]

    # -------- paths --------
    X_path = repo_root / "10_analysis" / "outputs" / "phase3_soil_dedup" / "X_deduplicated.csv"
    Y_path = repo_root / "10_analysis" / "outputs" / "phase2_preprocessing_fixed" / "Y_ml_filtered_log1p.csv"
    source_path = repo_root / "10_analysis" / "outputs" / "phase7_source_comparison" / "source_comparison.csv"

    output_dir = repo_root / "10_analysis" / "outputs" / "phase10_mb_prediction"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 10 - FINAL MB PREDICTION")
    print("=" * 70)

    # -------- load --------
    X = pd.read_csv(X_path)
    Y = pd.read_csv(Y_path)
    source_df = pd.read_csv(source_path)

    print(f"X shape : {X.shape}")
    print(f"Y shape : {Y.shape}")

    # ------------------------------------------------------------
    # Imputation des NaN dans X
    # ------------------------------------------------------------
    nan_before = int(X.isna().sum().sum())
    print(f"NaN in X before imputation: {nan_before}")

    X = X.copy()
    for col in X.columns:
        if X[col].isna().any():
            median_value = X[col].median()
            if pd.isna(median_value):
                median_value = 0
            X[col] = X[col].fillna(median_value)

    nan_after = int(X.isna().sum().sum())
    print(f"NaN in X after imputation : {nan_after}")

    # -------- select predictable MB --------
    selected = source_df[source_df["r2_fusion"] > 0.2].copy()
    metabolites = [m for m in selected["metabolite"] if m in Y.columns]

    print(f"Metabolites selected for prediction: {len(metabolites)}")

    # -------- prediction containers --------
    Y_pred_log = pd.DataFrame(index=X.index)
    Y_pred_original = pd.DataFrame(index=X.index)

    # -------- model per metabolite --------
    for i, m in enumerate(metabolites):
        print(f"[{i+1}/{len(metabolites)}] Predicting {m}")

        y = Y[m].values

        model = ExtraTreesRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            max_features="sqrt",
            min_samples_leaf=2
        )

        model.fit(X, y)

        y_pred_log = model.predict(X)
        y_pred_original = np.expm1(y_pred_log)

        Y_pred_log[m] = y_pred_log
        Y_pred_original[m] = y_pred_original

    # -------- save outputs --------
    Y_pred_log.to_csv(output_dir / "predicted_mb_log1p.csv", index=False)
    Y_pred_original.to_csv(output_dir / "predicted_mb_original_scale.csv", index=False)

    merged = pd.concat(
        [
            Y_pred_log.add_suffix("_log1p"),
            Y_pred_original.add_suffix("_original")
        ],
        axis=1
    )

    merged.to_csv(output_dir / "predicted_mb_full.csv", index=False)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Predicted metabolites: {len(metabolites)}")
    print(f"Samples: {X.shape[0]}")

    print("\nOutputs:")
    print(output_dir / "predicted_mb_log1p.csv")
    print(output_dir / "predicted_mb_original_scale.csv")
    print(output_dir / "predicted_mb_full.csv")

    print("\nPrediction completed successfully.")


if __name__ == "__main__":
    main()