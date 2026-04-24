from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


# ============================================================
# PHASE 6 - METABOLITE PREDICTABILITY SCREENING (CV)
# ============================================================


def compute_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, rmse


def classify_r2(r2):
    if r2 > 0.1:
        return "strong"
    elif r2 > 0:
        return "moderate"
    else:
        return "weak"


def main():
    repo_root = Path(__file__).resolve().parents[2]

    input_dir = repo_root / "10_analysis" / "outputs" / "phase3_smart_reduction_imputed"
    output_dir = repo_root / "10_analysis" / "outputs" / "phase6_screening"
    output_dir.mkdir(parents=True, exist_ok=True)

    x_path = input_dir / "X_final_smart.csv"
    y_path = repo_root / "10_analysis" / "outputs" / "phase2_preprocessing_fixed" / "Y_ml_filtered_log1p.csv"  # IMPORTANT: tous les MB

    print("=" * 70)
    print("PHASE 6 - METABOLITE SCREENING (CROSS-VALIDATION)")
    print("=" * 70)

    X = pd.read_csv(x_path)
    Y = pd.read_csv(y_path)

    print(f"X shape : {X.shape}")
    print(f"Y shape : {Y.shape}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    results = []

    for i, metabolite in enumerate(Y.columns):
        y = Y[metabolite].values

        r2_scores = []
        rmse_scores = []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2, rmse = compute_metrics(y_test, y_pred)

            r2_scores.append(r2)
            rmse_scores.append(rmse)

        mean_r2 = np.mean(r2_scores)
        std_r2 = np.std(r2_scores)
        mean_rmse = np.mean(rmse_scores)

        zero_fraction = (y == 0).sum() / len(y)

        results.append({
            "metabolite": metabolite,
            "mean_r2": mean_r2,
            "std_r2": std_r2,
            "mean_rmse": mean_rmse,
            "zero_fraction": zero_fraction,
            "predictability_class": classify_r2(mean_r2)
        })

        if i % 50 == 0:
            print(f"Processed {i}/{len(Y.columns)} metabolites")

    df = pd.DataFrame(results).sort_values("mean_r2", ascending=False)

    df.to_csv(output_dir / "metabolite_predictability_screening.csv", index=False)

    # summary
    summary = df["predictability_class"].value_counts()

    summary.to_csv(output_dir / "predictability_summary.csv")

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(summary)
    print()

    print("Top 20 metabolites:")
    print(df.head(20))

    print()
    print("Screening completed.")


if __name__ == "__main__":
    main()