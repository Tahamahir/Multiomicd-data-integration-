from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


# ============================================================
# PHASE 7 - SOURCE COMPARISON (SOIL vs MG vs FUSION)
# ============================================================


def compute_cv_r2(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

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

        scores.append(r2_score(y_test, y_pred))

    return np.mean(scores)


def classify_best(r2_soil, r2_mg, r2_fusion):
    values = {
        "soil": r2_soil,
        "mg": r2_mg,
        "fusion": r2_fusion
    }
    return max(values, key=values.get)


def main():
    repo_root = Path(__file__).resolve().parents[2]

    input_dir = repo_root / "10_analysis" / "outputs" / "phase3_smart_reduction_imputed"
    screening_dir = repo_root / "10_analysis" / "outputs" / "phase6_screening"
    output_dir = repo_root / "10_analysis" / "outputs" / "phase7_source_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------- fichiers --------
    X_path = input_dir / "X_final_smart.csv"
    Y_path = repo_root / "10_analysis" / "outputs" / "phase2_preprocessing_fixed" / "Y_ml_filtered_log1p.csv"
    screening_path = screening_dir / "metabolite_predictability_screening.csv"

    print("=" * 70)
    print("PHASE 7 - SOURCE COMPARISON")
    print("=" * 70)

    X = pd.read_csv(X_path)
    Y = pd.read_csv(Y_path)
    screening = pd.read_csv(screening_path)

    # -------- split X --------
    meta_cols = [c for c in X.columns if c.startswith("meta_PC")]
    soil_cols = [c for c in X.columns if not c.startswith("meta_PC")]

    X_mg = X[meta_cols]
    X_soil = X[soil_cols]
    X_fusion = X

    print(f"X_mg shape     : {X_mg.shape}")
    print(f"X_soil shape   : {X_soil.shape}")
    print(f"X_fusion shape : {X_fusion.shape}")

    # -------- sélectionner métabolites --------
    # on prend ceux avec R2 > 0.2 (forts)
    selected = screening[screening["mean_r2"] > 0.2]
    metabolites = selected["metabolite"].tolist()

    print(f"Metabolites selected: {len(metabolites)}")

    results = []

    for i, m in enumerate(metabolites):

        y = Y[m].values

        r2_soil = compute_cv_r2(X_soil, y)
        r2_mg = compute_cv_r2(X_mg, y)
        r2_fusion = compute_cv_r2(X_fusion, y)

        best = classify_best(r2_soil, r2_mg, r2_fusion)
        fusion_gain = r2_fusion - max(r2_soil, r2_mg)

        results.append({
            "metabolite": m,
            "r2_soil": r2_soil,
            "r2_mg": r2_mg,
            "r2_fusion": r2_fusion,
            "best_source": best,
            "fusion_gain": fusion_gain
        })

        if i % 10 == 0:
            print(f"Processed {i}/{len(metabolites)}")

    df = pd.DataFrame(results).sort_values("r2_fusion", ascending=False)

    df.to_csv(output_dir / "source_comparison.csv", index=False)

    # -------- résumé --------
    summary = df["best_source"].value_counts()
    summary.to_csv(output_dir / "source_summary.csv")

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(summary)
    print()

    print("Top metabolites:")
    print(df.head(20))

    print("\nSaved in:", output_dir)


if __name__ == "__main__":
    main()