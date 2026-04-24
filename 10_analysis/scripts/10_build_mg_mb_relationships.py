from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr


# ============================================================
# PHASE 8 - BUILD MG ↔ MB RELATIONSHIP TABLE
# ============================================================


def classify_relation(corr):
    if corr > 0.2:
        return "positive", "putative_production"
    elif corr < -0.2:
        return "negative", "putative_consumption"
    else:
        return "weak", "uncertain"


def main():
    repo_root = Path(__file__).resolve().parents[2]

    # -------- paths --------
    X_path = repo_root / "10_analysis" / "outputs" / "phase3_smart_reduction_imputed" / "X_final_smart.csv"
    Y_path = repo_root / "10_analysis" / "outputs" / "phase2_preprocessing_fixed" / "Y_ml_filtered_log1p.csv"
    source_path = repo_root / "10_analysis" / "outputs" / "phase7_source_comparison" / "source_comparison.csv"

    output_dir = repo_root / "10_analysis" / "outputs" / "phase8_mg_mb_relationships"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 8 - BUILD MG ↔ MB TABLE")
    print("=" * 70)

    X = pd.read_csv(X_path)
    Y = pd.read_csv(Y_path)
    source_df = pd.read_csv(source_path)

    # -------- split X --------
    meta_cols = [c for c in X.columns if c.startswith("meta_PC")]
    X_mg = X[meta_cols]

    # -------- sélectionner métabolites --------
    selected = source_df[
        (source_df["r2_fusion"] > 0.2) &
        (source_df["best_source"].isin(["mg", "fusion"]))
    ]

    metabolites = selected["metabolite"].tolist()

    print(f"Selected metabolites: {len(metabolites)}")

    all_rows = []

    for m in metabolites:

        print(f"Processing metabolite: {m}")

        y = Y[m].values

        # -------- modèle --------
        model = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X, y)

        importances = model.feature_importances_

        fi_df = pd.DataFrame({
            "feature": X.columns,
            "importance": importances
        }).sort_values("importance", ascending=False)

        # -------- garder MG uniquement --------
        fi_mg = fi_df[fi_df["feature"].str.startswith("meta_PC")]

        # -------- top features --------
        top_features = fi_mg.head(10)

        for _, row in top_features.iterrows():

            feature = row["feature"]
            importance = row["importance"]

            x_feat = X[feature].values

            corr, _ = spearmanr(x_feat, y)

            direction, role = classify_relation(corr)

            all_rows.append({
                "metabolite": m,
                "mg_feature": feature,
                "importance": importance,
                "spearman_corr": corr,
                "direction": direction,
                "putative_role": role
            })

    df = pd.DataFrame(all_rows)

    df = df.sort_values(["metabolite", "importance"], ascending=[True, False])

    df.to_csv(output_dir / "mg_mb_relationships.csv", index=False)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total relationships: {len(df)}")
    print(f"Metabolites covered: {df['metabolite'].nunique()}")

    print("\nSample:")
    print(df.head(20))

    print("\nSaved to:", output_dir)


if __name__ == "__main__":
    main()