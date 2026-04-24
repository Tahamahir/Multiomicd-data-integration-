from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from scipy.stats import spearmanr


# ============================================================
# PHASE 9 - SPECIES / MG FEATURE ↔ MB RELATIONSHIP TABLE
# ------------------------------------------------------------
# Objectif :
# - utiliser les vraies features métagénomiques, pas les PCA
# - sélectionner les métabolites prédictibles
# - entraîner un modèle par métabolite
# - extraire les features MG importantes
# - calculer Spearman(feature MG, métabolite)
# - annoter production/consommation putative
# ============================================================


def detect_soil_columns(columns):
    prefixes = [
        "soil_",
        "chem__",
        "psize_",
        "moist_",
        "nitrif_",
        "denit_",
    ]

    soil_cols = []
    for col in columns:
        c = col.lower().strip()
        if any(c.startswith(p) for p in prefixes):
            soil_cols.append(col)

    return soil_cols


def classify_relation(corr, weak_threshold=0.2):
    if pd.isna(corr):
        return "unknown", "uncertain"

    if corr >= weak_threshold:
        return "positive", "putative_production"
    elif corr <= -weak_threshold:
        return "negative", "putative_consumption"
    else:
        return "weak", "uncertain"


def confidence_label(importance, abs_corr):
    if importance >= 0.01 and abs_corr >= 0.30:
        return "high"
    elif importance >= 0.005 and abs_corr >= 0.20:
        return "medium"
    else:
        return "low"


def main():
    repo_root = Path(__file__).resolve().parents[2]

    # X brut filtré avec vraies features MG + soil
    x_path = repo_root / "10_analysis" / "outputs" / "phase3_soil_dedup" / "X_deduplicated.csv"

    # Y complet log1p, 652 métabolites
    y_path = repo_root / "10_analysis" / "outputs" / "phase2_preprocessing_fixed" / "Y_ml_filtered_log1p.csv"

    # comparaison des sources pour choisir les MB pertinents
    source_path = repo_root / "10_analysis" / "outputs" / "phase7_source_comparison" / "source_comparison.csv"

    output_dir = repo_root / "10_analysis" / "outputs" / "phase9_species_mb_relationships"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 9 - SPECIES / MG FEATURE ↔ MB RELATIONSHIPS")
    print("=" * 70)
    print(f"X input      : {x_path}")
    print(f"Y input      : {y_path}")
    print(f"Source input : {source_path}")
    print(f"Output dir   : {output_dir}")
    print()

    for p in [x_path, y_path, source_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    X = pd.read_csv(x_path, low_memory=False)
    Y = pd.read_csv(y_path, low_memory=False)
    source_df = pd.read_csv(source_path, low_memory=False)

    if len(X) != len(Y):
        raise ValueError("X and Y must have the same number of rows.")

    # ------------------------------------------------------------
    # 1. Séparer soil et vraies features MG
    # ------------------------------------------------------------
    soil_cols = detect_soil_columns(X.columns.tolist())
    mg_cols = [c for c in X.columns if c not in soil_cols]

    X_mg = X[mg_cols].copy()

    print(f"Total X columns       : {X.shape[1]}")
    print(f"Soil columns detected : {len(soil_cols)}")
    print(f"MG columns detected   : {len(mg_cols)}")
    print()

    # gérer NaN éventuels dans MG
    X_mg = X_mg.fillna(0)

    # ------------------------------------------------------------
    # 2. Sélection des métabolites candidats
    # ------------------------------------------------------------
    # On garde les MB prédictibles avec signal MG ou fusion
    selected = source_df[
        (source_df["r2_fusion"] > 0.20) &
        (source_df["best_source"].isin(["mg", "fusion"]))
    ].copy()

    metabolites = [m for m in selected["metabolite"].tolist() if m in Y.columns]

    print(f"Selected metabolites : {len(metabolites)}")
    print()

    if len(metabolites) == 0:
        raise ValueError("No selected metabolites found. Check thresholds or source_comparison.csv.")

    all_rows = []
    model_summary_rows = []

    # ------------------------------------------------------------
    # 3. Modèle par métabolite sur vraies MG features
    # ------------------------------------------------------------
    for idx, metabolite in enumerate(metabolites, start=1):
        print(f"[{idx}/{len(metabolites)}] Processing {metabolite}")

        y = Y[metabolite].values

        model = ExtraTreesRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            max_features="sqrt",
            min_samples_leaf=2
        )

        model.fit(X_mg, y)

        fi = pd.DataFrame({
            "mg_feature": X_mg.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        # On garde top 25 MG features par métabolite
        top_features = fi.head(25).copy()

        for _, row in top_features.iterrows():
            feature = row["mg_feature"]
            importance = float(row["importance"])

            corr, pval = spearmanr(X_mg[feature].values, y)
            abs_corr = abs(corr) if not pd.isna(corr) else np.nan

            direction, putative_role = classify_relation(corr)
            confidence = confidence_label(importance, abs_corr if not pd.isna(abs_corr) else 0)

            all_rows.append({
                "metabolite": metabolite,
                "mg_feature": feature,
                "importance": importance,
                "spearman_corr": corr,
                "spearman_pvalue": pval,
                "abs_spearman_corr": abs_corr,
                "direction": direction,
                "putative_role": putative_role,
                "confidence": confidence,
                "r2_soil": float(selected.loc[selected["metabolite"] == metabolite, "r2_soil"].iloc[0]),
                "r2_mg": float(selected.loc[selected["metabolite"] == metabolite, "r2_mg"].iloc[0]),
                "r2_fusion": float(selected.loc[selected["metabolite"] == metabolite, "r2_fusion"].iloc[0]),
                "best_source": selected.loc[selected["metabolite"] == metabolite, "best_source"].iloc[0],
                "fusion_gain": float(selected.loc[selected["metabolite"] == metabolite, "fusion_gain"].iloc[0]),
            })

        model_summary_rows.append({
            "metabolite": metabolite,
            "top_importance_sum": float(top_features["importance"].sum()),
            "n_top_features": int(len(top_features)),
            "best_source": selected.loc[selected["metabolite"] == metabolite, "best_source"].iloc[0],
            "r2_mg": float(selected.loc[selected["metabolite"] == metabolite, "r2_mg"].iloc[0]),
            "r2_fusion": float(selected.loc[selected["metabolite"] == metabolite, "r2_fusion"].iloc[0]),
        })

    relationships = pd.DataFrame(all_rows)

    # Trier : métabolite puis importance
    relationships = relationships.sort_values(
        ["metabolite", "importance"],
        ascending=[True, False]
    ).reset_index(drop=True)

    # Version filtrée : relations interprétables
    interpretable = relationships[
        (relationships["putative_role"].isin(["putative_production", "putative_consumption"])) &
        (relationships["confidence"].isin(["medium", "high"]))
    ].copy()

    # Résumés
    role_summary = (
        relationships["putative_role"]
        .value_counts()
        .rename_axis("putative_role")
        .reset_index(name="n_relationships")
    )

    confidence_summary = (
        relationships["confidence"]
        .value_counts()
        .rename_axis("confidence")
        .reset_index(name="n_relationships")
    )

    top_mg_features = (
        relationships.groupby("mg_feature")
        .agg(
            n_metabolites=("metabolite", "nunique"),
            mean_importance=("importance", "mean"),
            max_importance=("importance", "max"),
            mean_abs_corr=("abs_spearman_corr", "mean")
        )
        .reset_index()
        .sort_values(["n_metabolites", "max_importance"], ascending=[False, False])
    )

    model_summary = pd.DataFrame(model_summary_rows)

    # Sauvegardes
    relationships.to_csv(output_dir / "species_mb_relationships_all.csv", index=False)
    interpretable.to_csv(output_dir / "species_mb_relationships_interpretable.csv", index=False)
    role_summary.to_csv(output_dir / "relationship_role_summary.csv", index=False)
    confidence_summary.to_csv(output_dir / "relationship_confidence_summary.csv", index=False)
    top_mg_features.to_csv(output_dir / "top_mg_features_across_metabolites.csv", index=False)
    model_summary.to_csv(output_dir / "model_summary_by_metabolite.csv", index=False)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Metabolites covered                 : {relationships['metabolite'].nunique()}")
    print(f"Total MG-MB candidate relationships : {len(relationships)}")
    print(f"Interpretable relationships         : {len(interpretable)}")
    print()
    print("Role summary:")
    print(role_summary.to_string(index=False))
    print()
    print("Confidence summary:")
    print(confidence_summary.to_string(index=False))
    print()
    print("Top recurrent MG features:")
    print(top_mg_features.head(20).to_string(index=False))
    print()
    print("Main outputs:")
    print(output_dir / "species_mb_relationships_all.csv")
    print(output_dir / "species_mb_relationships_interpretable.csv")
    print(output_dir / "relationship_role_summary.csv")
    print(output_dir / "relationship_confidence_summary.csv")
    print(output_dir / "top_mg_features_across_metabolites.csv")
    print()
    print("Species/MG ↔ MB relationship table completed.")


if __name__ == "__main__":
    main()