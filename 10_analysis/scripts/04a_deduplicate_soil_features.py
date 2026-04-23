from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PHASE 3A - DEDUPLICATE SOIL VARIABLES
# ------------------------------------------------------------
# Objectif :
# - identifier variables sol fortement corrélées
# - garder une seule variable par groupe redondant
# - produire un X nettoyé
# ============================================================


def detect_soil_columns(columns):
    soil_cols = []

    prefixes = [
        "soil_",
        "chem__",
        "psize_",
        "moist_",
        "nitrif_",
        "denit_",
    ]

    for col in columns:
        c = col.lower()
        if any(c.startswith(p) for p in prefixes):
            soil_cols.append(col)

    return soil_cols


def main():
    repo_root = Path(__file__).resolve().parents[2]

    input_dir = repo_root / "10_analysis" / "outputs" / "phase2_preprocessing_fixed"
    output_dir = repo_root / "10_analysis" / "outputs" / "phase3_soil_dedup"
    output_dir.mkdir(parents=True, exist_ok=True)

    x_path = input_dir / "X_ml_filtered.csv"

    print("=" * 70)
    print("PHASE 3A - SOIL DEDUPLICATION")
    print("=" * 70)
    print(f"Input X : {x_path}")
    print(f"Output dir : {output_dir}")
    print()

    X = pd.read_csv(x_path, low_memory=False)

    # ------------------------------------------------------------
    # 1. Séparer soil / non-soil
    # ------------------------------------------------------------
    soil_cols = detect_soil_columns(X.columns.tolist())
    X_soil = X[soil_cols].copy()
    X_other = X.drop(columns=soil_cols).copy()

    print(f"Total X columns      : {X.shape[1]}")
    print(f"Soil columns         : {len(soil_cols)}")
    print(f"Other columns        : {X_other.shape[1]}")
    print()

    # ------------------------------------------------------------
    # 2. Corrélation sol (Spearman)
    # ------------------------------------------------------------
    print("Computing soil correlation matrix...")
    corr = X_soil.corr(method="spearman").abs()

    # ------------------------------------------------------------
    # 3. Détection groupes redondants
    # ------------------------------------------------------------
    threshold = 0.95

    to_drop = set()
    groups = []

    cols = corr.columns

    for i in range(len(cols)):
        if cols[i] in to_drop:
            continue

        group = [cols[i]]

        for j in range(i + 1, len(cols)):
            if corr.iloc[i, j] >= threshold:
                group.append(cols[j])
                to_drop.add(cols[j])

        if len(group) > 1:
            groups.append(group)

    # ------------------------------------------------------------
    # 4. Choix de la variable à garder
    # (on garde celle avec + de variance)
    # ------------------------------------------------------------
    kept_cols = []
    removed_pairs = []

    for group in groups:
        variances = X_soil[group].var()
        best_col = variances.idxmax()

        kept_cols.append(best_col)

        for col in group:
            if col != best_col:
                removed_pairs.append({
                    "kept": best_col,
                    "removed": col,
                    "correlation": corr.loc[best_col, col]
                })

    # ajouter les colonnes non redondantes
    non_redundant = [c for c in soil_cols if c not in to_drop]
    final_soil_cols = list(set(kept_cols + non_redundant))

    # ------------------------------------------------------------
    # 5. Reconstruction X
    # ------------------------------------------------------------
    X_soil_dedup = X_soil[final_soil_cols]
    X_final = pd.concat([X_other, X_soil_dedup], axis=1)

    # ------------------------------------------------------------
    # 6. Sauvegarde
    # ------------------------------------------------------------
    pd.DataFrame(groups).to_csv(output_dir / "soil_groups_redundant.csv", index=False)
    pd.DataFrame(removed_pairs).to_csv(output_dir / "soil_removed_pairs.csv", index=False)
    pd.DataFrame({"kept_soil_columns": final_soil_cols}).to_csv(
        output_dir / "soil_columns_kept.csv", index=False
    )

    X_final.to_csv(output_dir / "X_deduplicated.csv", index=False)

    # ------------------------------------------------------------
    # Résumé console
    # ------------------------------------------------------------
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Soil columns before        : {len(soil_cols)}")
    print(f"Soil columns after         : {len(final_soil_cols)}")
    print(f"Removed soil variables     : {len(soil_cols) - len(final_soil_cols)}")
    print(f"Redundant groups detected  : {len(groups)}")
    print(f"Final X shape              : {X_final.shape}")
    print()
    print("Main outputs:")
    print(output_dir / "X_deduplicated.csv")
    print(output_dir / "soil_removed_pairs.csv")
    print(output_dir / "soil_columns_kept.csv")
    print()
    print("Soil deduplication completed.")


if __name__ == "__main__":
    main()