from pathlib import Path
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================
# PHASE 3 - SMART DIMENSION REDUCTION
# ------------------------------------------------------------
# 1) charge X et Y preprocessed
# 2) sépare variables sol / metagenomics
# 3) standardise chaque bloc
# 4) applique PCA sur metagenomics seulement
# 5) garde top 50 métabolites les plus variables
# 6) sauvegarde matrices prêtes pour le premier modèle
# ============================================================


def detect_soil_columns(columns):
    """
    Détecte les colonnes liées au sol / chimie du sol.
    Tu pourras enrichir cette logique plus tard si besoin.
    """
    soil_cols = []
    keywords = [
        "soil",
        "chem__",
        "dry weight",
        "ph",
        "no3",
        "no2",
        "nh4",
        "moisture",
        "organic",
        "carbon",
        "nitrogen",
        "sand",
        "silt",
        "clay",
    ]

    for col in columns:
        col_low = col.lower()
        if any(k in col_low for k in keywords):
            soil_cols.append(col)

    return soil_cols


def main():
    repo_root = Path(__file__).resolve().parents[2]

    input_dir = repo_root / "10_analysis" / "outputs" / "phase2_preprocessing_fixed"
    output_dir = repo_root / "10_analysis" / "outputs" / "phase3_smart_reduction"
    output_dir.mkdir(parents=True, exist_ok=True)

    x_path = input_dir / "X_ml_filtered.csv"
    y_path = input_dir / "Y_ml_filtered_log1p.csv"
    ids_path = input_dir / "sample_ids.csv"

    print("=" * 70)
    print("PHASE 3 - SMART DIMENSION REDUCTION")
    print("=" * 70)
    print(f"X input   : {x_path}")
    print(f"Y input   : {y_path}")
    print(f"IDs input : {ids_path}")
    print(f"Output dir: {output_dir}")
    print()

    if not x_path.exists():
        raise FileNotFoundError(f"Missing file: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing file: {y_path}")
    if not ids_path.exists():
        raise FileNotFoundError(f"Missing file: {ids_path}")

    X = pd.read_csv(x_path, low_memory=False)
    Y = pd.read_csv(y_path, low_memory=False)
    sample_ids = pd.read_csv(ids_path, low_memory=False)

    if len(X) != len(Y) or len(X) != len(sample_ids):
        raise ValueError("X, Y and sample_ids must have the same number of rows.")

    print(f"X shape before split : {X.shape}")
    print(f"Y shape before reduction : {Y.shape}")
    print(f"Number of samples : {len(sample_ids)}")
    print()

    # ------------------------------------------------------------
    # 1. Séparer variables sol / metagenomics
    # ------------------------------------------------------------
    soil_cols = detect_soil_columns(X.columns.tolist())
    meta_cols = [c for c in X.columns if c not in soil_cols]

    X_soil = X[soil_cols].copy()
    X_meta = X[meta_cols].copy()

    print(f"Soil columns detected       : {len(soil_cols)}")
    print(f"Metagenomics columns        : {len(meta_cols)}")
    print(f"X_soil shape                : {X_soil.shape}")
    print(f"X_meta shape                : {X_meta.shape}")
    print()

    if X_meta.shape[1] == 0:
        raise ValueError("No metagenomics columns detected after soil split.")

    # ------------------------------------------------------------
    # 2. Standardisation
    # ------------------------------------------------------------
    meta_scaler = StandardScaler()
    X_meta_scaled = meta_scaler.fit_transform(X_meta)

    if X_soil.shape[1] > 0:
        soil_scaler = StandardScaler()
        X_soil_scaled = soil_scaler.fit_transform(X_soil)
        X_soil_scaled_df = pd.DataFrame(X_soil_scaled, columns=X_soil.columns)
    else:
        X_soil_scaled_df = pd.DataFrame(index=range(len(X)))

    # ------------------------------------------------------------
    # 3. PCA sur metagenomics seulement
    # ------------------------------------------------------------
    n_components = min(50, X_meta.shape[0] - 1, X_meta.shape[1])

    pca = PCA(n_components=n_components, random_state=42)
    X_meta_pca = pca.fit_transform(X_meta_scaled)

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    pca_cols = [f"meta_PC{i+1}" for i in range(X_meta_pca.shape[1])]
    X_meta_pca_df = pd.DataFrame(X_meta_pca, columns=pca_cols)

    print(f"PCA components kept         : {n_components}")
    print(f"Total explained variance    : {cumulative_variance[-1]:.4f}")
    print()

    # ------------------------------------------------------------
    # 4. Recomposer X final = PCA(meta) + soil standardisé
    # ------------------------------------------------------------
    X_final = pd.concat(
        [X_meta_pca_df.reset_index(drop=True), X_soil_scaled_df.reset_index(drop=True)],
        axis=1
    )

    print(f"X_final shape               : {X_final.shape}")
    print()

    # ------------------------------------------------------------
    # 5. Réduction de Y
    # Garder top 50 métabolites les plus variables
    # ------------------------------------------------------------
    y_variance = Y.var(axis=0).sort_values(ascending=False)
    n_top_y = min(50, Y.shape[1])
    top_y_cols = y_variance.head(n_top_y).index.tolist()

    Y_top = Y[top_y_cols].copy()

    y_scaler = StandardScaler()
    Y_top_scaled = pd.DataFrame(
        y_scaler.fit_transform(Y_top),
        columns=Y_top.columns
    )

    print(f"Top Y metabolites kept      : {n_top_y}")
    print(f"Y_top shape                 : {Y_top.shape}")
    print()

    # ------------------------------------------------------------
    # 6. Sauvegardes
    # ------------------------------------------------------------
    sample_ids.to_csv(output_dir / "sample_ids.csv", index=False)

    X_soil.to_csv(output_dir / "X_soil_raw.csv", index=False)
    X_meta.to_csv(output_dir / "X_metagenomics_raw.csv", index=False)
    X_meta_pca_df.to_csv(output_dir / "X_metagenomics_pca.csv", index=False)
    X_final.to_csv(output_dir / "X_final_smart.csv", index=False)

    Y_top.to_csv(output_dir / "Y_top50_log1p.csv", index=False)
    Y_top_scaled.to_csv(output_dir / "Y_top50_log1p_scaled.csv", index=False)

    pd.DataFrame({"soil_column": soil_cols}).to_csv(output_dir / "soil_columns_detected.csv", index=False)
    pd.DataFrame({"metagenomics_column": meta_cols}).to_csv(output_dir / "metagenomics_columns_detected.csv", index=False)

    pd.DataFrame({
        "component": pca_cols,
        "explained_variance_ratio": explained_variance,
        "cumulative_explained_variance": cumulative_variance
    }).to_csv(output_dir / "X_metagenomics_pca_explained_variance.csv", index=False)

    pd.DataFrame({
        "metabolite": top_y_cols,
        "variance": y_variance.head(n_top_y).values
    }).to_csv(output_dir / "Y_top50_selected_metabolites.csv", index=False)

    summary = {
        "n_samples": int(len(sample_ids)),
        "x_input_shape": [int(X.shape[0]), int(X.shape[1])],
        "y_input_shape": [int(Y.shape[0]), int(Y.shape[1])],
        "n_soil_columns": int(len(soil_cols)),
        "n_metagenomics_columns": int(len(meta_cols)),
        "x_meta_pca_shape": [int(X_meta_pca_df.shape[0]), int(X_meta_pca_df.shape[1])],
        "x_final_shape": [int(X_final.shape[0]), int(X_final.shape[1])],
        "y_top_shape": [int(Y_top.shape[0]), int(Y_top.shape[1])],
        "x_pca_n_components": int(n_components),
        "x_pca_total_explained_variance": float(cumulative_variance[-1]),
        "y_top_n_metabolites": int(n_top_y),
    }

    with open(output_dir / "phase3_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    pd.DataFrame([summary]).to_csv(output_dir / "phase3_summary.csv", index=False)

    # ------------------------------------------------------------
    # Console
    # ------------------------------------------------------------
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"X input shape                    : {X.shape}")
    print(f"Y input shape                    : {Y.shape}")
    print(f"Soil columns detected            : {len(soil_cols)}")
    print(f"Metagenomics columns detected    : {len(meta_cols)}")
    print(f"X PCA shape                      : {X_meta_pca_df.shape}")
    print(f"X final shape                    : {X_final.shape}")
    print(f"Y top metabolites shape          : {Y_top.shape}")
    print(f"Number of PCA components         : {n_components}")
    print(f"Total explained variance by PCA  : {cumulative_variance[-1]:.4f}")
    print()
    print("Main output files:")
    print(output_dir / "X_final_smart.csv")
    print(output_dir / "Y_top50_log1p.csv")
    print(output_dir / "Y_top50_log1p_scaled.csv")
    print(output_dir / "soil_columns_detected.csv")
    print(output_dir / "X_metagenomics_pca_explained_variance.csv")
    print()
    print("Phase 3 completed successfully.")


if __name__ == "__main__":
    main()