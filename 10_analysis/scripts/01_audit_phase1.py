from pathlib import Path
import pandas as pd
import numpy as np
import json


# ============================================================
# PHASE 1 - AUDIT INITIAL DU DATASET
# ------------------------------------------------------------
# Ce script :
# 1) charge X et Y depuis 09_final_ml
# 2) vérifie les dimensions
# 3) vérifie l'alignement des samples
# 4) calcule la sparsité
# 5) sépare colonnes numériques / non numériques
# 6) produit un résumé et quelques fichiers de diagnostic
# ============================================================


def detect_sample_id_column(df: pd.DataFrame, preferred=None):
    """
    Essaie de détecter la colonne identifiant échantillon.
    On teste d'abord quelques noms probables.
    """
    candidates = preferred or [
        "biosample_id",
        "nmdc_biosample_id",
        "biosample_id_norm",
        "sample_id",
        "id"
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def compute_global_sparsity(df_numeric: pd.DataFrame) -> float:
    """
    Calcule la proportion globale de zéros dans une matrice numérique.
    """
    if df_numeric.empty:
        return np.nan
    return float((df_numeric == 0).sum().sum() / (df_numeric.shape[0] * df_numeric.shape[1]))


def compute_missing_rate(df: pd.DataFrame) -> float:
    """
    Proportion globale de valeurs manquantes.
    """
    if df.empty:
        return np.nan
    return float(df.isna().sum().sum() / (df.shape[0] * df.shape[1]))


def main():
    # ------------------------------------------------------------
    # Définition des chemins
    # ------------------------------------------------------------
    repo_root = Path(__file__).resolve().parents[2]

    x_path = repo_root / "09_final_ml" / "X_strict.csv"
    y_path = repo_root / "09_final_ml" / "Y_strict.csv"

    output_dir = repo_root / "10_analysis" / "outputs" / "phase1_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 1 - AUDIT INITIAL DU DATASET")
    print("=" * 70)
    print(f"Repo root : {repo_root}")
    print(f"X file    : {x_path}")
    print(f"Y file    : {y_path}")
    print(f"Output dir: {output_dir}")
    print()

    # ------------------------------------------------------------
    # Vérification existence fichiers
    # ------------------------------------------------------------
    if not x_path.exists():
        raise FileNotFoundError(f"Fichier X introuvable : {x_path}")

    if not y_path.exists():
        raise FileNotFoundError(f"Fichier Y introuvable : {y_path}")

    # ------------------------------------------------------------
    # Chargement des données
    # low_memory=False pour éviter les warnings inutiles
    # ------------------------------------------------------------
    print("Chargement de X...")
    X = pd.read_csv(x_path, low_memory=False)

    print("Chargement de Y...")
    Y = pd.read_csv(y_path, low_memory=False)

    print("Chargement terminé.")
    print()

    # ------------------------------------------------------------
    # Détection colonne ID
    # ------------------------------------------------------------
    x_id_col = detect_sample_id_column(
        X,
        preferred=[
            "biosample_id",
            "nmdc_biosample_id",
            "biosample_id_norm",
            "sample_id",
            "id",
        ],
    )

    y_id_col = detect_sample_id_column(
        Y,
        preferred=[
            "biosample_id",
            "nmdc_biosample_id",
            "biosample_id_norm",
            "sample_id",
            "id",
        ],
    )

    print(f"Colonne ID détectée dans X : {x_id_col}")
    print(f"Colonne ID détectée dans Y : {y_id_col}")
    print()

    # ------------------------------------------------------------
    # Si aucune colonne ID n'est trouvée
    # on suppose que l'ordre des lignes doit être comparé tel quel
    # ------------------------------------------------------------
    if x_id_col is None:
        X_ids = pd.Series(np.arange(len(X)), name="row_index")
    else:
        X_ids = X[x_id_col].astype(str).str.strip()

    if y_id_col is None:
        Y_ids = pd.Series(np.arange(len(Y)), name="row_index")
    else:
        Y_ids = Y[y_id_col].astype(str).str.strip()

    # ------------------------------------------------------------
    # Colonnes numériques / non numériques
    # ------------------------------------------------------------
    X_numeric = X.select_dtypes(include=[np.number]).copy()
    Y_numeric = Y.select_dtypes(include=[np.number]).copy()

    X_non_numeric_cols = [c for c in X.columns if c not in X_numeric.columns]
    Y_non_numeric_cols = [c for c in Y.columns if c not in Y_numeric.columns]

    # ------------------------------------------------------------
    # Vérification dimensions
    # ------------------------------------------------------------
    n_samples_x = X.shape[0]
    n_samples_y = Y.shape[0]
    n_cols_x = X.shape[1]
    n_cols_y = Y.shape[1]
    n_numeric_x = X_numeric.shape[1]
    n_numeric_y = Y_numeric.shape[1]

    # ------------------------------------------------------------
    # Alignement des IDs
    # ------------------------------------------------------------
    same_sample_count = len(set(X_ids) & set(Y_ids))
    same_order = False

    if len(X_ids) == len(Y_ids):
        same_order = bool((X_ids.reset_index(drop=True) == Y_ids.reset_index(drop=True)).all())

    ids_only_in_x = sorted(list(set(X_ids) - set(Y_ids)))[:20]
    ids_only_in_y = sorted(list(set(Y_ids) - set(X_ids)))[:20]

    # ------------------------------------------------------------
    # Sparsité et valeurs manquantes
    # ------------------------------------------------------------
    x_global_sparsity = compute_global_sparsity(X_numeric)
    y_global_sparsity = compute_global_sparsity(Y_numeric)

    x_missing_rate = compute_missing_rate(X)
    y_missing_rate = compute_missing_rate(Y)

    # ------------------------------------------------------------
    # Colonnes très creuses
    # ------------------------------------------------------------
    x_col_zero_fraction = pd.Series(dtype=float)
    y_col_zero_fraction = pd.Series(dtype=float)

    if not X_numeric.empty:
        x_col_zero_fraction = (X_numeric == 0).mean().sort_values(ascending=False)

    if not Y_numeric.empty:
        y_col_zero_fraction = (Y_numeric == 0).mean().sort_values(ascending=False)

    x_top_sparse_cols = x_col_zero_fraction.head(30).reset_index()
    x_top_sparse_cols.columns = ["column", "zero_fraction"]

    y_top_sparse_cols = y_col_zero_fraction.head(30).reset_index()
    y_top_sparse_cols.columns = ["column", "zero_fraction"]

    # ------------------------------------------------------------
    # Variance des colonnes numériques
    # utile plus tard pour voir les colonnes constantes
    # ------------------------------------------------------------
    x_variance = pd.Series(dtype=float)
    y_variance = pd.Series(dtype=float)

    if not X_numeric.empty:
        x_variance = X_numeric.var(numeric_only=True).sort_values()

    if not Y_numeric.empty:
        y_variance = Y_numeric.var(numeric_only=True).sort_values()

    x_low_variance = x_variance.head(30).reset_index()
    x_low_variance.columns = ["column", "variance"]

    y_low_variance = y_variance.head(30).reset_index()
    y_low_variance.columns = ["column", "variance"]

    # ------------------------------------------------------------
    # Résumé final
    # ------------------------------------------------------------
    summary = {
        "n_samples_x": int(n_samples_x),
        "n_samples_y": int(n_samples_y),
        "n_columns_x_total": int(n_cols_x),
        "n_columns_y_total": int(n_cols_y),
        "n_numeric_columns_x": int(n_numeric_x),
        "n_numeric_columns_y": int(n_numeric_y),
        "x_id_column": x_id_col,
        "y_id_column": y_id_col,
        "n_common_sample_ids": int(same_sample_count),
        "same_order_between_x_and_y": bool(same_order),
        "x_global_sparsity_zero_fraction": x_global_sparsity,
        "y_global_sparsity_zero_fraction": y_global_sparsity,
        "x_missing_rate": x_missing_rate,
        "y_missing_rate": y_missing_rate,
        "n_non_numeric_columns_x": int(len(X_non_numeric_cols)),
        "n_non_numeric_columns_y": int(len(Y_non_numeric_cols)),
    }

    # ------------------------------------------------------------
    # Sauvegardes
    # ------------------------------------------------------------
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)

    pd.DataFrame({"column": X_non_numeric_cols}).to_csv(
        output_dir / "x_non_numeric_columns.csv", index=False
    )

    pd.DataFrame({"column": Y_non_numeric_cols}).to_csv(
        output_dir / "y_non_numeric_columns.csv", index=False
    )

    pd.DataFrame({"sample_id_only_in_x": ids_only_in_x}).to_csv(
        output_dir / "sample_ids_only_in_x_preview.csv", index=False
    )

    pd.DataFrame({"sample_id_only_in_y": ids_only_in_y}).to_csv(
        output_dir / "sample_ids_only_in_y_preview.csv", index=False
    )

    x_top_sparse_cols.to_csv(output_dir / "x_top_sparse_columns.csv", index=False)
    y_top_sparse_cols.to_csv(output_dir / "y_top_sparse_columns.csv", index=False)

    x_low_variance.to_csv(output_dir / "x_low_variance_columns.csv", index=False)
    y_low_variance.to_csv(output_dir / "y_low_variance_columns.csv", index=False)

    # Descriptifs numériques compacts
    if not X_numeric.empty:
        X_numeric.describe().T.to_csv(output_dir / "x_numeric_describe.csv")

    if not Y_numeric.empty:
        Y_numeric.describe().T.to_csv(output_dir / "y_numeric_describe.csv")

    # ------------------------------------------------------------
    # Affichage console
    # ------------------------------------------------------------
    print("=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)
    print(f"X shape                       : {X.shape}")
    print(f"Y shape                       : {Y.shape}")
    print(f"X numeric columns             : {n_numeric_x}")
    print(f"Y numeric columns             : {n_numeric_y}")
    print(f"X non-numeric columns         : {len(X_non_numeric_cols)}")
    print(f"Y non-numeric columns         : {len(Y_non_numeric_cols)}")
    print(f"Common sample IDs             : {same_sample_count}")
    print(f"Same order X/Y                : {same_order}")
    print(f"X sparsity (zero fraction)    : {x_global_sparsity:.4f}" if pd.notna(x_global_sparsity) else "X sparsity (zero fraction)    : NaN")
    print(f"Y sparsity (zero fraction)    : {y_global_sparsity:.4f}" if pd.notna(y_global_sparsity) else "Y sparsity (zero fraction)    : NaN")
    print(f"X missing rate                : {x_missing_rate:.4f}" if pd.notna(x_missing_rate) else "X missing rate                : NaN")
    print(f"Y missing rate                : {y_missing_rate:.4f}" if pd.notna(y_missing_rate) else "Y missing rate                : NaN")
    print()
    print("Fichiers générés dans :")
    print(output_dir)
    print()

    print("Top 10 colonnes les plus creuses dans X :")
    print(x_top_sparse_cols.head(10).to_string(index=False))
    print()

    print("Top 10 colonnes les plus creuses dans Y :")
    print(y_top_sparse_cols.head(10).to_string(index=False))
    print()

    print("Audit phase 1 terminé avec succès.")


if __name__ == "__main__":
    main()