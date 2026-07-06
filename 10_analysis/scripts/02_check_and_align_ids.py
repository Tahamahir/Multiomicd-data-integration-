from pathlib import Path
import pandas as pd
import re


# ============================================================
# PHASE 1 BIS - CHECK & ALIGN SAMPLE IDS BETWEEN X AND Y
# ------------------------------------------------------------
# Ce script :
# 1) charge X_strict.csv et Y_strict.csv
# 2) détecte les colonnes ID
# 3) normalise les IDs dans un format commun
# 4) compare les IDs avant/après normalisation
# 5) crée des versions alignées de X et Y si possible
# ============================================================


def detect_sample_id_column(df: pd.DataFrame, preferred=None):
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


def normalize_sample_id(x):
    """
    Normalise différents formats possibles vers un format stable:
    nmdc_bsm-xxxx
    """
    if pd.isna(x):
        return None

    s = str(x).strip()

    # minuscules
    s = s.lower()

    # enlever espaces internes inutiles
    s = re.sub(r"\s+", "", s)

    # normaliser les préfixes connus
    s = s.replace("nmdc:bsm-", "nmdc_bsm-")
    s = s.replace("nmdc_bsm_", "nmdc_bsm-")
    s = s.replace("nmdc:bsm_", "nmdc_bsm-")
    s = s.replace("nmdc-bsm-", "nmdc_bsm-")
    s = s.replace("nmdc:bsm", "nmdc_bsm")
    s = s.replace("nmdc_bsm:", "nmdc_bsm-")

    return s


def main():
    repo_root = Path(__file__).resolve().parents[2]

    x_path = repo_root / "09_final_ml" / "X_strict.csv"
    y_path = repo_root / "09_final_ml" / "Y_strict.csv"

    output_dir = repo_root / "10_analysis" / "outputs" / "phase1_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 1 BIS - CHECK & ALIGN IDS")
    print("=" * 70)
    print(f"X file    : {x_path}")
    print(f"Y file    : {y_path}")
    print(f"Output dir: {output_dir}")
    print()

    X = pd.read_csv(x_path, low_memory=False)
    Y = pd.read_csv(y_path, low_memory=False)

    x_id_col = detect_sample_id_column(X, preferred=[
        "nmdc_biosample_id", "biosample_id", "biosample_id_norm", "sample_id", "id"
    ])
    y_id_col = detect_sample_id_column(Y, preferred=[
        "biosample_id_norm", "biosample_id", "nmdc_biosample_id", "sample_id", "id"
    ])

    if x_id_col is None:
        raise ValueError("Impossible de détecter la colonne ID dans X")
    if y_id_col is None:
        raise ValueError("Impossible de détecter la colonne ID dans Y")

    print(f"Colonne ID X : {x_id_col}")
    print(f"Colonne ID Y : {y_id_col}")
    print()

    X["sample_id_raw"] = X[x_id_col].astype(str).str.strip()
    Y["sample_id_raw"] = Y[y_id_col].astype(str).str.strip()

    X["sample_id_norm"] = X["sample_id_raw"].apply(normalize_sample_id)
    Y["sample_id_norm"] = Y["sample_id_raw"].apply(normalize_sample_id)

    # Comparaison avant normalisation
    common_raw = set(X["sample_id_raw"]) & set(Y["sample_id_raw"])

    # Comparaison après normalisation
    common_norm = set(X["sample_id_norm"]) & set(Y["sample_id_norm"])

    print(f"Nombre IDs communs AVANT normalisation : {len(common_raw)}")
    print(f"Nombre IDs communs APRES normalisation : {len(common_norm)}")
    print()

    only_in_x_raw = sorted(list(set(X["sample_id_raw"]) - set(Y["sample_id_raw"])))[:10]
    only_in_y_raw = sorted(list(set(Y["sample_id_raw"]) - set(X["sample_id_raw"])))[:10]

    only_in_x_norm = sorted(list(set(X["sample_id_norm"]) - set(Y["sample_id_norm"])))[:10]
    only_in_y_norm = sorted(list(set(Y["sample_id_norm"]) - set(X["sample_id_norm"])))[:10]

    print("Exemples IDs seulement dans X (raw) :")
    print(only_in_x_raw)
    print()

    print("Exemples IDs seulement dans Y (raw) :")
    print(only_in_y_raw)
    print()

    print("Exemples IDs seulement dans X (normalized) :")
    print(only_in_x_norm)
    print()

    print("Exemples IDs seulement dans Y (normalized) :")
    print(only_in_y_norm)
    print()

    # Sauvegarder previews
    pd.DataFrame({"sample_id_only_in_x_raw": only_in_x_raw}).to_csv(
        output_dir / "ids_only_in_x_raw_preview.csv", index=False
    )
    pd.DataFrame({"sample_id_only_in_y_raw": only_in_y_raw}).to_csv(
        output_dir / "ids_only_in_y_raw_preview.csv", index=False
    )
    pd.DataFrame({"sample_id_only_in_x_norm": only_in_x_norm}).to_csv(
        output_dir / "ids_only_in_x_norm_preview.csv", index=False
    )
    pd.DataFrame({"sample_id_only_in_y_norm": only_in_y_norm}).to_csv(
        output_dir / "ids_only_in_y_norm_preview.csv", index=False
    )

    # Garder uniquement l'intersection normalisée
    common_ids_sorted = sorted(list(common_norm))

    X_aligned = X[X["sample_id_norm"].isin(common_ids_sorted)].copy()
    Y_aligned = Y[Y["sample_id_norm"].isin(common_ids_sorted)].copy()

    # Réordonner selon sample_id_norm
    X_aligned = X_aligned.sort_values("sample_id_norm").reset_index(drop=True)
    Y_aligned = Y_aligned.sort_values("sample_id_norm").reset_index(drop=True)

    same_order_after = False
    if len(X_aligned) == len(Y_aligned):
        same_order_after = bool(
            (X_aligned["sample_id_norm"] == Y_aligned["sample_id_norm"]).all()
        )

    print(f"Taille X_aligned : {X_aligned.shape}")
    print(f"Taille Y_aligned : {Y_aligned.shape}")
    print(f"Same order after alignment : {same_order_after}")
    print()

    # Sauvegarder versions alignées
    X_aligned.to_csv(output_dir / "X_aligned_phase1.csv", index=False)
    Y_aligned.to_csv(output_dir / "Y_aligned_phase1.csv", index=False)

    # Sauvegarder une table de correspondance
    X[["sample_id_raw", "sample_id_norm"]].drop_duplicates().to_csv(
        output_dir / "x_id_mapping.csv", index=False
    )
    Y[["sample_id_raw", "sample_id_norm"]].drop_duplicates().to_csv(
        output_dir / "y_id_mapping.csv", index=False
    )

    print("Fichiers générés :")
    print(output_dir / "X_aligned_phase1.csv")
    print(output_dir / "Y_aligned_phase1.csv")
    print(output_dir / "x_id_mapping.csv")
    print(output_dir / "y_id_mapping.csv")
    print()
    print("Phase 1 bis terminée.")


if __name__ == "__main__":
    main()