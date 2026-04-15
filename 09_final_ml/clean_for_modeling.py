import re
import pandas as pd
import numpy as np

X_FILE = "X_strict.csv"
Y_FILE = "Y_strict.csv"
Y_DICT_FILE = "Y_dictionary.csv"

# -----------------------------
# Helpers
# -----------------------------
def clean_text_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

def parse_left_censored(x):
    """
    Convertit:
    - "<0.19" -> 0.095
    - "<0.20" -> 0.10
    - "0.45"  -> 0.45
    - "" / nan -> np.nan
    """
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if x == "" or x.lower() == "nan":
        return np.nan
    if x.startswith("<"):
        try:
            v = float(x[1:])
            return v / 2.0
        except:
            return np.nan
    try:
        return float(x)
    except:
        return np.nan

def is_numeric_like_series(s: pd.Series) -> bool:
    """
    Détecte si une colonne texte ressemble à une colonne numérique,
    y compris avec des valeurs censurées du type <0.19
    """
    vals = s.dropna().astype(str).str.strip()
    if len(vals) == 0:
        return False
    pattern = r"^<?\s*-?\d+(\.\d+)?$"
    ok = vals.str.match(pattern).mean()
    return ok >= 0.8

# -----------------------------
# Load
# -----------------------------
X = pd.read_csv(X_FILE)
Y = pd.read_csv(Y_FILE)

# facultatif
try:
    Y_dict = pd.read_csv(Y_DICT_FILE)
except:
    Y_dict = None

# nettoyage texte
for df in [X, Y]:
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = clean_text_series(df[c])

# -----------------------------
# 1) Nettoyage Y
# -----------------------------
id_col_y = "biosample_id_norm"

# enlever colonnes placeholder du type "...|NAME:"
y_feature_cols = [c for c in Y.columns if c != id_col_y]
placeholder_cols = [c for c in y_feature_cols if c.endswith("|NAME:")]

Y_model = Y.drop(columns=placeholder_cols, errors="ignore").copy()

# enlever éventuelles colonnes constantes dans Y
y_feature_cols = [c for c in Y_model.columns if c != id_col_y]
constant_y = [c for c in y_feature_cols if Y_model[c].nunique(dropna=False) <= 1]
Y_model = Y_model.drop(columns=constant_y, errors="ignore")

# -----------------------------
# 2) Séparer metadata dans X
# -----------------------------
metadata_candidates = [
    "sample_key",
    "site",
    "genotype",
    "replicate",
    "compartment",
    "host_id",
    "nmdc_biosample_id",
    "metagenome_accession",
    "metabolomics_id",
    "keep_for_analysis",
    "notes",
    "gold_project_id",
    "metabolomics_present",
    "metabolomics_run_count",
    "cohort_status",
    "soil_match",
    "metagenomics_match",
    "biosample_id_norm",
    "biosample_id",
    "Field Sample ID",
]

metadata_cols = [c for c in metadata_candidates if c in X.columns]
metadata_model = X[metadata_cols].copy()

# -----------------------------
# 3) Enlever colonnes non-features connues
# -----------------------------
drop_from_X = set(metadata_cols)

# éviter redondance: on garde les colonnes étendues du sol,
# on enlève les versions simplifiées soil_*
for c in ["soil_pH", "soil_NH4", "soil_NO3", "soil_total_C", "soil_total_N"]:
    if c in X.columns:
        drop_from_X.add(c)

X_work = X.drop(columns=list(drop_from_X), errors="ignore").copy()

# -----------------------------
# 4) Convertir les colonnes texte numériques
# -----------------------------
for c in X_work.columns:
    if X_work[c].dtype == "object":
        if is_numeric_like_series(X_work[c]):
            X_work[c] = X_work[c].apply(parse_left_censored)

# convertir ce qui reste possible en numérique
for c in X_work.columns:
    if X_work[c].dtype == "object":
        converted = pd.to_numeric(X_work[c], errors="coerce")
        # si au moins 80% des valeurs deviennent numériques, on garde la conversion
        non_na_original = X_work[c].notna().sum()
        non_na_converted = converted.notna().sum()
        if non_na_original == 0:
            X_work[c] = converted
        elif non_na_converted / max(non_na_original, 1) >= 0.8:
            X_work[c] = converted

# -----------------------------
# 5) Garder seulement les colonnes numériques dans X
# -----------------------------
numeric_cols = X_work.select_dtypes(include=[np.number]).columns.tolist()
X_model = X_work[numeric_cols].copy()

# enlever colonnes constantes
constant_x = [c for c in X_model.columns if X_model[c].nunique(dropna=False) <= 1]
X_model = X_model.drop(columns=constant_x, errors="ignore")

# -----------------------------
# 6) Aligner X et Y sur les mêmes biosamples
# -----------------------------
if "biosample_id_norm" not in metadata_model.columns:
    raise ValueError("La colonne biosample_id_norm manque dans X/metadata.")

common_ids = sorted(
    set(metadata_model["biosample_id_norm"].astype(str))
    & set(Y_model["biosample_id_norm"].astype(str))
)

metadata_model = metadata_model[metadata_model["biosample_id_norm"].isin(common_ids)].copy()
X_model = X_model.loc[metadata_model.index].copy()

Y_model = Y_model[Y_model["biosample_id_norm"].isin(common_ids)].copy()

# remettre le même ordre
metadata_model = metadata_model.sort_values("biosample_id_norm").reset_index(drop=True)
Y_model = Y_model.sort_values("biosample_id_norm").reset_index(drop=True)

# réaligner X sur le même ordre que metadata
tmp = metadata_model[["biosample_id_norm"]].copy()
tmp["row_order"] = range(len(tmp))

metadata_idx = X.loc[X["biosample_id_norm"].isin(common_ids), "biosample_id_norm"].reset_index(drop=False)
metadata_idx.columns = ["orig_index", "biosample_id_norm"]

order_df = tmp.merge(metadata_idx, on="biosample_id_norm", how="left").sort_values("row_order")
X_model = X_work.loc[order_df["orig_index"]].reset_index(drop=True)
X_model = X_model.select_dtypes(include=[np.number]).copy()
X_model = X_model.drop(columns=[c for c in X_model.columns if X_model[c].nunique(dropna=False) <= 1], errors="ignore")

# -----------------------------
# 7) Sauvegarde
# -----------------------------
metadata_model.to_csv("metadata_model.csv", index=False)
X_model.to_csv("X_model.csv", index=False)
Y_model.to_csv("Y_model.csv", index=False)

# logs
print("Fichier créé : metadata_model.csv")
print("Fichier créé : X_model.csv")
print("Fichier créé : Y_model.csv")
print(f"Lignes finales : {len(common_ids)}")
print(f"Colonnes metadata : {metadata_model.shape[1]}")
print(f"Colonnes X_model : {X_model.shape[1]}")
print(f"Colonnes Y_model : {Y_model.shape[1]}")

print("\nColonnes Y supprimées (placeholder) :")
print(len(placeholder_cols))
for c in placeholder_cols[:20]:
    print("-", c)

print("\nColonnes X supprimées car constantes :")
print(len(constant_x))
for c in constant_x[:20]:
    print("-", c)

print("\nColonnes Y supprimées car constantes :")
print(len(constant_y))
for c in constant_y[:20]:
    print("-", c)