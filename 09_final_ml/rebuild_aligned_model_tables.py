import pandas as pd
import numpy as np
import re

X_FILE = "X_strict.csv"
Y_FILE = "Y_strict.csv"

def clean_text_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

def parse_left_censored(x):
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
    vals = s.dropna().astype(str).str.strip()
    if len(vals) == 0:
        return False
    pattern = r"^<?\s*-?\d+(\.\d+)?$"
    ok = vals.str.match(pattern).mean()
    return ok >= 0.8

# -----------------------------
# 1) Load
# -----------------------------
X = pd.read_csv(X_FILE)
Y = pd.read_csv(Y_FILE)

for df in [X, Y]:
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = clean_text_series(df[c])

# -----------------------------
# 2) Check ids
# -----------------------------
id_col = "biosample_id_norm"

if id_col not in X.columns:
    raise ValueError(f"{id_col} absent de {X_FILE}")
if id_col not in Y.columns:
    raise ValueError(f"{id_col} absent de {Y_FILE}")

# enlever placeholders Y
y_feature_cols = [c for c in Y.columns if c != id_col]
placeholder_cols = [c for c in y_feature_cols if c.endswith("|NAME:")]
Y = Y.drop(columns=placeholder_cols, errors="ignore")

# garder seulement l'intersection des ids
common_ids = sorted(set(X[id_col]) & set(Y[id_col]))

X = X[X[id_col].isin(common_ids)].copy()
Y = Y[Y[id_col].isin(common_ids)].copy()

# tri explicite par id
X = X.sort_values(id_col).reset_index(drop=True)
Y = Y.sort_values(id_col).reset_index(drop=True)

# audit dur
if not X[id_col].equals(Y[id_col]):
    raise ValueError("ERREUR: après tri, les ids de X et Y ne sont pas exactement dans le même ordre.")

# -----------------------------
# 3) Metadata
# -----------------------------
metadata_candidates = [
    "sample_key",
    "site",
    "genotype",
    "replicate",
    "compartment",
    "host_id",
    "nmdc_biosample_id",
    "biosample_id_norm",
    "metagenome_accession",
    "metabolomics_id",
    "gold_project_id",
    "cohort_status",
    "soil_match",
    "metagenomics_match",
]

metadata_cols = [c for c in metadata_candidates if c in X.columns]
metadata_model = X[metadata_cols].copy()

# -----------------------------
# 4) Build X_model_with_id
# -----------------------------
drop_from_X = set(metadata_cols) - {id_col}

# enlever versions simplifiées soil_* si les chem__ existent déjà
for c in ["soil_pH", "soil_NH4", "soil_NO3", "soil_total_C", "soil_total_N"]:
    if c in X.columns:
        drop_from_X.add(c)

# enlever colonnes non-features connues
for c in [
    "keep_for_analysis",
    "notes",
    "metabolomics_present",
    "metabolomics_run_count",
    "biosample_id",
    "Field Sample ID",
]:
    if c in X.columns:
        drop_from_X.add(c)

X_work = X.drop(columns=list(drop_from_X), errors="ignore").copy()

# convertir texte -> numérique si possible
for c in X_work.columns:
    if c == id_col:
        continue
    if X_work[c].dtype == "object":
        if is_numeric_like_series(X_work[c]):
            X_work[c] = X_work[c].apply(parse_left_censored)

for c in X_work.columns:
    if c == id_col:
        continue
    if X_work[c].dtype == "object":
        converted = pd.to_numeric(X_work[c], errors="coerce")
        non_na_original = X_work[c].notna().sum()
        non_na_converted = converted.notna().sum()
        if non_na_original == 0:
            X_work[c] = converted
        elif non_na_converted / max(non_na_original, 1) >= 0.8:
            X_work[c] = converted

# garder seulement numériques + id
numeric_cols = [c for c in X_work.columns if c == id_col or pd.api.types.is_numeric_dtype(X_work[c])]
X_model_with_id = X_work[numeric_cols].copy()

# enlever colonnes constantes, sauf id
constant_x = [
    c for c in X_model_with_id.columns
    if c != id_col and X_model_with_id[c].nunique(dropna=False) <= 1
]
X_model_with_id = X_model_with_id.drop(columns=constant_x, errors="ignore")

# -----------------------------
# 5) Build Y_model_with_id
# -----------------------------
constant_y = [
    c for c in Y.columns
    if c != id_col and Y[c].nunique(dropna=False) <= 1
]
Y_model_with_id = Y.drop(columns=constant_y, errors="ignore").copy()

# dernier audit
if not X_model_with_id[id_col].equals(Y_model_with_id[id_col]):
    raise ValueError("ERREUR: X_model_with_id et Y_model_with_id n'ont pas le même ordre d'ids.")

# -----------------------------
# 6) versions sans id
# -----------------------------
X_model = X_model_with_id.drop(columns=[id_col]).copy()
Y_model = Y_model_with_id.drop(columns=[id_col]).copy()

# -----------------------------
# 7) audit file
# -----------------------------
audit = pd.DataFrame({
    "row_index": range(len(X_model_with_id)),
    "biosample_id_norm_X": X_model_with_id[id_col],
    "biosample_id_norm_Y": Y_model_with_id[id_col],
})

audit["same_id"] = audit["biosample_id_norm_X"] == audit["biosample_id_norm_Y"]

# -----------------------------
# 8) save
# -----------------------------
metadata_model.to_csv("metadata_model.csv", index=False)
X_model_with_id.to_csv("X_model_with_id.csv", index=False)
Y_model_with_id.to_csv("Y_model_with_id.csv", index=False)
X_model.to_csv("X_model.csv", index=False)
Y_model.to_csv("Y_model.csv", index=False)
audit.to_csv("alignment_audit.csv", index=False)

print("Fichier créé : metadata_model.csv")
print("Fichier créé : X_model_with_id.csv")
print("Fichier créé : Y_model_with_id.csv")
print("Fichier créé : X_model.csv")
print("Fichier créé : Y_model.csv")
print("Fichier créé : alignment_audit.csv")

print(f"Lignes finales : {len(audit)}")
print(f"Colonnes X_model_with_id : {X_model_with_id.shape[1]}")
print(f"Colonnes Y_model_with_id : {Y_model_with_id.shape[1]}")
print(f"Ids parfaitement alignés : {audit['same_id'].all()}")

print("\nExemple des 10 premiers ids :")
print(audit.head(10).to_string(index=False))

print("\nColonnes Y supprimées (placeholders) :", len(placeholder_cols))
for c in placeholder_cols[:10]:
    print("-", c)

print("\nColonnes X supprimées (constantes) :", len(constant_x))
for c in constant_x[:10]:
    print("-", c)

print("\nColonnes Y supprimées (constantes) :", len(constant_y))
for c in constant_y[:10]:
    print("-", c)