import pandas as pd
from pathlib import Path

BASE = Path("metabolomics_csv_bundle")

rows = []

def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def normalize_name(x):
    x = clean_text(x)
    return x.replace("|", "_").replace(";", "_")

csv_files = sorted(BASE.rglob("*.csv"))

if not csv_files:
    raise ValueError("Aucun CSV métabolomique trouvé dans metabolomics_csv_bundle")

for path in csv_files:
    parts = path.parts

    biosample_id = next((p for p in parts if p.startswith("nmdc_bsm-")), "")
    dgms_id = next((p for p in parts if p.startswith("nmdc_dgms-")), "")

    if not biosample_id or not dgms_id:
        continue

    fname = path.name

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    if "HILICZ" in fname:
        method = "HILICZ"
    elif "IDX_C18" in fname:
        method = "C18"
    else:
        method = "UNK"

    if "_POS_" in fname:
        polarity = "positive"
    elif "_NEG_" in fname:
        polarity = "negative"
    else:
        polarity = clean_text(df.get("Polarity", "")).lower() or "unknown"

    assay = f"{method}_{polarity}"

    for col in [
        "Mass Feature ID", "Sample Name", "Polarity", "Retention Time (min)", "m/z",
        "Intensity", "Area", "Confidence Score", "Entropy Similarity",
        "Spectra with Annotation (n)", "name", "inchikey", "kegg", "chebi"
    ]:
        if col not in df.columns:
            df[col] = ""

    num_cols = [
        "Retention Time (min)", "m/z", "Intensity", "Area",
        "Confidence Score", "Entropy Similarity", "Spectra with Annotation (n)"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    annotated = (
        df["name"].astype(str).str.strip().ne("") |
        df["inchikey"].astype(str).str.strip().ne("") |
        df["kegg"].astype(str).str.strip().ne("") |
        df["chebi"].astype(str).str.strip().ne("")
    )
    df = df[annotated].copy()

    if df.empty:
        continue

    df = df.sort_values(
        by=["Confidence Score", "Entropy Similarity", "Spectra with Annotation (n)", "Area"],
        ascending=[False, False, False, False],
        na_position="last"
    )

    df_best = df.drop_duplicates(subset=["Mass Feature ID"], keep="first").copy()

    def make_metabolite_key(row):
        inchikey = clean_text(row["inchikey"])
        kegg = clean_text(row["kegg"])
        chebi = clean_text(row["chebi"])
        name = normalize_name(row["name"])

        if inchikey:
            ann = f"IK:{inchikey}"
        elif kegg:
            ann = f"KEGG:{kegg}"
        elif chebi:
            ann = f"CHEBI:{chebi}"
        else:
            ann = f"NAME:{name}"

        return f"{assay}|{ann}"

    df_best["metabolite_key"] = df_best.apply(make_metabolite_key, axis=1)
    df_best["biosample_id"] = biosample_id
    df_best["dgms_id"] = dgms_id
    df_best["source_file"] = fname
    df_best["assay"] = assay

    rows.append(
        df_best[
            [
                "biosample_id", "dgms_id", "source_file", "assay",
                "Mass Feature ID", "Sample Name", "Polarity",
                "Retention Time (min)", "m/z", "Intensity", "Area",
                "Confidence Score", "Entropy Similarity",
                "Spectra with Annotation (n)", "name", "inchikey",
                "kegg", "chebi", "metabolite_key"
            ]
        ]
    )

if not rows:
    raise ValueError("Aucune feature annotée trouvée dans les CSV métabolomiques.")

long_df = pd.concat(rows, ignore_index=True)
long_df.to_csv("metabolomics_annotated_long.csv", index=False)

matrix_area = long_df.pivot_table(
    index="biosample_id",
    columns="metabolite_key",
    values="Area",
    aggfunc="max",
    fill_value=0
).reset_index()

matrix_area.to_csv("metabolomics_target_matrix_area.csv", index=False)

feature_cols = [c for c in matrix_area.columns if c != "biosample_id"]
presence = (matrix_area[feature_cols] > 0).sum(axis=0)
min_samples = max(1, int(0.05 * matrix_area.shape[0]))

keep_cols = ["biosample_id"] + presence[presence >= min_samples].index.tolist()
matrix_area_filtered = matrix_area[keep_cols].copy()
matrix_area_filtered.to_csv("metabolomics_target_matrix_area_filtered.csv", index=False)

summary = (
    long_df.groupby(["biosample_id", "assay"])
    .size()
    .reset_index(name="n_annotated_features")
)
summary.to_csv("metabolomics_target_summary.csv", index=False)

print("Fichier créé : metabolomics_annotated_long.csv")
print("Fichier créé : metabolomics_target_matrix_area.csv")
print("Fichier créé : metabolomics_target_matrix_area_filtered.csv")
print("Fichier créé : metabolomics_target_summary.csv")
print(f"Biosamples traités : {long_df['biosample_id'].nunique()}")
print(f"Lignes long : {long_df.shape[0]}")
print(f"Taille matrice brute : {matrix_area.shape[0]} lignes x {matrix_area.shape[1]} colonnes")
print(f"Taille matrice filtrée : {matrix_area_filtered.shape[0]} lignes x {matrix_area_filtered.shape[1]} colonnes")
print(f"Seuil de prévalence : {min_samples} biosamples")
print("\\nRépartition des features annotées par assay :")
print(long_df.groupby('assay').size().to_string())