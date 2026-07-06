import pandas as pd

x_file = "cohort_ml_strict.csv"
y_file = "metabolomics_target_matrix_area_filtered.csv"

Xdf = pd.read_csv(x_file)
Ydf = pd.read_csv(y_file)

# nettoyage texte
for df in [Xdf, Ydf]:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

# normaliser les biosample ids
Xdf["biosample_id_norm"] = (
    Xdf["nmdc_biosample_id"]
    .astype(str)
    .str.replace("nmdc:bsm-", "nmdc_bsm-", regex=False)
)

Ydf["biosample_id_norm"] = Ydf["biosample_id"].astype(str).str.strip()

# merge
merged = Xdf.merge(
    Ydf,
    on="biosample_id_norm",
    how="inner",
    suffixes=("", "_y")
)

# contrôle
print(f"Lignes X strict : {Xdf.shape[0]}")
print(f"Lignes Y : {Ydf.shape[0]}")
print(f"Lignes après merge : {merged.shape[0]}")
print(f"Colonnes après merge : {merged.shape[1]}")

# colonnes metadata à garder
metadata_cols = [
    c for c in [
        "sample_key", "site", "genotype", "replicate", "compartment",
        "host_id", "nmdc_biosample_id", "metagenome_accession",
        "metabolomics_id", "soil_pH", "soil_NH4", "soil_NO3",
        "soil_total_C", "soil_total_N", "keep_for_analysis", "notes",
        "gold_project_id", "metabolomics_present", "metabolomics_run_count",
        "cohort_status", "soil_match", "metagenomics_match", "biosample_id_norm"
    ] if c in merged.columns
]

# colonnes X "simples" sol + metadata techniques
x_simple_cols = [
    c for c in [
        "soil_pH", "soil_NH4", "soil_NO3", "soil_total_C", "soil_total_N"
    ] if c in merged.columns
]

# colonnes métagénomiques = tout ce qui vient après biosample_id dans X strict,
# sauf colonnes metadata connues
exclude_cols = set(metadata_cols + ["biosample_id", "biosample_id_norm"])
y_cols = [c for c in Ydf.columns if c not in ["biosample_id", "biosample_id_norm"]]

# détecter les colonnes métagénomiques dans X
meta_geno_cols = [
    c for c in Xdf.columns
    if c not in exclude_cols
    and c not in ["biosample_id"]
    and c not in x_simple_cols
    and c not in y_cols
]

# on garde comme X final = sol + métagénomique
X_final_cols = metadata_cols + x_simple_cols + meta_geno_cols

# retirer doublons éventuels tout en gardant l'ordre
seen = set()
X_final_cols = [c for c in X_final_cols if not (c in seen or seen.add(c))]

# construire tables finales
metadata_df = merged[metadata_cols].copy()
X_final = merged[[c for c in X_final_cols if c in merged.columns]].copy()
Y_final = merged[["biosample_id_norm"] + [c for c in y_cols if c in merged.columns]].copy()

merged.to_csv("ml_dataset_strict.csv", index=False)
metadata_df.to_csv("metadata_strict.csv", index=False)
X_final.to_csv("X_strict.csv", index=False)
Y_final.to_csv("Y_strict.csv", index=False)

print("Fichier créé : ml_dataset_strict.csv")
print("Fichier créé : metadata_strict.csv")
print("Fichier créé : X_strict.csv")
print("Fichier créé : Y_strict.csv")
print(f"Colonnes metadata : {metadata_df.shape[1]}")
print(f"Colonnes X : {X_final.shape[1]}")
print(f"Colonnes Y : {Y_final.shape[1]}")