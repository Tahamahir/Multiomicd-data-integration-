import pandas as pd

cohort_file = "cohort_final_extended.csv"
meta_file = "metagenomics_species_matrix_pct_filtered.csv"

cohort = pd.read_csv(cohort_file)
meta = pd.read_csv(meta_file)

# nettoyage texte
for df in [cohort, meta]:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

# normaliser les IDs biosample
# cohort : nmdc:bsm-11-xxxx  -> nmdc_bsm-11-xxxx
cohort["biosample_id_norm"] = (
    cohort["nmdc_biosample_id"]
    .astype(str)
    .str.replace("nmdc:bsm-", "nmdc_bsm-", regex=False)
)

# meta : déjà sous forme nmdc_bsm-...
meta["biosample_id_norm"] = meta["biosample_id"].astype(str).str.strip()

# jointure par ID normalisé
merged = cohort.merge(
    meta,
    on="biosample_id_norm",
    how="left"
)

# indicateur de match
merged["metagenomics_match"] = merged["biosample_id"].notna().map({True: "yes", False: ""})

# sauvegarde full
merged.to_csv("cohort_ml_full.csv", index=False)

# dataset strict prêt ML
strict = merged[
    (merged["cohort_status"] == "ready") &
    (merged["soil_match"] == "yes") &
    (merged["metagenomics_match"] == "yes")
].copy()

strict.to_csv("cohort_ml_strict.csv", index=False)

print("Fichier créé : cohort_ml_full.csv")
print("Fichier créé : cohort_ml_strict.csv")
print(f"Lignes full : {merged.shape[0]}")
print(f"Colonnes full : {merged.shape[1]}")
print(f"Lignes strictes : {strict.shape[0]}")
print(f"Colonnes strictes : {strict.shape[1]}")
print(f"Lignes avec match métagénomique : {merged['metagenomics_match'].eq('yes').sum()}")
print(f"Lignes sans match métagénomique : {(merged['metagenomics_match'] == '').sum()}")

# contrôle rapide
print("\nExemple IDs cohorte normalisés :")
print(cohort["biosample_id_norm"].head().to_string(index=False))

print("\nExemple IDs métagénomique :")
print(meta["biosample_id_norm"].head().to_string(index=False))