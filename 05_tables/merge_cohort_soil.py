import pandas as pd

cohort_file = "cohort_final.csv"
soil_file = "soil_extended.csv"
output_file = "cohort_final_extended.csv"

cohort = pd.read_csv(cohort_file)
soil = pd.read_csv(soil_file)

# nettoyer
cohort["host_id"] = cohort["host_id"].astype(str).str.strip()
soil["Field Sample ID"] = soil["Field Sample ID"].astype(str).str.strip()

# normaliser la casse des IDs Clatskanie
soil["Field Sample ID"] = soil["Field Sample ID"].str.replace("-Cl", "-CL", regex=False)

merged = cohort.merge(
    soil,
    left_on="host_id",
    right_on="Field Sample ID",
    how="left"
)

merged["soil_match"] = merged["Field Sample ID"].notna().map({True: "yes", False: ""})

merged.to_csv(output_file, index=False)

print(f"Fichier créé : {output_file}")
print(f"Nombre de lignes : {merged.shape[0]}")
print(f"Nombre de colonnes : {merged.shape[1]}")
print(f"Lignes avec match sol : {merged['soil_match'].eq('yes').sum()}")
print(f"Lignes sans match sol : {(merged['soil_match'] == '').sum()}")