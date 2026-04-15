import pandas as pd

input_file = "cohort_final_extended.csv"

df = pd.read_csv(input_file)

# Nettoyage simple
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str).str.strip()

# Dataset strict :
# - cohorte ready
# - match sol oui
strict_df = df[
    (df["cohort_status"] == "ready") &
    (df["soil_match"] == "yes")
].copy()

# Dataset quasi complet :
# - cohorte ready ou check
full_df = df.copy()

# Sauvegardes
strict_df.to_csv("analysis_ready_strict.csv", index=False)
full_df.to_csv("analysis_full.csv", index=False)

print("Fichier créé : analysis_ready_strict.csv")
print("Fichier créé : analysis_full.csv")
print(f"Nombre de lignes strictes : {strict_df.shape[0]}")
print(f"Nombre de lignes full : {full_df.shape[0]}")

# Voir les lignes sans sol
missing_soil = df[df["soil_match"] != "yes"][["host_id", "cohort_status"]]
print("\nLignes sans sol :")
print(missing_soil.to_string(index=False))