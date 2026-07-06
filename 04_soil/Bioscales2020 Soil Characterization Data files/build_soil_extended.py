import pandas as pd
from pathlib import Path

BASE = Path(".")

FILES = {
    "chem": "Bio-Scales_Soil_Chemistry_Public_Release.csv",
    "denit": "Bio-Scales_Soil_Denitrification_Public_Release.csv",
    "moist": "Bio-Scales_Soil_Moisure_Public_Release.csv",
    "nitrif": "Bio-Scales_Soil_Nitrification_Public_Release.csv",
    "psize": "Bio-Scales_Soil_ParticleSize_Public_Release.csv",
}

KEY_CANDIDATES = [
    "Field Sample ID",
    "FieldSampleID",
    "Sample ID",
    "SampleID",
]

def find_key(df: pd.DataFrame) -> str:
    for c in KEY_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"Aucune colonne clé trouvée parmi: {KEY_CANDIDATES}\nColonnes trouvées: {list(df.columns)}")

merged = None
summary = []

for prefix, filename in FILES.items():
    path = BASE / filename
    df = pd.read_csv(path)

    key = find_key(df)

    # nettoyer les espaces autour de la clé
    df[key] = df[key].astype(str).str.strip()

    # renommer la clé en nom standard
    df = df.rename(columns={key: "Field Sample ID"})

    # préfixer toutes les autres colonnes
    rename_map = {}
    for col in df.columns:
        if col != "Field Sample ID":
            rename_map[col] = f"{prefix}__{col}"
    df = df.rename(columns=rename_map)

    # supprimer doublons éventuels sur la clé
    df = df.drop_duplicates(subset=["Field Sample ID"]).copy()

    summary.append((prefix, filename, df.shape[0], df.shape[1]))

    if merged is None:
        merged = df
    else:
        merged = merged.merge(df, on="Field Sample ID", how="outer")

# tri pour lisibilité
merged = merged.sort_values("Field Sample ID").reset_index(drop=True)

# sauvegarde
merged.to_csv("soil_extended.csv", index=False)

print("Fichier créé : soil_extended.csv")
print(f"Nombre de lignes : {merged.shape[0]}")
print(f"Nombre de colonnes : {merged.shape[1]}")
print("\nRésumé des fichiers lus :")
for prefix, filename, nrows, ncols in summary:
    print(f"- {prefix}: {filename} -> {nrows} lignes, {ncols} colonnes")