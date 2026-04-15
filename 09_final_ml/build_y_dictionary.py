import pandas as pd

y_file = "Y_strict.csv"
long_file = "metabolomics_annotated_long.csv"

Y = pd.read_csv(y_file)
long_df = pd.read_csv(long_file)

for df in [Y, long_df]:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

y_cols = [c for c in Y.columns if c != "biosample_id_norm"]

# garder seulement les colonnes utiles
keep = [
    c for c in [
        "metabolite_key", "assay", "name", "inchikey", "kegg", "chebi",
        "Confidence Score", "Entropy Similarity", "Spectra with Annotation (n)"
    ]
    if c in long_df.columns
]

df = long_df[keep].copy()

# ne garder que les metabolite_key présents dans Y
df = df[df["metabolite_key"].isin(y_cols)].copy()

# pour chaque metabolite_key, prendre la meilleure ligne représentative
sort_cols = [c for c in ["Confidence Score", "Entropy Similarity", "Spectra with Annotation (n)"] if c in df.columns]
if sort_cols:
    df = df.sort_values(by=sort_cols, ascending=[False] * len(sort_cols), na_position="last")

dict_df = df.drop_duplicates(subset=["metabolite_key"], keep="first").copy()

# ajouter la prévalence dans Y
presence = (Y[y_cols] > 0).sum(axis=0).reset_index()
presence.columns = ["metabolite_key", "n_samples_present"]

dict_df = dict_df.merge(presence, on="metabolite_key", how="left")

# réordonner
ordered_cols = [
    c for c in [
        "metabolite_key", "assay", "name", "inchikey", "kegg", "chebi",
        "n_samples_present", "Confidence Score", "Entropy Similarity", "Spectra with Annotation (n)"
    ]
    if c in dict_df.columns
]

dict_df = dict_df[ordered_cols].copy()
dict_df.to_csv("Y_dictionary.csv", index=False)

print("Fichier créé : Y_dictionary.csv")
print(f"Nombre de colonnes Y décrites : {dict_df.shape[0]}")