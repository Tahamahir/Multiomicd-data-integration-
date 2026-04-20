import pandas as pd

Y_REC_FILE = "audit_reports/Y_recommended_targets.csv"
Y_DICT_FILE = "Y_dictionary.csv"

yrec = pd.read_csv(Y_REC_FILE)
ydict = pd.read_csv(Y_DICT_FILE)

for df in [yrec, ydict]:
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()

# harmoniser
if "metabolite_key" not in yrec.columns and "target" in yrec.columns:
    yrec = yrec.rename(columns={"target": "metabolite_key"})
elif "metabolite_key" in yrec.columns and "target" in yrec.columns:
    # si les deux existent, on garde metabolite_key et on enlève target
    yrec = yrec.drop(columns=["target"])

# merge
df = yrec.merge(ydict, on="metabolite_key", how="left", suffixes=("", "_dict"))

# score simple pour ranking
for c in ["n_samples_present", "Confidence Score", "Entropy Similarity"]:
    if c not in df.columns:
        df[c] = pd.NA

df["n_samples_present"] = pd.to_numeric(df["n_samples_present"], errors="coerce")
df["Confidence Score"] = pd.to_numeric(df["Confidence Score"], errors="coerce")
df["Entropy Similarity"] = pd.to_numeric(df["Entropy Similarity"], errors="coerce")

# annotation lisible
def best_label(row):
    for c in ["name", "inchikey", "kegg", "chebi"]:
        if c in row and pd.notna(row[c]) and str(row[c]).strip() != "":
            return str(row[c]).strip()
    return row["metabolite_key"]

df["best_label"] = df.apply(best_label, axis=1)

# score global
df["target_score"] = (
    df["n_samples_present"].fillna(0) * 1.0
    + df["Confidence Score"].fillna(0) * 20.0
    + df["Entropy Similarity"].fillna(0) * 10.0
)

# panel 1 : top présence
panel_top_present = df.sort_values(
    by=["n_samples_present", "Confidence Score", "Entropy Similarity"],
    ascending=[False, False, False],
    na_position="last"
).head(30).copy()

# panel 2 : top annotation/score
panel_top_scored = df.sort_values(
    by=["target_score", "n_samples_present"],
    ascending=[False, False],
    na_position="last"
).head(30).copy()

# panel 3 : équilibré par assay
panel_balanced_parts = []
if "assay" in df.columns:
    for assay, sub in df.groupby("assay"):
        sub2 = sub.sort_values(
            by=["target_score", "n_samples_present"],
            ascending=[False, False],
            na_position="last"
        ).head(10)
        panel_balanced_parts.append(sub2)

panel_balanced = pd.concat(panel_balanced_parts, ignore_index=True) if panel_balanced_parts else pd.DataFrame()

# colonnes utiles
keep_cols = [
    c for c in [
        "metabolite_key",
        "assay",
        "best_label",
        "name",
        "inchikey",
        "kegg",
        "chebi",
        "n_samples_present",
        "Confidence Score",
        "Entropy Similarity",
        "target_score"
    ] if c in df.columns
]

panel_top_present = panel_top_present[keep_cols].copy()
panel_top_scored = panel_top_scored[keep_cols].copy()
panel_balanced = panel_balanced[keep_cols].copy()

panel_top_present.to_csv("panel_top_present_targets.csv", index=False)
panel_top_scored.to_csv("panel_top_scored_targets.csv", index=False)
panel_balanced.to_csv("panel_balanced_by_assay_targets.csv", index=False)

print("Fichier créé : panel_top_present_targets.csv")
print("Fichier créé : panel_top_scored_targets.csv")
print("Fichier créé : panel_balanced_by_assay_targets.csv")
print(f"Nb cibles recommandées au total : {df.shape[0]}")
print(f"Panel top présence : {panel_top_present.shape[0]}")
print(f"Panel top score : {panel_top_scored.shape[0]}")
print(f"Panel équilibré par assay : {panel_balanced.shape[0]}")

if "assay" in panel_balanced.columns:
    print("\nRépartition panel équilibré :")
    print(panel_balanced["assay"].value_counts().to_string())