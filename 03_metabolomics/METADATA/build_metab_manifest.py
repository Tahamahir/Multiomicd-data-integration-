import pandas as pd
from pathlib import Path
import re

BASE = Path(".")

FILES = [
    "20220211_JGI_MD_507130_BioS_final1_IDX_C18_USDAY59443_RENAME_negative_metadata (1).tab",
    "20220211_JGI_MD_507130_BioS_final1_IDX_C18_USDAY59443_RENAME_positive_metadata.tab",
    "20220119_JGI_MD_507130_BioS_final1_QE-139_HILICZ_USHXG01825_RENAME_positive_metadata.tab",
    # ajoute ici le HILICZ negative quand tu l'auras
    "20220119_JGI_MD_507130_BioS_final1_QE-139_HILICZ_USHXG01825_RENAME_negative_metadata.tab",
]

all_dfs = []

for fname in FILES:
    path = BASE / fname
    if not path.exists():
        print(f"[SKIP] fichier absent: {fname}")
        continue

    df = pd.read_csv(path, sep="\t")

    # colonnes attendues
    expected = ["filename", "ATTRIBUTE_sampletype", "CONTROL", "CASE", "ATTRIBUTE_media"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {fname}: {missing}")

    df = df.copy()
    df["source_metadata_file"] = fname

    # méthode
    if "IDX_C18" in fname:
        df["method"] = "C18"
    elif "HILICZ" in fname:
        df["method"] = "HILICZ"
    else:
        df["method"] = ""

    # polarité
    low = fname.lower()
    if "negative" in low:
        df["polarity"] = "negative"
    elif "positive" in low:
        df["polarity"] = "positive"
    else:
        df["polarity"] = ""

    # nom de fichier simple
    df["basename"] = df["filename"].astype(str).str.split("/").str[-1]

    # compartiment
    def get_compartment(x: str) -> str:
        x = str(x).lower()
        if "-rhizo-s1" in x:
            return "rhizosphere"
        if "-root-s1" in x:
            return "root"
        if "-leaf-s1" in x:
            return "leaf"
        return ""

    df["compartment"] = df["basename"].apply(get_compartment)

    # QC / ISTD
    # QC / ISTD / contrôles expérimentaux
    df["is_qc"] = (
        df["basename"].str.contains("QC", case=False, na=False) |
        df["ATTRIBUTE_sampletype"].astype(str).str.contains("qc", case=False, na=False)
    )

    df["is_istd"] = (
        df["basename"].str.contains("ISTD", case=False, na=False) |
        df["ATTRIBUTE_sampletype"].astype(str).str.contains("istd|estd", case=False, na=False)
    )

    df["is_exctrl"] = (
        df["ATTRIBUTE_sampletype"].astype(str).str.contains("exctrl", case=False, na=False)
    )

    df["is_biological"] = ~(df["is_qc"] | df["is_istd"] | df["is_exctrl"])

    # run id
    df["run_id"] = df["basename"].str.extract(r"Run(\d+)", expand=False)

    # sample label entre MSMS_<code>_ et _Rg...
    # ex: ..._MSMS_83RM_BESC-905-Clat-RM_1_Rg80to1200...
    df["sample_label"] = df["basename"].str.extract(r"MSMS_[^_]+_(.+?)_Rg", expand=False)

    # normalisation légère
    df["sample_label_norm"] = (
        df["sample_label"]
        .astype(str)
        .str.strip()
        .str.replace("-Clat-", "-CLAT-", regex=False)
        .str.replace("-Corv-", "-CORV-", regex=False)
        .str.upper()
    )

    all_dfs.append(df)

if not all_dfs:
    raise ValueError("Aucun fichier metadata trouvé.")

manifest = pd.concat(all_dfs, ignore_index=True)

# sauvegarde complète
manifest.to_csv("metab_manifest_all.csv", index=False)

# seulement les runs biologiques rhizosphère
rhizo_bio = manifest[
    (manifest["compartment"] == "rhizosphere") &
    (manifest["is_biological"])
].copy()

rhizo_bio.to_csv("metab_manifest_rhizo_bio.csv", index=False)

print("Fichier créé : metab_manifest_all.csv")
print("Fichier créé : metab_manifest_rhizo_bio.csv")
print(f"Total runs metadata : {manifest.shape[0]}")
print(f"Runs rhizosphere biologiques : {rhizo_bio.shape[0]}")

print("\nRépartition par méthode / polarité :")
print(
    rhizo_bio.groupby(["method", "polarity"])
    .size()
    .reset_index(name="n_runs")
    .to_string(index=False)
)