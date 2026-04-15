import pandas as pd
from pathlib import Path
import re

BASE = Path(".")
MANIFEST_FILE = BASE / "metagenomics_tsv_list.txt"
BRACKEN_DIR = BASE / "bracken_results"

# 1) mapping basename -> biosample_id depuis l'ancien arbre
manifest_lines = MANIFEST_FILE.read_text(encoding="utf-8").splitlines()

mapping_rows = []
for line in manifest_lines:
    line = line.strip()
    if not line.endswith(".tsv"):
        continue
    if "bracken_species" not in line:
        continue

    m = re.match(r"\./(nmdc_bsm-[^/]+)/[^/]+/([^/]+)$", line)
    if not m:
        continue

    biosample_id = m.group(1)
    old_basename = m.group(2)
    new_basename = old_basename.replace(
        "_kraken2_report_bracken_species.tsv",
        "_bracken_species.tsv"
    )

    mapping_rows.append({
        "biosample_id": biosample_id,
        "old_basename": old_basename,
        "new_basename": new_basename
    })

mapping_df = pd.DataFrame(mapping_rows).drop_duplicates()

if mapping_df.empty:
    raise ValueError("Aucun mapping biosample <-> fichier Bracken trouvé.")

mapping_df.to_csv("metagenomics_bracken_file_mapping.csv", index=False)

# 2) lire les fichiers Bracken locaux
bracken_files = sorted(BRACKEN_DIR.glob("*_bracken_species.tsv"))

if not bracken_files:
    raise ValueError("Aucun fichier *_bracken_species.tsv trouvé dans bracken_results/")

rows = []

for path in bracken_files:
    basename = path.name

    hit = mapping_df.loc[mapping_df["new_basename"] == basename]
    if hit.empty:
        continue

    biosample_id = hit.iloc[0]["biosample_id"]

    # format Bracken standard avec header
    df = pd.read_csv(path, sep="\t")

    # normaliser les noms de colonnes possibles
    colmap = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl == "name":
            colmap[c] = "name"
        elif cl in ["taxonomy_id", "taxid", "taxonomy id"]:
            colmap[c] = "taxid"
        elif cl in ["taxonomy_lvl", "taxonomy_lvl ", "taxonomy level", "taxonomy_lvl\n"]:
            colmap[c] = "taxonomy_lvl"
        elif cl in ["fraction_total_reads", "fraction total reads", "fraction"]:
            colmap[c] = "fraction_total_reads"
        elif cl in ["new_est_reads", "new est reads", "new_est_reads "]:
            colmap[c] = "new_est_reads"
        elif cl in ["kraken_assigned_reads", "kraken assigned reads"]:
            colmap[c] = "kraken_assigned_reads"
        elif cl in ["added_reads", "added reads"]:
            colmap[c] = "added_reads"

    df = df.rename(columns=colmap)

    required_cols = ["name", "taxid", "fraction_total_reads", "new_est_reads"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {basename}: {missing}. Colonnes trouvées: {list(df.columns)}")

    # si taxonomy_lvl existe, garder seulement S
    if "taxonomy_lvl" in df.columns:
        df = df[df["taxonomy_lvl"].astype(str).str.strip() == "S"].copy()

    df["name"] = df["name"].astype(str).str.strip()
    df["biosample_id"] = biosample_id
    df["source_file"] = basename

    # pct = fraction_total_reads * 100
    df["pct"] = pd.to_numeric(df["fraction_total_reads"], errors="coerce").fillna(0) * 100
    df["clade_reads"] = pd.to_numeric(df["new_est_reads"], errors="coerce").fillna(0)
    df["direct_reads"] = pd.to_numeric(df.get("kraken_assigned_reads", 0), errors="coerce").fillna(0)

    rows.append(df[["biosample_id", "source_file", "taxid", "name", "pct", "clade_reads", "direct_reads"]])

if not rows:
    raise ValueError("Aucune table d'espèces n'a pu être construite.")

long_df = pd.concat(rows, ignore_index=True)
long_df.to_csv("metagenomics_species_long.csv", index=False)

matrix_pct = long_df.pivot_table(
    index="biosample_id",
    columns="name",
    values="pct",
    aggfunc="sum",
    fill_value=0
).reset_index()

matrix_pct.to_csv("metagenomics_species_matrix_pct.csv", index=False)

feature_cols = [c for c in matrix_pct.columns if c != "biosample_id"]
presence = (matrix_pct[feature_cols] > 0).sum(axis=0)
min_samples = max(1, int(0.05 * matrix_pct.shape[0]))

keep_cols = ["biosample_id"] + presence[presence >= min_samples].index.tolist()
matrix_pct_filtered = matrix_pct[keep_cols].copy()

matrix_pct_filtered.to_csv("metagenomics_species_matrix_pct_filtered.csv", index=False)

print("Fichier créé : metagenomics_bracken_file_mapping.csv")
print("Fichier créé : metagenomics_species_long.csv")
print("Fichier créé : metagenomics_species_matrix_pct.csv")
print("Fichier créé : metagenomics_species_matrix_pct_filtered.csv")
print(f"Nombre de fichiers Bracken trouvés : {len(bracken_files)}")
print(f"Nombre de biosamples mappés : {long_df['biosample_id'].nunique()}")
print(f"Nombre de lignes long : {long_df.shape[0]}")
print(f"Taille matrice brute : {matrix_pct.shape[0]} lignes x {matrix_pct.shape[1]} colonnes")
print(f"Taille matrice filtrée : {matrix_pct_filtered.shape[0]} lignes x {matrix_pct_filtered.shape[1]} colonnes")
print(f"Seuil de prévalence : {min_samples} échantillons")