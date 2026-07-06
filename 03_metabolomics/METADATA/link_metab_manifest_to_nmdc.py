import pandas as pd

manifest_file = "metab_manifest_rhizo_bio.csv"
runs_map_file = "metabolomics_runs_map.csv"
output_file = "metab_manifest_rhizo_linked.csv"

manifest = pd.read_csv(manifest_file)
runs = pd.read_csv(runs_map_file)

manifest["basename"] = manifest["basename"].astype(str).str.strip()
manifest["ms_name"] = manifest["basename"].str.replace(".mzML", "", regex=False)

runs["ms_name"] = runs["ms_name"].astype(str).str.strip()

linked = manifest.merge(
    runs,
    on="ms_name",
    how="left"
)

linked.to_csv(output_file, index=False)

print(f"Fichier créé : {output_file}")
print(f"Nombre de lignes : {linked.shape[0]}")
print(f"Lignes avec dgms_id : {linked['dgms_id'].notna().sum()}")
print(f"Lignes sans dgms_id : {linked['dgms_id'].isna().sum()}")
