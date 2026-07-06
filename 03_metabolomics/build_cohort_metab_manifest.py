import pandas as pd

cohort_file = "cohort_final_extended.csv"
manifest_file = "metab_manifest_rhizo_linked.csv"

cohort = pd.read_csv(cohort_file)
manifest = pd.read_csv(manifest_file)

# nettoyer
for df in [cohort, manifest]:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

# garder seulement les runs reliés NMDC
manifest = manifest[manifest["dgms_id"].notna() & (manifest["dgms_id"] != "")].copy()

# ne garder que les colonnes utiles
manifest_small = manifest[
    [
        "biosample_id",
        "dgms_id",
        "basename",
        "method",
        "polarity",
        "run_id",
        "sample_label"
    ]
].copy()

# jointure avec la cohorte
linked = cohort.merge(
    manifest_small,
    left_on="nmdc_biosample_id",
    right_on="biosample_id",
    how="left"
)

linked.to_csv("cohort_metabolomics_long.csv", index=False)

# résumé par biosample
summary = (
    manifest_small.groupby("biosample_id")
    .agg(
        metabolomics_run_count=("dgms_id", "count"),
        dgms_ids=("dgms_id", lambda x: ";".join(sorted(set(map(str, x))))),
        methods=("method", lambda x: ";".join(sorted(set(map(str, x))))),
        polarities=("polarity", lambda x: ";".join(sorted(set(map(str, x))))),
    )
    .reset_index()
)

summary.to_csv("cohort_metabolomics_summary.csv", index=False)

print("Fichier créé : cohort_metabolomics_long.csv")
print("Fichier créé : cohort_metabolomics_summary.csv")
print(f"Lignes long : {linked.shape[0]}")
print(f"Biosamples résumés : {summary.shape[0]}")

print("\\nDistribution du nombre de runs par biosample :")
print(summary["metabolomics_run_count"].value_counts().sort_index().to_string())