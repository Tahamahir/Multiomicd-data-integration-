import pandas as pd
from pathlib import Path

ROOT = Path(".")
OUT = ROOT / "10_analysis/outputs/phase35_final_table"
OUT.mkdir(parents=True, exist_ok=True)

# Charger toutes les associations
df = pd.read_csv(ROOT / "10_analysis/outputs/phase33_mg_mb_associations/all_mg_mb_associations.csv")

# Filtrer les associations à haute confiance (rho >= 0.6, FDR < 0.01)
high_conf = df[(df["abs_rho"] >= 0.6) & (df["qvalue_fdr"] < 0.01)].copy()

# Ajouter la colonne "Putative role"
high_conf["Putative_role"] = high_conf["rho"].apply(lambda x: "putative_production_like" if x > 0 else "putative_consumption_like")

# Sélectionner et renommer les colonnes pour le tableau final
final_table = high_conf[["metabolite", "mg_feature", "rho", "qvalue_fdr", "Putative_role"]].copy()
final_table = final_table.rename(columns={
    "metabolite": "Metabolite",
    "mg_feature": "MG_feature",
    "rho": "Spearman_rho",
    "qvalue_fdr": "qvalue_FDR"
})

# Optionnel : trier par |rho| décroissant
final_table["abs_rho"] = final_table["Spearman_rho"].abs()
final_table = final_table.sort_values(by="abs_rho", ascending=False).drop(columns="abs_rho")

# Sauvegarder
final_table.to_csv(OUT / "MG_MB_high_confidence_table.csv", index=False)

print(f"Final table saved in {OUT / 'MG_MB_high_confidence_table.csv'}")
print(final_table.head(20))
