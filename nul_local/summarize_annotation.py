# summarize_annotations.py
# Résume les annotations PubChem pour le rapport

import pandas as pd
import numpy as np

INPUT = "metabolites_annotated.csv"

df = pd.read_csv(INPUT)

# Nettoyage
df["nom"] = df["nom"].fillna("non trouve")
df["formule"] = df["formule"].fillna("")
df["statut"] = df["statut"].fillna("")

# ------------------------------------------------------------
# Classification simple par familles chimiques
# ------------------------------------------------------------

def classify_compound(name):
    n = str(name).lower()

    if n in ["non trouve", "pas d'inchikey", "nan"]:
        return "Non annoté"

    if any(x in n for x in [
        "flavone", "flavonol", "flavan", "chalcone", "benzopyran"
    ]):
        return "Flavonoïdes / chalcones"

    if any(x in n for x in [
        "oleate", "tetracosanoic", "hexadecanoic", "oic acid", "cholesterol",
        "urs-12"
    ]):
        return "Lipides / acides gras"

    if any(x in n for x in [
        "phenyl", "benzaldehyde", "benzoic", "benzyl", "methoxy"
    ]):
        return "Composés aromatiques / phénoliques"

    if any(x in n for x in [
        "succinic", "malic", "quinate", "picolinic", "keto"
    ]):
        return "Acides organiques"

    if any(x in n for x in [
        "phenylalanine", "pantothenic", "betaine"
    ]):
        return "Acides aminés / vitamines / osmolytes"

    if any(x in n for x in [
        "glucopyranoside", "glucoside", "gastrodin", "arbutin"
    ]):
        return "Glycosides"

    return "Autres composés annotés"


df["classe_chimique"] = df["nom"].apply(classify_compound)

# ------------------------------------------------------------
# Résumé général
# ------------------------------------------------------------

n_total = len(df)
n_found = (df["statut"] == "trouve").sum()
n_missing = n_total - n_found
success_rate = 100 * n_found / n_total

summary = pd.DataFrame({
    "indicateur": [
        "Nombre total de métabolites",
        "Métabolites annotés",
        "Métabolites non annotés",
        "Taux d'annotation (%)"
    ],
    "valeur": [
        n_total,
        n_found,
        n_missing,
        round(success_rate, 1)
    ]
})

summary.to_csv("annotation_summary.csv", index=False, encoding="utf-8-sig")

# ------------------------------------------------------------
# Résumé par classe chimique
# ------------------------------------------------------------

class_summary = (
    df.groupby("classe_chimique")
      .size()
      .reset_index(name="nombre")
      .sort_values("nombre", ascending=False)
)

class_summary.to_csv("annotation_class_summary.csv", index=False, encoding="utf-8-sig")

# ------------------------------------------------------------
# Table des métabolites annotés avec R2 si disponible
# ------------------------------------------------------------

cols = ["metabolite", "plateforme", "inchikey", "nom", "formule", "poids", "classe_chimique", "statut"]
if "r2_real" in df.columns:
    cols.append("r2_real")
    df = df.sort_values("r2_real", ascending=False)

df[cols].to_csv("metabolites_annotated_clean.csv", index=False, encoding="utf-8-sig")

# Top métabolites annotés
if "r2_real" in df.columns:
    top = df[df["statut"] == "trouve"].head(15)
    top[cols].to_csv("top_annotated_metabolites_by_r2.csv", index=False, encoding="utf-8-sig")

# ------------------------------------------------------------
# Génération d'un tableau LaTeX court
# ------------------------------------------------------------

latex_df = df[df["statut"] == "trouve"].copy()

if "r2_real" in latex_df.columns:
    latex_df = latex_df.sort_values("r2_real", ascending=False).head(10)
    latex_cols = ["nom", "formule", "classe_chimique", "r2_real"]
else:
    latex_df = latex_df.head(10)
    latex_cols = ["nom", "formule", "classe_chimique"]

latex_df = latex_df[latex_cols]

with open("top_annotated_metabolites_table.tex", "w", encoding="utf-8") as f:
    f.write(latex_df.to_latex(
        index=False,
        escape=True,
        float_format="%.3f",
        caption="Principaux métabolites annotés à partir des InChIKeys.",
        label="tab:ch6_annotated_metabolites"
    ))

print("=== Résumé annotation ===")
print(summary.to_string(index=False))
print("\n=== Classes chimiques ===")
print(class_summary.to_string(index=False))
print("\nFichiers générés :")
print("- annotation_summary.csv")
print("- annotation_class_summary.csv")
print("- metabolites_annotated_clean.csv")
print("- top_annotated_metabolites_by_r2.csv si r2_real existe")
print("- top_annotated_metabolites_table.tex")