# annotate_metabolites.py — annote les metabolites via InChIKey (API PubChem)
import pandas as pd
import urllib.request
import json
import time

INPUT = "fdr_result_47.csv"        # contient la colonne 'metabolite'
OUTPUT = "metabolites_annotated.csv"

def parse_inchikey(metab):
    """Extrait l'InChIKey depuis 'C18_negative|IK:RGHHSNMVTDWUBI-UHFFFAOYSA-N'."""
    if "IK:" in metab:
        return metab.split("IK:")[1].strip()
    return None

def parse_platform(metab):
    """Extrait la plateforme analytique (C18_negative, HILICZ_positive...)."""
    return metab.split("|")[0].strip() if "|" in metab else ""

def query_pubchem(inchikey):
    """Interroge PubChem : InChIKey -> nom, formule, poids. Renvoie un dict."""
    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey"
    props = "IUPACName,MolecularFormula,MolecularWeight,Title"
    url = f"{base}/{inchikey}/property/{props}/JSON"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.loads(r.read().decode())
        p = data["PropertyTable"]["Properties"][0]
        cid = p.get("CID", "")
        # 'Title' est le nom courant ; IUPACName est le nom systematique
        name = p.get("Title") or p.get("IUPACName") or "—"
        return {
            "CID": cid,
            "nom": name,
            "iupac": p.get("IUPACName", ""),
            "formule": p.get("MolecularFormula", ""),
            "poids": p.get("MolecularWeight", ""),
            "statut": "trouve"
        }
    except Exception as e:
        return {"CID": "", "nom": "non trouve", "iupac": "", "formule": "",
                "poids": "", "statut": f"absent ({type(e).__name__})"}

# --- lecture ---
df = pd.read_csv(INPUT)
metabs = df["metabolite"].astype(str).tolist()
print(f"{len(metabs)} metabolites a annoter\n", flush=True)

rows = []
for i, m in enumerate(metabs):
    ik = parse_inchikey(m)
    plat = parse_platform(m)
    if ik is None:
        rows.append({"metabolite": m, "plateforme": plat, "inchikey": "",
                     "CID": "", "nom": "pas d'InChIKey", "formule": "",
                     "poids": "", "statut": "invalide"})
        print(f"[{i+1}/{len(metabs)}] pas d'InChIKey", flush=True)
        continue
    info = query_pubchem(ik)
    rows.append({"metabolite": m, "plateforme": plat, "inchikey": ik,
                 "CID": info["CID"], "nom": info["nom"],
                 "formule": info["formule"], "poids": info["poids"],
                 "statut": info["statut"]})
    print(f"[{i+1}/{len(metabs)}] {ik[:14]} -> {info['nom'][:45]}", flush=True)
    time.sleep(0.25)   # respecte la limite de l'API PubChem (5 req/s max)

out = pd.DataFrame(rows)
# on rattache les R2 si presents
if "r2_real" in df.columns:
    out = out.merge(df[["metabolite", "r2_real"]], on="metabolite", how="left")
out.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

# --- resume ---
n_ok = (out["statut"] == "trouve").sum()
print(f"\n=== RESULTAT ===")
print(f"Annotes avec succes : {n_ok}/{len(out)}")
print(f"Fichier ecrit : {OUTPUT}")
print("\nApercu :")
print(out[["nom", "formule", "poids"]].head(15).to_string(index=False))
