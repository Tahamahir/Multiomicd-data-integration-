import json
import glob
import csv
import re
import os

rows = []

json_files = sorted(glob.glob("response_*page*.json"))

for path in json_files:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for r in data.get("results", []):
        name = r.get("name", "")

        env_local = (
            ((r.get("env_local_scale") or {}).get("term") or {}).get("name", "")
        ).strip().lower()

        env_medium = (
            ((r.get("env_medium") or {}).get("term") or {}).get("name", "")
        ).strip().lower()

        # On garde seulement la rhizosphère
        if env_local != "rhizosphere":
            continue

        # Extraire le sample code depuis le nom
        # Exemple:
        # "... - BESC-904-Co3_16_51 rhizosphere"
        match = re.search(r" - (.+?) rhizosphere$", name)
        sample_code = match.group(1).strip() if match else ""

        # Déduire le site
        if "Corvallis" in name:
            site = "Corvallis"
        elif "Clatskanie" in name:
            site = "Clatskanie"
        else:
            site = ""

        rows.append({
            "nmdc_biosample_id": r.get("id", ""),
            "sample_name_nmdc": name,
            "sample_code": sample_code,
            "site": site,
            "compartment": "rhizosphere",
            "env_medium": env_medium
        })

# Trier pour rendre la table plus lisible
rows.sort(key=lambda x: (x["site"], x["sample_code"]))

output_file = "biosample_map_rhizosphere.csv"

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "nmdc_biosample_id",
            "sample_name_nmdc",
            "sample_code",
            "site",
            "compartment",
            "env_medium"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Fichier créé : {output_file}")
print(f"Nombre de lignes rhizosphere : {len(rows)}")