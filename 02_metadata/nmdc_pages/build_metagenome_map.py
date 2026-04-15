import json
import csv
import re

input_file = "metagenome_all.json"
output_file = "metagenome_map.csv"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

resources = data.get("resources", [])

rows = []

for r in resources:
    name = r.get("name", "")
    has_input = r.get("has_input", [])
    gold_ids = r.get("gold_sequencing_project_identifiers", [])

    nmdc_biosample_id = has_input[0] if has_input else ""
    gold_project_id = gold_ids[0].replace("gold:", "") if gold_ids else ""

    # sample_code = partie entre " - " et le dernier mot de compartiment
    sample_code = ""
    m = re.search(r" - (.+?) (rhizosphere|soil|endosphere)$", name)
    if m:
        sample_code = m.group(1).strip()

    # site
    if "Corvallis" in name:
        site = "Corvallis"
    elif "Clatskanie" in name:
        site = "Clatskanie"
    else:
        site = ""

    # compartment
    if name.endswith(" rhizosphere"):
        compartment = "rhizosphere"
    elif name.endswith(" soil"):
        compartment = "soil"
    elif name.endswith(" endosphere"):
        compartment = "endosphere"
    else:
        compartment = ""

    rows.append({
        "gold_project_id": gold_project_id,
        "nmdc_omics_id": r.get("id", ""),
        "nmdc_biosample_id": nmdc_biosample_id,
        "sample_name_nmdc": name,
        "sample_code": sample_code,
        "site": site,
        "compartment": compartment
    })

rows.sort(key=lambda x: (x["gold_project_id"], x["compartment"]))

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "gold_project_id",
            "nmdc_omics_id",
            "nmdc_biosample_id",
            "sample_name_nmdc",
            "sample_code",
            "site",
            "compartment"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Fichier créé : {output_file}")
print(f"Nombre de lignes : {len(rows)}")