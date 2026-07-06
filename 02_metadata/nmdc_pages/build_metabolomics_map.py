import json
import csv
import re
import time
import requests
from collections import defaultdict

INPUT_FILE = "metabolome_all.json"
RUNS_OUTPUT = "metabolomics_runs_map.csv"
SUMMARY_OUTPUT = "metabolomics_biosample_summary.csv"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

resources = data.get("resources", [])

runs = []
biosample_to_runs = defaultdict(list)

procsm_cache = {}

for r in resources:
    dgms_id = r.get("id", "")
    ms_name = r.get("name", "")
    has_input = r.get("has_input", [])
    procsm_id = has_input[0] if has_input else ""

    if not procsm_id:
        continue

    # cache pour éviter de redemander le même procsm plusieurs fois
    if procsm_id not in procsm_cache:
        url = f"https://api.microbiomedata.org/nmdcschema/ids/{procsm_id}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        procsm_cache[procsm_id] = resp.json()
        time.sleep(0.05)

    procsm = procsm_cache[procsm_id]
    procsm_name = procsm.get("name", "")
    procsm_desc = procsm.get("description", "")

    biosample_id = ""

    # cas principal: nmdc:bsm-..._resuspended
    m = re.match(r"^(nmdc:bsm-[^_]+)_resuspended$", procsm_name)
    if m:
        biosample_id = m.group(1)

    # secours: dans la description
    if not biosample_id:
        m2 = re.search(r"(nmdc:bsm-[A-Za-z0-9-]+)", procsm_desc)
        if m2:
            biosample_id = m2.group(1)

    runs.append({
        "dgms_id": dgms_id,
        "ms_name": ms_name,
        "procsm_id": procsm_id,
        "biosample_id": biosample_id
    })

    if biosample_id:
        biosample_to_runs[biosample_id].append(dgms_id)

# fichier 1 : détail par run
with open(RUNS_OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["dgms_id", "ms_name", "procsm_id", "biosample_id"]
    )
    writer.writeheader()
    writer.writerows(runs)

# fichier 2 : résumé par biosample
summary_rows = []
for biosample_id, dgms_list in sorted(biosample_to_runs.items()):
    summary_rows.append({
        "biosample_id": biosample_id,
        "metabolomics_run_count": len(dgms_list),
        "metabolomics_run_ids": ";".join(dgms_list)
    })

with open(SUMMARY_OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["biosample_id", "metabolomics_run_count", "metabolomics_run_ids"]
    )
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"Fichier créé : {RUNS_OUTPUT}")
print(f"Fichier créé : {SUMMARY_OUTPUT}")
print(f"Nombre de runs MS : {len(runs)}")
print(f"Nombre de biosamples avec métabolomique : {len(summary_rows)}")