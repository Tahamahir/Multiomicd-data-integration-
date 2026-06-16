#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 41 - Biological interpretation of XAI network results

Inputs:
    10_analysis/outputs/phase40_publication_figures/
        - filtered_feature_metabolite_links.csv
        - biomarker_hubs_publication.csv
        - node_metrics_publication.csv

Outputs:
    10_analysis/outputs/phase41_biological_interpretation/
        - top_microbe_hubs.csv
        - top_soil_hubs.csv
        - top_metabolite_hubs.csv
        - top_links_by_metabolite.csv
        - top_links_by_feature.csv
        - feature_category_summary.csv
        - metabolite_interpretation_summary.csv
        - biological_story_summary.txt
        - biological_interpretation_report.xlsx
"""

from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(".")
PHASE40_DIR = PROJECT_ROOT / "10_analysis/outputs/phase40_publication_figures"
OUT_DIR = PROJECT_ROOT / "10_analysis/outputs/phase41_biological_interpretation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


SOIL_KEYWORDS = {
    "carbon": ["total_c", "organic", "carbon", "toc"],
    "nitrogen": ["no3", "nh4", "nitrif", "denit", "total_n", "nitrogen", "nitrate", "ammonium"],
    "texture": ["clay", "sand", "silt", "psize"],
    "moisture": ["moist", "water"],
    "ph": ["ph"],
    "minerals": ["zn", "fe", "mn", "mg", "ca", "k", "p_", "phosph"],
}


MICROBE_FUNCTION_HINTS = {
    "Pseudomonas": "plant-associated and rhizosphere-adapted bacteria; often involved in nutrient cycling and secondary metabolite interactions",
    "Rhizobium": "symbiotic/plant-associated bacteria frequently linked to nitrogen-related processes",
    "Methylotenera": "methylotrophic bacteria potentially linked to one-carbon metabolism",
    "Enterobacter": "plant-associated bacteria with broad metabolic capabilities",
    "Agrobacterium": "soil and plant-associated bacteria",
    "Paenibacillus": "soil bacteria often linked to plant growth promotion and nutrient mobilization",
    "Kitasatospora": "Actinobacteria-related taxon; often associated with secondary metabolism",
    "Streptomyces": "Actinobacteria known for secondary metabolite biosynthesis",
    "Nocardiopsis": "Actinobacteria often associated with stress tolerance and secondary metabolism",
    "Serratia": "plant-associated bacteria with potential rhizosphere activity",
    "Massilia": "soil/rhizosphere-associated bacteria",
    "Cupriavidus": "soil bacteria often associated with stress tolerance and metal resistance",
    "Flavobacterium": "soil and rhizosphere bacteria involved in organic matter transformation",
}


def short_label(x: str) -> str:
    x = str(x)
    if "|IK:" in x:
        mode = x.split("|IK:")[0]
        ik = x.split("|IK:")[1].split("-")[0]
        return f"{mode}|IK:{ik[:10]}"
    return x


def classify_soil_category(feature: str) -> str:
    f = str(feature).lower()
    for cat, keys in SOIL_KEYWORDS.items():
        if any(k in f for k in keys):
            return cat
    return "soil_other"


def classify_node_type_from_label(label: str) -> str:
    l = str(label).lower()
    if "c18" in l or "hilic" in l or "|ik:" in l and ("negative" in l or "positive" in l):
        return "metabolite"
    if l.startswith("soil") or "nitrif" in l or "denit" in l or "psize" in l or "moist" in l or "chem__" in l:
        return "soil"
    return "microbe"


def infer_feature_category(feature: str, node_type: str) -> str:
    if node_type == "soil":
        return classify_soil_category(feature)

    f = str(feature)
    for genus in MICROBE_FUNCTION_HINTS:
        if genus.lower() in f.lower():
            return f"microbe_{genus}"

    return "microbe_other"


def extract_genus_or_clean_name(feature: str) -> str:
    f = str(feature)

    if ";" in f:
        last = f.split(";")[-1].strip()
        if last:
            return last

    for genus in MICROBE_FUNCTION_HINTS:
        if genus.lower() in f.lower():
            return genus

    return short_label(f)


def load_inputs():
    links_path = PHASE40_DIR / "filtered_feature_metabolite_links.csv"
    hubs_path = PHASE40_DIR / "biomarker_hubs_publication.csv"
    metrics_path = PHASE40_DIR / "node_metrics_publication.csv"

    if not links_path.exists():
        raise FileNotFoundError(f"Missing: {links_path}")
    if not hubs_path.exists():
        raise FileNotFoundError(f"Missing: {hubs_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing: {metrics_path}")

    links = pd.read_csv(links_path)
    hubs = pd.read_csv(hubs_path)
    metrics = pd.read_csv(metrics_path)

    return links, hubs, metrics


def prepare_links(links):
    links = links.copy()

    if "feature_type" not in links.columns:
        links["feature_type"] = links["feature"].apply(
            lambda x: "soil" if any(k in str(x).lower() for k in ["soil", "nitrif", "denit", "psize", "moist", "chem__"]) else "microbe"
        )

    links["feature_category"] = links.apply(
        lambda r: infer_feature_category(r["feature"], r["feature_type"]), axis=1
    )

    links["feature_clean"] = links["feature"].apply(extract_genus_or_clean_name)
    links["metabolite_short"] = links["metabolite"].apply(short_label)

    links["importance"] = pd.to_numeric(links["importance"], errors="coerce")
    links = links.dropna(subset=["importance"])

    return links


def prepare_metrics(metrics):
    metrics = metrics.copy()

    if "node_type" not in metrics.columns:
        metrics["node_type"] = metrics["label"].apply(classify_node_type_from_label)

    metrics["label_short"] = metrics["label"].apply(short_label)
    metrics["clean_name"] = metrics["label"].apply(extract_genus_or_clean_name)

    metrics["biological_category"] = metrics.apply(
        lambda r: infer_feature_category(r["label"], r["node_type"]), axis=1
    )

    return metrics


def generate_tables(links, metrics):
    top_microbe_hubs = (
        metrics[metrics["node_type"] == "microbe"]
        .sort_values("hub_score", ascending=False)
        .head(30)
    )

    top_soil_hubs = (
        metrics[metrics["node_type"] == "soil"]
        .sort_values("hub_score", ascending=False)
        .head(30)
    )

    top_metabolite_hubs = (
        metrics[metrics["node_type"] == "metabolite"]
        .sort_values("hub_score", ascending=False)
        .head(30)
    )

    top_links_by_metabolite = (
        links.sort_values(["metabolite", "importance"], ascending=[True, False])
        .groupby("metabolite")
        .head(5)
        .sort_values(["metabolite", "importance"], ascending=[True, False])
    )

    top_links_by_feature = (
        links.sort_values(["feature", "importance"], ascending=[True, False])
        .groupby("feature")
        .head(5)
        .sort_values(["feature", "importance"], ascending=[True, False])
    )

    feature_category_summary = (
        links.groupby(["feature_type", "feature_category"])
        .agg(
            n_links=("importance", "count"),
            mean_importance=("importance", "mean"),
            max_importance=("importance", "max"),
            n_metabolites=("metabolite", "nunique"),
            n_features=("feature", "nunique"),
        )
        .reset_index()
        .sort_values(["feature_type", "mean_importance"], ascending=[True, False])
    )

    metabolite_interpretation_summary = (
        links.groupby("metabolite")
        .agg(
            n_links=("importance", "count"),
            mean_importance=("importance", "mean"),
            max_importance=("importance", "max"),
            n_microbe_links=("feature_type", lambda x: (x == "microbe").sum()),
            n_soil_links=("feature_type", lambda x: (x == "soil").sum()),
            top_feature=("feature_clean", lambda x: x.iloc[0] if len(x) > 0 else ""),
        )
        .reset_index()
    )

    metabolite_interpretation_summary["metabolite_short"] = metabolite_interpretation_summary["metabolite"].apply(short_label)

    return {
        "top_microbe_hubs": top_microbe_hubs,
        "top_soil_hubs": top_soil_hubs,
        "top_metabolite_hubs": top_metabolite_hubs,
        "top_links_by_metabolite": top_links_by_metabolite,
        "top_links_by_feature": top_links_by_feature,
        "feature_category_summary": feature_category_summary,
        "metabolite_interpretation_summary": metabolite_interpretation_summary,
    }


def generate_story(links, metrics, tables):
    top_microbes = tables["top_microbe_hubs"].head(10)
    top_soil = tables["top_soil_hubs"].head(10)
    top_mets = tables["top_metabolite_hubs"].head(10)

    n_links = len(links)
    n_metabolites = links["metabolite"].nunique()
    n_features = links["feature"].nunique()
    n_microbe_links = (links["feature_type"] == "microbe").sum()
    n_soil_links = (links["feature_type"] == "soil").sum()

    microbe_ratio = n_microbe_links / max(n_links, 1)
    soil_ratio = n_soil_links / max(n_links, 1)

    soil_categories = (
        links[links["feature_type"] == "soil"]["feature_category"]
        .value_counts()
        .head(5)
        .to_dict()
    )

    story = []
    story.append("Phase 41 - Biological interpretation summary")
    story.append("============================================")
    story.append("")
    story.append("1. Global structure")
    story.append(f"- The interpreted XAI network contains {n_links} selected featuremetabolite links.")
    story.append(f"- These links connect {n_features} microbial/soil features to {n_metabolites} metabolites.")
    story.append(f"- Microbial links represent {microbe_ratio:.1%} of selected links, while soil-related links represent {soil_ratio:.1%}.")
    story.append("")
    story.append("2. Main biological hubs")
    story.append("- Top metabolite hubs:")
    for _, r in top_mets.iterrows():
        story.append(f"  * {r['label_short']} | hub_score={r['hub_score']:.3f}, degree={r['degree']}")

    story.append("")
    story.append("- Top microbial hubs:")
    for _, r in top_microbes.iterrows():
        name = r["clean_name"]
        hint = MICROBE_FUNCTION_HINTS.get(name, "microbial feature with recurrent predictive contribution")
        story.append(f"  * {name} | hub_score={r['hub_score']:.3f}, degree={r['degree']} | {hint}")

    story.append("")
    story.append("- Top soil hubs:")
    for _, r in top_soil.iterrows():
        story.append(f"  * {r['label']} | hub_score={r['hub_score']:.3f}, degree={r['degree']}")

    story.append("")
    story.append("3. Soil interpretation")
    if soil_categories:
        story.append("- The most represented soil-related categories are:")
        for cat, count in soil_categories.items():
            story.append(f"  * {cat}: {count} links")
    else:
        story.append("- No soil variable was retained among the selected XAI links.")

    story.append("")
    story.append("4. Biological conclusion")
    story.append(
        "- The XAI-based interpretation suggests that metabolite prediction is not driven by a single variable, "
        "but by a structured combination of microbial and soil-related predictors."
    )
    story.append(
        "- Microbial features dominate the network globally, while soil variables such as carbon, nitrogen, "
        "texture or moisture-related variables act as environmental modulators."
    )
    story.append(
        "- These results should be interpreted as predictive and associative signals, not as direct causal mechanisms."
    )

    return "\n".join(story)


def save_outputs(tables, story):
    for name, df in tables.items():
        df.to_csv(OUT_DIR / f"{name}.csv", index=False)

    with open(OUT_DIR / "biological_story_summary.txt", "w") as f:
        f.write(story)

    xlsx_path = OUT_DIR / "biological_interpretation_report.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for name, df in tables.items():
            sheet = name[:31]
            df.to_excel(writer, sheet_name=sheet, index=False)

        story_df = pd.DataFrame({"biological_story": story.split("\n")})
        story_df.to_excel(writer, sheet_name="story_summary", index=False)


def main():
    print("[PHASE 41] Loading Phase 40 outputs...")
    links, hubs, metrics = load_inputs()

    print("[PHASE 41] Preparing data...")
    links = prepare_links(links)
    metrics = prepare_metrics(metrics)

    print("[PHASE 41] Generating biological interpretation tables...")
    tables = generate_tables(links, metrics)

    print("[PHASE 41] Generating biological story...")
    story = generate_story(links, metrics, tables)

    print("[PHASE 41] Saving outputs...")
    save_outputs(tables, story)

    print("\n[DONE] Phase 41 completed.")
    print(f"Output folder: {OUT_DIR}")
    print("\nMain files:")
    print(f"- {OUT_DIR / 'biological_story_summary.txt'}")
    print(f"- {OUT_DIR / 'biological_interpretation_report.xlsx'}")


if __name__ == "__main__":
    main()
