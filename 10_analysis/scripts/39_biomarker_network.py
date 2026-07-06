#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import networkx as nx

import matplotlib.pyplot as plt


def load_xai(project_root):
    path = Path(project_root) / "10_analysis/outputs/phase38_xai"
    files = list(path.glob("perm_importance_*.csv"))
    return path, files


def build_biomarkers(files):
    all_features = []

    for f in files:
        df = pd.read_csv(f)

        # take top features per metabolite
        top = df.sort_values("importance", ascending=False).head(20)

        for _, row in top.iterrows():
            all_features.append({
                "feature": row["feature"],
                "importance": row["importance"],
                "metabolite": f.stem.replace("perm_importance_", "")
            })

    df_all = pd.DataFrame(all_features)

    biomarker_score = df_all.groupby("feature").agg(
        mean_importance=("importance", "mean"),
        freq=("feature", "count")
    ).reset_index()

    biomarker_score["score"] = biomarker_score["mean_importance"] * biomarker_score["freq"]

    return df_all, biomarker_score.sort_values("score", ascending=False)


def build_network(df_all):
    G = nx.Graph()

    for _, row in df_all.iterrows():
        microbe = row["feature"]
        metabolite = row["metabolite"]

        if G.has_edge(microbe, metabolite):
            G[microbe][metabolite]["weight"] += row["importance"]
        else:
            G.add_edge(microbe, metabolite, weight=row["importance"])

    return G


def plot_network(G, out_path):
    plt.figure(figsize=(12, 10))

    pos = nx.spring_layout(G, k=0.5)

    weights = [G[u][v]["weight"] for u, v in G.edges()]

    nx.draw(
        G,
        pos,
        with_labels=False,
        node_size=30,
        width=[w * 2 for w in weights]
    )

    plt.title("Microbiome  Metabolome Network")
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    project_root = Path(args.project_root)

    out_dir = project_root / "10_analysis/outputs/phase39_biomarkers"
    out_dir.mkdir(parents=True, exist_ok=True)

    path, files = load_xai(project_root)

    df_all, biomarker_score = build_biomarkers(files)

    biomarker_score.to_csv(out_dir / "biomarkers_ranked.csv", index=False)
    df_all.to_csv(out_dir / "feature_metabolite_links.csv", index=False)

    G = build_network(df_all)

    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    plot_network(G, out_dir / "network.png")

    print("Saved results in:", out_dir)


if __name__ == "__main__":
    main()
