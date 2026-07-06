#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import networkx as nx
import matplotlib.pyplot as plt


def load_all_xai(project_root):
    path = Path(project_root) / "10_analysis/outputs/phase38_xai"

    perm_files = list(path.glob("perm_importance_*.csv"))

    if len(perm_files) == 0:
        raise ValueError("No permutation importance files found in phase38_xai")

    return perm_files


def extract_links(files):
    rows = []

    for f in files:
        try:
            df = pd.read_csv(f)

            if "importance" not in df.columns:
                continue

            top = df.sort_values("importance", ascending=False).head(20)

            metabolite = f.name.replace("perm_importance_", "").replace(".csv", "")

            for _, r in top.iterrows():
                rows.append({
                    "feature": r["feature"],
                    "importance": r["importance"],
                    "metabolite": metabolite
                })

        except Exception as e:
            print(f"Skipping {f}: {e}")

    return pd.DataFrame(rows)


def build_network(df):
    G = nx.Graph()

    for _, row in df.iterrows():
        m = row["metabolite"]
        f = row["feature"]
        w = float(row["importance"])

        if G.has_edge(f, m):
            G[f][m]["weight"] += w
        else:
            G.add_edge(f, m, weight=w)

    return G


def compute_biomarkers(df):
    biom = df.groupby("feature").agg(
        mean_imp=("importance", "mean"),
        freq=("feature", "count")
    ).reset_index()

    biom["score"] = biom["mean_imp"] * biom["freq"]

    return biom.sort_values("score", ascending=False)


def plot_network(G, out_path):
    plt.figure(figsize=(14, 10))

    pos = nx.spring_layout(G, seed=42)

    weights = [G[u][v]["weight"] for u, v in G.edges()]

    nx.draw(
        G,
        pos,
        node_size=40,
        width=[w * 2 for w in weights],
        with_labels=False,
        alpha=0.7
    )

    plt.title("Microbiome  Metabolome Biomarker Network")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    project_root = Path(args.project_root)

    out_dir = project_root / "10_analysis/outputs/phase39_biomarkers"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = load_all_xai(project_root)

    df = extract_links(files)

    if df.empty:
        raise ValueError("No valid XAI data extracted")

    biom = compute_biomarkers(df)

    biom.to_csv(out_dir / "biomarkers_ranked.csv", index=False)
    df.to_csv(out_dir / "links.csv", index=False)

    G = build_network(df)

    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    plot_network(G, out_dir / "network.png")

    print("Saved in:", out_dir)


if __name__ == "__main__":
    main()
