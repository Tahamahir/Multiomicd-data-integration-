#!/usr/bin/env python3

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# Louvain (robust import)
try:
    from community import community_louvain
except:
    raise ImportError("Install with: pip install python-louvain")


# =========================================================
# 1. LOAD DATA
# =========================================================
def load_links(path):
    df = pd.read_csv(path)

    # sécurité
    required = {"feature", "metabolite", "importance"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")

    return df


# =========================================================
# 2. FILTER NETWORK (CRITICAL STEP)
# =========================================================
def filter_edges(df, quantile=0.80):
    threshold = df["importance"].quantile(quantile)
    df_f = df[df["importance"] >= threshold].copy()

    print(f"[INFO] Edges before: {len(df)}, after filtering: {len(df_f)}")
    print(f"[INFO] Threshold used: {threshold:.4f}")

    return df_f


# =========================================================
# 3. BUILD GRAPH
# =========================================================
def build_graph(df):
    G = nx.Graph()

    for _, row in df.iterrows():
        f = row["feature"]
        m = row["metabolite"]
        w = float(row["importance"])

        if G.has_edge(f, m):
            G[f][m]["weight"] += w
        else:
            G.add_edge(f, m, weight=w)

    return G


# =========================================================
# 4. COMMUNITY DETECTION (BIO MODULES)
# =========================================================
def detect_modules(G):
    partition = community_louvain.best_partition(G, weight="weight")
    return partition


# =========================================================
# 5. HUBS ANALYSIS
# =========================================================
def compute_hubs(G):
    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G, weight="weight")

    df = pd.DataFrame({
        "node": list(degree.keys()),
        "degree": list(degree.values()),
        "betweenness": [betweenness[n] for n in degree.keys()]
    })

    df["score"] = df["degree"] * 0.7 + df["betweenness"] * 0.3
    return df.sort_values("score", ascending=False)


# =========================================================
# 6. FIGURE 1  CLEAN NETWORK (MODULES)
# =========================================================
def plot_network(G, partition, out_file):

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42, k=0.4)

    nodes = list(G.nodes())
    colors = [partition[n] for n in nodes]

    nx.draw_networkx_nodes(
        G, pos,
        node_color=colors,
        node_size=40,
        cmap=plt.cm.Set3,
        alpha=0.9
    )

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_edges(
        G, pos,
        alpha=0.2,
        width=[w * 2 for w in weights]
    )

    plt.title("MicrobiomeMetabolome Interaction Network (Biological Modules)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


# =========================================================
# 7. FIGURE 2  HUBS
# =========================================================
def plot_hubs(df, out_file):

    top = df.head(20)

    plt.figure(figsize=(10, 5))
    plt.bar(top["node"], top["score"])

    plt.xticks(rotation=90, fontsize=8)
    plt.ylabel("Hub Score")
    plt.title("Top Biological Hubs (MicrobiomeMetabolome Network)")
    plt.tight_layout()

    plt.savefig(out_file, dpi=300)
    plt.close()


# =========================================================
# 8. MAIN PIPELINE
# =========================================================
def main():

    project_root = Path(".")
    in_path = project_root / "10_analysis/outputs/phase39_biomarkers/links.csv"

    out_dir = project_root / "10_analysis/outputs/phase40_modules"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[PHASE 40] Loading data...")
    df = load_links(in_path)

    print("[PHASE 40] Filtering network...")
    df_f = filter_edges(df)

    print("[PHASE 40] Building graph...")
    G = build_graph(df_f)

    print(f"[INFO] Nodes: {G.number_of_nodes()}")
    print(f"[INFO] Edges: {G.number_of_edges()}")

    print("[PHASE 40] Detecting modules...")
    partition = detect_modules(G)

    print("[PHASE 40] Computing hubs...")
    hubs = compute_hubs(G)

    # Save outputs
    hubs.to_csv(out_dir / "biological_hubs.csv", index=False)
    nx.write_gexf(G, out_dir / "clean_network.gexf")

    print("[PHASE 40] Generating figures...")

    plot_network(G, partition, out_dir / "network_modules.png")
    plot_hubs(hubs, out_dir / "top_hubs.png")

    print("\n[SUCCESS] Phase 40 completed!")
    print("Output:", out_dir)


if __name__ == "__main__":
    main()
