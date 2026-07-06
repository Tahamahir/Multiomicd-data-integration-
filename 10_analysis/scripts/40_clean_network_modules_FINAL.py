#!/usr/bin/env python3

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from community import community_louvain


# =========================================================
# LOAD DATA
# =========================================================
def load_data(path):
    df = pd.read_csv(path)
    return df


# =========================================================
# BUILD GRAPH (NO EARLY FILTER)
# =========================================================
def build_graph(df):

    G = nx.Graph()

    for _, r in df.iterrows():
        w = float(r["importance"])

        G.add_edge(
            r["feature"],
            r["metabolite"],
            weight=w
        )

    return G


# =========================================================
# BACKBONE EXTRACTION (KEY IDEA NEW APPROACH)
# =========================================================
def backbone_graph(G):

    # 1. Maximum Spanning Tree (core structure)
    mst = nx.maximum_spanning_tree(G, weight="weight")

    # 2. add top 15% strongest edges
    edges = sorted(G.edges(data=True), key=lambda x: x[2]["weight"], reverse=True)
    top_edges = edges[:int(len(edges) * 0.15)]

    H = nx.Graph()
    H.add_edges_from(mst.edges(data=True))
    H.add_edges_from([(u, v, d) for u, v, d in top_edges])

    return H


# =========================================================
# MODULES
# =========================================================
def detect_modules(G):
    return community_louvain.best_partition(G, weight="weight")


# =========================================================
# HUB SCORE (MULTI METRIC)
# =========================================================
def compute_hubs(G):

    deg = dict(G.degree())
    bet = nx.betweenness_centrality(G, weight="weight")

    hubs = {}

    for n in G.nodes():
        hubs[n] = 0.7 * deg.get(n, 0) + 0.3 * bet.get(n, 0)

    return hubs


# =========================================================
# NETWORK PLOT (NEW STYLE CLEAN SCIENTIFIC)
# =========================================================
def plot_network(G, partition, hubs, out_file):

    plt.figure(figsize=(20, 14))

    pos = nx.spring_layout(G, seed=42, k=1.4, iterations=400)

    # ----------------------------
    # NODE SIZE = HUB SCORE
    # ----------------------------
    node_size = [max(20, hubs[n] * 120) for n in G.nodes()]

    # ----------------------------
    # NODE COLOR = MODULES
    # ----------------------------
    node_color = [partition[n] for n in G.nodes()]

    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_size,
        node_color=node_color,
        cmap=plt.cm.tab20,
        alpha=0.95
    )

    # ----------------------------
    # EDGE WIDTH = IMPORTANCE
    # ----------------------------
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights)

    nx.draw_networkx_edges(
        G, pos,
        width=[1 + 4 * (w / max_w) for w in weights],
        alpha=0.35,
        edge_color="gray"
    )

    # ----------------------------
    # LABEL ONLY TOP HUBS
    # ----------------------------
    top_nodes = sorted(hubs, key=hubs.get, reverse=True)[:10]

    labels = {n: n.split("|")[-1][:12] for n in top_nodes}

    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=10
    )

    plt.title("MicrobiomeMetabolome Backbone + Modules (NEW METHOD)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_file, dpi=600)
    plt.close()


# =========================================================
# HUBS FIGURE
# =========================================================
def plot_hubs(hubs, out_file):

    top = dict(sorted(hubs.items(), key=lambda x: x[1], reverse=True)[:20])

    plt.figure(figsize=(12,6))
    plt.bar(top.keys(), top.values(), color="#3498db")

    plt.xticks(rotation=90)
    plt.ylabel("Hub Score")
    plt.title("Top Biological Hubs (Multi-metric)")

    plt.tight_layout()
    plt.savefig(out_file, dpi=600)
    plt.close()


# =========================================================
# MAIN PIPELINE
# =========================================================
def main():

    root = Path(".")

    input_file = root / "10_analysis/outputs/phase39_biomarkers/links.csv"
    out_dir = root / "10_analysis/outputs/phase40_modules"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1] Loading data")
    df = load_data(input_file)

    print("[2] Building full graph")
    G = build_graph(df)

    print("[3] Backbone extraction (MST + strong edges)")
    G = backbone_graph(G)

    print("[INFO] nodes:", G.number_of_nodes())
    print("[INFO] edges:", G.number_of_edges())

    print("[4] Detecting modules")
    partition = detect_modules(G)

    print("[5] Computing hubs")
    hubs = compute_hubs(G)

    print("[6] Saving results")

    nx.write_gexf(G, out_dir / "network.gexf")
    pd.DataFrame.from_dict(hubs, orient="index").to_csv(out_dir / "hubs.csv")

    print("[7] Plotting figures")

    plot_network(G, partition, hubs, out_dir / "network_FINAL.png")
    plot_hubs(hubs, out_dir / "hubs_FINAL.png")

    print("DONE ")


if __name__ == "__main__":
    main()
