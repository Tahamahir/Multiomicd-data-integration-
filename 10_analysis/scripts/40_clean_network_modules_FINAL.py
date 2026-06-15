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

    required = {"feature", "metabolite", "importance"}
    if not required.issubset(df.columns):
        raise ValueError("Missing columns")

    return df


# =========================================================
# FILTER (balanced)
# =========================================================
def filter_edges(df):

    threshold = df["importance"].quantile(0.65)
    df = df[df["importance"] >= threshold].copy()

    return df


# =========================================================
# GRAPH BUILD
# =========================================================
def build_graph(df):

    G = nx.Graph()

    for _, r in df.iterrows():
        G.add_edge(r["feature"], r["metabolite"], weight=float(r["importance"]))

    return G


# =========================================================
# MODULES
# =========================================================
def detect_modules(G):
    return community_louvain.best_partition(G, weight="weight")


# =========================================================
# HUBS
# =========================================================
def compute_hubs(G):

    degree = dict(G.degree())
    return degree


# =========================================================
# FULL PAPER FIGURE
# =========================================================
def plot_network(G, partition, hubs, out_file):

    plt.figure(figsize=(20, 14))

    pos = nx.spring_layout(G, seed=42, k=1.3, iterations=300)

    # -----------------------------
    # NODE TYPES
    # -----------------------------
    microbes = [n for n in G.nodes() if "IK:" in str(n)]
    metabolites = [n for n in G.nodes() if "C18" in str(n) or "HILIC" in str(n)]

    # -----------------------------
    # NODE SIZE = HUBS
    # -----------------------------
    node_size = []
    for n in G.nodes():
        node_size.append(hubs.get(n, 1) * 20)

    # -----------------------------
    # COLOR = MODULES
    # -----------------------------
    node_colors = [partition[n] for n in G.nodes()]

    # -----------------------------
    # DRAW NODES
    # -----------------------------
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        cmap=plt.cm.tab20,
        node_size=node_size,
        alpha=0.9
    )

    # -----------------------------
    # EDGE FILTER (BACKBONE STYLE)
    # -----------------------------
    edges = sorted(G.edges(data=True), key=lambda x: x[2]["weight"], reverse=True)
    edges = edges[:int(len(edges) * 0.6)]

    nx.draw_networkx_edges(
        G, pos,
        edgelist=[(u, v) for u, v, _ in edges],
        width=2,
        alpha=0.4,
        edge_color="gray"
    )

    # -----------------------------
    # HUB LABELS ONLY
    # -----------------------------
    top_nodes = sorted(hubs, key=hubs.get, reverse=True)[:12]

    labels = {n: n.split("|")[-1][:10] for n in top_nodes}

    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=9
    )

    # -----------------------------
    # LEGEND MANUAL
    # -----------------------------
    plt.scatter([], [], c="green", label="Microbes")
    plt.scatter([], [], c="blue", label="Metabolites")
    plt.legend(loc="upper right")

    plt.title("MicrobiomeMetabolome Interaction Network (Publication Panel)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_file, dpi=600, bbox_inches="tight")
    plt.close()


# =========================================================
# HUBS FIGURE
# =========================================================
def plot_hubs(hubs, out_file):

    top = dict(sorted(hubs.items(), key=lambda x: x[1], reverse=True)[:20])

    plt.figure(figsize=(12,6))

    plt.bar(top.keys(), top.values(), color="#2ecc71")

    plt.xticks(rotation=90)
    plt.ylabel("Degree (Hub Score)")
    plt.title("Top Biological Hubs")

    plt.tight_layout()
    plt.savefig(out_file, dpi=600)
    plt.close()


# =========================================================
# MAIN
# =========================================================
def main():

    root = Path(".")

    input_file = root / "10_analysis/outputs/phase39_biomarkers/links.csv"
    out_dir = root / "10_analysis/outputs/phase40_modules"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[PHASE 40] Loading data...")
    df = load_data(input_file)

    print("[PHASE 40] Filtering edges...")
    df = filter_edges(df)

    print("[PHASE 40] Building graph...")
    G = build_graph(df)

    print("[PHASE 40] Detecting modules...")
    partition = detect_modules(G)

    print("[PHASE 40] Computing hubs...")
    hubs = compute_hubs(G)

    print("[PHASE 40] Plotting publication figure...")

    plot_network(G, partition, hubs, out_dir / "network_PUBLICATION.png")
    plot_hubs(hubs, out_dir / "hubs.png")

    print("[DONE] Publication-ready figures generated ")


if __name__ == "__main__":
    main()
