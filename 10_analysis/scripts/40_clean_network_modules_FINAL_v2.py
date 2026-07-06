#!/usr/bin/env python3

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from community import community_louvain
except:
    raise ImportError("pip install python-louvain")


# =========================================================
# LOAD DATA
# =========================================================
def load_data(path):
    df = pd.read_csv(path)

    required = {"feature", "metabolite", "importance"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required}")

    return df


# =========================================================
# FILTER NETWORK (IMPORTANT FIX)
# =========================================================
def filter_edges(df):
    thr = df["importance"].quantile(0.80)
    df = df[df["importance"] >= thr].copy()
    print(f"[INFO] edges after filtering: {len(df)}")
    return df


# =========================================================
# BUILD GRAPH
# =========================================================
def build_graph(df):
    G = nx.Graph()

    for _, r in df.iterrows():
        G.add_edge(r["feature"], r["metabolite"], weight=float(r["importance"]))

    return G


# =========================================================
# MODULE DETECTION
# =========================================================
def detect_modules(G):
    return community_louvain.best_partition(G, weight="weight")


# =========================================================
# HUBS
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
# NETWORK FIGURE (FIXED + PAPER READY)
# =========================================================
def plot_network(G, partition, out_file):

    plt.figure(figsize=(18, 12))

    # FIXED LAYOUT (better separation)
    pos = nx.spring_layout(G, seed=42, k=0.6, iterations=200)

    # NODE TYPES
    microbes = [n for n in G.nodes() if "IK:" in str(n)]
    metabolites = [n for n in G.nodes() if "C18" in str(n) or "HILIC" in str(n)]

    # MODULE COLORS
    node_colors = [partition[n] for n in G.nodes()]

    # DRAW NODES (with modules)
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        cmap=plt.cm.tab20,
        node_size=60,
        alpha=0.95
    )

    # FILTER STRONG EDGES ONLY
    edges = sorted(
        G.edges(data=True),
        key=lambda x: x[2]["weight"],
        reverse=True
    )

    edges = edges[:int(len(edges) * 0.6)]  # keep top 60%

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[(u, v) for u, v, _ in edges],
        alpha=0.4,
        width=2,
        edge_color="gray"
    )

    # LABEL TOP HUBS ONLY
    deg = dict(G.degree())
    top_nodes = sorted(deg, key=deg.get, reverse=True)[:15]

    labels = {n: n.split("|")[-1][:12] for n in top_nodes}

    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=8
    )

    plt.title("MicrobiomeMetabolome Interaction Network (Improved Biological Modules)", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_file, dpi=600, bbox_inches="tight")
    plt.close()


# =========================================================
# HUBS FIGURE (FIXED)
# =========================================================
def plot_hubs(df, out_file):

    df = df.head(20)

    plt.figure(figsize=(12,6))

    colors = [
        "#2ecc71" if "IK:" in str(x) else "#3498db"
        for x in df["node"]
    ]

    plt.bar(df["node"], df["score"], color=colors)

    plt.xticks(rotation=90, fontsize=8)
    plt.ylabel("Hub Score")
    plt.title("Top Biological Hubs (MicrobiomeMetabolome Network)")

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

    print(f"[INFO] Nodes: {G.number_of_nodes()}")
    print(f"[INFO] Edges: {G.number_of_edges()}")

    print("[PHASE 40] Detecting modules...")
    partition = detect_modules(G)

    print("[PHASE 40] Computing hubs...")
    hubs = compute_hubs(G)

    # SAVE OUTPUTS
    hubs.to_csv(out_dir / "biological_hubs.csv", index=False)
    nx.write_gexf(G, out_dir / "clean_network.gexf")

    print("[PHASE 40] Generating figures...")

    plot_network(G, partition, out_dir / "network.png")
    plot_hubs(hubs, out_dir / "hubs.png")

    print("\n[DONE] Phase 40 completed successfully!")
    print("Output folder:", out_dir)


if __name__ == "__main__":
    main()
