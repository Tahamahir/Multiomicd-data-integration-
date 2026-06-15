#!/usr/bin/env python3

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from community import community_louvain
except:
    raise ImportError("Install: pip install python-louvain")


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
# FILTER EDGES (TOP 20%)
# =========================================================
def filter_edges(df):
    thr = df["importance"].quantile(0.80)
    df = df[df["importance"] >= thr].copy()
    print(f"[INFO] Filtered edges: {len(df)}")
    return df


# =========================================================
# BUILD GRAPH
# =========================================================
def build_graph(df):
    G = nx.Graph()

    for _, r in df.iterrows():
        G.add_edge(r["feature"], r["metabolite"], weight=r["importance"])

    return G


# =========================================================
# MODULE DETECTION
# =========================================================
def modules(G):
    return community_louvain.best_partition(G, weight="weight")


# =========================================================
# HUBS
# =========================================================
def hubs(G):
    deg = dict(G.degree())
    btw = nx.betweenness_centrality(G, weight="weight")

    df = pd.DataFrame({
        "node": list(deg.keys()),
        "degree": list(deg.values()),
        "betweenness": [btw[n] for n in deg.keys()]
    })

    df["score"] = df["degree"]*0.7 + df["betweenness"]*0.3
    return df.sort_values("score", ascending=False)


# =========================================================
# NETWORK FIGURE (PAPER READY)
# =========================================================
def plot_network(G, part, out):

    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, seed=42, k=0.35)

    microbes = [n for n in G.nodes() if "IK:" in str(n)]
    metabolites = [n for n in G.nodes() if "C18" in str(n) or "HILIC" in str(n)]

    nx.draw_networkx_nodes(G, pos,
        nodelist=microbes,
        node_color="#2ecc71",
        node_size=50,
        label="Microbes"
    )

    nx.draw_networkx_nodes(G, pos,
        nodelist=metabolites,
        node_color="#3498db",
        node_size=50,
        label="Metabolites"
    )

    weights = [G[u][v]["weight"] for u,v in G.edges()]
    nx.draw_networkx_edges(G, pos,
        alpha=0.15,
        width=[w*2 for w in weights],
        edge_color="gray"
    )

    top = sorted(dict(G.degree()), key=dict(G.degree()).get, reverse=True)[:15]
    labels = {n:n.split("|")[-1][:12] for n in top}

    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    plt.title("MicrobiomeMetabolome Network (Clean Modules)")
    plt.legend()
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=600)
    plt.close()


# =========================================================
# HUBS FIGURE
# =========================================================
def plot_hubs(df, out):

    df = df.head(20)

    plt.figure(figsize=(12,6))

    colors = ["#2ecc71" if "IK:" in str(x) else "#3498db"
              for x in df["node"]]

    plt.bar(df["node"], df["score"], color=colors)
    plt.xticks(rotation=90, fontsize=8)

    plt.title("Top Biological Hubs")
    plt.ylabel("Score")

    plt.tight_layout()
    plt.savefig(out, dpi=600)
    plt.close()


# =========================================================
# MAIN
# =========================================================
def main():

    root = Path(".")
    inp = root / "10_analysis/outputs/phase39_biomarkers/links.csv"
    outdir = root / "10_analysis/outputs/phase40_modules"
    outdir.mkdir(parents=True, exist_ok=True)

    print("[PHASE 40] Loading data")
    df = load_data(inp)

    print("[PHASE 40] Filtering")
    df = filter_edges(df)

    print("[PHASE 40] Graph building")
    G = build_graph(df)

    print("[INFO] Nodes:", G.number_of_nodes())
    print("[INFO] Edges:", G.number_of_edges())

    print("[PHASE 40] Modules detection")
    part = modules(G)

    print("[PHASE 40] Hubs")
    hub_df = hubs(G)

    hub_df.to_csv(outdir/"biological_hubs.csv", index=False)
    nx.write_gexf(G, outdir/"clean_network.gexf")

    print("[PHASE 40] Figures")

    plot_network(G, part, outdir/"network.png")
    plot_hubs(hub_df, outdir/"hubs.png")

    print("[DONE] Saved in", outdir)


if __name__ == "__main__":
    main()
