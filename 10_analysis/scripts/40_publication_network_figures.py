#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 40 - Publication-ready MicrobiomeMetabolome Figures

This script generates clean and informative figures from Phase 39 outputs:
1. A readable bipartite network figure
2. A heatmap of strongest featuremetabolite links
3. A clean hub ranking barplot
4. GEXF network for Gephi/Cytoscape
5. CSV tables for reporting

Input:
    10_analysis/outputs/phase39_biomarkers/links.csv

Output:
    10_analysis/outputs/phase40_publication_figures/
"""

from pathlib import Path
from collections import defaultdict
import textwrap
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(".")
INPUT_LINKS = PROJECT_ROOT / "10_analysis/outputs/phase39_biomarkers/links.csv"

OUT_DIR = PROJECT_ROOT / "10_analysis/outputs/phase40_publication_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_FEATURES_PER_METABOLITE = 3
MAX_EDGES_FOR_NETWORK = 80
TOP_FEATURES_HEATMAP = 20
TOP_METABOLITES_HEATMAP = 15
TOP_HUBS = 15

SOIL_PREFIXES = (
    "soil_",
    "chem__",
    "psize__",
    "moist__",
    "nitrif__",
    "denit__",
)


# ============================================================
# HELPERS
# ============================================================

def classify_feature(feature: str) -> str:
    """Classify feature nodes as soil or microbe using name patterns."""
    f = str(feature).lower()
    if f.startswith(SOIL_PREFIXES) or "soil" in f:
        return "soil"
    return "microbe"


def short_label(label: str, max_len: int = 22) -> str:
    """Shorten long biological/metabolite labels for figures."""
    label = str(label)

    if "|IK:" in label:
        prefix = label.split("|IK:")[0]
        ik = label.split("|IK:")[1].split("-")[0]
        prefix_short = prefix.replace("_negative", "-").replace("_positive", "+")
        return f"{prefix_short}\nIK:{ik[:8]}"

    if "nitrif__" in label:
        return label.replace("nitrif__", "nitrif:")

    if "denit__" in label:
        return label.replace("denit__", "denit:")

    if "soil_" in label.lower():
        return label.replace("soil_", "soil:")

    if ";" in label:
        last = label.split(";")[-1]
        if len(last.strip()) > 0:
            label = last.strip()

    if "|" in label and len(label) > max_len:
        label = label.split("|")[-1]

    if len(label) > max_len:
        return label[:max_len - 3] + "..."

    return label


def wrap_label(label: str, width: int = 18) -> str:
    return "\n".join(textwrap.wrap(str(label), width=width))


def safe_importance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce")
    df = df.dropna(subset=["importance"])
    df = df[df["importance"] > 0]
    return df


# ============================================================
# LOAD AND PREPARE LINKS
# ============================================================

def load_links() -> pd.DataFrame:
    if not INPUT_LINKS.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_LINKS}")

    df = pd.read_csv(INPUT_LINKS)

    required = {"feature", "metabolite", "importance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in links.csv: {missing}")

    df = safe_importance(df)

    # Aggregate duplicate links if any
    df = (
        df.groupby(["feature", "metabolite"], as_index=False)
        .agg(
            importance=("importance", "mean"),
            importance_sum=("importance", "sum"),
            n_occurrences=("importance", "count"),
        )
    )

    df["feature_type"] = df["feature"].apply(classify_feature)

    return df


def select_publication_edges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select a readable network:
    - top N features per metabolite
    - then cap global number of edges
    """
    selected = []

    for metabolite, sub in df.groupby("metabolite"):
        top = sub.sort_values("importance", ascending=False).head(TOP_FEATURES_PER_METABOLITE)
        selected.append(top)

    edges = pd.concat(selected, ignore_index=True)
    edges = edges.sort_values("importance", ascending=False).head(MAX_EDGES_FOR_NETWORK)
    edges = edges.sort_values(["metabolite", "importance"], ascending=[True, False])

    return edges


# ============================================================
# BUILD BIPARTITE GRAPH
# ============================================================

def build_bipartite_graph(edges: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()

    for _, row in edges.iterrows():
        feature = str(row["feature"])
        metabolite = str(row["metabolite"])
        feature_type = row["feature_type"]
        w = float(row["importance"])

        f_node = f"F::{feature}"
        m_node = f"M::{metabolite}"

        G.add_node(
            f_node,
            node_type=feature_type,
            original_label=feature,
            display_label=short_label(feature),
            bipartite="feature",
        )

        G.add_node(
            m_node,
            node_type="metabolite",
            original_label=metabolite,
            display_label=short_label(metabolite),
            bipartite="metabolite",
        )

        G.add_edge(f_node, m_node, weight=w)

    return G


def compute_node_metrics(G: nx.Graph) -> pd.DataFrame:
    degree = dict(G.degree())
    weighted_degree = dict(G.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)

    rows = []
    for node in G.nodes():
        rows.append({
            "node_id": node,
            "label": G.nodes[node].get("original_label", node),
            "display_label": G.nodes[node].get("display_label", node),
            "node_type": G.nodes[node].get("node_type", "unknown"),
            "degree": degree.get(node, 0),
            "weighted_degree": weighted_degree.get(node, 0.0),
            "betweenness": betweenness.get(node, 0.0),
        })

    metrics = pd.DataFrame(rows)
    metrics["hub_score"] = (
        0.50 * metrics["degree"].rank(pct=True)
        + 0.35 * metrics["weighted_degree"].rank(pct=True)
        + 0.15 * metrics["betweenness"].rank(pct=True)
    )

    metrics = metrics.sort_values("hub_score", ascending=False)
    return metrics


# ============================================================
# FIGURE 1: CLEAN BIPARTITE NETWORK
# ============================================================

def plot_bipartite_network(G: nx.Graph, metrics: pd.DataFrame, out_file: Path):
    """
    Manual bipartite layout:
    - metabolites on the left
    - features on the right
    - node size = hub score
    - edge width = importance
    """

    plt.figure(figsize=(18, 12))
    ax = plt.gca()

    metabolite_nodes = [n for n, d in G.nodes(data=True) if d.get("bipartite") == "metabolite"]
    feature_nodes = [n for n, d in G.nodes(data=True) if d.get("bipartite") == "feature"]

    # Sort nodes by hub score for stable plotting
    score_map = dict(zip(metrics["node_id"], metrics["hub_score"]))

    metabolite_nodes = sorted(metabolite_nodes, key=lambda n: score_map.get(n, 0), reverse=True)
    feature_nodes = sorted(feature_nodes, key=lambda n: score_map.get(n, 0), reverse=True)

    pos = {}

    # left side: metabolites
    for i, node in enumerate(metabolite_nodes):
        y = 1.0 - i / max(1, len(metabolite_nodes) - 1)
        pos[node] = (0.0, y)

    # right side: features
    for i, node in enumerate(feature_nodes):
        y = 1.0 - i / max(1, len(feature_nodes) - 1)
        pos[node] = (1.0, y)

    # Edge drawing
    weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
    if len(weights) == 0:
        raise ValueError("No edges to plot.")

    w_min, w_max = weights.min(), weights.max()
    denom = max(w_max - w_min, 1e-12)

    for u, v in G.edges():
        w = G[u][v]["weight"]
        alpha = 0.18 + 0.45 * ((w - w_min) / denom)
        width = 0.6 + 2.6 * ((w - w_min) / denom)

        x1, y1 = pos[u]
        x2, y2 = pos[v]

        ax.plot(
            [x1, x2],
            [y1, y2],
            color="gray",
            alpha=alpha,
            linewidth=width,
            zorder=1,
        )

    # Node plotting by type
    type_colors = {
        "microbe": "#2ecc71",
        "soil": "#e67e22",
        "metabolite": "#3498db",
    }

    type_markers = {
        "microbe": "o",
        "soil": "s",
        "metabolite": "D",
    }

    for node_type in ["metabolite", "microbe", "soil"]:
        nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == node_type]
        if not nodes:
            continue

        xs = [pos[n][0] for n in nodes]
        ys = [pos[n][1] for n in nodes]
        sizes = [120 + 850 * score_map.get(n, 0.1) for n in nodes]

        ax.scatter(
            xs,
            ys,
            s=sizes,
            c=type_colors[node_type],
            marker=type_markers[node_type],
            edgecolors="white",
            linewidths=1.0,
            alpha=0.95,
            label=node_type.capitalize(),
            zorder=3,
        )

    # Label only top hubs + all soil variables
    top_nodes = metrics.head(14)["node_id"].tolist()
    soil_nodes = metrics[metrics["node_type"] == "soil"].head(8)["node_id"].tolist()
    label_nodes = list(dict.fromkeys(top_nodes + soil_nodes))

    for node in label_nodes:
        if node not in pos:
            continue
        x, y = pos[node]
        label = G.nodes[node].get("display_label", node)
        ha = "right" if x == 0.0 else "left"
        dx = -0.018 if x == 0.0 else 0.018

        ax.text(
            x + dx,
            y,
            label,
            fontsize=8,
            ha=ha,
            va="center",
            zorder=4,
        )

    ax.set_title(
        "Clean microbiomemetabolome bipartite network\n"
        f"Top {TOP_FEATURES_PER_METABOLITE} features per metabolite, max {MAX_EDGES_FOR_NETWORK} links",
        fontsize=15,
        pad=20,
    )

    ax.text(0.0, 1.06, "Metabolites", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.text(1.0, 1.06, "Microbial / soil features", ha="center", va="bottom", fontsize=13, fontweight="bold")

    legend_elements = [
        Line2D([0], [0], marker="D", color="w", label="Metabolites",
               markerfacecolor=type_colors["metabolite"], markersize=10),
        Line2D([0], [0], marker="o", color="w", label="Microbial features",
               markerfacecolor=type_colors["microbe"], markersize=10),
        Line2D([0], [0], marker="s", color="w", label="Soil variables",
               markerfacecolor=type_colors["soil"], markersize=10),
        Line2D([0], [0], color="gray", lw=2, label="XAI link strength"),
    ]

    ax.legend(handles=legend_elements, loc="lower center", ncol=4, frameon=False, fontsize=10)

    ax.set_xlim(-0.22, 1.22)
    ax.set_ylim(-0.05, 1.10)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_file, dpi=600, bbox_inches="tight")
    plt.close()


# ============================================================
# FIGURE 2: HEATMAP OF TOP LINKS
# ============================================================

def plot_heatmap(df: pd.DataFrame, out_file: Path):
    """
    Heatmap is usually more informative than a dense network for PFE/paper.
    Rows = top features
    Columns = top metabolites
    Values = XAI importance
    """

    feature_score = (
        df.groupby("feature")
        .agg(mean_importance=("importance", "mean"), freq=("feature", "count"))
        .reset_index()
    )
    feature_score["score"] = feature_score["mean_importance"] * feature_score["freq"]

    metabolite_score = (
        df.groupby("metabolite")
        .agg(mean_importance=("importance", "mean"), freq=("metabolite", "count"))
        .reset_index()
    )
    metabolite_score["score"] = metabolite_score["mean_importance"] * metabolite_score["freq"]

    top_features = feature_score.sort_values("score", ascending=False).head(TOP_FEATURES_HEATMAP)["feature"].tolist()
    top_metabolites = metabolite_score.sort_values("score", ascending=False).head(TOP_METABOLITES_HEATMAP)["metabolite"].tolist()

    sub = df[df["feature"].isin(top_features) & df["metabolite"].isin(top_metabolites)].copy()

    matrix = (
        sub.pivot_table(
            index="feature",
            columns="metabolite",
            values="importance",
            aggfunc="mean",
            fill_value=0,
        )
    )

    matrix = matrix.reindex(index=top_features, columns=top_metabolites).fillna(0)

    fig_w = max(12, 0.55 * len(top_metabolites))
    fig_h = max(10, 0.35 * len(top_features))

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(matrix.values, aspect="auto", interpolation="nearest")

    plt.colorbar(im, label="XAI importance")

    plt.xticks(
        ticks=np.arange(len(matrix.columns)),
        labels=[short_label(c, 16) for c in matrix.columns],
        rotation=90,
        fontsize=8,
    )

    plt.yticks(
        ticks=np.arange(len(matrix.index)),
        labels=[short_label(i, 28) for i in matrix.index],
        fontsize=8,
    )

    plt.title("Top featuremetabolite links based on XAI importance", fontsize=14)
    plt.xlabel("Metabolites")
    plt.ylabel("Microbial / soil features")

    plt.tight_layout()
    plt.savefig(out_file, dpi=600, bbox_inches="tight")
    plt.close()


# ============================================================
# FIGURE 3: HUB SCORE BARPLOT
# ============================================================

def plot_hub_scores(metrics: pd.DataFrame, out_file: Path):
    top = metrics.head(TOP_HUBS).copy()
    top = top.sort_values("hub_score", ascending=True)

    color_map = {
        "microbe": "#2ecc71",
        "soil": "#e67e22",
        "metabolite": "#3498db",
    }

    colors = [color_map.get(t, "#7f8c8d") for t in top["node_type"]]

    labels = [short_label(x, 34) for x in top["label"]]

    plt.figure(figsize=(12, 8))
    plt.barh(labels, top["hub_score"], color=colors)

    plt.xlabel("Hub score")
    plt.title("Top biological hubs in the microbiomemetabolome network", fontsize=14)

    legend_elements = [
        Line2D([0], [0], marker="s", color="w", label="Microbial features",
               markerfacecolor=color_map["microbe"], markersize=10),
        Line2D([0], [0], marker="s", color="w", label="Soil variables",
               markerfacecolor=color_map["soil"], markersize=10),
        Line2D([0], [0], marker="s", color="w", label="Metabolites",
               markerfacecolor=color_map["metabolite"], markersize=10),
    ]

    plt.legend(handles=legend_elements, loc="lower right", frameon=False)

    plt.tight_layout()
    plt.savefig(out_file, dpi=600, bbox_inches="tight")
    plt.close()


# ============================================================
# SAVE SUMMARY
# ============================================================

def save_summary(df_all: pd.DataFrame, edges: pd.DataFrame, G: nx.Graph, metrics: pd.DataFrame):
    summary_file = OUT_DIR / "network_summary.txt"

    n_features = df_all["feature"].nunique()
    n_metabolites = df_all["metabolite"].nunique()
    n_edges_all = len(df_all)

    n_features_plot = edges["feature"].nunique()
    n_metabolites_plot = edges["metabolite"].nunique()
    n_edges_plot = len(edges)

    n_microbe = (metrics["node_type"] == "microbe").sum()
    n_soil = (metrics["node_type"] == "soil").sum()
    n_met = (metrics["node_type"] == "metabolite").sum()

    with open(summary_file, "w") as f:
        f.write("Phase 40 publication figures summary\n")
        f.write("====================================\n\n")
        f.write(f"Input file: {INPUT_LINKS}\n")
        f.write(f"Output folder: {OUT_DIR}\n\n")

        f.write("Original network\n")
        f.write(f"- Unique features: {n_features}\n")
        f.write(f"- Unique metabolites: {n_metabolites}\n")
        f.write(f"- Total feature-metabolite links: {n_edges_all}\n\n")

        f.write("Publication network\n")
        f.write(f"- Plotted features: {n_features_plot}\n")
        f.write(f"- Plotted metabolites: {n_metabolites_plot}\n")
        f.write(f"- Plotted links: {n_edges_plot}\n")
        f.write(f"- Graph nodes: {G.number_of_nodes()}\n")
        f.write(f"- Graph edges: {G.number_of_edges()}\n\n")

        f.write("Node types in publication network\n")
        f.write(f"- Microbial features: {n_microbe}\n")
        f.write(f"- Soil variables: {n_soil}\n")
        f.write(f"- Metabolites: {n_met}\n\n")

        f.write("Top 10 hubs\n")
        for _, row in metrics.head(10).iterrows():
            f.write(
                f"- {row['node_type']}: {row['label']} | "
                f"degree={row['degree']}, weighted_degree={row['weighted_degree']:.4f}, "
                f"hub_score={row['hub_score']:.4f}\n"
            )


# ============================================================
# MAIN
# ============================================================

def main():
    print("[PHASE 40] Loading links...")
    df_all = load_links()

    print(f"[INFO] Raw links after aggregation: {len(df_all)}")
    print(f"[INFO] Unique features: {df_all['feature'].nunique()}")
    print(f"[INFO] Unique metabolites: {df_all['metabolite'].nunique()}")

    print("[PHASE 40] Selecting publication edges...")
    edges = select_publication_edges(df_all)
    edges.to_csv(OUT_DIR / "filtered_feature_metabolite_links.csv", index=False)

    print(f"[INFO] Publication edges: {len(edges)}")
    print(f"[INFO] Publication features: {edges['feature'].nunique()}")
    print(f"[INFO] Publication metabolites: {edges['metabolite'].nunique()}")

    print("[PHASE 40] Building bipartite graph...")
    G = build_bipartite_graph(edges)

    print("[PHASE 40] Computing node metrics...")
    metrics = compute_node_metrics(G)
    metrics.to_csv(OUT_DIR / "node_metrics_publication.csv", index=False)

    hub_table = metrics.head(50).copy()
    hub_table.to_csv(OUT_DIR / "biomarker_hubs_publication.csv", index=False)

    print("[PHASE 40] Exporting GEXF...")
    nx.write_gexf(G, OUT_DIR / "publication_network.gexf")

    print("[PHASE 40] Plotting bipartite network...")
    plot_bipartite_network(G, metrics, OUT_DIR / "publication_network_bipartite.png")

    print("[PHASE 40] Plotting heatmap...")
    plot_heatmap(df_all, OUT_DIR / "publication_heatmap_top_links.png")

    print("[PHASE 40] Plotting hubs...")
    plot_hub_scores(metrics, OUT_DIR / "publication_hub_scores.png")

    print("[PHASE 40] Saving summary...")
    save_summary(df_all, edges, G, metrics)

    print("\n[DONE] Publication figures generated successfully.")
    print(f"Output folder: {OUT_DIR}")


if __name__ == "__main__":
    main()
