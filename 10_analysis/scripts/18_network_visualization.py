from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# ============================================================
# PHASE 18 - NETWORK VISUALIZATION MG ↔ MB
# ------------------------------------------------------------
# Objectif :
# - construire un réseau biparti microbe/metabolite
# - identifier les hubs MG et MB
# - générer fichiers Cytoscape/Gephi
# - générer figures simples pour rapport
# ============================================================


def shorten_label(x, max_len=35):
    x = str(x)
    if len(x) <= max_len:
        return x
    return x[:max_len] + "..."


def main():
    repo_root = Path(__file__).resolve().parents[2]

    input_path = (
        repo_root
        / "10_analysis"
        / "outputs"
        / "phase17_final_best_model_pipeline"
        / "species_mb_relationships_interpretable_optimized.csv"
    )

    output_dir = (
        repo_root
        / "10_analysis"
        / "outputs"
        / "phase18_network_visualization"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 18 - MG ↔ MB NETWORK VISUALIZATION")
    print("=" * 70)
    print(f"Input file : {input_path}")
    print(f"Output dir : {output_dir}")
    print()

    if not input_path.exists():
        raise FileNotFoundError(f"Missing file: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)

    required_cols = [
        "metabolite",
        "mg_feature",
        "importance",
        "spearman_corr",
        "abs_spearman_corr",
        "putative_role",
        "confidence",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"Input relationships: {len(df)}")
    print(f"Metabolites        : {df['metabolite'].nunique()}")
    print(f"MG features        : {df['mg_feature'].nunique()}")
    print()

    # ------------------------------------------------------------
    # 1. Filtrage pour réseau principal
    # ------------------------------------------------------------
    # On garde les relations medium/high déjà interprétables,
    # puis on filtre un peu par corrélation pour éviter un réseau trop dense.
    network_df = df[
        (df["confidence"].isin(["medium", "high"])) &
        (df["abs_spearman_corr"] >= 0.30)
    ].copy()

    if network_df.empty:
        print("Warning: filtered network is empty. Relaxing abs_spearman_corr threshold to 0.20")
        network_df = df[df["confidence"].isin(["medium", "high"])].copy()

    print(f"Filtered network relationships: {len(network_df)}")
    print(f"Filtered metabolites           : {network_df['metabolite'].nunique()}")
    print(f"Filtered MG features           : {network_df['mg_feature'].nunique()}")
    print()

    # ------------------------------------------------------------
    # 2. Construire edges
    # ------------------------------------------------------------
    edges = network_df.copy()
    edges = edges.rename(columns={
        "mg_feature": "source",
        "metabolite": "target",
    })

    edges["edge_type"] = edges["putative_role"]
    edges["weight"] = edges["abs_spearman_corr"]
    edges["direction"] = np.where(edges["spearman_corr"] >= 0, "positive", "negative")

    edges_out = edges[
        [
            "source",
            "target",
            "edge_type",
            "direction",
            "importance",
            "spearman_corr",
            "abs_spearman_corr",
            "weight",
            "confidence",
        ]
    ].copy()

    edges_out.to_csv(output_dir / "network_edges.csv", index=False)

    # ------------------------------------------------------------
    # 3. Construire nodes
    # ------------------------------------------------------------
    mg_nodes = pd.DataFrame({
        "node": sorted(network_df["mg_feature"].unique()),
        "node_type": "MG"
    })

    mb_nodes = pd.DataFrame({
        "node": sorted(network_df["metabolite"].unique()),
        "node_type": "MB"
    })

    nodes = pd.concat([mg_nodes, mb_nodes], ignore_index=True)

    # degree
    degree_counts = pd.concat([
        edges_out["source"],
        edges_out["target"]
    ]).value_counts()

    nodes["degree"] = nodes["node"].map(degree_counts).fillna(0).astype(int)

    nodes.to_csv(output_dir / "network_nodes.csv", index=False)

    # ------------------------------------------------------------
    # 4. Hubs MG et MB
    # ------------------------------------------------------------
    mg_hubs = (
        network_df.groupby("mg_feature")
        .agg(
            n_metabolites=("metabolite", "nunique"),
            n_relationships=("metabolite", "count"),
            mean_importance=("importance", "mean"),
            max_importance=("importance", "max"),
            mean_abs_corr=("abs_spearman_corr", "mean"),
            max_abs_corr=("abs_spearman_corr", "max"),
        )
        .reset_index()
        .sort_values(["n_metabolites", "max_abs_corr"], ascending=[False, False])
    )

    mb_hubs = (
        network_df.groupby("metabolite")
        .agg(
            n_mg_features=("mg_feature", "nunique"),
            n_relationships=("mg_feature", "count"),
            mean_importance=("importance", "mean"),
            max_importance=("importance", "max"),
            mean_abs_corr=("abs_spearman_corr", "mean"),
            max_abs_corr=("abs_spearman_corr", "max"),
        )
        .reset_index()
        .sort_values(["n_mg_features", "max_abs_corr"], ascending=[False, False])
    )

    mg_hubs.to_csv(output_dir / "mg_hubs.csv", index=False)
    mb_hubs.to_csv(output_dir / "mb_hubs.csv", index=False)

    # ------------------------------------------------------------
    # 5. Résumé roles
    # ------------------------------------------------------------
    role_summary = (
        network_df["putative_role"]
        .value_counts()
        .rename_axis("putative_role")
        .reset_index(name="n_relationships")
    )

    confidence_summary = (
        network_df["confidence"]
        .value_counts()
        .rename_axis("confidence")
        .reset_index(name="n_relationships")
    )

    network_summary = {
        "n_relationships": int(len(network_df)),
        "n_mg_features": int(network_df["mg_feature"].nunique()),
        "n_metabolites": int(network_df["metabolite"].nunique()),
        "n_putative_production": int((network_df["putative_role"] == "putative_production").sum()),
        "n_putative_consumption": int((network_df["putative_role"] == "putative_consumption").sum()),
        "mean_abs_spearman_corr": float(network_df["abs_spearman_corr"].mean()),
        "max_abs_spearman_corr": float(network_df["abs_spearman_corr"].max()),
    }

    pd.DataFrame([network_summary]).to_csv(output_dir / "network_summary.csv", index=False)
    role_summary.to_csv(output_dir / "network_role_summary.csv", index=False)
    confidence_summary.to_csv(output_dir / "network_confidence_summary.csv", index=False)

    # ------------------------------------------------------------
    # 6. Barplots hubs
    # ------------------------------------------------------------
    top_n = 20

    top_mg = mg_hubs.head(top_n).copy()
    plt.figure(figsize=(10, 6))
    plt.barh(
        [shorten_label(x, 45) for x in top_mg["mg_feature"]],
        top_mg["n_metabolites"]
    )
    plt.gca().invert_yaxis()
    plt.xlabel("Number of connected metabolites")
    plt.title("Top MG hubs")
    plt.tight_layout()
    plt.savefig(output_dir / "mg_hub_barplot.png", dpi=250)
    plt.close()

    top_mb = mb_hubs.head(top_n).copy()
    plt.figure(figsize=(10, 6))
    plt.barh(
        [shorten_label(x, 45) for x in top_mb["metabolite"]],
        top_mb["n_mg_features"]
    )
    plt.gca().invert_yaxis()
    plt.xlabel("Number of connected MG features")
    plt.title("Top metabolite hubs")
    plt.tight_layout()
    plt.savefig(output_dir / "mb_hub_barplot.png", dpi=250)
    plt.close()

    # Role barplot
    plt.figure(figsize=(7, 5))
    plt.bar(role_summary["putative_role"], role_summary["n_relationships"])
    plt.ylabel("Number of relationships")
    plt.title("Putative roles in MG-MB network")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "role_summary_barplot.png", dpi=250)
    plt.close()

    # ------------------------------------------------------------
    # 7. Réseau NetworkX filtré pour figure lisible
    # ------------------------------------------------------------
    # Pour la figure PNG, on prend les top relations pour éviter un graphe illisible.
    plot_df = network_df.sort_values("abs_spearman_corr", ascending=False).head(120).copy()

    G = nx.Graph()

    for _, row in plot_df.iterrows():
        mg = row["mg_feature"]
        mb = row["metabolite"]

        G.add_node(mg, node_type="MG")
        G.add_node(mb, node_type="MB")
        G.add_edge(
            mg,
            mb,
            weight=row["abs_spearman_corr"],
            role=row["putative_role"],
            corr=row["spearman_corr"]
        )

    pos = nx.spring_layout(G, seed=42, k=0.45)

    node_colors = []
    node_sizes = []

    for n in G.nodes():
        node_type = G.nodes[n]["node_type"]
        degree = G.degree(n)

        if node_type == "MG":
            node_colors.append("skyblue")
        else:
            node_colors.append("orange")

        node_sizes.append(80 + degree * 40)

    edge_colors = []
    edge_widths = []

    for u, v, data in G.edges(data=True):
        if data["role"] == "putative_production":
            edge_colors.append("green")
        elif data["role"] == "putative_consumption":
            edge_colors.append("red")
        else:
            edge_colors.append("gray")

        edge_widths.append(0.5 + data["weight"] * 2)

    plt.figure(figsize=(16, 12))
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.55
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9
    )

    # labels seulement pour hubs pour éviter surcharge
    labels = {}
    for n in G.nodes():
        if G.degree(n) >= 3:
            labels[n] = shorten_label(n, 25)

    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7)

    plt.title("MG-MB bipartite network (top associations)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "mg_mb_network.png", dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Network relationships       : {network_summary['n_relationships']}")
    print(f"MG nodes                    : {network_summary['n_mg_features']}")
    print(f"MB nodes                    : {network_summary['n_metabolites']}")
    print(f"Putative production edges   : {network_summary['n_putative_production']}")
    print(f"Putative consumption edges  : {network_summary['n_putative_consumption']}")
    print(f"Mean abs Spearman corr      : {network_summary['mean_abs_spearman_corr']:.4f}")
    print(f"Max abs Spearman corr       : {network_summary['max_abs_spearman_corr']:.4f}")
    print()
    print("Top 10 MG hubs:")
    print(mg_hubs.head(10).to_string(index=False))
    print()
    print("Top 10 MB hubs:")
    print(mb_hubs.head(10).to_string(index=False))
    print()
    print("Main outputs:")
    print(output_dir / "network_nodes.csv")
    print(output_dir / "network_edges.csv")
    print(output_dir / "mg_hubs.csv")
    print(output_dir / "mb_hubs.csv")
    print(output_dir / "network_summary.csv")
    print(output_dir / "mg_mb_network.png")
    print(output_dir / "mg_hub_barplot.png")
    print(output_dir / "mb_hub_barplot.png")
    print()
    print("Network visualization completed successfully.")


if __name__ == "__main__":
    main()