from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

ROOT = Path(".")
OUT = ROOT / "10_analysis/outputs/phase34_mg_mb_network"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def safe_name(x):
    return (
        str(x)
        .replace("/", "_")
        .replace("|", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def shorten_label(text, max_len=45):
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def load_assoc(file_name):
    path = ROOT / "10_analysis/outputs/phase33_mg_mb_associations" / file_name
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def make_bipartite_network(
    df,
    out_prefix,
    title,
    min_degree_keep=1,
    top_mg=None,
    top_mb=None,
):
    df = df.copy()

    # degrees
    mg_degree = df.groupby("mg_feature").size().sort_values(ascending=False)
    mb_degree = df.groupby("metabolite").size().sort_values(ascending=False)

    if top_mg is not None:
        keep_mg = set(mg_degree.head(top_mg).index)
        df = df[df["mg_feature"].isin(keep_mg)]

    if top_mb is not None:
        keep_mb = set(mb_degree.head(top_mb).index)
        df = df[df["metabolite"].isin(keep_mb)]

    # recalc after filter
    mg_degree = df.groupby("mg_feature").size().sort_values(ascending=False)
    mb_degree = df.groupby("metabolite").size().sort_values(ascending=False)

    if min_degree_keep > 1:
        keep_mg = set(mg_degree[mg_degree >= min_degree_keep].index)
        keep_mb = set(mb_degree[mb_degree >= min_degree_keep].index)
        df = df[
            df["mg_feature"].isin(keep_mg)
            & df["metabolite"].isin(keep_mb)
        ]
        mg_degree = df.groupby("mg_feature").size().sort_values(ascending=False)
        mb_degree = df.groupby("metabolite").size().sort_values(ascending=False)

    # build graph
    G = nx.Graph()

    mg_nodes = sorted(df["mg_feature"].unique(), key=lambda x: (-mg_degree[x], x))
    mb_nodes = sorted(df["metabolite"].unique(), key=lambda x: (-mb_degree[x], x))

    for node in mg_nodes:
        G.add_node(node, bipartite="MG", degree=int(mg_degree[node]))

    for node in mb_nodes:
        G.add_node(node, bipartite="MB", degree=int(mb_degree[node]))

    for _, row in df.iterrows():
        G.add_edge(
            row["mg_feature"],
            row["metabolite"],
            rho=float(row["rho"]),
            abs_rho=float(row["abs_rho"]),
            qvalue=float(row["qvalue_fdr"]),
            role=row["putative_role"],
        )

    # positions: MG left, MB right
    pos = {}
    n_mg = len(mg_nodes)
    n_mb = len(mb_nodes)

    for i, node in enumerate(mg_nodes):
        pos[node] = (0, n_mg - i)

    for i, node in enumerate(mb_nodes):
        pos[node] = (1, n_mb - i)

    # normalize y spacing if sizes differ
    if n_mg > 0:
        mg_y = np.linspace(1, 0, n_mg)
        for i, node in enumerate(mg_nodes):
            pos[node] = (0, mg_y[i])
    if n_mb > 0:
        mb_y = np.linspace(1, 0, n_mb)
        for i, node in enumerate(mb_nodes):
            pos[node] = (1, mb_y[i])

    edge_colors = []
    edge_widths = []

    for u, v, d in G.edges(data=True):
        if d["rho"] >= 0:
            edge_colors.append("forestgreen")
        else:
            edge_colors.append("firebrick")

        # width between ~1 and ~5
        edge_widths.append(1 + 8 * (d["abs_rho"] - df["abs_rho"].min()) / (df["abs_rho"].max() - df["abs_rho"].min() + 1e-9))

    mg_sizes = [250 + 120 * G.nodes[n]["degree"] for n in mg_nodes]
    mb_sizes = [350 + 140 * G.nodes[n]["degree"] for n in mb_nodes]

    plt.figure(figsize=(18, max(10, 0.35 * max(n_mg, n_mb))))

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.55,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=mg_nodes,
        node_color="lightblue",
        node_size=mg_sizes,
        edgecolors="black",
        linewidths=0.6,
        label="MG features",
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=mb_nodes,
        node_color="khaki",
        node_size=mb_sizes,
        edgecolors="black",
        linewidths=0.8,
        label="Metabolites",
    )

    labels = {}
    for n in mg_nodes:
        labels[n] = shorten_label(n, 38)
    for n in mb_nodes:
        labels[n] = shorten_label(n, 38)

    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=8,
    )

    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.legend(scatterpoints=1, loc="upper center", ncol=2)
    plt.tight_layout()

    plt.savefig(FIG / f"{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIG / f"{out_prefix}.pdf", bbox_inches="tight")
    plt.close()

    # Save node summary
    mg_summary = pd.DataFrame({
        "mg_feature": mg_degree.index,
        "degree": mg_degree.values
    })
    mb_summary = pd.DataFrame({
        "metabolite": mb_degree.index,
        "degree": mb_degree.values
    })

    mg_summary.to_csv(OUT / f"{out_prefix}_mg_degree.csv", index=False)
    mb_summary.to_csv(OUT / f"{out_prefix}_mb_degree.csv", index=False)
    df.to_csv(OUT / f"{out_prefix}_edges.csv", index=False)

    summary = {
        "n_edges": int(len(df)),
        "n_mg_nodes": int(len(mg_nodes)),
        "n_mb_nodes": int(len(mb_nodes)),
        "n_positive_edges": int((df["rho"] > 0).sum()),
        "n_negative_edges": int((df["rho"] < 0).sum()),
        "mean_abs_rho": float(df["abs_rho"].mean()),
        "max_abs_rho": float(df["abs_rho"].max()),
    }

    pd.DataFrame([summary]).to_csv(OUT / f"{out_prefix}_summary.csv", index=False)

    print(f"\n[{out_prefix}]")
    print(pd.DataFrame([summary]).to_string(index=False))
    print("\nTop MG nodes:")
    print(mg_summary.head(15).to_string(index=False))
    print("\nTop MB nodes:")
    print(mb_summary.head(15).to_string(index=False))


def main():
    # 1) high-confidence network (rho >= 0.70)
    high = load_assoc("high_confidence_associations_rho070_fdr001.csv")
    make_bipartite_network(
        high,
        out_prefix="network_rho070_full",
        title="High-confidence MGMB network (FDR < 0.01, |rho| e 0.70)",
        min_degree_keep=1,
    )

    # 2) simplified high-confidence network (top nodes only, more readable)
    make_bipartite_network(
        high,
        out_prefix="network_rho070_simplified",
        title="Simplified high-confidence MGMB network",
        top_mg=25,
        top_mb=15,
        min_degree_keep=1,
    )

    # 3) strong network (rho >= 0.60) simplified
    strong = load_assoc("strong_associations_rho060_fdr001.csv")
    make_bipartite_network(
        strong,
        out_prefix="network_rho060_simplified",
        title="Strong MGMB network (FDR < 0.01, |rho| e 0.60)",
        top_mg=30,
        top_mb=20,
        min_degree_keep=2,
    )

    print("\nOutputs saved in:")
    print(OUT)
    print(FIG)


if __name__ == "__main__":
    main()
