import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from community import community_louvain


def load_network(df_path):
    df = pd.read_csv(df_path)
    return df


def build_filtered_graph(df):
    # =% FILTER TOP EDGES
    threshold = df["importance"].quantile(0.80)
    df = df[df["importance"] >= threshold]

    G = nx.Graph()

    for _, row in df.iterrows():
        G.add_edge(row["feature"], row["metabolite"], weight=row["importance"])

    return G


def detect_modules(G):
    partition = community_louvain.best_partition(G, weight='weight')
    return partition


def plot_colored_network(G, partition, out_path):
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(14, 10))

    colors = [partition[node] for node in G.nodes()]

    nx.draw(
        G,
        pos,
        node_color=colors,
        node_size=50,
        edge_color="gray",
        alpha=0.7,
        with_labels=False
    )

    plt.title("Clean MicrobiomeMetabolome Modules")
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_hubs(G, out_path):
    degree = dict(G.degree())

    top = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:15]

    names, values = zip(*top)

    plt.figure(figsize=(10, 5))
    plt.bar(names, values)
    plt.xticks(rotation=90)
    plt.title("Top Biological Hubs")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    project_root = Path(".")
    out_dir = project_root / "10_analysis/outputs/phase40_modules"
    out_dir.mkdir(exist_ok=True, parents=True)

    df = load_network("10_analysis/outputs/phase39_biomarkers/links.csv")

    G = build_filtered_graph(df)

    partition = detect_modules(G)

    nx.write_gexf(G, out_dir / "clean_network.gexf")

    plot_colored_network(G, partition, out_dir / "modules_network.png")

    plot_hubs(G, out_dir / "top_hubs.png")

    print("Nodes:", len(G.nodes()))
    print("Edges:", len(G.edges()))
    print("Saved in:", out_dir)


if __name__ == "__main__":
    main()
