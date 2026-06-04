from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT = Path("10_analysis/outputs/phase30_statistical_validation")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close()


def plot_top_pipelines():
    df = pd.read_csv(OUT / "pipeline_summary.csv")
    df = df.sort_values("mean_r2", ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    plt.barh(df["pipeline"][::-1], df["mean_r2"][::-1])
    plt.xlabel("Mean R²")
    plt.ylabel("Pipeline")
    plt.title("Top 15 pipelines ranked by mean R²")
    plt.grid(axis="x", alpha=0.3)
    savefig("top15_pipelines_mean_r2")


def plot_mean_ranks():
    df = pd.read_csv(OUT / "friedman_mean_ranks.csv")
    df = df.sort_values("mean_rank", ascending=True).head(15)

    plt.figure(figsize=(10, 6))
    plt.barh(df["pipeline"][::-1], df["mean_rank"][::-1])
    plt.xlabel("Mean Friedman rank (lower is better)")
    plt.ylabel("Pipeline")
    plt.title("Top 15 pipelines ranked by Friedman mean rank")
    plt.grid(axis="x", alpha=0.3)
    savefig("top15_friedman_mean_ranks")


def plot_validation_progression():
    data = pd.DataFrame({
        "validation": [
            "Initial 5-fold CV",
            "Repeated CV 5×10",
            "Nested CV light",
        ],
        "mean_r2": [
            0.338264,
            0.328298,
            0.316325,
        ],
    })

    plt.figure(figsize=(7, 5))
    plt.plot(data["validation"], data["mean_r2"], marker="o")
    plt.ylabel("Mean R²")
    plt.xlabel("Validation strategy")
    plt.title("Performance under increasingly robust validation")
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    savefig("validation_progression")


def plot_champion_vs_competitors():
    df = pd.read_csv(OUT / "champion_pairwise_comparison.csv")

    keep = [
        "T270_mi500_spca75_a10_w7_rf_e",
        "T267_mi500_spca75_a10_w7_rf_b",
        "T268_mi500_spca75_a10_w7_rf_c",
        "DR20_late_mi500_sparsepca100_rf",
        "T286_mi500_spca75_a20_w7_rf_a",
        "B15_late_mi500_rf",
        "DR01_late_mi500_rf_none",
        "B14_late_rf",
        "B16_borutaLight300_rf",
        "B04_xgboost_light",
        "B09_pls2",
    ]

    df = df[df["competitor"].isin(keep)].copy()
    df = df.sort_values("delta_champion_minus_competitor", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(df["competitor"], df["delta_champion_minus_competitor"])
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel(" Mean R² vs champion")
    plt.ylabel("Competitor pipeline")
    plt.title("Champion improvement over selected competitors")
    plt.grid(axis="x", alpha=0.3)
    savefig("champion_delta_vs_competitors")


def plot_wilcoxon_significance():
    df = pd.read_csv(OUT / "champion_pairwise_comparison.csv")

    keep = [
        "T270_mi500_spca75_a10_w7_rf_e",
        "T267_mi500_spca75_a10_w7_rf_b",
        "T268_mi500_spca75_a10_w7_rf_c",
        "DR20_late_mi500_sparsepca100_rf",
        "T286_mi500_spca75_a20_w7_rf_a",
        "B15_late_mi500_rf",
        "DR01_late_mi500_rf_none",
        "B14_late_rf",
        "B16_borutaLight300_rf",
        "B04_xgboost_light",
        "B09_pls2",
    ]

    df = df[df["competitor"].isin(keep)].copy()
    df["minus_log10_p"] = -np.log10(df["p_corrected_holm"].clip(lower=1e-300))
    df = df.sort_values("minus_log10_p", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(df["competitor"], df["minus_log10_p"])
    plt.axvline(-np.log10(0.05), linestyle="--", linewidth=1)
    plt.xlabel("-log10(Holm-corrected p-value)")
    plt.ylabel("Competitor pipeline")
    plt.title("Wilcoxon post-hoc significance vs champion")
    plt.grid(axis="x", alpha=0.3)
    savefig("wilcoxon_significance_vs_champion")


def plot_r2_distribution_top10():
    matrix = pd.read_csv(OUT / "r2_matrix_metabolite_by_pipeline.csv", index_col=0)
    summary = pd.read_csv(OUT / "pipeline_summary.csv")

    top = summary.sort_values("mean_r2", ascending=False).head(10)["pipeline"].tolist()
    data = [matrix[p].dropna().values for p in top]

    plt.figure(figsize=(11, 6))
    plt.boxplot(data, labels=top, vert=True, showfliers=True)
    plt.ylabel("R² per metabolite")
    plt.xlabel("Pipeline")
    plt.title("Distribution of R² across metabolites for top 10 pipelines")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    savefig("top10_pipeline_r2_distribution")


def main():
    plot_top_pipelines()
    plot_mean_ranks()
    plot_validation_progression()
    plot_champion_vs_competitors()
    plot_wilcoxon_significance()
    plot_r2_distribution_top10()

    print("Figures saved in:")
    print(FIG)

    for f in sorted(FIG.glob("*")):
        print(f)


if __name__ == "__main__":
    main()
