from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_assay(metabolite_name):
    """
    Extrait le type d'assay depuis le nom du métabolite.
    Exemples:
    C18_negative|IK:...
    HILICZ_positive|IK:...
    """
    if "|" in metabolite_name:
        return metabolite_name.split("|")[0]
    return "unknown"


def save_hist(values, title, xlabel, output_path, bins=40):
    values = pd.Series(values).dropna()

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_scatter(x, y, title, xlabel, ylabel, output_path):
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, s=25)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_bar(labels, values, title, ylabel, output_path):
    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    repo_root = Path(__file__).resolve().parents[2]

    input_path = (
        repo_root
        / "10_analysis"
        / "outputs"
        / "phase6_screening"
        / "metabolite_predictability_screening.csv"
    )

    output_dir = (
        repo_root
        / "10_analysis"
        / "outputs"
        / "phase6_screening_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 6B - ANALYZE SCREENING RESULTS")
    print("=" * 70)
    print(f"Input : {input_path}")
    print(f"Output: {output_dir}")
    print()

    if not input_path.exists():
        raise FileNotFoundError(f"Missing file: {input_path}")

    df = pd.read_csv(input_path)

    required_cols = [
        "metabolite",
        "mean_r2",
        "std_r2",
        "mean_rmse",
        "zero_fraction",
        "predictability_class",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["assay"] = df["metabolite"].apply(get_assay)

    # ------------------------------------------------------------
    # Résumés principaux
    # ------------------------------------------------------------
    class_summary = (
        df["predictability_class"]
        .value_counts()
        .rename_axis("predictability_class")
        .reset_index(name="n_metabolites")
    )

    r2_thresholds = [0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    threshold_rows = []
    for thr in r2_thresholds:
        threshold_rows.append({
            "r2_threshold": thr,
            "n_metabolites_above_threshold": int((df["mean_r2"] > thr).sum())
        })
    r2_threshold_summary = pd.DataFrame(threshold_rows)

    assay_summary = (
        df.groupby("assay")
        .agg(
            n_metabolites=("metabolite", "count"),
            mean_r2=("mean_r2", "mean"),
            median_r2=("mean_r2", "median"),
            max_r2=("mean_r2", "max"),
            n_r2_gt_0=("mean_r2", lambda x: int((x > 0).sum())),
            n_r2_gt_0_1=("mean_r2", lambda x: int((x > 0.1).sum())),
            mean_zero_fraction=("zero_fraction", "mean"),
        )
        .reset_index()
        .sort_values("max_r2", ascending=False)
    )

    # ------------------------------------------------------------
    # Corrélation performance vs zero_fraction
    # ------------------------------------------------------------
    corr_zero_r2_pearson = df["zero_fraction"].corr(df["mean_r2"], method="pearson")
    corr_zero_r2_spearman = df["zero_fraction"].corr(df["mean_r2"], method="spearman")

    summary = {
        "n_total_metabolites": int(len(df)),
        "n_r2_gt_0": int((df["mean_r2"] > 0).sum()),
        "n_r2_gt_0_1": int((df["mean_r2"] > 0.1).sum()),
        "n_r2_gt_0_2": int((df["mean_r2"] > 0.2).sum()),
        "n_r2_gt_0_3": int((df["mean_r2"] > 0.3).sum()),
        "best_r2": float(df["mean_r2"].max()),
        "median_r2": float(df["mean_r2"].median()),
        "mean_r2": float(df["mean_r2"].mean()),
        "pearson_corr_zero_fraction_vs_r2": float(corr_zero_r2_pearson),
        "spearman_corr_zero_fraction_vs_r2": float(corr_zero_r2_spearman),
    }

    # ------------------------------------------------------------
    # Fichiers CSV
    # ------------------------------------------------------------
    df.sort_values("mean_r2", ascending=False).to_csv(
        output_dir / "screening_ranked_with_assay.csv",
        index=False
    )

    class_summary.to_csv(output_dir / "predictability_class_summary.csv", index=False)
    r2_threshold_summary.to_csv(output_dir / "r2_threshold_summary.csv", index=False)
    assay_summary.to_csv(output_dir / "assay_summary.csv", index=False)

    pd.DataFrame([summary]).to_csv(output_dir / "screening_analysis_summary.csv", index=False)

    df[df["mean_r2"] > 0].to_csv(output_dir / "metabolites_r2_positive.csv", index=False)
    df[df["mean_r2"] > 0.1].to_csv(output_dir / "metabolites_r2_gt_0_1.csv", index=False)
    df[df["mean_r2"] > 0.2].to_csv(output_dir / "metabolites_r2_gt_0_2.csv", index=False)
    df[df["mean_r2"] > 0.3].to_csv(output_dir / "metabolites_r2_gt_0_3.csv", index=False)

    df.head(50).to_csv(output_dir / "top50_predictable_metabolites.csv", index=False)

    # ------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------
    save_hist(
        df["mean_r2"],
        "Distribution of mean cross-validated R2 across metabolites",
        "Mean CV R2",
        output_dir / "mean_r2_distribution.png",
        bins=50
    )

    save_hist(
        df["zero_fraction"],
        "Distribution of zero fraction across metabolites",
        "Zero fraction",
        output_dir / "zero_fraction_distribution.png",
        bins=40
    )

    save_scatter(
        df["zero_fraction"],
        df["mean_r2"],
        "Zero fraction vs mean CV R2",
        "Zero fraction",
        "Mean CV R2",
        output_dir / "zero_fraction_vs_mean_r2.png"
    )

    # barplot class summary
    save_bar(
        class_summary["predictability_class"].astype(str).tolist(),
        class_summary["n_metabolites"].tolist(),
        "Predictability classes",
        "Number of metabolites",
        output_dir / "predictability_class_barplot.png"
    )

    # assay max R2
    save_bar(
        assay_summary["assay"].astype(str).tolist(),
        assay_summary["max_r2"].tolist(),
        "Max R2 by assay",
        "Max mean CV R2",
        output_dir / "assay_max_r2_barplot.png"
    )

    # assay count R2>0.1
    save_bar(
        assay_summary["assay"].astype(str).tolist(),
        assay_summary["n_r2_gt_0_1"].tolist(),
        "Number of metabolites with R2 > 0.1 by assay",
        "Count",
        output_dir / "assay_n_r2_gt_0_1_barplot.png"
    )

    # ------------------------------------------------------------
    # Console
    # ------------------------------------------------------------
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total metabolites                 : {summary['n_total_metabolites']}")
    print(f"Metabolites with R2 > 0           : {summary['n_r2_gt_0']}")
    print(f"Metabolites with R2 > 0.1         : {summary['n_r2_gt_0_1']}")
    print(f"Metabolites with R2 > 0.2         : {summary['n_r2_gt_0_2']}")
    print(f"Metabolites with R2 > 0.3         : {summary['n_r2_gt_0_3']}")
    print(f"Best R2                           : {summary['best_r2']:.4f}")
    print(f"Mean R2                           : {summary['mean_r2']:.4f}")
    print(f"Median R2                         : {summary['median_r2']:.4f}")
    print(f"Spearman corr zero_fraction vs R2 : {summary['spearman_corr_zero_fraction_vs_r2']:.4f}")
    print()
    print("Predictability class summary:")
    print(class_summary.to_string(index=False))
    print()
    print("Assay summary:")
    print(assay_summary.to_string(index=False))
    print()
    print("Top 20 predictable metabolites:")
    print(df.sort_values("mean_r2", ascending=False).head(20).to_string(index=False))
    print()
    print("Main outputs:")
    print(output_dir / "screening_analysis_summary.csv")
    print(output_dir / "screening_ranked_with_assay.csv")
    print(output_dir / "assay_summary.csv")
    print(output_dir / "top50_predictable_metabolites.csv")
    print(output_dir / "mean_r2_distribution.png")
    print(output_dir / "zero_fraction_vs_mean_r2.png")
    print()
    print("Screening analysis completed successfully.")


if __name__ == "__main__":
    main()