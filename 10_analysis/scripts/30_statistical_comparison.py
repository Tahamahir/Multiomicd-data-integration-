from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")


ROOT = Path("10_analysis/outputs")
OUT = ROOT / "phase30_statistical_validation"
OUT.mkdir(parents=True, exist_ok=True)


TARGET_FOLDERS = [
    "phase22_pipeline_optimization",
    "phase22B_pipeline_optimization",
    "phase25_dimension_reduction_late_rf",
    "phase26_tune_champion_late_sparsepca_rf",
]


def load_metric_files():
    rows = []

    for folder in TARGET_FOLDERS:
        folder_path = ROOT / folder

        if not folder_path.exists():
            print(f"[SKIP] Missing folder: {folder_path}")
            continue

        for file_path in sorted(folder_path.glob("*_metrics_per_metabolite.csv")):
            try:
                df = pd.read_csv(file_path)

                if "metabolite" not in df.columns or "r2" not in df.columns:
                    continue

                if "experiment_id" in df.columns:
                    pipeline_name = str(df["experiment_id"].iloc[0])
                else:
                    pipeline_name = file_path.name.replace(
                        "_metrics_per_metabolite.csv", ""
                    )

                tmp = pd.DataFrame({
                    "pipeline": pipeline_name,
                    "metabolite": df["metabolite"],
                    "r2": df["r2"],
                })

                rows.append(tmp)

            except Exception as e:
                print(f"[WARNING] Could not read {file_path}: {e}")

    if not rows:
        raise RuntimeError("No valid metrics files found.")

    return pd.concat(rows, ignore_index=True)


def build_matrix(df):
    pivot = df.pivot_table(
        index="metabolite",
        columns="pipeline",
        values="r2",
        aggfunc="mean",
    )

    pivot = pivot.dropna(axis=0, how="any")
    pivot = pivot.dropna(axis=1, how="any")

    return pivot


def compute_pipeline_summary(pivot):
    summary = []

    for p in pivot.columns:
        scores = pivot[p]

        summary.append({
            "pipeline": p,
            "mean_r2": float(scores.mean()),
            "median_r2": float(scores.median()),
            "std_r2": float(scores.std()),
            "min_r2": float(scores.min()),
            "max_r2": float(scores.max()),
            "n_metabolites": int(scores.shape[0]),
            "n_r2_gt_0": int((scores > 0).sum()),
            "n_r2_gt_02": int((scores > 0.2).sum()),
            "n_r2_gt_04": int((scores > 0.4).sum()),
            "n_r2_gt_06": int((scores > 0.6).sum()),
        })

    return (
        pd.DataFrame(summary)
        .sort_values(["mean_r2", "median_r2"], ascending=False)
        .reset_index(drop=True)
    )


def compute_friedman(pivot):
    pipelines = list(pivot.columns)
    scores = [pivot[p].values for p in pipelines]

    stat, pvalue = friedmanchisquare(*scores)

    return pd.DataFrame([{
        "n_pipelines": len(pipelines),
        "n_metabolites": pivot.shape[0],
        "friedman_stat": float(stat),
        "pvalue": float(pvalue),
    }])


def compute_ranks(pivot):
    ranks = pivot.rank(axis=1, ascending=False)
    mean_ranks = ranks.mean(axis=0).sort_values()

    out = mean_ranks.reset_index()
    out.columns = ["pipeline", "mean_rank"]

    summary = compute_pipeline_summary(pivot)

    out = out.merge(
        summary[["pipeline", "mean_r2", "median_r2"]],
        on="pipeline",
        how="left",
    )

    return out.sort_values("mean_rank").reset_index(drop=True)


def safe_wilcoxon(a, b):
    diff = a - b

    if np.allclose(diff, 0, equal_nan=True):
        return 0.0, 1.0

    try:
        stat, pvalue = wilcoxon(
            a,
            b,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            mode="auto",
        )
        return float(stat), float(pvalue)

    except Exception:
        return np.nan, 1.0


def compute_wilcoxon(pivot):
    pipelines = list(pivot.columns)
    rows = []

    for i in range(len(pipelines)):
        for j in range(i + 1, len(pipelines)):
            a = pipelines[i]
            b = pipelines[j]

            scores_a = pivot[a].values
            scores_b = pivot[b].values

            stat, pvalue = safe_wilcoxon(scores_a, scores_b)

            rows.append({
                "pipeline_A": a,
                "pipeline_B": b,
                "wilcoxon_stat": stat,
                "p": pvalue,
                "mean_r2_A": float(np.mean(scores_a)),
                "mean_r2_B": float(np.mean(scores_b)),
                "delta_mean_r2_A_minus_B": float(np.mean(scores_a) - np.mean(scores_b)),
                "median_r2_A": float(np.median(scores_a)),
                "median_r2_B": float(np.median(scores_b)),
            })

    wil = pd.DataFrame(rows)

    reject, p_corr, _, _ = multipletests(
        wil["p"].values,
        alpha=0.05,
        method="holm",
    )

    wil["p_corrected_holm"] = p_corr
    wil["significant_holm_005"] = reject

    wil = wil.sort_values(
        ["p_corrected_holm", "delta_mean_r2_A_minus_B"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return wil


def compare_champion(wil, champion):
    rows = []

    for _, r in wil.iterrows():
        a = r["pipeline_A"]
        b = r["pipeline_B"]

        if a == champion:
            rows.append({
                "champion": champion,
                "competitor": b,
                "champion_mean_r2": r["mean_r2_A"],
                "competitor_mean_r2": r["mean_r2_B"],
                "delta_champion_minus_competitor": r["delta_mean_r2_A_minus_B"],
                "p": r["p"],
                "p_corrected_holm": r["p_corrected_holm"],
                "significant_holm_005": r["significant_holm_005"],
            })

        elif b == champion:
            rows.append({
                "champion": champion,
                "competitor": a,
                "champion_mean_r2": r["mean_r2_B"],
                "competitor_mean_r2": r["mean_r2_A"],
                "delta_champion_minus_competitor": -r["delta_mean_r2_A_minus_B"],
                "p": r["p"],
                "p_corrected_holm": r["p_corrected_holm"],
                "significant_holm_005": r["significant_holm_005"],
            })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values("delta_champion_minus_competitor", ascending=False)
        .reset_index(drop=True)
    )


def main():
    print("=" * 70)
    print("PHASE 30 - STATISTICAL COMPARISON OF PIPELINES")
    print("=" * 70)

    df = load_metric_files()
    pivot = build_matrix(df)

    print(f"Performance matrix shape: {pivot.shape}")
    print(f"Metabolites compared     : {pivot.shape[0]}")
    print(f"Pipelines compared       : {pivot.shape[1]}")

    pivot.to_csv(OUT / "r2_matrix_metabolite_by_pipeline.csv")

    pipeline_summary = compute_pipeline_summary(pivot)
    pipeline_summary.to_csv(OUT / "pipeline_summary.csv", index=False)

    friedman = compute_friedman(pivot)
    friedman.to_csv(OUT / "friedman_test.csv", index=False)

    ranks = compute_ranks(pivot)
    ranks.to_csv(OUT / "friedman_mean_ranks.csv", index=False)

    wil = compute_wilcoxon(pivot)
    wil.to_csv(OUT / "wilcoxon_pairwise_holm.csv", index=False)

    champion = pipeline_summary.iloc[0]["pipeline"]
    champion_comparison = compare_champion(wil, champion)
    champion_comparison.to_csv(OUT / "champion_pairwise_comparison.csv", index=False)

    summary = {
        "n_metabolites": int(pivot.shape[0]),
        "n_pipelines": int(pivot.shape[1]),
        "best_pipeline_by_mean_r2": champion,
        "best_mean_r2": float(pipeline_summary.iloc[0]["mean_r2"]),
        "best_median_r2": float(pipeline_summary.iloc[0]["median_r2"]),
        "friedman_pvalue": float(friedman.iloc[0]["pvalue"]),
        "friedman_significant_005": bool(friedman.iloc[0]["pvalue"] < 0.05),
    }

    with open(OUT / "statistical_validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("Friedman test:")
    print(friedman.to_string(index=False))

    print()
    print("Top pipelines:")
    print(pipeline_summary.head(15).to_string(index=False))

    print()
    print("Mean ranks:")
    print(ranks.head(15).to_string(index=False))

    print()
    print(f"Champion: {champion}")
    print("Champion pairwise comparison:")
    print(champion_comparison.head(20).to_string(index=False))

    print()
    print("Outputs saved in:")
    print(OUT)


if __name__ == "__main__":
    main()
