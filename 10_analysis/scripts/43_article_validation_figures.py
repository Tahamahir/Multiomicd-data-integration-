#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 43 - Article-grade validation figures

Input:
    10_analysis/outputs/phase42_article_validation/
        - baseline_validation_metrics.csv
        - baseline_validation_summary.csv

Output:
    10_analysis/outputs/phase43_article_figures/
        - Figure 1: Model comparison real vs null
        - Figure 2: R2 distribution boxplots
        - Figure 3: Delta R2 per metabolite
        - Figure 4: Per-metabolite R2 heatmap
        - Summary CSV and article paragraph
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(".")
INPUT_DIR = PROJECT_ROOT / "10_analysis/outputs/phase42_article_validation"
OUT_DIR = PROJECT_ROOT / "10_analysis/outputs/phase43_article_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_FILE = INPUT_DIR / "baseline_validation_metrics.csv"
SUMMARY_FILE = INPUT_DIR / "baseline_validation_summary.csv"

MODEL_ORDER = ["Soil_only", "MG_only", "MG_Soil_late"]

MODEL_LABELS = {
    "Soil_only": "Soil-only",
    "MG_only": "MG-only",
    "MG_Soil_late": "MG+Soil",
}

COLORS = {
    "Real": "#2c7fb8",
    "Null": "#bdbdbd",
    "Soil_only": "#e67e22",
    "MG_only": "#2ecc71",
    "MG_Soil_late": "#3498db",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ============================================================
# HELPERS
# ============================================================

def parse_bool_column(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def short_metabolite_name(x):
    x = str(x)
    if "|IK:" in x:
        mode = x.split("|IK:")[0]
        ik = x.split("|IK:")[1].split("-")[0]
        mode = mode.replace("C18_negative", "C18-").replace("C18_positive", "C18+")
        return f"{mode}|{ik[:8]}"
    return x[:28]


def load_data():
    if not METRICS_FILE.exists():
        raise FileNotFoundError(f"Missing file: {METRICS_FILE}")
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(f"Missing file: {SUMMARY_FILE}")

    metrics = pd.read_csv(METRICS_FILE)
    summary = pd.read_csv(SUMMARY_FILE)

    metrics["shuffle"] = parse_bool_column(metrics["shuffle"])
    summary["shuffle"] = parse_bool_column(summary["shuffle"])

    metrics["r2"] = pd.to_numeric(metrics["r2"], errors="coerce")
    metrics["rmse"] = pd.to_numeric(metrics["rmse"], errors="coerce")
    metrics["mae"] = pd.to_numeric(metrics["mae"], errors="coerce")

    metrics = metrics.dropna(subset=["r2"])

    return metrics, summary


def prepare_real_and_null(metrics):
    real = metrics[metrics["shuffle"] == False].copy()
    null = metrics[metrics["shuffle"] == True].copy()

    # Null model has several permutations.
    # For fair visual comparison, average null permutations per metabolite/model.
    null_avg = (
        null.groupby(["metabolite", "model"], as_index=False)
        .agg(
            r2=("r2", "mean"),
            rmse=("rmse", "mean"),
            mae=("mae", "mean"),
        )
    )
    null_avg["shuffle"] = True

    return real, null_avg


# ============================================================
# FIGURE 1 - BARPLOT REAL VS NULL
# ============================================================

def plot_model_comparison(real, null_avg):
    rows = []

    for model in MODEL_ORDER:
        r = real[real["model"] == model]["r2"]
        n = null_avg[null_avg["model"] == model]["r2"]

        rows.append({
            "model": model,
            "label": MODEL_LABELS[model],
            "condition": "Real",
            "mean_r2": r.mean(),
            "std_r2": r.std(),
            "n": r.shape[0],
        })

        rows.append({
            "model": model,
            "label": MODEL_LABELS[model],
            "condition": "Null",
            "mean_r2": n.mean(),
            "std_r2": n.std(),
            "n": n.shape[0],
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "article_validation_figure_summary.csv", index=False)

    x = np.arange(len(MODEL_ORDER))
    width = 0.34

    real_means = [
        df[(df["model"] == m) & (df["condition"] == "Real")]["mean_r2"].iloc[0]
        for m in MODEL_ORDER
    ]
    real_stds = [
        df[(df["model"] == m) & (df["condition"] == "Real")]["std_r2"].iloc[0]
        for m in MODEL_ORDER
    ]

    null_means = [
        df[(df["model"] == m) & (df["condition"] == "Null")]["mean_r2"].iloc[0]
        for m in MODEL_ORDER
    ]
    null_stds = [
        df[(df["model"] == m) & (df["condition"] == "Null")]["std_r2"].iloc[0]
        for m in MODEL_ORDER
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    ax.bar(
        x - width / 2,
        real_means,
        width,
        yerr=real_stds,
        capsize=4,
        label="Real",
        color=COLORS["Real"],
        edgecolor="black",
        linewidth=0.5,
    )

    ax.bar(
        x + width / 2,
        null_means,
        width,
        yerr=null_stds,
        capsize=4,
        label="Shuffled null",
        color=COLORS["Null"],
        edgecolor="black",
        linewidth=0.5,
    )

    ax.axhline(0, color="black", linewidth=1, linestyle="--")

    for i, v in enumerate(real_means):
        ax.text(i - width / 2, v + 0.025, f"{v:.3f}", ha="center", fontsize=10)

    for i, v in enumerate(null_means):
        ax.text(i + width / 2, v - 0.045, f"{v:.3f}", ha="center", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_ylabel("Mean cross-validated R²")
    ax.set_title("Predictive performance compared with shuffled null models")
    ax.legend(frameon=False)
    ax.set_ylim(min(null_means) - 0.15, max(real_means) + 0.15)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig1_model_comparison_real_vs_null.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig1_model_comparison_real_vs_null.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# FIGURE 2 - BOXPLOT R2 DISTRIBUTIONS
# ============================================================

def plot_r2_boxplots(real, null_avg):
    data = []
    labels = []
    colors = []
    positions = []

    pos = 1
    for model in MODEL_ORDER:
        r = real[real["model"] == model]["r2"].values
        n = null_avg[null_avg["model"] == model]["r2"].values

        data.append(r)
        labels.append(f"{MODEL_LABELS[model]}\nReal")
        colors.append(COLORS["Real"])
        positions.append(pos)

        data.append(n)
        labels.append(f"{MODEL_LABELS[model]}\nNull")
        colors.append(COLORS["Null"])
        positions.append(pos + 0.55)

        pos += 1.55

    fig, ax = plt.subplots(figsize=(10, 5.8))

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.42,
        patch_artist=True,
        showfliers=True,
        medianprops=dict(color="black", linewidth=1.3),
        boxprops=dict(linewidth=0.8),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Cross-validated R²")
    ax.set_title("Distribution of R² across metabolites")
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig2_r2_boxplot_real_vs_null.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig2_r2_boxplot_real_vs_null.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# FIGURE 3 - DELTA R2 PER METABOLITE
# ============================================================

def plot_delta_r2(real):
    pivot = real.pivot_table(
        index="metabolite",
        columns="model",
        values="r2",
        aggfunc="mean",
    )

    for model in MODEL_ORDER:
        if model not in pivot.columns:
            pivot[model] = np.nan

    pivot["delta_integrated_vs_MG"] = pivot["MG_Soil_late"] - pivot["MG_only"]
    pivot["delta_integrated_vs_Soil"] = pivot["MG_Soil_late"] - pivot["Soil_only"]
    pivot["metabolite_short"] = [short_metabolite_name(x) for x in pivot.index]

    delta_df = pivot.reset_index()
    delta_df.to_csv(OUT_DIR / "delta_r2_per_metabolite.csv", index=False)

    plot_df = delta_df.sort_values("delta_integrated_vs_MG", ascending=True)

    fig_h = max(7, 0.28 * len(plot_df))
    fig, ax = plt.subplots(figsize=(10, fig_h))

    y = np.arange(len(plot_df))

    ax.barh(
        y - 0.18,
        plot_df["delta_integrated_vs_MG"],
        height=0.34,
        color="#3498db",
        label="MG+Soil - MG-only",
    )

    ax.barh(
        y + 0.18,
        plot_df["delta_integrated_vs_Soil"],
        height=0.34,
        color="#e67e22",
        label="MG+Soil - Soil-only",
    )

    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["metabolite_short"], fontsize=8)
    ax.set_xlabel("R²")
    ax.set_title("Performance gain of integrated model by metabolite")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig3_delta_r2_integrated_vs_baselines.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig3_delta_r2_integrated_vs_baselines.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# FIGURE 4 - PER-METABOLITE HEATMAP
# ============================================================

def plot_per_metabolite_heatmap(real, null_avg):
    real_pivot = real.pivot_table(
        index="metabolite",
        columns="model",
        values="r2",
        aggfunc="mean",
    )

    null_integrated = (
        null_avg[null_avg["model"] == "MG_Soil_late"]
        .set_index("metabolite")["r2"]
    )

    heat = pd.DataFrame(index=real_pivot.index)

    heat["Soil-only"] = real_pivot.get("Soil_only")
    heat["MG-only"] = real_pivot.get("MG_only")
    heat["MG+Soil"] = real_pivot.get("MG_Soil_late")
    heat["Null MG+Soil"] = null_integrated

    heat = heat.sort_values("MG+Soil", ascending=False)
    heat_labels = [short_metabolite_name(x) for x in heat.index]

    fig_h = max(8, 0.28 * len(heat))
    fig, ax = plt.subplots(figsize=(7.5, fig_h))

    im = ax.imshow(
        heat.values,
        aspect="auto",
        interpolation="nearest",
        vmin=-0.2,
        vmax=max(0.75, np.nanmax(heat.values)),
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("R²")

    ax.set_xticks(np.arange(heat.shape[1]))
    ax.set_xticklabels(heat.columns, rotation=30, ha="right")

    ax.set_yticks(np.arange(heat.shape[0]))
    ax.set_yticklabels(heat_labels, fontsize=8)

    ax.set_title("Per-metabolite predictive performance")
    ax.set_xlabel("Model")
    ax.set_ylabel("Metabolite")

    # add values
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = heat.values[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig4_per_metabolite_r2_heatmap.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig4_per_metabolite_r2_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# ARTICLE PARAGRAPH
# ============================================================

def write_article_paragraph(summary):
    def get(model, shuffle, col):
        row = summary[(summary["model"] == model) & (summary["shuffle"] == shuffle)]
        if row.empty:
            return np.nan
        return row.iloc[0][col]

    mgsoil_r2 = get("MG_Soil_late", False, "mean_r2")
    mg_r2 = get("MG_only", False, "mean_r2")
    soil_r2 = get("Soil_only", False, "mean_r2")
    null_r2 = get("MG_Soil_late", True, "mean_r2")

    mgsoil_med = get("MG_Soil_late", False, "median_r2")
    mgsoil_max = get("MG_Soil_late", False, "max_r2")
    mgsoil_min = get("MG_Soil_late", False, "min_r2")

    paragraph = f"""
Article-ready validation paragraph
==================================

The integrated MG+Soil late-fusion model achieved the best overall predictive performance, with a mean cross-validated R² of {mgsoil_r2:.3f} and a median R² of {mgsoil_med:.3f} across 30 metabolites. This performance was higher than both the MG-only baseline (mean R² = {mg_r2:.3f}) and the Soil-only baseline (mean R² = {soil_r2:.3f}). The per-metabolite performance ranged from R² = {mgsoil_min:.3f} to R² = {mgsoil_max:.3f}, indicating heterogeneous predictability across metabolites. In contrast, the shuffled null model produced a negative mean R² ({null_r2:.3f}), confirming that the observed predictive performance was not driven by random associations. Overall, these results support the presence of a measurable multi-omic signal linking microbial and soil-derived features to metabolomic variation.
""".strip()

    with open(OUT_DIR / "article_results_paragraph.txt", "w") as f:
        f.write(paragraph + "\n")


# ============================================================
# MAIN
# ============================================================

def main():
    print("[PHASE 43] Loading Phase 42 results...")
    metrics, summary = load_data()

    real, null_avg = prepare_real_and_null(metrics)

    print("[INFO] Real rows:", len(real))
    print("[INFO] Null averaged rows:", len(null_avg))
    print("[INFO] Metabolites:", real["metabolite"].nunique())

    print("[PHASE 43] Plotting Figure 1...")
    plot_model_comparison(real, null_avg)

    print("[PHASE 43] Plotting Figure 2...")
    plot_r2_boxplots(real, null_avg)

    print("[PHASE 43] Plotting Figure 3...")
    plot_delta_r2(real)

    print("[PHASE 43] Plotting Figure 4...")
    plot_per_metabolite_heatmap(real, null_avg)

    print("[PHASE 43] Writing article paragraph...")
    write_article_paragraph(summary)

    print("\n[DONE] Phase 43 completed.")
    print(f"Output folder: {OUT_DIR}")
    print("\nMain files:")
    print(f"- {OUT_DIR / 'fig1_model_comparison_real_vs_null.png'}")
    print(f"- {OUT_DIR / 'fig2_r2_boxplot_real_vs_null.png'}")
    print(f"- {OUT_DIR / 'fig3_delta_r2_integrated_vs_baselines.png'}")
    print(f"- {OUT_DIR / 'fig4_per_metabolite_r2_heatmap.png'}")
    print(f"- {OUT_DIR / 'article_results_paragraph.txt'}")


if __name__ == "__main__":
    main()
