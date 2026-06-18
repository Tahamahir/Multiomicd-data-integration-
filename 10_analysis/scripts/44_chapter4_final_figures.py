#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 44 - Final Chapter 4 EDA Figures

Goal:
Generate professional, readable and report-ready figures for:
Chapter 4 — Description des données et préparation du dataset.

The script uses existing EDA outputs if available and falls back to
known confirmed values from previous analyses.

Output folder:
10_analysis/outputs/phase44_chapter4_figures_final/
"""

from pathlib import Path
import re
import textwrap
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(".")
OUTPUTS_ROOT = ROOT / "10_analysis/outputs"
OUT_DIR = OUTPUTS_ROOT / "phase44_chapter4_figures_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Confirmed values from your EDA and final ML dataset
CONFIRMED = {
    "n_samples": 105,

    # Initial aligned EDA dataset
    "x_initial": 3066,
    "x_initial_soil": 27,
    "x_initial_mg": 3039,
    "y_initial": 652,

    # Final ML dataset
    "x_final": 3059,
    "x_final_soil": 20,
    "x_final_mg": 3039,
    "y_final": 47,

    # EDA metrics
    "x_mean_zero_fraction": 0.18403690243220577,
    "y_mean_zero_fraction": 0.6261904761904761,
    "n_x_features_zero_fraction_ge_0_90": 137,
    "n_y_metabolites_zero_fraction_ge_0_90": 0,
    "n_x_features_variance_lt_1e_4": 2197,
    "n_soil_strong_pairs_abs_ge_0_90": 18,
}

SOIL_COLUMNS_CONFIRMED = [
    "soil_pH",
    "soil_NH4",
    "soil_NO3",
    "soil_total_C",
    "soil_total_N",
    "chem__LBC (mg/kg CaCO3/pH dry weight)",
    "chem__LBCeq (mg/kg CaCO3/pH dry weight)",
    "chem__pH",
    "chem__Ca (Mehlich 1 mg/kg dry weight)",
    "chem__K (Mehlich 1 mg/kg dry weight)",
    "chem__Mg (Mehlich 1 mg/kg dry weight)",
    "chem__Mn (Mehlich 1 mg/kg dry weight)",
    "chem__P (Mehlich 1 mg/kg dry weight)",
    "chem__Zn (Mehlich 1 mg/kg dry weight)",
    "chem__NH4-N (mg/kg dry weight)",
    "chem__NO3-N (mg/kg dry weight)",
    "chem__C (%)",
    "chem__N (%)",
    "denit__Rate (microgN/kilogram dry weight soil/day)",
    "moist__Replicate 1 Mass Water / Mass Dry Soil %",
    "moist__Replicate 2 Mass Water / Mass Dry Soil %",
    "moist__Replicate 3 Mass Water / Mass Dry Soil %",
    "nitrif__Technical Replicate",
    "nitrif__Rate (mg NO3-N/kilogram dry weight soil/day)",
    "psize__Soil Sand Content %",
    "psize__Soil Clay Content %",
    "psize__Soil Silt Content %",
]

# Manual known summaries from your outputs
X_ZERO_THRESHOLD_SUMMARY = pd.DataFrame({
    "threshold": [0.5, 0.7, 0.8, 0.9, 0.95],
    "n_features_above_threshold": [539, 374, 287, 137, 0],
})

X_LOW_VARIANCE_SUMMARY = pd.DataFrame({
    "threshold": [1e-8, 1e-6, 1e-4, 1e-3, 1e-2],
    "n_features_below_threshold": [0, 1083, 2197, 2671, 2901],
})

Y_PRESENCE_SENSITIVITY = pd.DataFrame({
    "threshold": [0.0, 1e-9, 1e-6, 1e-3],
    "mean_presence_fraction": [0.37380952380952376] * 4,
    "median_presence_fraction": [0.2761904761904762] * 4,
    "n_metabolites_present_in_ge_10pct": [652] * 4,
    "n_metabolites_present_in_ge_20pct": [435] * 4,
})

# Professional style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ============================================================
# UTILITIES
# ============================================================

def find_file(filename: str):
    """Search a file recursively under 10_analysis/outputs."""
    candidates = list(OUTPUTS_ROOT.rglob(filename))
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def read_csv_if_exists(filename: str):
    path = find_file(filename)
    if path is None:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def save_fig(fig, name: str):
    fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def short_label(label: str, max_len: int = 24):
    label = str(label)
    label = label.replace("chem__", "")
    label = label.replace("soil_", "soil:")
    label = label.replace("psize__", "texture:")
    label = label.replace("moist__", "moist:")
    label = label.replace("nitrif__", "nitrif:")
    label = label.replace("denit__", "denit:")

    if len(label) <= max_len:
        return label

    return label[:max_len - 3] + "..."


def classify_soil_column(col: str):
    c = col.lower()
    if "ph" in c or "lbc" in c:
        return "pH / acidity"
    if "nh4" in c or "no3" in c or "nitrif" in c or "denit" in c or "total_n" in c or "chem__n" in c:
        return "Nitrogen cycle"
    if "total_c" in c or "chem__c" in c:
        return "Carbon"
    if any(x in c for x in ["ca", "mg", "mn", "zn", " k ", "__k", "__p", "mehlich"]):
        return "Minerals"
    if "moist" in c or "water" in c:
        return "Moisture"
    if "sand" in c or "clay" in c or "silt" in c or "psize" in c:
        return "Texture"
    return "Other"


def wrap_text(s, width=20):
    return "\n".join(textwrap.wrap(str(s), width=width))


# ============================================================
# FIGURE 4.1 — DATASET OVERVIEW
# ============================================================

def fig4_1_dataset_overview():
    data = pd.DataFrame({
        "Block": [
            "Initial X\n(all features)",
            "Initial MG",
            "Initial Soil",
            "Initial Y\n(metabolites)",
            "Final X\n(ML dataset)",
            "Final MG",
            "Final Soil",
            "Final Y\n(targets)",
        ],
        "Variables": [
            CONFIRMED["x_initial"],
            CONFIRMED["x_initial_mg"],
            CONFIRMED["x_initial_soil"],
            CONFIRMED["y_initial"],
            CONFIRMED["x_final"],
            CONFIRMED["x_final_mg"],
            CONFIRMED["x_final_soil"],
            CONFIRMED["y_final"],
        ],
        "Stage": [
            "Initial", "Initial", "Initial", "Initial",
            "Final", "Final", "Final", "Final"
        ]
    })

    colors = ["#9ecae1" if s == "Initial" else "#3182bd" for s in data["Stage"]]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(data["Block"], data["Variables"], color=colors, edgecolor="black", linewidth=0.6)

    ax.set_yscale("log")
    ax.set_ylabel("Number of variables (log scale)")
    ax.set_title("Dataset structure before and after preprocessing")
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, data["Variables"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.12,
            f"{value}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    ax.text(
        0.5, 0.93,
        f"{CONFIRMED['n_samples']} aligned samples across all data blocks",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f7f7f7", edgecolor="#cccccc")
    )

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    save_fig(fig, "fig4_1_dataset_overview")

    data.to_csv(OUT_DIR / "fig4_1_dataset_overview_data.csv", index=False)


# ============================================================
# FIGURE 4.2 — X SPARSITY SUMMARY
# ============================================================

def fig4_2_x_sparsity_summary():
    df = X_ZERO_THRESHOLD_SUMMARY.copy()
    df["threshold_percent"] = df["threshold"] * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.1, 1]})

    ax = axes[0]
    bars = ax.bar(
        [f">{int(t)}%" for t in df["threshold_percent"]],
        df["n_features_above_threshold"],
        color="#2c7fb8",
        edgecolor="black",
        linewidth=0.6
    )
    ax.set_ylabel("Number of X features")
    ax.set_xlabel("Zero-fraction threshold")
    ax.set_title("Features with high zero fraction")
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, df["n_features_above_threshold"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(df["n_features_above_threshold"]) * 0.03,
            f"{int(value)}",
            ha="center",
            fontsize=10,
            fontweight="bold"
        )

    ax = axes[1]
    labels = ["Non-zero\naverage", "Zero\naverage"]
    values = [
        1 - CONFIRMED["x_mean_zero_fraction"],
        CONFIRMED["x_mean_zero_fraction"]
    ]
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
        colors=["#74c476", "#fdae6b"],
        wedgeprops=dict(edgecolor="black", linewidth=0.6),
        textprops=dict(fontsize=11)
    )
    ax.set_title("Average zero fraction in X")

    fig.suptitle("Sparsity analysis of explanatory variables", fontsize=15)
    plt.tight_layout()
    save_fig(fig, "fig4_2_x_sparsity_summary")

    df.to_csv(OUT_DIR / "fig4_2_x_sparsity_summary_data.csv", index=False)


# ============================================================
# FIGURE 4.3 — LOW VARIANCE SUMMARY
# ============================================================

def fig4_3_x_low_variance_summary():
    df = X_LOW_VARIANCE_SUMMARY.copy()
    df["threshold_label"] = ["$10^{-8}$", "$10^{-6}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$"]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars = ax.bar(
        df["threshold_label"],
        df["n_features_below_threshold"],
        color="#756bb1",
        edgecolor="black",
        linewidth=0.6
    )

    ax.set_xlabel("Variance threshold")
    ax.set_ylabel("Number of X features below threshold")
    ax.set_title("Low-variance features in the explanatory matrix")
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, df["n_features_below_threshold"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(df["n_features_below_threshold"]) * 0.025,
            f"{int(value)}",
            ha="center",
            fontsize=10,
            fontweight="bold"
        )

    ax.text(
        0.5, 0.88,
        "Large number of weakly variable features → feature selection is required",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f7f7f7", edgecolor="#cccccc")
    )

    plt.tight_layout()
    save_fig(fig, "fig4_3_x_low_variance_summary")

    df.to_csv(OUT_DIR / "fig4_3_x_low_variance_summary_data.csv", index=False)


# ============================================================
# FIGURE 4.4 — METABOLOMICS SPARSITY + LOG EFFECT
# ============================================================

def fig4_4_metabolomics_sparsity_and_log():
    y_raw = read_csv_if_exists("y_nonzero_values_raw.csv")
    y_log = read_csv_if_exists("y_nonzero_values_log.csv")

    raw_values = None
    log_values = None

    if y_raw is not None:
        raw_values = y_raw.select_dtypes(include=[np.number]).values.flatten()
        raw_values = raw_values[np.isfinite(raw_values)]

    if y_log is not None:
        log_values = y_log.select_dtypes(include=[np.number]).values.flatten()
        log_values = log_values[np.isfinite(log_values)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), gridspec_kw={"width_ratios": [0.8, 1, 1]})

    # Panel A: presence / zero fraction
    ax = axes[0]
    labels = ["Present\nmean", "Zero\nmean"]
    values = [
        1 - CONFIRMED["y_mean_zero_fraction"],
        CONFIRMED["y_mean_zero_fraction"]
    ]
    ax.pie(
        values,
        labels=labels,
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
        colors=["#74c476", "#fb6a4a"],
        wedgeprops=dict(edgecolor="black", linewidth=0.6),
        textprops=dict(fontsize=10)
    )
    ax.set_title("Average metabolite presence")

    # Panel B: raw distribution
    ax = axes[1]
    if raw_values is not None and len(raw_values) > 0:
        cut = np.percentile(raw_values, 99.5)
        raw_plot = raw_values[raw_values <= cut]
        ax.hist(raw_plot, bins=45, color="#9ecae1", edgecolor="black", linewidth=0.4)
        ax.set_xlabel("Raw non-zero intensity")
    else:
        ax.text(0.5, 0.5, "Raw non-zero values\nnot found", ha="center", va="center", fontsize=12)
        ax.set_xlabel("Raw intensity")
    ax.set_ylabel("Frequency")
    ax.set_title("Raw non-zero intensities")
    ax.grid(axis="y", alpha=0.25)

    # Panel C: log distribution
    ax = axes[2]
    if log_values is not None and len(log_values) > 0:
        ax.hist(log_values, bins=45, color="#3182bd", edgecolor="black", linewidth=0.4)
        ax.set_xlabel("log1p non-zero intensity")
    else:
        ax.text(0.5, 0.5, "Log-transformed values\nnot found", ha="center", va="center", fontsize=12)
        ax.set_xlabel("log1p intensity")
    ax.set_ylabel("Frequency")
    ax.set_title("After log1p transformation")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Metabolomic sparsity and effect of log1p transformation", fontsize=15)
    plt.tight_layout()
    save_fig(fig, "fig4_4_metabolomics_sparsity_and_log_transform")


# ============================================================
# FIGURE 4.5 — SOIL VARIABLE CATEGORIES
# ============================================================

def fig4_5_soil_variable_categories():
    soil_cols_df = read_csv_if_exists("soil_columns_detected.csv")

    if soil_cols_df is not None and soil_cols_df.shape[1] >= 1:
        soil_cols = soil_cols_df.iloc[:, 0].dropna().astype(str).tolist()
    else:
        soil_cols = SOIL_COLUMNS_CONFIRMED

    categories = pd.Series([classify_soil_column(c) for c in soil_cols]).value_counts()

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars = ax.bar(
        categories.index,
        categories.values,
        color="#fd8d3c",
        edgecolor="black",
        linewidth=0.6
    )

    ax.set_ylabel("Number of soil variables")
    ax.set_title("Detected soil variables by category")
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, categories.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.15,
            f"{int(value)}",
            ha="center",
            fontsize=10,
            fontweight="bold"
        )

    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    save_fig(fig, "fig4_5_soil_variable_categories")

    pd.DataFrame({"category": categories.index, "n_variables": categories.values}).to_csv(
        OUT_DIR / "fig4_5_soil_variable_categories_data.csv", index=False
    )


# ============================================================
# FIGURE 4.6 — SOIL REDUNDANCY HEATMAP
# ============================================================

def clean_corr_matrix(df):
    """Try to convert different correlation CSV structures into numeric matrix."""
    if df is None:
        return None

    d = df.copy()

    # If first column is names, set it as index
    first_col = d.columns[0]
    if not np.issubdtype(d[first_col].dtype, np.number):
        d = d.set_index(first_col)

    # Convert all to numeric
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # Keep numeric columns and rows
    d = d.select_dtypes(include=[np.number])
    d = d.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if d.empty:
        return None

    return d


def fig4_6_soil_redundancy_heatmap():
    corr_raw = read_csv_if_exists("soil_spearman_correlation_matrix.csv")
    if corr_raw is None:
        corr_raw = read_csv_if_exists("soil_correlation_matrix.csv")

    corr = clean_corr_matrix(corr_raw)

    if corr is None or corr.shape[0] < 2:
        # Create simplified matrix from confirmed redundant pairs if correlation file missing
        labels = [
            "soil_NO3", "chem__NO3-N",
            "soil_NH4", "chem__NH4-N",
            "soil_total_C", "chem__C",
            "soil_total_N", "chem__N",
            "soil_pH", "chem__pH",
            "Clay", "Silt",
        ]
        mat = np.eye(len(labels))
        pairs = [
            (0, 1, 1.0),
            (2, 3, 1.0),
            (4, 5, 1.0),
            (6, 7, 1.0),
            (8, 9, 1.0),
            (10, 11, -0.94),
        ]
        for i, j, v in pairs:
            mat[i, j] = v
            mat[j, i] = v
        corr = pd.DataFrame(mat, index=labels, columns=labels)

    # If matrix too large, keep variables with strongest average absolute correlations
    if corr.shape[0] > 18:
        score = corr.abs().replace(1, np.nan).mean(axis=1).sort_values(ascending=False)
        keep = score.head(18).index.tolist()
        corr = corr.loc[keep, keep]

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm", interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Spearman correlation")

    labels = [short_label(c, 18) for c in corr.columns]

    ax.set_xticks(np.arange(corr.shape[1]))
    ax.set_xticklabels(labels, rotation=70, ha="right")
    ax.set_yticks(np.arange(corr.shape[0]))
    ax.set_yticklabels(labels)

    ax.set_title("Redundancy structure among soil variables")

    # Write strong correlations values only
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            v = corr.values[i, j]
            if i != j and np.isfinite(v) and abs(v) >= 0.90:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="black")

    plt.tight_layout()
    save_fig(fig, "fig4_6_soil_redundancy_heatmap")

    corr.to_csv(OUT_DIR / "fig4_6_soil_redundancy_heatmap_data.csv")


# ============================================================
# FIGURE 4.7 — PREPROCESSING DECISION FLOW
# ============================================================

def add_box(ax, xy, text, color="#f7f7f7", width=2.8, height=0.82, fontsize=10):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.08,rounding_size=0.08",
        facecolor=color,
        edgecolor="#444444",
        linewidth=1.0
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize
    )
    return box


def add_arrow(ax, start, end):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.2,
        color="#444444"
    )
    ax.add_patch(arrow)


def fig4_7_preprocessing_decision_flow():
    fig, ax = plt.subplots(figsize=(14, 7.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Column 1: raw blocks
    add_box(ax, (0.4, 5.8), "Initial aligned data\n105 samples", "#deebf7", width=2.7)
    add_box(ax, (0.4, 4.2), "X matrix\n3066 features", "#deebf7", width=2.7)
    add_box(ax, (0.4, 2.6), "Y matrix\n652 metabolites", "#deebf7", width=2.7)

    # Column 2: EDA diagnostics
    add_box(ax, (4.0, 5.8), "High dimensionality\nfeatures >> samples", "#fff7bc", width=3.2)
    add_box(ax, (4.0, 4.2), "Sparsity in X\n137 features >90% zeros", "#fff7bc", width=3.2)
    add_box(ax, (4.0, 2.6), "Sparsity in Y\n62.6% mean zeros", "#fff7bc", width=3.2)
    add_box(ax, (4.0, 1.0), "Soil redundancy\n18 pairs |rho| ≥ 0.90", "#fff7bc", width=3.2)

    # Column 3: preprocessing decisions
    add_box(ax, (8.0, 5.8), "Feature control\nand model-based selection", "#e5f5e0", width=3.2)
    add_box(ax, (8.0, 4.2), "Low-variance and\nrare feature handling", "#e5f5e0", width=3.2)
    add_box(ax, (8.0, 2.6), "Metabolite filtering\n+ log1p transform", "#e5f5e0", width=3.2)
    add_box(ax, (8.0, 1.0), "Soil consolidation\n+ standardization", "#e5f5e0", width=3.2)

    # Column 4: final dataset
    add_box(ax, (11.8, 4.25), "Final ML dataset\nX: 3059 features\nY: 47 metabolites", "#c7e9c0", width=1.9, height=1.35, fontsize=10)

    # Arrows left to EDA
    add_arrow(ax, (3.1, 6.2), (4.0, 6.2))
    add_arrow(ax, (3.1, 4.6), (4.0, 4.6))
    add_arrow(ax, (3.1, 3.0), (4.0, 3.0))

    # EDA to decisions
    for y in [6.2, 4.6, 3.0, 1.4]:
        add_arrow(ax, (7.2, y), (8.0, y))

    # Decisions to final
    add_arrow(ax, (11.2, 6.2), (11.8, 5.15))
    add_arrow(ax, (11.2, 4.6), (11.8, 4.95))
    add_arrow(ax, (11.2, 3.0), (11.8, 4.75))
    add_arrow(ax, (11.2, 1.4), (11.8, 4.55))

    ax.text(
        7, 7.55,
        "EDA-driven preprocessing rationale for Chapter 4",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()
    save_fig(fig, "fig4_7_preprocessing_decision_flow")


# ============================================================
# SUMMARY FILES
# ============================================================

def write_summary():
    summary = pd.DataFrame([
        {"metric": "n_samples", "value": CONFIRMED["n_samples"]},
        {"metric": "initial_x_features", "value": CONFIRMED["x_initial"]},
        {"metric": "initial_mg_features", "value": CONFIRMED["x_initial_mg"]},
        {"metric": "initial_soil_variables", "value": CONFIRMED["x_initial_soil"]},
        {"metric": "initial_y_metabolites", "value": CONFIRMED["y_initial"]},
        {"metric": "final_x_features", "value": CONFIRMED["x_final"]},
        {"metric": "final_mg_features", "value": CONFIRMED["x_final_mg"]},
        {"metric": "final_soil_variables", "value": CONFIRMED["x_final_soil"]},
        {"metric": "final_y_metabolites", "value": CONFIRMED["y_final"]},
        {"metric": "x_mean_zero_fraction_percent", "value": CONFIRMED["x_mean_zero_fraction"] * 100},
        {"metric": "y_mean_zero_fraction_percent", "value": CONFIRMED["y_mean_zero_fraction"] * 100},
        {"metric": "x_features_zero_fraction_ge_90", "value": CONFIRMED["n_x_features_zero_fraction_ge_0_90"]},
        {"metric": "x_features_variance_lt_1e_4", "value": CONFIRMED["n_x_features_variance_lt_1e_4"]},
        {"metric": "soil_strong_pairs_abs_ge_0_90", "value": CONFIRMED["n_soil_strong_pairs_abs_ge_0_90"]},
    ])
    summary.to_csv(OUT_DIR / "chapter4_figures_summary.csv", index=False)

    readme = """
Phase 44 - Final Chapter 4 Figures
==================================

Generated figures:
- fig4_1_dataset_overview: initial vs final dataset dimensions
- fig4_2_x_sparsity_summary: sparsity in explanatory variables
- fig4_3_x_low_variance_summary: low-variance features
- fig4_4_metabolomics_sparsity_and_log_transform: Y sparsity and log1p effect
- fig4_5_soil_variable_categories: soil variables grouped by category
- fig4_6_soil_redundancy_heatmap: Spearman redundancy among soil variables
- fig4_7_preprocessing_decision_flow: rationale linking EDA to preprocessing decisions

These figures are designed for Chapter 4:
Description des données et préparation du dataset.
""".strip()

    with open(OUT_DIR / "chapter4_figures_readme.txt", "w") as f:
        f.write(readme + "\n")


# ============================================================
# MAIN
# ============================================================

def main():
    print("[PHASE 44 FINAL] Generating Chapter 4 figures...")

    print("[1/7] Dataset overview")
    fig4_1_dataset_overview()

    print("[2/7] X sparsity summary")
    fig4_2_x_sparsity_summary()

    print("[3/7] X low variance summary")
    fig4_3_x_low_variance_summary()

    print("[4/7] Metabolomics sparsity and log transformation")
    fig4_4_metabolomics_sparsity_and_log()

    print("[5/7] Soil variable categories")
    fig4_5_soil_variable_categories()

    print("[6/7] Soil redundancy heatmap")
    fig4_6_soil_redundancy_heatmap()

    print("[7/7] Preprocessing decision flow")
    fig4_7_preprocessing_decision_flow()

    print("[SUMMARY] Writing summary files")
    write_summary()

    print("\n[DONE] Chapter 4 final figures generated.")
    print(f"Output folder: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
