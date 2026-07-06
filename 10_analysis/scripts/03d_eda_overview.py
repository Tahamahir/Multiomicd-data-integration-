from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PHASE 2 EDA - OVERVIEW
# ------------------------------------------------------------
# Ce script :
# 1) charge X_ml_filtered, Y_ml_filtered_untransformed, Y_ml_filtered_log1p
# 2) fait une exploration visuelle simple et utile
# 3) sauvegarde figures + tableaux de corrélation
# ============================================================


def detect_soil_columns(columns):
    """
    Détecte les colonnes liées au sol / chimie du sol.
    """
    soil_cols = []
    keywords = [
        "soil",
        "chem__",
        "dry weight",
        "ph",
        "no3",
        "no2",
        "nh4",
        "moisture",
        "organic",
        "carbon",
        "nitrogen",
        "sand",
        "silt",
        "clay",
    ]

    for col in columns:
        c = col.lower()
        if any(k in c for k in keywords):
            soil_cols.append(col)

    return soil_cols


def save_histogram(values, title, xlabel, output_path, bins=50, max_points=50000):
    """
    Histogramme simple. On sous-échantillonne si trop de points.
    """
    values = pd.Series(values).dropna()
    if len(values) == 0:
        return

    if len(values) > max_points:
        values = values.sample(max_points, random_state=42)

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_boxplot(df, title, output_path, max_cols=20):
    """
    Boxplot pour un nombre raisonnable de colonnes.
    """
    if df.empty:
        return

    df_plot = df.copy()
    if df_plot.shape[1] > max_cols:
        df_plot = df_plot.iloc[:, :max_cols]

    plt.figure(figsize=(max(10, df_plot.shape[1] * 0.6), 6))
    plt.boxplot([df_plot[c].dropna().values for c in df_plot.columns], tick_labels=df_plot.columns)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_heatmap(corr_df, title, output_path):
    """
    Heatmap simple avec matplotlib.
    """
    if corr_df.empty:
        return

    fig_w = max(8, corr_df.shape[1] * 0.5)
    fig_h = max(6, corr_df.shape[0] * 0.5)

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(corr_df.values, aspect="auto")
    plt.colorbar(im)
    plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=90)
    plt.yticks(range(len(corr_df.index)), corr_df.index)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_scatter(x, y, xlabel, ylabel, title, output_path):
    x = pd.Series(x)
    y = pd.Series(y)
    mask = ~(x.isna() | y.isna())
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return

    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s=15)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    repo_root = Path(__file__).resolve().parents[2]

    input_dir = repo_root / "10_analysis" / "outputs" / "phase2_preprocessing_fixed"
    output_dir = repo_root / "10_analysis" / "outputs" / "phase2_eda"
    output_dir.mkdir(parents=True, exist_ok=True)

    x_path = input_dir / "X_ml_filtered.csv"
    y_raw_path = input_dir / "Y_ml_filtered_untransformed.csv"
    y_log_path = input_dir / "Y_ml_filtered_log1p.csv"
    ids_path = input_dir / "sample_ids.csv"

    print("=" * 70)
    print("PHASE 2 EDA - OVERVIEW")
    print("=" * 70)
    print(f"X input   : {x_path}")
    print(f"Y raw     : {y_raw_path}")
    print(f"Y log1p   : {y_log_path}")
    print(f"IDs input : {ids_path}")
    print(f"Output dir: {output_dir}")
    print()

    for p in [x_path, y_raw_path, y_log_path, ids_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    X = pd.read_csv(x_path, low_memory=False)
    Y_raw = pd.read_csv(y_raw_path, low_memory=False)
    Y_log = pd.read_csv(y_log_path, low_memory=False)
    sample_ids = pd.read_csv(ids_path, low_memory=False)

    if not (len(X) == len(Y_raw) == len(Y_log) == len(sample_ids)):
        raise ValueError("X, Y_raw, Y_log and sample_ids must have the same number of rows.")

    print(f"X shape      : {X.shape}")
    print(f"Y raw shape  : {Y_raw.shape}")
    print(f"Y log shape  : {Y_log.shape}")
    print()

    # ------------------------------------------------------------
    # 1. Détection variables sol
    # ------------------------------------------------------------
    soil_cols = detect_soil_columns(X.columns.tolist())
    X_soil = X[soil_cols].copy() if len(soil_cols) > 0 else pd.DataFrame(index=X.index)
    X_meta = X.drop(columns=soil_cols).copy()

    print(f"Soil columns detected        : {len(soil_cols)}")
    print(f"Metagenomics columns         : {X_meta.shape[1]}")
    print()

    pd.DataFrame({"soil_column": soil_cols}).to_csv(output_dir / "soil_columns_detected.csv", index=False)

    # ------------------------------------------------------------
    # 2. Résumés numériques simples
    # ------------------------------------------------------------
    summary = {
        "n_samples": int(len(sample_ids)),
        "x_n_columns": int(X.shape[1]),
        "x_soil_n_columns": int(X_soil.shape[1]),
        "x_metagenomics_n_columns": int(X_meta.shape[1]),
        "y_raw_n_columns": int(Y_raw.shape[1]),
        "y_log_n_columns": int(Y_log.shape[1]),
        "x_global_zero_fraction": float((X == 0).sum().sum() / (X.shape[0] * X.shape[1])),
        "y_raw_global_zero_fraction": float((Y_raw == 0).sum().sum() / (Y_raw.shape[0] * Y_raw.shape[1])),
        "y_log_global_zero_fraction": float((Y_log == 0).sum().sum() / (Y_log.shape[0] * Y_log.shape[1])),
    }

    pd.DataFrame([summary]).to_csv(output_dir / "eda_summary.csv", index=False)

    # ------------------------------------------------------------
    # 3. Distributions globales
    # ------------------------------------------------------------
    save_histogram(
        X.to_numpy().ravel(),
        "Global distribution of X values",
        "X values",
        output_dir / "x_global_distribution.png"
    )

    save_histogram(
        Y_raw.to_numpy().ravel(),
        "Global distribution of Y values (before log1p)",
        "Y raw values",
        output_dir / "y_raw_global_distribution.png"
    )

    save_histogram(
        Y_log.to_numpy().ravel(),
        "Global distribution of Y values (after log1p)",
        "Y log1p values",
        output_dir / "y_log_global_distribution.png"
    )

    # ------------------------------------------------------------
    # 4. Sparsité par colonne
    # ------------------------------------------------------------
    x_zero_fraction = (X == 0).mean(axis=0).sort_values(ascending=False)
    y_zero_fraction = (Y_raw == 0).mean(axis=0).sort_values(ascending=False)

    x_zero_fraction.to_csv(output_dir / "x_zero_fraction_by_column.csv", header=["zero_fraction"])
    y_zero_fraction.to_csv(output_dir / "y_zero_fraction_by_column.csv", header=["zero_fraction"])

    save_histogram(
        x_zero_fraction.values,
        "Distribution of zero fraction across X columns",
        "Zero fraction per X column",
        output_dir / "x_zero_fraction_distribution.png",
        bins=40
    )

    save_histogram(
        y_zero_fraction.values,
        "Distribution of zero fraction across Y columns",
        "Zero fraction per Y column",
        output_dir / "y_zero_fraction_distribution.png",
        bins=40
    )

    # ------------------------------------------------------------
    # 5. Variance par colonne
    # ------------------------------------------------------------
    x_variance = X.var(axis=0).sort_values(ascending=False)
    y_variance_raw = Y_raw.var(axis=0).sort_values(ascending=False)
    y_variance_log = Y_log.var(axis=0).sort_values(ascending=False)

    x_variance.to_csv(output_dir / "x_variance_by_column.csv", header=["variance"])
    y_variance_raw.to_csv(output_dir / "y_raw_variance_by_column.csv", header=["variance"])
    y_variance_log.to_csv(output_dir / "y_log_variance_by_column.csv", header=["variance"])

    save_histogram(
        x_variance.values,
        "Distribution of X variances",
        "Variance",
        output_dir / "x_variance_distribution.png",
        bins=40
    )

    save_histogram(
        y_variance_log.values,
        "Distribution of Y variances after log1p",
        "Variance",
        output_dir / "y_log_variance_distribution.png",
        bins=40
    )

    # ------------------------------------------------------------
    # 6. Variables de sol : describe + boxplots + heatmap corr
    # ------------------------------------------------------------
    if not X_soil.empty:
        X_soil.describe().T.to_csv(output_dir / "soil_describe.csv")

        # garder top 20 variables de sol les plus variables pour boxplot
        soil_var = X_soil.var(axis=0).sort_values(ascending=False)
        top_soil_cols = soil_var.head(min(20, len(soil_var))).index.tolist()
        X_soil[top_soil_cols].to_csv(output_dir / "soil_top20_by_variance.csv", index=False)

        save_boxplot(
            X_soil[top_soil_cols],
            "Top soil variables by variance",
            output_dir / "soil_top20_boxplot.png",
            max_cols=20
        )

        # corrélation entre variables de sol
        soil_corr = X_soil.corr()
        soil_corr.to_csv(output_dir / "soil_correlation_matrix.csv")
        save_heatmap(
            soil_corr,
            "Correlation matrix of soil variables",
            output_dir / "soil_correlation_heatmap.png"
        )

    # ------------------------------------------------------------
    # 7. Top métabolites : describe + heatmap corr
    # ------------------------------------------------------------
    y_top_var_cols = y_variance_log.head(min(20, len(y_variance_log))).index.tolist()
    Y_top20 = Y_log[y_top_var_cols].copy()
    Y_top20.describe().T.to_csv(output_dir / "y_top20_describe.csv")

    y_top20_corr = Y_top20.corr()
    y_top20_corr.to_csv(output_dir / "y_top20_correlation_matrix.csv")
    save_heatmap(
        y_top20_corr,
        "Correlation matrix of top 20 metabolites (after log1p)",
        output_dir / "y_top20_correlation_heatmap.png"
    )

    # ------------------------------------------------------------
    # 8. Relations X (soil) ↔ Y (top metabolites)
    # ------------------------------------------------------------
    if not X_soil.empty:
        # top 15 soil vars par variance
        soil_var = X_soil.var(axis=0).sort_values(ascending=False)
        top_soil_cols = soil_var.head(min(15, len(soil_var))).index.tolist()

        # top 20 métabolites par variance
        top_y_cols = y_variance_log.head(min(20, len(y_variance_log))).index.tolist()

        soil_y_corr = pd.DataFrame(index=top_soil_cols, columns=top_y_cols, dtype=float)

        for s in top_soil_cols:
            for y in top_y_cols:
                soil_y_corr.loc[s, y] = X_soil[s].corr(Y_log[y])

        soil_y_corr.to_csv(output_dir / "soil_vs_top20_metabolites_correlation.csv")
        save_heatmap(
            soil_y_corr,
            "Soil variables vs top 20 metabolites correlations",
            output_dir / "soil_vs_top20_metabolites_heatmap.png"
        )

        # top associations absolues
        corr_long = (
            soil_y_corr.stack()
            .reset_index()
            .rename(columns={"level_0": "soil_variable", "level_1": "metabolite", 0: "correlation"})
        )
        corr_long["abs_correlation"] = corr_long["correlation"].abs()
        corr_long = corr_long.sort_values("abs_correlation", ascending=False)
        corr_long.to_csv(output_dir / "soil_vs_top20_metabolites_correlation_long.csv", index=False)

        top_pairs = corr_long.head(min(6, len(corr_long))).copy()
        top_pairs.to_csv(output_dir / "top_soil_metabolite_pairs.csv", index=False)

        # scatter plots pour les 6 meilleures associations
        for i, row in top_pairs.iterrows():
            soil_var_name = row["soil_variable"]
            metab_name = row["metabolite"]

            safe_name = f"{soil_var_name[:40]}__{metab_name[:40]}"
            safe_name = safe_name.replace("/", "_").replace("\\", "_").replace(":", "_").replace("|", "_")

            save_scatter(
                X_soil[soil_var_name],
                Y_log[metab_name],
                soil_var_name,
                metab_name,
                f"{soil_var_name} vs {metab_name}",
                output_dir / f"scatter_{safe_name}.png"
            )

    # ------------------------------------------------------------
    # 9. Sauvegarder un aperçu des colonnes les plus informatives
    # ------------------------------------------------------------
    pd.DataFrame({
        "column": x_variance.head(50).index,
        "variance": x_variance.head(50).values
    }).to_csv(output_dir / "x_top50_by_variance.csv", index=False)

    pd.DataFrame({
        "column": y_variance_log.head(50).index,
        "variance": y_variance_log.head(50).values
    }).to_csv(output_dir / "y_top50_by_variance_after_log.csv", index=False)

    # ------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Number of samples                  : {len(sample_ids)}")
    print(f"X shape                            : {X.shape}")
    print(f"Y raw shape                        : {Y_raw.shape}")
    print(f"Y log shape                        : {Y_log.shape}")
    print(f"Soil columns detected              : {len(soil_cols)}")
    print(f"Metagenomics columns detected      : {X_meta.shape[1]}")
    print(f"X global zero fraction             : {summary['x_global_zero_fraction']:.4f}")
    print(f"Y raw global zero fraction         : {summary['y_raw_global_zero_fraction']:.4f}")
    print(f"Y log global zero fraction         : {summary['y_log_global_zero_fraction']:.4f}")
    print()
    print("Main outputs:")
    print(output_dir / "eda_summary.csv")
    print(output_dir / "x_global_distribution.png")
    print(output_dir / "y_raw_global_distribution.png")
    print(output_dir / "y_log_global_distribution.png")
    print(output_dir / "soil_correlation_heatmap.png")
    print(output_dir / "y_top20_correlation_heatmap.png")
    print(output_dir / "soil_vs_top20_metabolites_heatmap.png")
    print()
    print("EDA completed successfully.")


if __name__ == "__main__":
    main()