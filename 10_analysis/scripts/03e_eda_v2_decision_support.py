from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression


# ============================================================
# PHASE 2 EDA V2 - DECISION SUPPORT (SAFE VERSION)
# ------------------------------------------------------------
# Objectif :
# 1) identifier les variables X peu informatives
# 2) caractériser la zero-inflation de Y
# 3) explorer relations non linéaires soil <-> metabolites
# 4) mesurer la redondance des variables sol
# 5) produire une shortlist de métabolites prioritaires
# ------------------------------------------------------------
# Version sécurisée :
# - détection soil plus stricte
# - calculs lourds limités
# - sauvegardes progressives
# ============================================================


def detect_soil_columns(columns):
    """
    Détection PLUS STRICTE des variables sol / chimie du sol.
    On évite les mots-clés trop courts comme 'ca' ou 'p_' qui attrapent
    des colonnes métagénomiques par erreur.
    """
    soil_cols = []

    explicit_prefixes = [
        "soil_",
        "chem__",
        "psize_",
        "moist_",
        "nitrif_",
        "denit_",
    ]

    explicit_contains = [
        "dry weight",
        "mehlich",
        "caco3",
        "mass dry soil",
        "mass water",
        "soil clay",
        "soil silt",
        "soil sand",
        "soil content",
        "field sample id",   # à garder pour repérage, même si ensuite non numérique
    ]

    for col in columns:
        c = col.lower().strip()

        if any(c.startswith(pref) for pref in explicit_prefixes):
            soil_cols.append(col)
            continue

        if any(token in c for token in explicit_contains):
            soil_cols.append(col)
            continue

    # retirer doublons éventuels en conservant l'ordre
    seen = set()
    final_cols = []
    for col in soil_cols:
        if col not in seen:
            final_cols.append(col)
            seen.add(col)

    return final_cols


def save_histogram(values, title, xlabel, output_path, bins=50, max_points=50000):
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


def save_heatmap(df, title, output_path):
    if df.empty:
        return

    fig_w = max(8, df.shape[1] * 0.45)
    fig_h = max(6, df.shape[0] * 0.45)

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(df.values, aspect="auto")
    plt.colorbar(im)
    plt.xticks(range(len(df.columns)), df.columns, rotation=90)
    plt.yticks(range(len(df.index)), df.index)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    repo_root = Path(__file__).resolve().parents[2]

    input_dir = repo_root / "10_analysis" / "outputs" / "phase2_preprocessing_fixed"
    output_dir = repo_root / "10_analysis" / "outputs" / "phase2_eda_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    x_path = input_dir / "X_ml_filtered.csv"
    y_raw_path = input_dir / "Y_ml_filtered_untransformed.csv"
    y_log_path = input_dir / "Y_ml_filtered_log1p.csv"
    ids_path = input_dir / "sample_ids.csv"

    print("=" * 70)
    print("PHASE 2 EDA V2 - DECISION SUPPORT (SAFE VERSION)")
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
    # 1. Séparer soil / non-soil
    # ------------------------------------------------------------
    soil_cols = detect_soil_columns(X.columns.tolist())
    X_soil = X[soil_cols].copy() if len(soil_cols) > 0 else pd.DataFrame(index=X.index)
    X_other = X.drop(columns=soil_cols).copy()

    print(f"Soil columns detected        : {len(soil_cols)}")
    print(f"Other X columns              : {X_other.shape[1]}")
    print()

    pd.DataFrame({"soil_column": soil_cols}).to_csv(output_dir / "soil_columns_detected.csv", index=False)

    # ------------------------------------------------------------
    # 2. Diagnostic low variance X
    # ------------------------------------------------------------
    print("Step 2/9 - Computing X variance diagnostics...")
    x_variance = X.var(axis=0).sort_values()
    x_low_variance = x_variance.reset_index()
    x_low_variance.columns = ["feature", "variance"]
    x_low_variance.to_csv(output_dir / "x_variance_all.csv", index=False)
    x_low_variance.head(200).to_csv(output_dir / "x_low_variance_top200.csv", index=False)

    low_variance_thresholds = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2]
    low_variance_summary = []
    for thr in low_variance_thresholds:
        low_variance_summary.append({
            "threshold": thr,
            "n_features_below_threshold": int((x_variance < thr).sum())
        })
    pd.DataFrame(low_variance_summary).to_csv(output_dir / "x_low_variance_summary.csv", index=False)

    save_histogram(
        x_variance.values,
        "Distribution of X variance",
        "Variance",
        output_dir / "x_variance_distribution_v2.png",
        bins=50
    )

    # ------------------------------------------------------------
    # 3. Diagnostic high sparsity X
    # ------------------------------------------------------------
    print("Step 3/9 - Computing X sparsity diagnostics...")
    x_zero_fraction = (X == 0).mean(axis=0).sort_values(ascending=False)
    x_zero_df = x_zero_fraction.reset_index()
    x_zero_df.columns = ["feature", "zero_fraction"]
    x_zero_df.to_csv(output_dir / "x_zero_fraction_all.csv", index=False)
    x_zero_df.head(200).to_csv(output_dir / "x_high_zero_fraction_top200.csv", index=False)

    x_sparsity_thresholds = [0.50, 0.70, 0.80, 0.90, 0.95]
    x_sparsity_summary = []
    for thr in x_sparsity_thresholds:
        x_sparsity_summary.append({
            "threshold": thr,
            "n_features_above_threshold": int((x_zero_fraction >= thr).sum())
        })
    pd.DataFrame(x_sparsity_summary).to_csv(output_dir / "x_high_zero_fraction_summary.csv", index=False)

    save_histogram(
        x_zero_fraction.values,
        "Distribution of X zero fraction",
        "Zero fraction",
        output_dir / "x_zero_fraction_distribution_v2.png",
        bins=50
    )

    # ------------------------------------------------------------
    # 4. Zero-inflation Y
    # ------------------------------------------------------------
    print("Step 4/9 - Computing Y zero-inflation diagnostics...")
    y_zero_fraction = (Y_raw == 0).mean(axis=0).sort_values(ascending=False)
    y_zero_df = y_zero_fraction.reset_index()
    y_zero_df.columns = ["metabolite", "zero_fraction"]
    y_zero_df.to_csv(output_dir / "y_zero_fraction_all.csv", index=False)
    y_zero_df.head(200).to_csv(output_dir / "y_high_zero_fraction_top200.csv", index=False)

    y_sparsity_thresholds = [0.50, 0.70, 0.80, 0.90, 0.95]
    y_sparsity_summary = []
    for thr in y_sparsity_thresholds:
        y_sparsity_summary.append({
            "threshold": thr,
            "n_metabolites_above_threshold": int((y_zero_fraction >= thr).sum())
        })
    pd.DataFrame(y_sparsity_summary).to_csv(output_dir / "y_zero_fraction_summary.csv", index=False)

    save_histogram(
        y_zero_fraction.values,
        "Distribution of Y zero fraction",
        "Zero fraction",
        output_dir / "y_zero_fraction_distribution_v2.png",
        bins=50
    )

    # ------------------------------------------------------------
    # 5. Distribution des non-zéros uniquement dans Y
    # ------------------------------------------------------------
    print("Step 5/9 - Studying non-zero Y values...")
    y_nonzero_values_raw = Y_raw.values[Y_raw.values > 0]
    y_nonzero_values_log = Y_log.values[Y_log.values > 0]

    pd.DataFrame({"y_raw_nonzero_values": y_nonzero_values_raw}).to_csv(
        output_dir / "y_nonzero_values_raw.csv", index=False
    )
    pd.DataFrame({"y_log_nonzero_values": y_nonzero_values_log}).to_csv(
        output_dir / "y_nonzero_values_log.csv", index=False
    )

    save_histogram(
        y_nonzero_values_raw,
        "Distribution of Y non-zero values (raw)",
        "Y raw non-zero values",
        output_dir / "y_nonzero_distribution_raw.png",
        bins=60
    )

    save_histogram(
        y_nonzero_values_log,
        "Distribution of Y non-zero values (log1p)",
        "Y log1p non-zero values",
        output_dir / "y_nonzero_distribution_log.png",
        bins=60
    )

    # ------------------------------------------------------------
    # 6. Tests de seuils de présence sur Y
    # ------------------------------------------------------------
    print("Step 6/9 - Testing Y presence thresholds...")
    y_presence_threshold_candidates = [0, 1e-9, 1e-6, 1e-3]
    threshold_rows = []

    for thr in y_presence_threshold_candidates:
        presence_fraction = (Y_raw > thr).mean(axis=0)
        threshold_rows.append({
            "threshold": thr,
            "mean_presence_fraction": float(presence_fraction.mean()),
            "median_presence_fraction": float(presence_fraction.median()),
            "n_metabolites_present_in_>=10pct_samples": int((presence_fraction >= 0.10).sum()),
            "n_metabolites_present_in_>=20pct_samples": int((presence_fraction >= 0.20).sum()),
        })

    pd.DataFrame(threshold_rows).to_csv(output_dir / "y_presence_threshold_sensitivity.csv", index=False)

    # ------------------------------------------------------------
    # 7. Redondance des variables sol
    # ------------------------------------------------------------
    print("Step 7/9 - Studying soil redundancy...")
    strong_corr_pairs = pd.DataFrame(columns=["var1", "var2", "correlation", "abs_correlation"])

    if not X_soil.empty:
        # Limiter le calcul aux variables sol les plus variables pour éviter explosion mémoire
        soil_var = X_soil.var(axis=0).sort_values(ascending=False)
        top_soil_for_corr = soil_var.head(min(80, len(soil_var))).index.tolist()

        X_soil_corr_subset = X_soil[top_soil_for_corr].copy()
        soil_corr = X_soil_corr_subset.corr(method="spearman")
        soil_corr.to_csv(output_dir / "soil_spearman_correlation_matrix.csv")

        save_heatmap(
            soil_corr,
            "Soil variables Spearman correlation matrix (top variable subset)",
            output_dir / "soil_spearman_correlation_heatmap.png"
        )

        pairs = []
        cols = soil_corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c = soil_corr.iloc[i, j]
                if pd.notna(c):
                    pairs.append({
                        "var1": cols[i],
                        "var2": cols[j],
                        "correlation": c,
                        "abs_correlation": abs(c)
                    })

        strong_corr_pairs = pd.DataFrame(pairs).sort_values("abs_correlation", ascending=False)
        strong_corr_pairs.to_csv(output_dir / "soil_all_variable_pairs_spearman.csv", index=False)
        strong_corr_pairs[strong_corr_pairs["abs_correlation"] >= 0.90].to_csv(
            output_dir / "soil_strong_pairs_abs_ge_0_90.csv", index=False
        )
        strong_corr_pairs[strong_corr_pairs["abs_correlation"] >= 0.80].to_csv(
            output_dir / "soil_strong_pairs_abs_ge_0_80.csv", index=False
        )

    # ------------------------------------------------------------
    # 8. Top métabolites par variance et présence
    # ------------------------------------------------------------
    print("Step 8/9 - Ranking metabolites...")
    y_variance_log = Y_log.var(axis=0).sort_values(ascending=False)
    y_presence = (Y_raw > 0).mean(axis=0)

    metabolite_priority = pd.DataFrame({
        "metabolite": Y_log.columns,
        "variance_log": y_variance_log.reindex(Y_log.columns).values,
        "presence_fraction": y_presence.reindex(Y_log.columns).values,
        "zero_fraction": (1 - y_presence.reindex(Y_log.columns)).values,
    })

    metabolite_priority["priority_score"] = (
        metabolite_priority["variance_log"].rank(pct=True) * 0.6 +
        metabolite_priority["presence_fraction"].rank(pct=True) * 0.4
    )

    metabolite_priority = metabolite_priority.sort_values("priority_score", ascending=False)
    metabolite_priority.to_csv(output_dir / "metabolite_priority_ranking.csv", index=False)
    metabolite_priority.head(50).to_csv(output_dir / "metabolite_priority_top50.csv", index=False)

    # ------------------------------------------------------------
    # 9. Relations soil <-> metabolites
    # ------------------------------------------------------------
    print("Step 9/9 - Computing soil <-> metabolite associations...")
    spearman_long = pd.DataFrame()
    mi_long = pd.DataFrame()

    if not X_soil.empty:
        top_soil_cols = X_soil.var(axis=0).sort_values(ascending=False).head(min(25, X_soil.shape[1])).index.tolist()
        top_metabolites = metabolite_priority.head(min(20, len(metabolite_priority)))["metabolite"].tolist()

        # -------- Spearman --------
        soil_vs_metab_spearman = pd.DataFrame(index=top_soil_cols, columns=top_metabolites, dtype=float)

        for s in top_soil_cols:
            for m in top_metabolites:
                soil_vs_metab_spearman.loc[s, m] = X_soil[s].corr(Y_log[m], method="spearman")

        soil_vs_metab_spearman.to_csv(output_dir / "soil_vs_top_metabolites_spearman_matrix.csv")
        save_heatmap(
            soil_vs_metab_spearman,
            "Soil vs top metabolites (Spearman)",
            output_dir / "soil_vs_top_metabolites_spearman_heatmap.png"
        )

        spearman_long = (
            soil_vs_metab_spearman.stack()
            .reset_index()
            .rename(columns={"level_0": "soil_variable", "level_1": "metabolite", 0: "spearman"})
        )
        spearman_long["abs_spearman"] = spearman_long["spearman"].abs()
        spearman_long = spearman_long.sort_values("abs_spearman", ascending=False)
        spearman_long.to_csv(output_dir / "soil_vs_top_metabolites_spearman_long.csv", index=False)
        spearman_long.head(50).to_csv(output_dir / "soil_vs_top_metabolites_spearman_top50.csv", index=False)

        # -------- Mutual Information --------
        # On limite encore plus pour éviter les calculs trop lourds
        mi_soil_cols = X_soil.var(axis=0).sort_values(ascending=False).head(min(20, X_soil.shape[1])).index.tolist()
        mi_metabolites = metabolite_priority.head(min(10, len(metabolite_priority)))["metabolite"].tolist()

        X_soil_mi = X_soil[mi_soil_cols].copy().fillna(X_soil[mi_soil_cols].median())

        mi_rows = []
        for m in mi_metabolites:
            mi = mutual_info_regression(
                X_soil_mi,
                Y_log[m],
                random_state=42
            )
            for soil_var, mi_value in zip(X_soil_mi.columns, mi):
                mi_rows.append({
                    "soil_variable": soil_var,
                    "metabolite": m,
                    "mutual_information": mi_value
                })

        mi_long = pd.DataFrame(mi_rows).sort_values("mutual_information", ascending=False)
        mi_long.to_csv(output_dir / "soil_vs_top_metabolites_mutual_information_long.csv", index=False)
        mi_long.head(100).to_csv(output_dir / "soil_vs_top_metabolites_mutual_information_top100.csv", index=False)

        mi_pivot = mi_long.pivot(index="soil_variable", columns="metabolite", values="mutual_information")
        save_heatmap(
            mi_pivot,
            "Soil vs top metabolites (Mutual Information)",
            output_dir / "soil_vs_top_metabolites_mutual_information_heatmap.png"
        )

    # ------------------------------------------------------------
    # Résumé final
    # ------------------------------------------------------------
    summary = {
        "n_samples": int(len(sample_ids)),
        "x_n_columns": int(X.shape[1]),
        "x_soil_n_columns": int(X_soil.shape[1]),
        "x_other_n_columns": int(X_other.shape[1]),
        "y_n_columns": int(Y_raw.shape[1]),
        "x_mean_zero_fraction": float((X == 0).mean(axis=0).mean()),
        "y_mean_zero_fraction": float((Y_raw == 0).mean(axis=0).mean()),
        "n_x_features_zero_fraction_ge_0_90": int((x_zero_fraction >= 0.90).sum()),
        "n_y_metabolites_zero_fraction_ge_0_90": int((y_zero_fraction >= 0.90).sum()),
        "n_x_features_variance_lt_1e_4": int((x_variance < 1e-4).sum()),
        "n_soil_strong_pairs_abs_ge_0_90": int((strong_corr_pairs["abs_correlation"] >= 0.90).sum()) if not strong_corr_pairs.empty else 0,
        "n_spearman_pairs_evaluated": int(len(spearman_long)) if not spearman_long.empty else 0,
        "n_mutual_info_pairs_evaluated": int(len(mi_long)) if not mi_long.empty else 0,
    }

    pd.DataFrame([summary]).to_csv(output_dir / "eda_v2_summary.csv", index=False)

    # ------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Number of samples                       : {len(sample_ids)}")
    print(f"X shape                                 : {X.shape}")
    print(f"Y shape                                 : {Y_raw.shape}")
    print(f"Soil columns detected                   : {len(soil_cols)}")
    print(f"Mean zero fraction in X                 : {summary['x_mean_zero_fraction']:.4f}")
    print(f"Mean zero fraction in Y                 : {summary['y_mean_zero_fraction']:.4f}")
    print(f"X features with zero fraction >= 0.90   : {summary['n_x_features_zero_fraction_ge_0_90']}")
    print(f"Y metabolites with zero fraction >=0.90 : {summary['n_y_metabolites_zero_fraction_ge_0_90']}")
    print(f"X features variance < 1e-4              : {summary['n_x_features_variance_lt_1e_4']}")
    print(f"Soil strong pairs |corr|>=0.90          : {summary['n_soil_strong_pairs_abs_ge_0_90']}")
    print(f"Spearman pairs evaluated                : {summary['n_spearman_pairs_evaluated']}")
    print(f"Mutual information pairs evaluated      : {summary['n_mutual_info_pairs_evaluated']}")
    print()
    print("Main outputs:")
    print(output_dir / "eda_v2_summary.csv")
    print(output_dir / "x_low_variance_summary.csv")
    print(output_dir / "x_high_zero_fraction_summary.csv")
    print(output_dir / "y_zero_fraction_summary.csv")
    print(output_dir / "y_presence_threshold_sensitivity.csv")
    print(output_dir / "soil_strong_pairs_abs_ge_0_90.csv")
    print(output_dir / "soil_vs_top_metabolites_spearman_top50.csv")
    print(output_dir / "soil_vs_top_metabolites_mutual_information_top100.csv")
    print(output_dir / "metabolite_priority_top50.csv")
    print()
    print("EDA V2 completed successfully.")


if __name__ == "__main__":
    main()