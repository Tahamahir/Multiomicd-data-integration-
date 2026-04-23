from pathlib import Path
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


# ============================================================
# PHASE 5 - INTERPRET TOP METABOLITES
# ------------------------------------------------------------
# Ce script :
# 1) lit les top métabolites du benchmark baseline
# 2) recharge X et Y
# 3) entraîne un modèle par métabolite
# 4) extrait les top features importantes
# 5) résume la contribution soil vs MG
# ============================================================


def compute_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, rmse


def classify_feature(feature_name):
    """
    Classe les features en :
    - soil
    - metagenomics_pca
    """
    if str(feature_name).startswith("meta_PC"):
        return "metagenomics_pca"
    return "soil"


def main():
    repo_root = Path(__file__).resolve().parents[2]

    # ------------------------------------------------------------
    # Réglages
    # ------------------------------------------------------------
    selected_config = "soil_plus_mg"   # options: soil_only, mg_only, soil_plus_mg
    n_top_metabolites = 10

    # ------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------
    benchmark_dir = repo_root / "10_analysis" / "outputs" / "phase4_baseline_three_configs"
    reduction_dir = repo_root / "10_analysis" / "outputs" / "phase3_smart_reduction_imputed"
    output_dir = repo_root / "10_analysis" / "outputs" / "phase5_interpret_top_metabolites"
    output_dir.mkdir(parents=True, exist_ok=True)

    x_path = reduction_dir / "X_final_smart.csv"
    y_path = reduction_dir / "Y_top50_log1p_scaled.csv"
    ids_path = reduction_dir / "sample_ids.csv"
    top_path = benchmark_dir / selected_config / "top10_metabolites_by_r2.csv"

    print("=" * 70)
    print("PHASE 5 - INTERPRET TOP METABOLITES")
    print("=" * 70)
    print(f"Selected config : {selected_config}")
    print(f"X input         : {x_path}")
    print(f"Y input         : {y_path}")
    print(f"IDs input       : {ids_path}")
    print(f"Top metabolites : {top_path}")
    print(f"Output dir      : {output_dir}")
    print()

    for p in [x_path, y_path, ids_path, top_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    X = pd.read_csv(x_path, low_memory=False)
    Y = pd.read_csv(y_path, low_memory=False)
    sample_ids = pd.read_csv(ids_path, low_memory=False)
    top_df = pd.read_csv(top_path, low_memory=False)

    if not (len(X) == len(Y) == len(sample_ids)):
        raise ValueError("X, Y and sample_ids must have the same number of rows.")

    # ------------------------------------------------------------
    # Choisir le bloc X selon la config
    # ------------------------------------------------------------
    meta_cols = [c for c in X.columns if c.startswith("meta_PC")]
    soil_cols = [c for c in X.columns if not c.startswith("meta_PC")]

    if selected_config == "soil_only":
        X_used = X[soil_cols].copy()
    elif selected_config == "mg_only":
        X_used = X[meta_cols].copy()
    elif selected_config == "soil_plus_mg":
        X_used = X.copy()
    else:
        raise ValueError("selected_config must be one of: soil_only, mg_only, soil_plus_mg")

    # ------------------------------------------------------------
    # Choisir top métabolites
    # ------------------------------------------------------------
    top_metabolites = top_df["metabolite"].head(n_top_metabolites).tolist()

    print(f"X_used shape      : {X_used.shape}")
    print(f"Y shape           : {Y.shape}")
    print(f"Top metabolites   : {len(top_metabolites)}")
    print()

    all_feature_importances = []
    all_metrics = []
    all_predictions = []
    block_contrib_rows = []

    # ------------------------------------------------------------
    # Split unique pour cohérence
    # ------------------------------------------------------------
    X_train, X_test, Y_train, Y_test, ids_train, ids_test = train_test_split(
        X_used, Y, sample_ids,
        test_size=0.2,
        random_state=42
    )

    # ------------------------------------------------------------
    # Un modèle par métabolite
    # ------------------------------------------------------------
    for metabolite in top_metabolites:
        print(f"Training single-target model for: {metabolite}")

        y_train = Y_train[metabolite]
        y_test = Y_test[metabolite]

        model = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2, rmse = compute_metrics(y_test, y_pred)

        all_metrics.append({
            "metabolite": metabolite,
            "r2": r2,
            "rmse": rmse
        })

        # predictions
        pred_df = pd.DataFrame({
            "sample_id": ids_test.iloc[:, 0].reset_index(drop=True),
            "metabolite": metabolite,
            "y_true": y_test.reset_index(drop=True),
            "y_pred": y_pred
        })
        all_predictions.append(pred_df)

        # feature importances
        fi_df = pd.DataFrame({
            "metabolite": metabolite,
            "feature": X_used.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        fi_df["feature_type"] = fi_df["feature"].apply(classify_feature)
        all_feature_importances.append(fi_df)

        # résumé block contribution
        soil_importance = fi_df.loc[fi_df["feature_type"] == "soil", "importance"].sum()
        mg_importance = fi_df.loc[fi_df["feature_type"] == "metagenomics_pca", "importance"].sum()

        block_contrib_rows.append({
            "metabolite": metabolite,
            "r2": r2,
            "rmse": rmse,
            "soil_importance_sum": soil_importance,
            "metagenomics_pca_importance_sum": mg_importance
        })

    # ------------------------------------------------------------
    # Concat outputs
    # ------------------------------------------------------------
    metrics_df = pd.DataFrame(all_metrics).sort_values("r2", ascending=False).reset_index(drop=True)
    feature_importance_df = pd.concat(all_feature_importances, ignore_index=True)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    block_contrib_df = pd.DataFrame(block_contrib_rows).sort_values("r2", ascending=False).reset_index(drop=True)

    # top 20 features par métabolite
    top_features_per_metabolite = (
        feature_importance_df
        .groupby("metabolite", group_keys=False)
        .head(20)
        .reset_index(drop=True)
    )

    # top 10 globalement fréquentes parmi les top 20
    top_feature_counts = (
        top_features_per_metabolite.groupby(["feature", "feature_type"])
        .size()
        .reset_index(name="count_in_top20_lists")
        .sort_values(["count_in_top20_lists", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    cfg_out = output_dir / selected_config
    cfg_out.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(cfg_out / "single_target_metrics.csv", index=False)
    predictions_df.to_csv(cfg_out / "single_target_predictions.csv", index=False)
    feature_importance_df.to_csv(cfg_out / "all_feature_importances.csv", index=False)
    top_features_per_metabolite.to_csv(cfg_out / "top20_features_per_metabolite.csv", index=False)
    top_feature_counts.to_csv(cfg_out / "top_feature_counts_across_metabolites.csv", index=False)
    block_contrib_df.to_csv(cfg_out / "block_contribution_summary.csv", index=False)

    summary = {
        "selected_config": selected_config,
        "n_samples": int(len(X_used)),
        "n_features": int(X_used.shape[1]),
        "n_top_metabolites": int(len(top_metabolites)),
        "mean_r2_across_selected_metabolites": float(metrics_df["r2"].mean()),
        "median_r2_across_selected_metabolites": float(metrics_df["r2"].median()),
        "n_positive_r2_selected_metabolites": int((metrics_df["r2"] > 0).sum()),
    }

    with open(cfg_out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    pd.DataFrame([summary]).to_csv(cfg_out / "summary.csv", index=False)

    # ------------------------------------------------------------
    # Console
    # ------------------------------------------------------------
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Selected config                     : {selected_config}")
    print(f"X_used shape                        : {X_used.shape}")
    print(f"Top metabolites analyzed            : {len(top_metabolites)}")
    print(f"Mean R2 across selected metabolites : {metrics_df['r2'].mean():.4f}")
    print(f"Median R2 across selected metabolites : {metrics_df['r2'].median():.4f}")
    print(f"Selected metabolites with R2 > 0    : {(metrics_df['r2'] > 0).sum()}")
    print()
    print("Top metabolites by single-target R2:")
    print(metrics_df.to_string(index=False))
    print()
    print("Top repeated important features across metabolites:")
    print(top_feature_counts.head(20).to_string(index=False))
    print()
    print("Block contribution summary:")
    print(block_contrib_df.to_string(index=False))
    print()
    print("Main outputs:")
    print(cfg_out / "summary.csv")
    print(cfg_out / "single_target_metrics.csv")
    print(cfg_out / "top20_features_per_metabolite.csv")
    print(cfg_out / "top_feature_counts_across_metabolites.csv")
    print(cfg_out / "block_contribution_summary.csv")
    print()
    print("Interpretation step completed successfully.")


if __name__ == "__main__":
    main()