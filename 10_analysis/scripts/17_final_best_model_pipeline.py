from pathlib import Path
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from scipy.stats import spearmanr


# ============================================================
# PHASE 17 - FINAL OPTIMIZED PIPELINE
# ------------------------------------------------------------
# Objectif :
# - comparer RF tuned, XGB tuned, ExtraTrees tuned
# - choisir le meilleur modèle par métabolite
# - entraîner le modèle final par MB
# - prédire les abondances MB
# - générer feature importance
# - reconstruire table MG ↔ MB finale optimisée
# ============================================================


def impute_X(X):
    X = X.copy()
    for col in X.columns:
        if X[col].isna().any():
            med = X[col].median()
            if pd.isna(med):
                med = 0
            X[col] = X[col].fillna(med)
    return X


def detect_soil_columns(columns):
    prefixes = [
        "soil_",
        "chem__",
        "psize_",
        "moist_",
        "nitrif_",
        "denit_",
    ]

    soil_cols = []
    for col in columns:
        c = str(col).lower().strip()
        if any(c.startswith(p) for p in prefixes):
            soil_cols.append(col)

    return soil_cols


def classify_relation(corr, threshold=0.2):
    if pd.isna(corr):
        return "unknown", "uncertain"

    if corr >= threshold:
        return "positive", "putative_production"
    elif corr <= -threshold:
        return "negative", "putative_consumption"
    else:
        return "weak", "uncertain"


def confidence_label(importance, abs_corr):
    if importance >= 0.01 and abs_corr >= 0.30:
        return "high"
    elif importance >= 0.005 and abs_corr >= 0.20:
        return "medium"
    else:
        return "low"


def load_params(path):
    df = pd.read_csv(path, low_memory=False)
    out = {}
    for _, row in df.iterrows():
        out[row["metabolite"]] = json.loads(row["best_params"])
    return out


def clean_xgb_columns(columns):
    cleaned = []
    seen = {}

    for c in columns:
        new_c = (
            str(c)
            .replace("[", "_")
            .replace("]", "_")
            .replace("<", "_")
            .replace(">", "_")
        )

        # éviter doublons après nettoyage
        if new_c not in seen:
            seen[new_c] = 0
            cleaned.append(new_c)
        else:
            seen[new_c] += 1
            cleaned.append(f"{new_c}_{seen[new_c]}")

    return cleaned


def build_model(model_name, params):
    if model_name == "ExtraTrees_tuned":
        return ExtraTreesRegressor(
            random_state=42,
            n_jobs=-1,
            **params
        )

    if model_name == "RandomForest_tuned":
        return RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
            **params
        )

    if model_name == "XGBoost_tuned":
        return XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            **params
        )

    raise ValueError(f"Unknown model name: {model_name}")


def main():
    repo_root = Path(__file__).resolve().parents[2]

    # -----------------------------
    # Inputs
    # -----------------------------
    x_path = repo_root / "10_analysis" / "outputs" / "phase3_soil_dedup" / "X_deduplicated.csv"
    y_path = repo_root / "10_analysis" / "outputs" / "phase2_preprocessing_fixed" / "Y_ml_filtered_log1p.csv"

    et_results_path = repo_root / "10_analysis" / "outputs" / "phase13_tuning_extratrees" / "tuning_results.csv"
    et_params_path = repo_root / "10_analysis" / "outputs" / "phase13_tuning_extratrees" / "best_params_by_metabolite.csv"

    rf_results_path = repo_root / "10_analysis" / "outputs" / "phase15_tuning_randomforest" / "rf_tuning_results.csv"
    rf_params_path = repo_root / "10_analysis" / "outputs" / "phase15_tuning_randomforest" / "rf_best_params_by_metabolite.csv"

    xgb_results_path = repo_root / "10_analysis" / "outputs" / "phase16_tuning_xgboost" / "xgb_tuning_results.csv"
    xgb_params_path = repo_root / "10_analysis" / "outputs" / "phase16_tuning_xgboost" / "xgb_best_params_by_metabolite.csv"

    output_dir = repo_root / "10_analysis" / "outputs" / "phase17_final_best_model_pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 17 - FINAL OPTIMIZED BEST-MODEL PIPELINE")
    print("=" * 70)
    print(f"X input   : {x_path}")
    print(f"Y input   : {y_path}")
    print(f"Output dir: {output_dir}")
    print()

    for p in [
        x_path, y_path,
        et_results_path, et_params_path,
        rf_results_path, rf_params_path,
        xgb_results_path, xgb_params_path,
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    # -----------------------------
    # Load data
    # -----------------------------
    X_raw = pd.read_csv(x_path, low_memory=False)
    Y = pd.read_csv(y_path, low_memory=False)

    if len(X_raw) != len(Y):
        raise ValueError("X and Y must have the same number of rows.")

    print(f"X raw shape : {X_raw.shape}")
    print(f"Y shape     : {Y.shape}")
    print(f"NaN in X before imputation: {int(X_raw.isna().sum().sum())}")

    X_raw = impute_X(X_raw)

    print(f"NaN in X after imputation : {int(X_raw.isna().sum().sum())}")
    print()

    # X version cleaned for XGBoost
    X_xgb = X_raw.copy()
    original_cols = X_xgb.columns.tolist()
    xgb_cols = clean_xgb_columns(original_cols)
    xgb_feature_name_map = pd.DataFrame({
        "original_feature": original_cols,
        "xgb_feature": xgb_cols
    })
    X_xgb.columns = xgb_cols
    xgb_feature_name_map.to_csv(output_dir / "xgb_feature_name_map.csv", index=False)

    # -----------------------------
    # Load tuning results
    # -----------------------------
    et_results = pd.read_csv(et_results_path, low_memory=False)
    rf_results = pd.read_csv(rf_results_path, low_memory=False)
    xgb_results = pd.read_csv(xgb_results_path, low_memory=False)

    et_params = load_params(et_params_path)
    rf_params = load_params(rf_params_path)
    xgb_params = load_params(xgb_params_path)

    # Standardiser colonnes
    et_perf = et_results[["metabolite", "tuned_mean_r2", "tuned_mean_rmse"]].copy()
    et_perf["model"] = "ExtraTrees_tuned"

    rf_perf = rf_results[["metabolite", "tuned_mean_r2", "tuned_mean_rmse"]].copy()
    rf_perf["model"] = "RandomForest_tuned"

    xgb_perf = xgb_results[["metabolite", "tuned_mean_r2", "tuned_mean_rmse"]].copy()
    xgb_perf["model"] = "XGBoost_tuned"

    all_perf = pd.concat([et_perf, rf_perf, xgb_perf], ignore_index=True)

    # Best model per metabolite
    best_models = (
        all_perf
        .sort_values(["metabolite", "tuned_mean_r2"], ascending=[True, False])
        .groupby("metabolite", as_index=False)
        .first()
        .sort_values("tuned_mean_r2", ascending=False)
        .reset_index(drop=True)
    )

    best_models.to_csv(output_dir / "best_model_per_metabolite_final.csv", index=False)
    all_perf.to_csv(output_dir / "all_model_performance_combined.csv", index=False)

    metabolites = [m for m in best_models["metabolite"].tolist() if m in Y.columns]

    print(f"Metabolites in final optimized pipeline: {len(metabolites)}")
    print()

    # -----------------------------
    # Préparer soil / MG pour associations
    # -----------------------------
    soil_cols = detect_soil_columns(X_raw.columns.tolist())
    mg_cols = [c for c in X_raw.columns if c not in soil_cols]

    print(f"Soil columns detected: {len(soil_cols)}")
    print(f"MG columns detected  : {len(mg_cols)}")
    print()

    # -----------------------------
    # Containers
    # -----------------------------
    pred_log = pd.DataFrame(index=X_raw.index)
    pred_original = pd.DataFrame(index=X_raw.index)

    feature_importance_rows = []
    relationship_rows = []
    model_summary_rows = []

    # -----------------------------
    # Final fit per metabolite
    # -----------------------------
    for idx, metabolite in enumerate(metabolites, start=1):
        row = best_models[best_models["metabolite"] == metabolite].iloc[0]
        model_name = row["model"]
        best_r2 = float(row["tuned_mean_r2"])
        best_rmse = float(row["tuned_mean_rmse"])

        print(f"[{idx}/{len(metabolites)}] {metabolite}")
        print(f"  best model: {model_name} | CV R2={best_r2:.4f}")

        y = Y[metabolite].values

        if model_name == "ExtraTrees_tuned":
            params = et_params[metabolite]
            X_model = X_raw
            feature_names_model = X_raw.columns.tolist()

        elif model_name == "RandomForest_tuned":
            params = rf_params[metabolite]
            X_model = X_raw
            feature_names_model = X_raw.columns.tolist()

        elif model_name == "XGBoost_tuned":
            params = xgb_params[metabolite]
            X_model = X_xgb
            feature_names_model = X_xgb.columns.tolist()

        else:
            raise ValueError(f"Unknown model: {model_name}")

        model = build_model(model_name, params)
        model.fit(X_model, y)

        y_pred_log = model.predict(X_model)
        y_pred_original = np.expm1(y_pred_log)

        pred_log[metabolite] = y_pred_log
        pred_original[metabolite] = y_pred_original

        # Feature importance
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            importances = np.zeros(len(feature_names_model))

        fi = pd.DataFrame({
            "model_feature": feature_names_model,
            "importance": importances
        })

        # Remapper features XGBoost vers noms originaux
        if model_name == "XGBoost_tuned":
            reverse_map = dict(zip(xgb_feature_name_map["xgb_feature"], xgb_feature_name_map["original_feature"]))
            fi["feature"] = fi["model_feature"].map(reverse_map)
        else:
            fi["feature"] = fi["model_feature"]

        fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)

        for rank, fi_row in fi.head(50).iterrows():
            feature = fi_row["feature"]
            importance = float(fi_row["importance"])
            feature_type = "soil" if feature in soil_cols else "mg"

            feature_importance_rows.append({
                "metabolite": metabolite,
                "best_model": model_name,
                "cv_r2": best_r2,
                "cv_rmse": best_rmse,
                "feature": feature,
                "feature_type": feature_type,
                "importance_rank": int(rank + 1),
                "importance": importance,
            })

        # MG ↔ MB relationships: top 25 MG features
        fi_mg = fi[fi["feature"].isin(mg_cols)].head(25)

        for _, rel_row in fi_mg.iterrows():
            feature = rel_row["feature"]
            importance = float(rel_row["importance"])

            corr, pval = spearmanr(X_raw[feature].values, y)
            abs_corr = abs(corr) if not pd.isna(corr) else np.nan

            direction, role = classify_relation(corr)
            confidence = confidence_label(
                importance,
                abs_corr if not pd.isna(abs_corr) else 0
            )

            relationship_rows.append({
                "metabolite": metabolite,
                "best_model": model_name,
                "cv_r2": best_r2,
                "cv_rmse": best_rmse,
                "mg_feature": feature,
                "importance": importance,
                "spearman_corr": corr,
                "spearman_pvalue": pval,
                "abs_spearman_corr": abs_corr,
                "direction": direction,
                "putative_role": role,
                "confidence": confidence,
            })

        model_summary_rows.append({
            "metabolite": metabolite,
            "best_model": model_name,
            "cv_r2": best_r2,
            "cv_rmse": best_rmse,
            "n_features": int(X_model.shape[1]),
            "top10_importance_sum": float(fi.head(10)["importance"].sum()),
            "top25_importance_sum": float(fi.head(25)["importance"].sum()),
        })

    # -----------------------------
    # Save predictions
    # -----------------------------
    pred_log.to_csv(output_dir / "predicted_mb_log1p_optimized.csv", index=False)
    pred_original.to_csv(output_dir / "predicted_mb_original_scale_optimized.csv", index=False)

    pred_full = pd.concat(
        [
            pred_log.add_suffix("_log1p"),
            pred_original.add_suffix("_original")
        ],
        axis=1
    )
    pred_full.to_csv(output_dir / "predicted_mb_full_optimized.csv", index=False)

    # -----------------------------
    # Save feature importances
    # -----------------------------
    feature_importances = pd.DataFrame(feature_importance_rows)
    feature_importances.to_csv(output_dir / "feature_importances_top50_optimized.csv", index=False)

    # -----------------------------
    # Save relationships
    # -----------------------------
    relationships = pd.DataFrame(relationship_rows)

    relationships_all_path = output_dir / "species_mb_relationships_all_optimized.csv"
    relationships.to_csv(relationships_all_path, index=False)

    interpretable = relationships[
        (relationships["putative_role"].isin(["putative_production", "putative_consumption"])) &
        (relationships["confidence"].isin(["medium", "high"]))
    ].copy()

    interpretable.to_csv(output_dir / "species_mb_relationships_interpretable_optimized.csv", index=False)

    role_summary = (
        relationships["putative_role"]
        .value_counts()
        .rename_axis("putative_role")
        .reset_index(name="n_relationships")
    )
    role_summary.to_csv(output_dir / "relationship_role_summary_optimized.csv", index=False)

    confidence_summary = (
        relationships["confidence"]
        .value_counts()
        .rename_axis("confidence")
        .reset_index(name="n_relationships")
    )
    confidence_summary.to_csv(output_dir / "relationship_confidence_summary_optimized.csv", index=False)

    top_mg_features = (
        relationships.groupby("mg_feature")
        .agg(
            n_metabolites=("metabolite", "nunique"),
            mean_importance=("importance", "mean"),
            max_importance=("importance", "max"),
            mean_abs_corr=("abs_spearman_corr", "mean")
        )
        .reset_index()
        .sort_values(["n_metabolites", "max_importance"], ascending=[False, False])
    )
    top_mg_features.to_csv(output_dir / "top_mg_features_across_metabolites_optimized.csv", index=False)

    # -----------------------------
    # Save model summary
    # -----------------------------
    model_summary = pd.DataFrame(model_summary_rows)
    model_summary.to_csv(output_dir / "final_model_summary_by_metabolite.csv", index=False)

    best_model_counts = (
        best_models["model"]
        .value_counts()
        .rename_axis("model")
        .reset_index(name="n_best_metabolites")
    )
    best_model_counts.to_csv(output_dir / "final_best_model_counts.csv", index=False)

    global_summary = {
        "n_metabolites": int(len(metabolites)),
        "mean_final_cv_r2": float(best_models["tuned_mean_r2"].mean()),
        "median_final_cv_r2": float(best_models["tuned_mean_r2"].median()),
        "max_final_cv_r2": float(best_models["tuned_mean_r2"].max()),
        "mean_final_cv_rmse": float(best_models["tuned_mean_rmse"].mean()),
        "n_r2_gt_0_2": int((best_models["tuned_mean_r2"] > 0.2).sum()),
        "n_r2_gt_0_4": int((best_models["tuned_mean_r2"] > 0.4).sum()),
        "n_r2_gt_0_6": int((best_models["tuned_mean_r2"] > 0.6).sum()),
        "n_relationships_all": int(len(relationships)),
        "n_relationships_interpretable": int(len(interpretable)),
    }

    pd.DataFrame([global_summary]).to_csv(output_dir / "final_pipeline_summary.csv", index=False)

    with open(output_dir / "final_pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)

    # -----------------------------
    # Console summary
    # -----------------------------
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Final metabolites predicted        : {global_summary['n_metabolites']}")
    print(f"Mean final CV R2                   : {global_summary['mean_final_cv_r2']:.4f}")
    print(f"Median final CV R2                 : {global_summary['median_final_cv_r2']:.4f}")
    print(f"Max final CV R2                    : {global_summary['max_final_cv_r2']:.4f}")
    print(f"Mean final CV RMSE                 : {global_summary['mean_final_cv_rmse']:.4f}")
    print(f"R2 > 0.2                           : {global_summary['n_r2_gt_0_2']}")
    print(f"R2 > 0.4                           : {global_summary['n_r2_gt_0_4']}")
    print(f"R2 > 0.6                           : {global_summary['n_r2_gt_0_6']}")
    print(f"Total MG-MB relationships          : {global_summary['n_relationships_all']}")
    print(f"Interpretable MG-MB relationships  : {global_summary['n_relationships_interpretable']}")
    print()
    print("Best model counts:")
    print(best_model_counts.to_string(index=False))
    print()
    print("Main outputs:")
    print(output_dir / "best_model_per_metabolite_final.csv")
    print(output_dir / "predicted_mb_original_scale_optimized.csv")
    print(output_dir / "species_mb_relationships_interpretable_optimized.csv")
    print(output_dir / "final_pipeline_summary.csv")
    print()
    print("Final optimized pipeline completed successfully.")


if __name__ == "__main__":
    main()