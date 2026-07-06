from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from xgboost import XGBRegressor


# ============================================================
# PHASE 19 - SHAP INTERPRETABILITY (SAFE VERSION)
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
    prefixes = ["soil_", "chem__", "psize_", "moist_", "nitrif_", "denit_"]
    return [col for col in columns if any(str(col).lower().strip().startswith(p) for p in prefixes)]


def clean_filename(name, max_len=80):
    bad_chars = ["|", ":", "/", "\\", " ", "(", ")", "[", "]", "<", ">", ","]
    out = str(name)
    for ch in bad_chars:
        out = out.replace(ch, "_")
    return out[:max_len]


def clean_xgb_columns(columns):
    cleaned = []
    seen = {}
    for c in columns:
        new_c = str(c).replace("[", "_").replace("]", "_").replace("<", "_").replace(">", "_")
        if new_c not in seen:
            seen[new_c] = 0
            cleaned.append(new_c)
        else:
            seen[new_c] += 1
            cleaned.append(f"{new_c}_{seen[new_c]}")
    return cleaned


def load_params(path):
    df = pd.read_csv(path, low_memory=False)
    out = {}
    for _, row in df.iterrows():
        out[row["metabolite"]] = json.loads(row["best_params"])
    return out


def build_model(model_name, params):
    if model_name == "ExtraTrees_tuned":
        return ExtraTreesRegressor(random_state=42, n_jobs=-1, **params)

    if model_name == "RandomForest_tuned":
        return RandomForestRegressor(random_state=42, n_jobs=-1, **params)

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

    x_path = repo_root / "10_analysis" / "outputs" / "phase3_soil_dedup" / "X_deduplicated.csv"
    y_path = repo_root / "10_analysis" / "outputs" / "phase2_preprocessing_fixed" / "Y_ml_filtered_log1p.csv"

    best_models_path = (
        repo_root / "10_analysis" / "outputs"
        / "phase17_final_best_model_pipeline"
        / "best_model_per_metabolite_final.csv"
    )

    et_params_path = (
        repo_root / "10_analysis" / "outputs"
        / "phase13_tuning_extratrees"
        / "best_params_by_metabolite.csv"
    )

    rf_params_path = (
        repo_root / "10_analysis" / "outputs"
        / "phase15_tuning_randomforest"
        / "rf_best_params_by_metabolite.csv"
    )

    xgb_params_path = (
        repo_root / "10_analysis" / "outputs"
        / "phase16_tuning_xgboost"
        / "xgb_best_params_by_metabolite.csv"
    )

    output_dir = repo_root / "10_analysis" / "outputs" / "phase19_shap_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 19 - SHAP INTERPRETABILITY SAFE VERSION")
    print("=" * 70)

    for p in [x_path, y_path, best_models_path, et_params_path, rf_params_path, xgb_params_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    X_raw = pd.read_csv(x_path, low_memory=False)
    Y = pd.read_csv(y_path, low_memory=False)
    best_models = pd.read_csv(best_models_path, low_memory=False)

    if len(X_raw) != len(Y):
        raise ValueError("X and Y must have the same number of rows.")

    print(f"X shape before imputation : {X_raw.shape}")
    print(f"Y shape                   : {Y.shape}")
    print(f"NaN in X before           : {int(X_raw.isna().sum().sum())}")

    X_raw = impute_X(X_raw)

    print(f"NaN in X after            : {int(X_raw.isna().sum().sum())}")

    soil_cols = detect_soil_columns(X_raw.columns.tolist())
    mg_cols = [c for c in X_raw.columns if c not in soil_cols]

    print(f"Soil columns detected     : {len(soil_cols)}")
    print(f"MG columns detected       : {len(mg_cols)}")
    print()

    X_xgb = X_raw.copy()
    original_cols = X_xgb.columns.tolist()
    xgb_cols = clean_xgb_columns(original_cols)
    X_xgb.columns = xgb_cols
    xgb_reverse_map = dict(zip(xgb_cols, original_cols))

    et_params = load_params(et_params_path)
    rf_params = load_params(rf_params_path)
    xgb_params = load_params(xgb_params_path)

    # Top 10 meilleurs métabolites
    n_top_metabolites = 10
    selected = best_models.sort_values("tuned_mean_r2", ascending=False).head(n_top_metabolites).copy()
    metabolites = [m for m in selected["metabolite"].tolist() if m in Y.columns]

    print(f"Selected top metabolites for interpretation: {len(metabolites)}")
    print()

    all_importance_rows = []
    block_summary_rows = []

    max_display_features = 25

    for idx, metabolite in enumerate(metabolites, start=1):
        model_row = selected[selected["metabolite"] == metabolite].iloc[0]

        model_name = model_row["model"]
        cv_r2 = float(model_row["tuned_mean_r2"])
        cv_rmse = float(model_row["tuned_mean_rmse"])

        print(f"[{idx}/{len(metabolites)}] Interpretation for {metabolite}")
        print(f"  model : {model_name}")
        print(f"  R2    : {cv_r2:.4f}")

        y = Y[metabolite].values

        if model_name == "ExtraTrees_tuned":
            params = et_params[metabolite]
            X_model = X_raw
            feature_names_model = X_raw.columns.tolist()
            reverse_map = None

        elif model_name == "RandomForest_tuned":
            params = rf_params[metabolite]
            X_model = X_raw
            feature_names_model = X_raw.columns.tolist()
            reverse_map = None

        elif model_name == "XGBoost_tuned":
            params = xgb_params[metabolite]
            X_model = X_xgb
            feature_names_model = X_xgb.columns.tolist()
            reverse_map = xgb_reverse_map

        else:
            raise ValueError(f"Unknown model: {model_name}")

        model = build_model(model_name, params)
        model.fit(X_model, y)

        # --------------------------------------------------------
        # SHAP ou fallback importance
        # --------------------------------------------------------
        shap_values = None

        try:
            if model_name == "XGBoost_tuned":
                raise RuntimeError("SHAP skipped for XGBoost due to SHAP/XGBoost compatibility issue")

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_model)

            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            shap_values = np.array(shap_values)
            mean_abs_importance = np.abs(shap_values).mean(axis=0)
            interpretation_mode = "true_shap"

        except Exception as e:
            print(f"  WARNING: true SHAP failed or skipped for {model_name}")
            print(f"  Using feature_importances_ fallback")
            print(f"  Error: {e}")

            if hasattr(model, "feature_importances_"):
                mean_abs_importance = model.feature_importances_
            else:
                mean_abs_importance = np.zeros(X_model.shape[1])

            interpretation_mode = "fallback_feature_importance"

        importance_df = pd.DataFrame({
            "model_feature": feature_names_model,
            "mean_abs_importance": mean_abs_importance
        })

        if reverse_map is not None:
            importance_df["feature"] = importance_df["model_feature"].map(reverse_map)
        else:
            importance_df["feature"] = importance_df["model_feature"]

        importance_df["feature_type"] = importance_df["feature"].apply(
            lambda f: "soil" if f in soil_cols else "mg"
        )

        importance_df["metabolite"] = metabolite
        importance_df["best_model"] = model_name
        importance_df["cv_r2"] = cv_r2
        importance_df["cv_rmse"] = cv_rmse
        importance_df["interpretation_mode"] = interpretation_mode

        importance_df = importance_df.sort_values("mean_abs_importance", ascending=False).reset_index(drop=True)
        importance_df["rank"] = np.arange(1, len(importance_df) + 1)

        all_importance_rows.append(importance_df)

        total_imp = importance_df["mean_abs_importance"].sum()
        soil_imp = importance_df.loc[importance_df["feature_type"] == "soil", "mean_abs_importance"].sum()
        mg_imp = importance_df.loc[importance_df["feature_type"] == "mg", "mean_abs_importance"].sum()

        block_summary_rows.append({
            "metabolite": metabolite,
            "best_model": model_name,
            "cv_r2": cv_r2,
            "cv_rmse": cv_rmse,
            "interpretation_mode": interpretation_mode,
            "total_importance": float(total_imp),
            "soil_importance_sum": float(soil_imp),
            "mg_importance_sum": float(mg_imp),
            "soil_importance_fraction": float(soil_imp / total_imp) if total_imp > 0 else np.nan,
            "mg_importance_fraction": float(mg_imp / total_imp) if total_imp > 0 else np.nan,
        })

        safe_name = clean_filename(metabolite)

        # Barplot
        top_plot = importance_df.head(max_display_features).copy()

        plt.figure(figsize=(10, 7))
        plt.barh(
            top_plot["feature"].apply(lambda x: str(x)[:50]),
            top_plot["mean_abs_importance"]
        )
        plt.gca().invert_yaxis()
        plt.xlabel("Mean |SHAP| / Feature importance")
        plt.title(f"Top influential features\n{metabolite[:70]}\nMode: {interpretation_mode}")
        plt.tight_layout()
        plt.savefig(figures_dir / f"importance_barplot_{safe_name}.png", dpi=250)
        plt.close()

        # Beeswarm seulement si vrai SHAP
        if shap_values is not None:
            plt.figure()
            shap.summary_plot(
                shap_values,
                X_model,
                feature_names=feature_names_model,
                max_display=20,
                show=False
            )
            plt.title(f"SHAP beeswarm: {metabolite[:60]}")
            plt.tight_layout()
            plt.savefig(
                figures_dir / f"shap_beeswarm_{safe_name}.png",
                dpi=250,
                bbox_inches="tight"
            )
            plt.close()

        print(f"  done with mode: {interpretation_mode}")
        print()

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------
    all_importance = pd.concat(all_importance_rows, ignore_index=True)
    block_summary = pd.DataFrame(block_summary_rows)

    all_importance.to_csv(output_dir / "feature_interpretability_all.csv", index=False)

    top30 = all_importance[all_importance["rank"] <= 30].copy()
    top30.to_csv(output_dir / "top30_features_per_metabolite.csv", index=False)

    block_summary.to_csv(output_dir / "interpretability_block_summary.csv", index=False)

    recurrent = (
        top30.groupby(["feature", "feature_type"])
        .agg(
            n_metabolites=("metabolite", "nunique"),
            mean_importance=("mean_abs_importance", "mean"),
            max_importance=("mean_abs_importance", "max"),
            mean_cv_r2=("cv_r2", "mean"),
        )
        .reset_index()
        .sort_values(["n_metabolites", "max_importance"], ascending=[False, False])
    )

    recurrent.to_csv(output_dir / "recurrent_top_features.csv", index=False)

    mode_counts = (
        block_summary["interpretation_mode"]
        .value_counts()
        .rename_axis("interpretation_mode")
        .reset_index(name="n_metabolites")
    )
    mode_counts.to_csv(output_dir / "interpretation_mode_counts.csv", index=False)

    summary = {
        "n_metabolites_analyzed": int(len(metabolites)),
        "n_total_features": int(X_raw.shape[1]),
        "n_soil_features": int(len(soil_cols)),
        "n_mg_features": int(len(mg_cols)),
        "mean_cv_r2_analyzed": float(block_summary["cv_r2"].mean()),
        "mean_soil_importance_fraction": float(block_summary["soil_importance_fraction"].mean()),
        "mean_mg_importance_fraction": float(block_summary["mg_importance_fraction"].mean()),
    }

    pd.DataFrame([summary]).to_csv(output_dir / "interpretability_summary.csv", index=False)

    with open(output_dir / "interpretability_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Metabolites analyzed        : {summary['n_metabolites_analyzed']}")
    print(f"Total features              : {summary['n_total_features']}")
    print(f"Soil features               : {summary['n_soil_features']}")
    print(f"MG features                 : {summary['n_mg_features']}")
    print(f"Mean CV R2 analyzed         : {summary['mean_cv_r2_analyzed']:.4f}")
    print(f"Mean soil importance frac   : {summary['mean_soil_importance_fraction']:.4f}")
    print(f"Mean MG importance frac     : {summary['mean_mg_importance_fraction']:.4f}")
    print()
    print("Interpretation mode counts:")
    print(mode_counts.to_string(index=False))
    print()
    print("Block summary:")
    print(block_summary.to_string(index=False))
    print()
    print("Top recurrent features:")
    print(recurrent.head(20).to_string(index=False))
    print()
    print("Main outputs:")
    print(output_dir / "feature_interpretability_all.csv")
    print(output_dir / "top30_features_per_metabolite.csv")
    print(output_dir / "interpretability_block_summary.csv")
    print(output_dir / "recurrent_top_features.csv")
    print(figures_dir)
    print()
    print("SHAP / interpretability analysis completed successfully.")


if __name__ == "__main__":
    main()