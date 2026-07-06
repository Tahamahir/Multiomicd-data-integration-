from pathlib import Path
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor


def impute_X(X):
    X = X.copy()
    for col in X.columns:
        if X[col].isna().any():
            med = X[col].median()
            if pd.isna(med):
                med = 0
            X[col] = X[col].fillna(med)
    return X


def load_best_extratrees_params(best_params_path):
    """
    Charge les meilleurs paramètres ExtraTrees obtenus dans phase13.
    """
    params_df = pd.read_csv(best_params_path)

    params_dict = {}

    for _, row in params_df.iterrows():
        metabolite = row["metabolite"]
        params = json.loads(row["best_params"])
        params_dict[metabolite] = params

    return params_dict


def evaluate_model(model, X, y, cv):
    """
    Retourne mean/std R2 et mean RMSE en cross-validation.
    """
    r2_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="r2",
        n_jobs=-1
    )

    rmse_scores = -cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    return {
        "mean_r2": float(np.mean(r2_scores)),
        "std_r2": float(np.std(r2_scores)),
        "mean_rmse": float(np.mean(rmse_scores)),
        "std_rmse": float(np.std(rmse_scores)),
    }


def main():
    repo_root = Path(__file__).resolve().parents[2]

    x_path = repo_root / "10_analysis" / "outputs" / "phase3_soil_dedup" / "X_deduplicated.csv"
    y_path = repo_root / "10_analysis" / "outputs" / "phase2_preprocessing_fixed" / "Y_ml_filtered_log1p.csv"
    source_path = repo_root / "10_analysis" / "outputs" / "phase7_source_comparison" / "source_comparison.csv"
    best_params_path = repo_root / "10_analysis" / "outputs" / "phase13_tuning_extratrees" / "best_params_by_metabolite.csv"

    output_dir = repo_root / "10_analysis" / "outputs" / "phase14_model_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 14 - BENCHMARK OTHER MODELS")
    print("=" * 70)
    print(f"X input          : {x_path}")
    print(f"Y input          : {y_path}")
    print(f"Source input     : {source_path}")
    print(f"Best params input: {best_params_path}")
    print(f"Output dir       : {output_dir}")
    print()

    for p in [x_path, y_path, source_path, best_params_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    X = pd.read_csv(x_path, low_memory=False)
    Y = pd.read_csv(y_path, low_memory=False)
    source_df = pd.read_csv(source_path, low_memory=False)

    print(f"X shape before imputation : {X.shape}")
    print(f"Y shape                   : {Y.shape}")
    print(f"NaN in X before           : {int(X.isna().sum().sum())}")

    X = impute_X(X)

    print(f"NaN in X after            : {int(X.isna().sum().sum())}")
    print()

    best_et_params = load_best_extratrees_params(best_params_path)

    selected = source_df[source_df["r2_fusion"] > 0.20].copy()
    metabolites = [m for m in selected["metabolite"].tolist() if m in Y.columns]

    print(f"Selected metabolites for benchmark: {len(metabolites)}")
    print()

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    all_results = []

    # Optionnel : tester si xgboost est installé
    xgb_available = False
    try:
        from xgboost import XGBRegressor
        xgb_available = True
        print("XGBoost detected: YES")
    except Exception:
        print("XGBoost detected: NO")
    print()

    for idx, metabolite in enumerate(metabolites, start=1):
        print(f"[{idx}/{len(metabolites)}] Benchmarking {metabolite}")

        y = Y[metabolite].values

        # -------------------------------
        # 1. ExtraTrees tuned
        # -------------------------------
        et_params = best_et_params.get(metabolite, {
            "n_estimators": 500,
            "max_features": "sqrt",
            "min_samples_leaf": 2,
            "max_depth": None,
            "bootstrap": False,
        })

        models = {
            "ExtraTrees_tuned": ExtraTreesRegressor(
                random_state=42,
                n_jobs=-1,
                **et_params
            ),

            "RandomForest_default": RandomForestRegressor(
                n_estimators=500,
                random_state=42,
                n_jobs=-1,
                max_features="sqrt",
                min_samples_leaf=2
            ),

            "RandomForest_regularized": RandomForestRegressor(
                n_estimators=700,
                random_state=42,
                n_jobs=-1,
                max_features=0.3,
                min_samples_leaf=3,
                max_depth=20
            ),

            "HistGradientBoosting": HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=0.1,
                random_state=42
            )
        }

        if xgb_available:
            models["XGBoost"] = XGBRegressor(
                n_estimators=400,
                learning_rate=0.03,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1
            )

        for model_name, model in models.items():
            metrics = evaluate_model(model, X, y, cv)

            all_results.append({
                "metabolite": metabolite,
                "model": model_name,
                "mean_r2": metrics["mean_r2"],
                "std_r2": metrics["std_r2"],
                "mean_rmse": metrics["mean_rmse"],
                "std_rmse": metrics["std_rmse"],
            })

            print(
                f"  {model_name:25s} "
                f"R2={metrics['mean_r2']:.4f} "
                f"RMSE={metrics['mean_rmse']:.4f}"
            )

        print()

    results = pd.DataFrame(all_results)

    # Best model per metabolite
    best_model_per_metabolite = (
        results.sort_values(["metabolite", "mean_r2"], ascending=[True, False])
        .groupby("metabolite", as_index=False)
        .first()
        .sort_values("mean_r2", ascending=False)
        .reset_index(drop=True)
    )

    # Model summary
    model_summary = (
        results.groupby("model")
        .agg(
            n_metabolites=("metabolite", "nunique"),
            mean_r2=("mean_r2", "mean"),
            median_r2=("mean_r2", "median"),
            max_r2=("mean_r2", "max"),
            mean_rmse=("mean_rmse", "mean"),
            n_r2_gt_0_2=("mean_r2", lambda x: int((x > 0.2).sum())),
            n_r2_gt_0_4=("mean_r2", lambda x: int((x > 0.4).sum())),
            n_r2_gt_0_6=("mean_r2", lambda x: int((x > 0.6).sum())),
        )
        .reset_index()
        .sort_values("mean_r2", ascending=False)
    )

    best_counts = (
        best_model_per_metabolite["model"]
        .value_counts()
        .rename_axis("model")
        .reset_index(name="n_best_metabolites")
    )

    # Save
    results.to_csv(output_dir / "model_benchmark_all_results.csv", index=False)
    best_model_per_metabolite.to_csv(output_dir / "best_model_per_metabolite.csv", index=False)
    model_summary.to_csv(output_dir / "model_summary.csv", index=False)
    best_counts.to_csv(output_dir / "best_model_counts.csv", index=False)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("Model summary:")
    print(model_summary.to_string(index=False))
    print()

    print("Best model counts:")
    print(best_counts.to_string(index=False))
    print()

    print("Top 20 best model per metabolite:")
    print(best_model_per_metabolite.head(20).to_string(index=False))
    print()

    print("Main outputs:")
    print(output_dir / "model_benchmark_all_results.csv")
    print(output_dir / "best_model_per_metabolite.csv")
    print(output_dir / "model_summary.csv")
    print(output_dir / "best_model_counts.csv")
    print()
    print("Model benchmark completed successfully.")


if __name__ == "__main__":
    main()