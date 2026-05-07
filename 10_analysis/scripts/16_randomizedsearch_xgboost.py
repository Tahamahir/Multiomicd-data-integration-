from pathlib import Path
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from xgboost import XGBRegressor


# ============================================================
# PHASE 16 - RANDOMIZED SEARCH XGBOOST
# ------------------------------------------------------------
# Objectif :
# - tuner XGBoost sur les métabolites prédictibles
# - comparer ensuite avec ExtraTrees tuned et RandomForest tuned
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


def main():
    repo_root = Path(__file__).resolve().parents[2]

    x_path = repo_root / "10_analysis" / "outputs" / "phase3_soil_dedup" / "X_deduplicated.csv"
    y_path = repo_root / "10_analysis" / "outputs" / "phase2_preprocessing_fixed" / "Y_ml_filtered_log1p.csv"
    source_path = repo_root / "10_analysis" / "outputs" / "phase7_source_comparison" / "source_comparison.csv"

    output_dir = repo_root / "10_analysis" / "outputs" / "phase16_tuning_xgboost"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 16 - RANDOMIZED SEARCH XGBOOST")
    print("=" * 70)
    print(f"X input      : {x_path}")
    print(f"Y input      : {y_path}")
    print(f"Source input : {source_path}")
    print(f"Output dir   : {output_dir}")
    print()

    for p in [x_path, y_path, source_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    X = pd.read_csv(x_path, low_memory=False)
    Y = pd.read_csv(y_path, low_memory=False)
    source_df = pd.read_csv(source_path, low_memory=False)

    if len(X) != len(Y):
        raise ValueError("X and Y must have the same number of rows.")

    print(f"X shape before imputation : {X.shape}")
    print(f"Y shape                   : {Y.shape}")
    print(f"NaN in X before           : {int(X.isna().sum().sum())}")

    X = impute_X(X)
    X.columns = [
        str(c)
        .replace("[", "_")
        .replace("]", "_")
        .replace("<", "_")
        .replace(">", "_")
        for c in X.columns
    ]

    print(f"NaN in X after            : {int(X.isna().sum().sum())}")
    print()

    selected = source_df[source_df["r2_fusion"] > 0.20].copy()
    metabolites = [m for m in selected["metabolite"].tolist() if m in Y.columns]

    print(f"Selected metabolites for XGBoost tuning: {len(metabolites)}")
    print()

    if len(metabolites) == 0:
        raise ValueError("No metabolites selected. Check r2_fusion threshold.")

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    baseline_model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    param_dist = {
        "n_estimators": [200, 300, 500, 700],
        "learning_rate": [0.01, 0.03, 0.05, 0.08],
        "max_depth": [2, 3, 4, 5],
        "min_child_weight": [1, 3, 5, 10],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.5, 0.7, 0.8, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
        "reg_alpha": [0.0, 0.01, 0.1, 0.5, 1.0],
        "gamma": [0.0, 0.01, 0.1, 0.5],
    }

    all_rows = []
    best_params_rows = []

    for idx, metabolite in enumerate(metabolites, start=1):
        print(f"[{idx}/{len(metabolites)}] Tuning XGBoost for {metabolite}")

        y = Y[metabolite].values

        baseline_r2_scores = cross_val_score(
            baseline_model,
            X,
            y,
            cv=cv,
            scoring="r2",
            n_jobs=-1
        )

        baseline_rmse_scores = -cross_val_score(
            baseline_model,
            X,
            y,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )

        baseline_mean_r2 = float(np.mean(baseline_r2_scores))
        baseline_std_r2 = float(np.std(baseline_r2_scores))
        baseline_mean_rmse = float(np.mean(baseline_rmse_scores))

        search = RandomizedSearchCV(
            estimator=XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1
            ),
            param_distributions=param_dist,
            n_iter=30,
            scoring="r2",
            cv=cv,
            random_state=42,
            n_jobs=-1,
            verbose=0,
            refit=True
        )

        search.fit(X, y)

        tuned_mean_r2 = float(search.best_score_)
        tuned_params = search.best_params_

        tuned_model = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            **tuned_params
        )

        tuned_rmse_scores = -cross_val_score(
            tuned_model,
            X,
            y,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )

        tuned_mean_rmse = float(np.mean(tuned_rmse_scores))
        improvement = tuned_mean_r2 - baseline_mean_r2

        all_rows.append({
            "metabolite": metabolite,
            "baseline_mean_r2": baseline_mean_r2,
            "baseline_std_r2": baseline_std_r2,
            "baseline_mean_rmse": baseline_mean_rmse,
            "tuned_mean_r2": tuned_mean_r2,
            "tuned_mean_rmse": tuned_mean_rmse,
            "r2_improvement": improvement,
            "best_n_estimators": tuned_params.get("n_estimators"),
            "best_learning_rate": tuned_params.get("learning_rate"),
            "best_max_depth": tuned_params.get("max_depth"),
            "best_min_child_weight": tuned_params.get("min_child_weight"),
            "best_subsample": tuned_params.get("subsample"),
            "best_colsample_bytree": tuned_params.get("colsample_bytree"),
            "best_reg_lambda": tuned_params.get("reg_lambda"),
            "best_reg_alpha": tuned_params.get("reg_alpha"),
            "best_gamma": tuned_params.get("gamma"),
        })

        best_params_rows.append({
            "metabolite": metabolite,
            "best_params": json.dumps(tuned_params)
        })

        print(f"  baseline XGB R2 : {baseline_mean_r2:.4f}")
        print(f"  tuned XGB R2    : {tuned_mean_r2:.4f}")
        print(f"  improvement     : {improvement:+.4f}")
        print()

    results = pd.DataFrame(all_rows).sort_values("tuned_mean_r2", ascending=False)
    best_params_df = pd.DataFrame(best_params_rows)

    results.to_csv(output_dir / "xgb_tuning_results.csv", index=False)
    best_params_df.to_csv(output_dir / "xgb_best_params_by_metabolite.csv", index=False)

    summary = {
        "n_metabolites_tuned": int(len(results)),
        "mean_baseline_xgb_r2": float(results["baseline_mean_r2"].mean()),
        "mean_tuned_xgb_r2": float(results["tuned_mean_r2"].mean()),
        "median_baseline_xgb_r2": float(results["baseline_mean_r2"].median()),
        "median_tuned_xgb_r2": float(results["tuned_mean_r2"].median()),
        "mean_r2_improvement": float(results["r2_improvement"].mean()),
        "n_improved_metabolites": int((results["r2_improvement"] > 0).sum()),
        "n_worse_metabolites": int((results["r2_improvement"] < 0).sum()),
        "n_tuned_r2_gt_0_2": int((results["tuned_mean_r2"] > 0.2).sum()),
        "n_tuned_r2_gt_0_4": int((results["tuned_mean_r2"] > 0.4).sum()),
        "n_tuned_r2_gt_0_6": int((results["tuned_mean_r2"] > 0.6).sum()),
    }

    pd.DataFrame([summary]).to_csv(output_dir / "xgb_tuning_summary.csv", index=False)

    with open(output_dir / "xgb_tuning_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Metabolites tuned        : {summary['n_metabolites_tuned']}")
    print(f"Mean baseline XGB R2     : {summary['mean_baseline_xgb_r2']:.4f}")
    print(f"Mean tuned XGB R2        : {summary['mean_tuned_xgb_r2']:.4f}")
    print(f"Mean R2 improvement      : {summary['mean_r2_improvement']:+.4f}")
    print(f"Improved metabolites     : {summary['n_improved_metabolites']}")
    print(f"Worse metabolites        : {summary['n_worse_metabolites']}")
    print(f"Tuned XGB R2 > 0.2       : {summary['n_tuned_r2_gt_0_2']}")
    print(f"Tuned XGB R2 > 0.4       : {summary['n_tuned_r2_gt_0_4']}")
    print(f"Tuned XGB R2 > 0.6       : {summary['n_tuned_r2_gt_0_6']}")
    print()
    print("Top 20 tuned XGBoost metabolites:")
    print(results.head(20).to_string(index=False))
    print()
    print("Main outputs:")
    print(output_dir / "xgb_tuning_results.csv")
    print(output_dir / "xgb_best_params_by_metabolite.csv")
    print(output_dir / "xgb_tuning_summary.csv")
    print()
    print("XGBoost tuning completed successfully.")


if __name__ == "__main__":
    main()