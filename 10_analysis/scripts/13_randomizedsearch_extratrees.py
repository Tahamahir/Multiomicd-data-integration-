from pathlib import Path
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, make_scorer


def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


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

    output_dir = repo_root / "10_analysis" / "outputs" / "phase13_tuning_extratrees"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 13 - RANDOMIZED SEARCH EXTRA TREES")
    print("=" * 70)
    print(f"X input      : {x_path}")
    print(f"Y input      : {y_path}")
    print(f"Source input : {source_path}")
    print(f"Output dir   : {output_dir}")
    print()

    X = pd.read_csv(x_path, low_memory=False)
    Y = pd.read_csv(y_path, low_memory=False)
    source_df = pd.read_csv(source_path, low_memory=False)

    if len(X) != len(Y):
        raise ValueError("X and Y must have the same number of rows.")

    print(f"X shape before imputation : {X.shape}")
    print(f"Y shape                   : {Y.shape}")
    print(f"NaN in X before           : {int(X.isna().sum().sum())}")

    X = impute_X(X)

    print(f"NaN in X after            : {int(X.isna().sum().sum())}")
    print()

    # Métabolites déjà prédictibles
    selected = source_df[source_df["r2_fusion"] > 0.20].copy()
    metabolites = [m for m in selected["metabolite"].tolist() if m in Y.columns]

    print(f"Selected metabolites for tuning: {len(metabolites)}")
    print()

    if len(metabolites) == 0:
        raise ValueError("No metabolites selected. Check r2_fusion threshold.")

    # Baseline ExtraTrees utilisée avant
    baseline_model = ExtraTreesRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=2
    )

    # Recherche aléatoire raisonnable
    param_dist = {
        "n_estimators": [300, 500, 800],
        "max_features": ["sqrt", "log2", 0.2, 0.3, 0.5],
        "min_samples_leaf": [1, 2, 3, 5],
        "max_depth": [None, 5, 10, 20, 40],
        "bootstrap": [False, True],
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    all_rows = []
    best_params_rows = []

    for idx, metabolite in enumerate(metabolites, start=1):
        print(f"[{idx}/{len(metabolites)}] Tuning {metabolite}")

        y = Y[metabolite].values

        # Baseline CV
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
            estimator=ExtraTreesRegressor(random_state=42, n_jobs=-1),
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

        # RMSE du meilleur modèle en CV
        tuned_model = ExtraTreesRegressor(
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
            "best_max_features": tuned_params.get("max_features"),
            "best_min_samples_leaf": tuned_params.get("min_samples_leaf"),
            "best_max_depth": tuned_params.get("max_depth"),
            "best_bootstrap": tuned_params.get("bootstrap"),
        })

        best_params_rows.append({
            "metabolite": metabolite,
            "best_params": json.dumps(tuned_params)
        })

        print(f"  baseline R2 : {baseline_mean_r2:.4f}")
        print(f"  tuned R2    : {tuned_mean_r2:.4f}")
        print(f"  improvement : {improvement:+.4f}")
        print()

    results = pd.DataFrame(all_rows).sort_values("tuned_mean_r2", ascending=False)
    best_params_df = pd.DataFrame(best_params_rows)

    results.to_csv(output_dir / "tuning_results.csv", index=False)
    best_params_df.to_csv(output_dir / "best_params_by_metabolite.csv", index=False)

    summary = {
        "n_metabolites_tuned": int(len(results)),
        "mean_baseline_r2": float(results["baseline_mean_r2"].mean()),
        "mean_tuned_r2": float(results["tuned_mean_r2"].mean()),
        "median_baseline_r2": float(results["baseline_mean_r2"].median()),
        "median_tuned_r2": float(results["tuned_mean_r2"].median()),
        "mean_r2_improvement": float(results["r2_improvement"].mean()),
        "n_improved_metabolites": int((results["r2_improvement"] > 0).sum()),
        "n_worse_metabolites": int((results["r2_improvement"] < 0).sum()),
        "n_tuned_r2_gt_0_2": int((results["tuned_mean_r2"] > 0.2).sum()),
        "n_tuned_r2_gt_0_4": int((results["tuned_mean_r2"] > 0.4).sum()),
        "n_tuned_r2_gt_0_6": int((results["tuned_mean_r2"] > 0.6).sum()),
    }

    pd.DataFrame([summary]).to_csv(output_dir / "tuning_summary.csv", index=False)

    with open(output_dir / "tuning_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Metabolites tuned        : {summary['n_metabolites_tuned']}")
    print(f"Mean baseline R2         : {summary['mean_baseline_r2']:.4f}")
    print(f"Mean tuned R2            : {summary['mean_tuned_r2']:.4f}")
    print(f"Mean R2 improvement      : {summary['mean_r2_improvement']:+.4f}")
    print(f"Improved metabolites     : {summary['n_improved_metabolites']}")
    print(f"Worse metabolites        : {summary['n_worse_metabolites']}")
    print(f"Tuned R2 > 0.2           : {summary['n_tuned_r2_gt_0_2']}")
    print(f"Tuned R2 > 0.4           : {summary['n_tuned_r2_gt_0_4']}")
    print(f"Tuned R2 > 0.6           : {summary['n_tuned_r2_gt_0_6']}")
    print()
    print("Top 20 tuned metabolites:")
    print(results.head(20).to_string(index=False))
    print()
    print("Main outputs:")
    print(output_dir / "tuning_results.csv")
    print(output_dir / "best_params_by_metabolite.csv")
    print(output_dir / "tuning_summary.csv")
    print()
    print("ExtraTrees tuning completed successfully.")


if __name__ == "__main__":
    main()