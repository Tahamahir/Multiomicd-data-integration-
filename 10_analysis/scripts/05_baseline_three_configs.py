from pathlib import Path
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


# ============================================================
# PHASE 4 - BASELINE BENCHMARK (3 CONFIGS)
# ------------------------------------------------------------
# Configurations comparées :
# 1) soil only
# 2) metagenomics PCA only
# 3) soil + metagenomics PCA
# ============================================================


def compute_metrics(y_true, y_pred, y_columns):
    global_r2_uniform = r2_score(y_true, y_pred, multioutput="uniform_average")
    global_r2_variance = r2_score(y_true, y_pred, multioutput="variance_weighted")
    global_rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    rows = []
    for i, col in enumerate(y_columns):
        rows.append({
            "metabolite": col,
            "r2": r2_score(y_true[:, i], y_pred[:, i]),
            "rmse": np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        })

    per_target = pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)

    summary = {
        "global_r2_uniform_average": float(global_r2_uniform),
        "global_r2_variance_weighted": float(global_r2_variance),
        "global_rmse": float(global_rmse),
        "n_positive_r2_targets": int((per_target["r2"] > 0).sum()),
        "n_negative_r2_targets": int((per_target["r2"] < 0).sum()),
    }

    return summary, per_target


def train_and_evaluate(X, Y, sample_ids, config_name, output_dir):
    """
    Entraîne et évalue une configuration.
    """
    X_train, X_test, Y_train, Y_test, ids_train, ids_test = train_test_split(
        X, Y, sample_ids,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    print(f"Training config: {config_name}")
    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")

    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    summary, per_target = compute_metrics(
        Y_test.to_numpy(),
        Y_pred,
        Y.columns.tolist()
    )

    # feature importance
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # predictions
    y_true_df = Y_test.reset_index(drop=True).copy()
    y_pred_df = pd.DataFrame(Y_pred, columns=Y.columns)

    y_true_df.columns = [f"true__{c}" for c in y_true_df.columns]
    y_pred_df.columns = [f"pred__{c}" for c in y_pred_df.columns]

    pred_df = pd.concat(
        [ids_test.reset_index(drop=True), y_true_df, y_pred_df],
        axis=1
    )

    # output dir config
    cfg_dir = output_dir / config_name
    cfg_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([summary]).to_csv(cfg_dir / "summary.csv", index=False)
    per_target.to_csv(cfg_dir / "per_metabolite_metrics.csv", index=False)
    per_target.head(10).to_csv(cfg_dir / "top10_metabolites_by_r2.csv", index=False)
    per_target.tail(10).to_csv(cfg_dir / "bottom10_metabolites_by_r2.csv", index=False)
    feature_importance.to_csv(cfg_dir / "feature_importance.csv", index=False)
    pred_df.to_csv(cfg_dir / "test_predictions.csv", index=False)

    with open(cfg_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary, per_target, feature_importance


def main():
    repo_root = Path(__file__).resolve().parents[2]

    input_dir = repo_root / "10_analysis" / "outputs" / "phase3_smart_reduction_imputed"
    output_dir = repo_root / "10_analysis" / "outputs" / "phase4_baseline_three_configs"
    output_dir.mkdir(parents=True, exist_ok=True)

    x_path = input_dir / "X_final_smart.csv"
    y_path = input_dir / "Y_top50_log1p_scaled.csv"
    ids_path = input_dir / "sample_ids.csv"

    print("=" * 70)
    print("PHASE 4 - BASELINE BENCHMARK (3 CONFIGS)")
    print("=" * 70)
    print(f"X input   : {x_path}")
    print(f"Y input   : {y_path}")
    print(f"IDs input : {ids_path}")
    print(f"Output dir: {output_dir}")
    print()

    for p in [x_path, y_path, ids_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    X = pd.read_csv(x_path, low_memory=False)
    Y = pd.read_csv(y_path, low_memory=False)
    sample_ids = pd.read_csv(ids_path, low_memory=False)

    if not (len(X) == len(Y) == len(sample_ids)):
        raise ValueError("X, Y and sample_ids must have the same number of rows.")

    # ------------------------------------------------------------
    # Split X into soil vs metagenomics PCA
    # ------------------------------------------------------------
    meta_cols = [c for c in X.columns if c.startswith("meta_PC")]
    soil_cols = [c for c in X.columns if not c.startswith("meta_PC")]

    X_meta = X[meta_cols].copy()
    X_soil = X[soil_cols].copy()
    X_both = X.copy()

    print(f"Total X shape          : {X.shape}")
    print(f"Meta-only shape        : {X_meta.shape}")
    print(f"Soil-only shape        : {X_soil.shape}")
    print(f"Y shape                : {Y.shape}")
    print()

    # ------------------------------------------------------------
    # Train & evaluate 3 configurations
    # ------------------------------------------------------------
    results = []

    for config_name, X_cfg in [
        ("soil_only", X_soil),
        ("mg_only", X_meta),
        ("soil_plus_mg", X_both),
    ]:
        summary, per_target, feature_importance = train_and_evaluate(
            X_cfg, Y, sample_ids, config_name, output_dir
        )

        results.append({
            "config": config_name,
            **summary
        })

        print(f"Finished: {config_name}")
        print(f"  Global R2 (uniform)   : {summary['global_r2_uniform_average']:.4f}")
        print(f"  Global R2 (variance)  : {summary['global_r2_variance_weighted']:.4f}")
        print(f"  Global RMSE           : {summary['global_rmse']:.4f}")
        print(f"  Positive R2 targets   : {summary['n_positive_r2_targets']}")
        print()

    comparison_df = pd.DataFrame(results).sort_values("global_r2_variance_weighted", ascending=False)
    comparison_df.to_csv(output_dir / "configs_comparison.csv", index=False)

    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(comparison_df.to_string(index=False))
    print()
    print("Main outputs:")
    print(output_dir / "configs_comparison.csv")
    print(output_dir / "soil_only")
    print(output_dir / "mg_only")
    print(output_dir / "soil_plus_mg")
    print()
    print("Baseline benchmark completed successfully.")


if __name__ == "__main__":
    main()