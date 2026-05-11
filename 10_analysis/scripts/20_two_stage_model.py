from pathlib import Path
import pandas as pd
import numpy as np
import json

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor


# ============================================================
# PHASE 20 - TWO-STAGE MODEL FOR ZERO-INFLATED METABOLITES
# ------------------------------------------------------------
# Stage 1 : classification presence / absence
# Stage 2 : regression abundance if present
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


def build_direct_model(model_name, metabolite, et_params, rf_params, xgb_params):
    if model_name == "ExtraTrees_tuned":
        return ExtraTreesRegressor(
            random_state=42,
            n_jobs=-1,
            **et_params[metabolite]
        )

    if model_name == "RandomForest_tuned":
        return RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
            **rf_params[metabolite]
        )

    if model_name == "XGBoost_tuned":
        return XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            **xgb_params[metabolite]
        )

    raise ValueError(f"Unknown model: {model_name}")


def safe_r2(y_true, y_pred):
    if np.std(y_true) == 0:
        return np.nan
    return r2_score(y_true, y_pred)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def main():
    repo_root = Path(__file__).resolve().parents[2]

    x_path = repo_root / "10_analysis" / "outputs" / "phase3_soil_dedup" / "X_deduplicated.csv"
    y_path = repo_root / "10_analysis" / "outputs" / "phase2_preprocessing_fixed" / "Y_ml_filtered_log1p.csv"

    best_models_path = repo_root / "10_analysis" / "outputs" / "phase17_final_best_model_pipeline" / "best_model_per_metabolite_final.csv"

    et_params_path = repo_root / "10_analysis" / "outputs" / "phase13_tuning_extratrees" / "best_params_by_metabolite.csv"
    rf_params_path = repo_root / "10_analysis" / "outputs" / "phase15_tuning_randomforest" / "rf_best_params_by_metabolite.csv"
    xgb_params_path = repo_root / "10_analysis" / "outputs" / "phase16_tuning_xgboost" / "xgb_best_params_by_metabolite.csv"

    output_dir = repo_root / "10_analysis" / "outputs" / "phase20_two_stage_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 20 - TWO-STAGE MODEL")
    print("=" * 70)

    for p in [x_path, y_path, best_models_path, et_params_path, rf_params_path, xgb_params_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    X_raw = pd.read_csv(x_path, low_memory=False)
    Y = pd.read_csv(y_path, low_memory=False)
    best_models = pd.read_csv(best_models_path, low_memory=False)

    X_raw = impute_X(X_raw)

    X_xgb = X_raw.copy()
    X_xgb.columns = clean_xgb_columns(X_xgb.columns)

    et_params = load_params(et_params_path)
    rf_params = load_params(rf_params_path)
    xgb_params = load_params(xgb_params_path)

    metabolites = [m for m in best_models["metabolite"].tolist() if m in Y.columns]

    print(f"X shape: {X_raw.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"Metabolites tested: {len(metabolites)}")
    print()

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    all_rows = []

    for idx, metabolite in enumerate(metabolites, start=1):
        print(f"[{idx}/{len(metabolites)}] Two-stage test: {metabolite}")

        model_row = best_models[best_models["metabolite"] == metabolite].iloc[0]
        direct_model_name = model_row["model"]

        y = Y[metabolite].values
        y_binary = (y > 0).astype(int)

        zero_fraction = float((y == 0).mean())
        presence_fraction = float((y > 0).mean())

        direct_r2_scores = []
        direct_rmse_scores = []

        two_stage_r2_scores = []
        two_stage_rmse_scores = []

        auc_scores = []
        f1_scores = []
        acc_scores = []

        for fold, (train_idx, test_idx) in enumerate(cv.split(X_raw), start=1):

            # --------------------------
            # Direct model
            # --------------------------
            if direct_model_name == "XGBoost_tuned":
                X_train_direct = X_xgb.iloc[train_idx]
                X_test_direct = X_xgb.iloc[test_idx]
            else:
                X_train_direct = X_raw.iloc[train_idx]
                X_test_direct = X_raw.iloc[test_idx]

            y_train = y[train_idx]
            y_test = y[test_idx]

            direct_model = build_direct_model(
                direct_model_name,
                metabolite,
                et_params,
                rf_params,
                xgb_params
            )

            direct_model.fit(X_train_direct, y_train)
            y_pred_direct = direct_model.predict(X_test_direct)

            direct_r2_scores.append(safe_r2(y_test, y_pred_direct))
            direct_rmse_scores.append(rmse(y_test, y_pred_direct))

            # --------------------------
            # Two-stage model
            # --------------------------
            X_train = X_raw.iloc[train_idx]
            X_test = X_raw.iloc[test_idx]

            y_train_bin = y_binary[train_idx]
            y_test_bin = y_binary[test_idx]

            # Si le train contient une seule classe, impossible de classifier
            if len(np.unique(y_train_bin)) < 2:
                y_pred_two_stage = np.repeat(np.mean(y_train), len(test_idx))
                two_stage_r2_scores.append(safe_r2(y_test, y_pred_two_stage))
                two_stage_rmse_scores.append(rmse(y_test, y_pred_two_stage))
                continue

            clf = RandomForestClassifier(
                n_estimators=500,
                random_state=42,
                n_jobs=-1,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced"
            )

            clf.fit(X_train, y_train_bin)

            proba_present = clf.predict_proba(X_test)[:, 1]
            pred_present = (proba_present >= 0.5).astype(int)

            # classification metrics
            if len(np.unique(y_test_bin)) == 2:
                auc_scores.append(roc_auc_score(y_test_bin, proba_present))

            f1_scores.append(f1_score(y_test_bin, pred_present, zero_division=0))
            acc_scores.append(accuracy_score(y_test_bin, pred_present))

            # regression only on positive training samples
            positive_train_mask = y_train > 0

            if positive_train_mask.sum() < 5:
                # pas assez de positifs pour entraîner régression
                positive_mean = y_train[y_train > 0].mean() if (y_train > 0).sum() > 0 else 0
                y_pred_reg = np.repeat(positive_mean, len(test_idx))
            else:
                reg = RandomForestRegressor(
                    n_estimators=500,
                    random_state=42,
                    n_jobs=-1,
                    max_features="sqrt",
                    min_samples_leaf=2
                )

                reg.fit(X_train.loc[positive_train_mask], y_train[positive_train_mask])
                y_pred_reg = reg.predict(X_test)

            # final two-stage prediction
            y_pred_two_stage = np.where(pred_present == 1, y_pred_reg, 0)

            two_stage_r2_scores.append(safe_r2(y_test, y_pred_two_stage))
            two_stage_rmse_scores.append(rmse(y_test, y_pred_two_stage))

        mean_direct_r2 = float(np.nanmean(direct_r2_scores))
        mean_direct_rmse = float(np.nanmean(direct_rmse_scores))

        mean_two_stage_r2 = float(np.nanmean(two_stage_r2_scores))
        mean_two_stage_rmse = float(np.nanmean(two_stage_rmse_scores))

        mean_auc = float(np.nanmean(auc_scores)) if len(auc_scores) > 0 else np.nan
        mean_f1 = float(np.nanmean(f1_scores)) if len(f1_scores) > 0 else np.nan
        mean_acc = float(np.nanmean(acc_scores)) if len(acc_scores) > 0 else np.nan

        improvement = mean_two_stage_r2 - mean_direct_r2

        all_rows.append({
            "metabolite": metabolite,
            "direct_model": direct_model_name,
            "zero_fraction": zero_fraction,
            "presence_fraction": presence_fraction,

            "direct_mean_r2": mean_direct_r2,
            "direct_mean_rmse": mean_direct_rmse,

            "two_stage_mean_r2": mean_two_stage_r2,
            "two_stage_mean_rmse": mean_two_stage_rmse,

            "r2_improvement": improvement,

            "stage1_mean_auc": mean_auc,
            "stage1_mean_f1": mean_f1,
            "stage1_mean_accuracy": mean_acc,

            "two_stage_better": improvement > 0
        })

        print(f"  direct R2    : {mean_direct_r2:.4f}")
        print(f"  two-stage R2 : {mean_two_stage_r2:.4f}")
        print(f"  improvement  : {improvement:+.4f}")
        print(f"  AUC/F1/ACC   : {mean_auc:.3f} / {mean_f1:.3f} / {mean_acc:.3f}")
        print()

    results = pd.DataFrame(all_rows).sort_values("r2_improvement", ascending=False)

    results.to_csv(output_dir / "two_stage_results.csv", index=False)

    improved = results[results["r2_improvement"] > 0].copy()
    worsened = results[results["r2_improvement"] <= 0].copy()

    improved.to_csv(output_dir / "two_stage_improved_metabolites.csv", index=False)
    worsened.to_csv(output_dir / "two_stage_not_improved_metabolites.csv", index=False)

    summary = {
        "n_metabolites_tested": int(len(results)),
        "mean_direct_r2": float(results["direct_mean_r2"].mean()),
        "mean_two_stage_r2": float(results["two_stage_mean_r2"].mean()),
        "median_direct_r2": float(results["direct_mean_r2"].median()),
        "median_two_stage_r2": float(results["two_stage_mean_r2"].median()),
        "mean_r2_improvement": float(results["r2_improvement"].mean()),
        "n_improved": int((results["r2_improvement"] > 0).sum()),
        "n_not_improved": int((results["r2_improvement"] <= 0).sum()),
        "mean_stage1_auc": float(results["stage1_mean_auc"].mean()),
        "mean_stage1_f1": float(results["stage1_mean_f1"].mean()),
        "mean_stage1_accuracy": float(results["stage1_mean_accuracy"].mean()),
    }

    pd.DataFrame([summary]).to_csv(output_dir / "two_stage_summary.csv", index=False)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Metabolites tested        : {summary['n_metabolites_tested']}")
    print(f"Mean direct R2            : {summary['mean_direct_r2']:.4f}")
    print(f"Mean two-stage R2         : {summary['mean_two_stage_r2']:.4f}")
    print(f"Mean R2 improvement       : {summary['mean_r2_improvement']:+.4f}")
    print(f"Improved metabolites      : {summary['n_improved']}")
    print(f"Not improved metabolites  : {summary['n_not_improved']}")
    print(f"Mean stage1 AUC           : {summary['mean_stage1_auc']:.4f}")
    print(f"Mean stage1 F1            : {summary['mean_stage1_f1']:.4f}")
    print(f"Mean stage1 accuracy      : {summary['mean_stage1_accuracy']:.4f}")
    print()
    print("Top improved metabolites:")
    print(results.head(15).to_string(index=False))
    print()
    print("Main outputs:")
    print(output_dir / "two_stage_results.csv")
    print(output_dir / "two_stage_improved_metabolites.csv")
    print(output_dir / "two_stage_summary.csv")
    print()
    print("Two-stage model completed successfully.")


if __name__ == "__main__":
    main()