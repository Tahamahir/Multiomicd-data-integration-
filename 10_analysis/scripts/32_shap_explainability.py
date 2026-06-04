from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

ROOT = Path(".")
OUT = ROOT / "10_analysis/outputs/phase32_shap_explainability"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

TOP_METABOLITES = 5
MI_K = 500


def split_blocks(X):
    prefixes = ["soil_", "chem__", "psize__", "moist__", "nitrif__", "denit__"]
    soil_cols = [
        c for c in X.columns
        if any(str(c).lower().strip().startswith(p) for p in prefixes)
    ]
    mg_cols = [c for c in X.columns if c not in soil_cols]
    return mg_cols, soil_cols


def load_data():
    X = pd.read_csv(
        ROOT / "10_analysis/outputs/phase3_soil_dedup/X_deduplicated.csv",
        low_memory=False,
    )

    Y = pd.read_csv(
        ROOT / "10_analysis/outputs/phase2_preprocessing_fixed/Y_ml_filtered_log1p.csv",
        low_memory=False,
    )

    metrics = pd.read_csv(
        ROOT / "10_analysis/outputs/phase26_tune_champion_late_sparsepca_rf/"
        "T266_mi500_spca75_a10_w7_rf_a_metrics_per_metabolite.csv"
    )

    top_mets = (
        metrics.sort_values("r2", ascending=False)
        .head(TOP_METABOLITES)["metabolite"]
        .tolist()
    )

    top_mets = [m for m in top_mets if m in Y.columns]

    return X, Y[top_mets], top_mets


def preprocess_block(X_block, strategy="constant"):
    if strategy == "constant":
        imp = SimpleImputer(strategy="constant", fill_value=0)
    else:
        imp = SimpleImputer(strategy="median")

    X_imp = imp.fit_transform(X_block)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    return X_scaled, imp, scaler


def select_mi_features(X_scaled, y, feature_names, k=500):
    scores = mutual_info_regression(X_scaled, y, random_state=42)
    idx = np.argsort(scores)[::-1][:min(k, X_scaled.shape[1])]
    selected_names = [feature_names[i] for i in idx]
    return X_scaled[:, idx], selected_names, scores[idx]


def train_rf(X, y):
    model = RandomForestRegressor(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def shap_summary(model, X, feature_names, title, out_prefix):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    mean_abs = np.abs(shap_values).mean(axis=0)

    imp = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)

    imp.to_csv(OUT / f"{out_prefix}_feature_importance.csv", index=False)

    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIG / f"{out_prefix}_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIG / f"{out_prefix}_beeswarm.pdf", bbox_inches="tight")
    plt.close()

    top = imp.head(20).iloc[::-1]

    plt.figure(figsize=(8, 7))
    plt.barh(top["feature"], top["mean_abs_shap"])
    plt.xlabel("Mean |SHAP value|")
    plt.title(title + "  Top 20 features")
    plt.tight_layout()
    plt.savefig(FIG / f"{out_prefix}_top20_bar.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIG / f"{out_prefix}_top20_bar.pdf", bbox_inches="tight")
    plt.close()

    return imp


def main():
    X, Y, metabolites = load_data()
    mg_cols, soil_cols = split_blocks(X)

    print(f"Loaded X={X.shape}, Y={Y.shape}")
    print(f"MG={len(mg_cols)}, Soil={len(soil_cols)}")
    print("Metabolites explained:", metabolites)

    all_rows = []

    for metabolite in metabolites:
        print(f"\n[SHAP] {metabolite}")

        y = Y[metabolite].values

        # MG block
        X_mg_scaled, _, _ = preprocess_block(X[mg_cols], strategy="constant")
        X_mg_sel, mg_selected, _ = select_mi_features(
            X_mg_scaled,
            y,
            mg_cols,
            k=MI_K,
        )

        model_mg = train_rf(X_mg_sel, y)

        mg_imp = shap_summary(
            model_mg,
            X_mg_sel,
            mg_selected,
            f"SHAP MG features  {metabolite}",
            f"{metabolite}_MG",
        )

        mg_imp["metabolite"] = metabolite
        mg_imp["block"] = "MG"

        # Soil block
        X_soil_scaled, _, _ = preprocess_block(X[soil_cols], strategy="median")

        model_soil = train_rf(X_soil_scaled, y)

        soil_imp = shap_summary(
            model_soil,
            X_soil_scaled,
            soil_cols,
            f"SHAP soil features  {metabolite}",
            f"{metabolite}_SOIL",
        )

        soil_imp["metabolite"] = metabolite
        soil_imp["block"] = "SOIL"

        all_rows.append(mg_imp)
        all_rows.append(soil_imp)

    all_imp = pd.concat(all_rows, ignore_index=True)

    all_imp.to_csv(OUT / "all_shap_feature_importance.csv", index=False)

    global_imp = (
        all_imp.groupby(["block", "feature"], as_index=False)["mean_abs_shap"]
        .mean()
        .sort_values("mean_abs_shap", ascending=False)
    )

    global_imp.to_csv(OUT / "global_mean_shap_importance.csv", index=False)

    for block in ["MG", "SOIL"]:
        sub = global_imp[global_imp["block"] == block].head(25).iloc[::-1]

        plt.figure(figsize=(9, 8))
        plt.barh(sub["feature"], sub["mean_abs_shap"])
        plt.xlabel("Mean |SHAP value| across explained metabolites")
        plt.title(f"Global SHAP importance  {block}")
        plt.tight_layout()
        plt.savefig(FIG / f"global_{block}_top25_shap.png", dpi=300, bbox_inches="tight")
        plt.savefig(FIG / f"global_{block}_top25_shap.pdf", bbox_inches="tight")
        plt.close()

    print("\nSaved outputs in:")
    print(OUT)
    print(FIG)


if __name__ == "__main__":
    main()
