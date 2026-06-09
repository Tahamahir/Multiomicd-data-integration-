from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


ROOT = Path(".")
OUT = ROOT / "10_analysis/outputs/phase33_mg_mb_associations"
FIG = OUT / "figures"

OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


TOP_METABOLITES = 30
TOP_MI_FEATURES_PER_MB = 200
FDR_ALPHA = 0.05
MIN_ABS_RHO = 0.25


def bh_fdr(pvalues):
    pvalues = np.asarray(pvalues)
    n = len(pvalues)

    order = np.argsort(pvalues)
    ranked = pvalues[order]

    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    out = np.empty_like(q)
    out[order] = q

    return out


def split_blocks(X):
    soil_prefixes = [
        "soil_",
        "chem__",
        "psize__",
        "moist__",
        "nitrif__",
        "denit__",
    ]

    soil_cols = []
    for c in X.columns:
        cl = str(c).lower().strip()
        if any(cl.startswith(p) for p in soil_prefixes):
            soil_cols.append(c)

    mg_cols = [c for c in X.columns if c not in soil_cols]
    return mg_cols, soil_cols


def load_data():
    X = pd.read_csv(
        ROOT / "10_analysis/outputs/phase3_soil_dedup/X_deduplicated.csv",
        low_memory=False,
    )

    Y_all = pd.read_csv(
        ROOT / "10_analysis/outputs/phase2_preprocessing_fixed/Y_ml_filtered_log1p.csv",
        low_memory=False,
    )

    metrics = pd.read_csv(
        ROOT / "10_analysis/outputs/phase26_tune_champion_late_sparsepca_rf/"
        "T266_mi500_spca75_a10_w7_rf_a_metrics_per_metabolite.csv"
    )

    top_mets = (
        metrics
        .sort_values("r2", ascending=False)
        .head(TOP_METABOLITES)["metabolite"]
        .tolist()
    )

    top_mets = [m for m in top_mets if m in Y_all.columns]

    return X, Y_all[top_mets], top_mets


def safe_name(x):
    return (
        str(x)
        .replace("/", "_")
        .replace("|", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def select_mi_candidates(X_mg_scaled, y, mg_cols, k):
    mi = mutual_info_regression(
        X_mg_scaled,
        y,
        random_state=42,
    )

    idx = np.argsort(mi)[::-1][:min(k, len(mg_cols))]

    selected = []
    for i in idx:
        selected.append({
            "feature": mg_cols[i],
            "mi_score": mi[i],
            "feature_index": i,
        })

    return selected


def classify_association(rho):
    if rho > 0:
        return "putative_production_like"
    elif rho < 0:
        return "putative_consumption_like"
    else:
        return "neutral"


def main():
    print("Loading data...")

    X, Y, metabolites = load_data()
    mg_cols, soil_cols = split_blocks(X)

    print(f"X={X.shape}")
    print(f"Y_top={Y.shape}")
    print(f"MG features={len(mg_cols)}")
    print(f"Soil features={len(soil_cols)}")

    # Prepare MG matrix
    imp = SimpleImputer(strategy="constant", fill_value=0)
    X_mg = imp.fit_transform(X[mg_cols])

    scaler = StandardScaler()
    X_mg_scaled = scaler.fit_transform(X_mg)

    rows = []

    for mb_i, metabolite in enumerate(metabolites, start=1):
        print(f"[{mb_i}/{len(metabolites)}] {metabolite}")

        y = Y[metabolite].values

        candidates = select_mi_candidates(
            X_mg_scaled,
            y,
            mg_cols,
            TOP_MI_FEATURES_PER_MB,
        )

        for cand in candidates:
            feature = cand["feature"]
            idx = cand["feature_index"]

            x = X_mg[:, idx]

            if np.nanstd(x) == 0:
                continue

            rho, p = spearmanr(x, y)

            if np.isnan(rho) or np.isnan(p):
                continue

            rows.append({
                "metabolite": metabolite,
                "mg_feature": feature,
                "rho": rho,
                "pvalue": p,
                "abs_rho": abs(rho),
                "mi_score": cand["mi_score"],
                "direction": "positive" if rho > 0 else "negative",
                "putative_role": classify_association(rho),
            })

    assoc = pd.DataFrame(rows)

    assoc["qvalue_fdr"] = bh_fdr(assoc["pvalue"].values)

    assoc["significant"] = (
        (assoc["qvalue_fdr"] < FDR_ALPHA)
        &
        (assoc["abs_rho"] >= MIN_ABS_RHO)
    )

    assoc = assoc.sort_values(
        ["significant", "abs_rho", "mi_score"],
        ascending=[False, False, False],
    )

    assoc.to_csv(
        OUT / "all_mg_mb_associations.csv",
        index=False,
    )

    sig = assoc[assoc["significant"]].copy()

    sig.to_csv(
        OUT / "significant_mg_mb_associations.csv",
        index=False,
    )

    top_by_mb = (
        sig
        .sort_values(["metabolite", "abs_rho"], ascending=[True, False])
        .groupby("metabolite")
        .head(10)
        .reset_index(drop=True)
    )

    top_by_mb.to_csv(
        OUT / "top10_significant_associations_per_metabolite.csv",
        index=False,
    )

    summary = {
        "n_metabolites": len(metabolites),
        "n_mg_features": len(mg_cols),
        "n_tested_pairs": int(len(assoc)),
        "n_significant_pairs": int(len(sig)),
        "n_positive_significant": int((sig["rho"] > 0).sum()),
        "n_negative_significant": int((sig["rho"] < 0).sum()),
        "fdr_alpha": FDR_ALPHA,
        "min_abs_rho": MIN_ABS_RHO,
    }

    with open(OUT / "association_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame([summary]).to_csv(
        OUT / "association_summary.csv",
        index=False,
    )

    print(json.dumps(summary, indent=2))

    make_plots(assoc, sig, top_by_mb)

    print("\nSaved outputs in:")
    print(OUT)


def make_plots(assoc, sig, top_by_mb):
    # 1. Count positive / negative significant associations
    if len(sig) > 0:
        counts = sig["putative_role"].value_counts()

        plt.figure(figsize=(7, 5))
        plt.bar(counts.index, counts.values)
        plt.ylabel("Number of significant associations")
        plt.title("Putative production-like vs consumption-like MGMB associations")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(FIG / "association_role_counts.png", dpi=300, bbox_inches="tight")
        plt.savefig(FIG / "association_role_counts.pdf", bbox_inches="tight")
        plt.close()

    # 2. Top positive associations
    pos = sig[sig["rho"] > 0].sort_values("rho", ascending=False).head(20)

    if len(pos) > 0:
        labels = [
            f"{r['mg_feature']}  {r['metabolite'][:25]}..."
            for _, r in pos.iterrows()
        ]

        plt.figure(figsize=(10, 8))
        plt.barh(labels[::-1], pos["rho"].values[::-1])
        plt.xlabel("Spearman rho")
        plt.title("Top putative production-like associations")
        plt.tight_layout()
        plt.savefig(FIG / "top_positive_associations.png", dpi=300, bbox_inches="tight")
        plt.savefig(FIG / "top_positive_associations.pdf", bbox_inches="tight")
        plt.close()

    # 3. Top negative associations
    neg = sig[sig["rho"] < 0].sort_values("rho", ascending=True).head(20)

    if len(neg) > 0:
        labels = [
            f"{r['mg_feature']}  {r['metabolite'][:25]}..."
            for _, r in neg.iterrows()
        ]

        plt.figure(figsize=(10, 8))
        plt.barh(labels[::-1], neg["rho"].values[::-1])
        plt.xlabel("Spearman rho")
        plt.title("Top putative consumption-like associations")
        plt.tight_layout()
        plt.savefig(FIG / "top_negative_associations.png", dpi=300, bbox_inches="tight")
        plt.savefig(FIG / "top_negative_associations.pdf", bbox_inches="tight")
        plt.close()

    # 4. Heatmap of top associations
    if len(sig) > 0:
        top_features = (
            sig
            .groupby("mg_feature")["abs_rho"]
            .max()
            .sort_values(ascending=False)
            .head(30)
            .index
            .tolist()
        )

        top_mets = (
            sig
            .groupby("metabolite")["abs_rho"]
            .max()
            .sort_values(ascending=False)
            .head(20)
            .index
            .tolist()
        )

        heat = sig[
            sig["mg_feature"].isin(top_features)
            &
            sig["metabolite"].isin(top_mets)
        ].pivot_table(
            index="mg_feature",
            columns="metabolite",
            values="rho",
            aggfunc="mean",
        )

        heat = heat.reindex(index=top_features, columns=top_mets)
        heat = heat.fillna(0)

        plt.figure(figsize=(14, 10))
        plt.imshow(heat.values, aspect="auto", vmin=-1, vmax=1)
        plt.colorbar(label="Spearman rho")
        plt.xticks(
            range(len(heat.columns)),
            [c[:25] + "..." for c in heat.columns],
            rotation=60,
            ha="right",
            fontsize=8,
        )
        plt.yticks(
            range(len(heat.index)),
            heat.index,
            fontsize=8,
        )
        plt.title("MGMB association heatmap")
        plt.xlabel("Metabolite")
        plt.ylabel("MG feature")
        plt.tight_layout()
        plt.savefig(FIG / "mg_mb_association_heatmap.png", dpi=300, bbox_inches="tight")
        plt.savefig(FIG / "mg_mb_association_heatmap.pdf", bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()
