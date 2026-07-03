#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 42b - Re-validation 47 metabolites + nul + FDR (ecriture incrementale)."""

from pathlib import Path
import argparse
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import SparsePCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

X_PATH = "10_analysis/outputs/phase3_soil_dedup/X_deduplicated.csv"
Y_PATH = "10_analysis/outputs/phase2_preprocessing_fixed/Y_ml_filtered_log1p.csv"
METAB_LIST_PATH = ("10_analysis/outputs/phase26_tune_champion_late_sparsepca_rf/"
                   "T266_mi500_spca75_a10_w7_rf_a_metrics_per_metabolite.csv")
METAB_GLOB_FALLBACK = ("10_analysis/outputs/phase26_tune_champion_late_sparsepca_rf/"
                       "*_metrics_per_metabolite.csv")
OUT_SUBDIR = "10_analysis/outputs/phase42b_revalidation_full"

SOIL_PREFIXES = ("soil_", "chem__", "psize__", "moist__", "nitrif__", "denit__")
MI_K = 500
N_COMPONENTS = 75
W_MG = 0.7
W_SOIL = 0.3
N_SPLITS = 5
MODELS = ["MG_only", "Soil_only", "MG_Soil_late"]
DEFAULT_N_REPEATS = 2
DEFAULT_N_PERMUTATIONS = 100


def split_blocks(X):
    soil_cols = [c for c in X.columns
                 if any(str(c).lower().strip().startswith(p) for p in SOIL_PREFIXES)]
    mg_cols = [c for c in X.columns if c not in soil_cols]
    return mg_cols, soil_cols


def load_metabolite_list(project_root):
    root = Path(project_root)
    p = root / METAB_LIST_PATH
    if not p.exists():
        cand = sorted(root.glob(METAB_GLOB_FALLBACK))
        if not cand:
            raise FileNotFoundError("Liste des 47 metabolites introuvable (phase 26).")
        p = cand[0]
    return pd.read_csv(p)["metabolite"].dropna().astype(str).unique().tolist()


def load_data(project_root):
    root = Path(project_root)
    X = pd.read_csv(root / X_PATH, low_memory=False)
    Y_all = pd.read_csv(root / Y_PATH, low_memory=False)
    metabolites = [m for m in load_metabolite_list(project_root) if m in Y_all.columns]
    mg_cols, soil_cols = split_blocks(X)
    return X, Y_all, metabolites, mg_cols, soil_cols


def preprocess_block(Xtr_df, Xte_df, strategy):
    imp = (SimpleImputer(strategy="median") if strategy == "median"
           else SimpleImputer(strategy="constant", fill_value=0))
    sc = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(Xtr_df))
    Xte = sc.transform(imp.transform(Xte_df))
    return Xtr, Xte


def select_mi(Xtr, ytr, Xte, k):
    if Xtr.shape[1] <= k:
        return Xtr, Xte
    mi = mutual_info_regression(Xtr, ytr, random_state=42, discrete_features=False)
    mi = np.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0)
    idx = np.argsort(mi)[::-1][:k]
    return Xtr[:, idx], Xte[:, idx]


def sparsepca_reduce(Xtr, Xte, n_components):
    n_components = min(n_components, Xtr.shape[0] - 1, Xtr.shape[1])
    if n_components < 1:
        return Xtr, Xte
    red = SparsePCA(n_components=n_components, alpha=1, random_state=42,
                    n_jobs=-1, max_iter=100)
    return red.fit_transform(Xtr), red.transform(Xte)


def build_rf(seed):
    return RandomForestRegressor(n_estimators=800, min_samples_leaf=2,
                                 max_features="sqrt", random_state=seed, n_jobs=-1)


def one_repeat_oof(X, y, mg_cols, soil_cols, seed):
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    pred = {m: np.zeros(len(y)) for m in MODELS}
    for fold_id, (tr, te) in enumerate(cv.split(X)):
        ytr = y[tr]
        Xtr_mg, Xte_mg = preprocess_block(X.iloc[tr][mg_cols], X.iloc[te][mg_cols], "constant")
        Xtr_mg, Xte_mg = select_mi(Xtr_mg, ytr, Xte_mg, MI_K)
        Xtr_mg, Xte_mg = sparsepca_reduce(Xtr_mg, Xte_mg, N_COMPONENTS)
        m_mg = build_rf(seed + fold_id)
        m_mg.fit(Xtr_mg, ytr)
        p_mg = m_mg.predict(Xte_mg)
        pred["MG_only"][te] = p_mg
        if soil_cols:
            Xtr_s, Xte_s = preprocess_block(X.iloc[tr][soil_cols], X.iloc[te][soil_cols], "median")
            m_s = build_rf(1000 + seed + fold_id)
            m_s.fit(Xtr_s, ytr)
            p_s = m_s.predict(Xte_s)
        else:
            p_s = np.zeros_like(p_mg)
        pred["Soil_only"][te] = p_s
        pred["MG_Soil_late"][te] = W_MG * p_mg + W_SOIL * p_s
    return pred


def metrics_over_repeats(X, y, mg_cols, soil_cols, base_seed, n_repeats):
    acc = {m: {"r2": [], "rmse": [], "mae": []} for m in MODELS}
    for r in range(n_repeats):
        pred = one_repeat_oof(X, y, mg_cols, soil_cols, seed=base_seed + r)
        for m in MODELS:
            acc[m]["r2"].append(r2_score(y, pred[m]))
            acc[m]["rmse"].append(np.sqrt(mean_squared_error(y, pred[m])))
            acc[m]["mae"].append(mean_absolute_error(y, pred[m]))
    return {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in acc.items()}


def run_one_metabolite(X, Y_all, metabolite, mg_cols, soil_cols,
                       n_repeats, n_permutations, out_file):
    y = Y_all[metabolite].values.astype(float)
    cols = ["metabolite", "model", "kind", "perm", "r2", "rmse", "mae"]
    if out_file.exists():
        out_file.unlink()

    def append_rows(rows):
        df = pd.DataFrame(rows)[cols]
        write_header = not out_file.exists()
        df.to_csv(out_file, mode="a", header=write_header, index=False)

    real = metrics_over_repeats(X, y, mg_cols, soil_cols, base_seed=42, n_repeats=n_repeats)
    append_rows([{"metabolite": metabolite, "model": mm, "kind": "real",
                  "perm": -1, **real[mm]} for mm in MODELS])
    print(f"    [reel ecrit] MG_Soil_late r2={real['MG_Soil_late']['r2']:.3f}", flush=True)

    n_ok = 0
    for b in range(n_permutations):
        try:
            rng = np.random.default_rng(1000 + b)
            y_perm = rng.permutation(y)
            nul = metrics_over_repeats(X, y_perm, mg_cols, soil_cols,
                                       base_seed=5000 + 7 * b, n_repeats=n_repeats)
            append_rows([{"metabolite": metabolite, "model": mm, "kind": "null",
                          "perm": b, **nul[mm]} for mm in MODELS])
            n_ok += 1
            if b == 0 or (b + 1) % 5 == 0:
                print(f"    [perm {b + 1}/{n_permutations} ecrite] "
                      f"nul r2={nul['MG_Soil_late']['r2']:.3f}", flush=True)
        except Exception as e:
            print(f"    [WARN perm {b}] {repr(e)}", flush=True)

    print(f"    [nul termine] {n_ok}/{n_permutations} permutations ecrites", flush=True)


def aggregate_and_fdr(out_dir):
    files = sorted(out_dir.glob("metrics_task_*.csv"))
    if not files:
        raise FileNotFoundError("Aucun fichier metrics_task_*.csv trouve.")
    allm = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    allm.to_csv(out_dir / "revalidation_all_metrics.csv", index=False)

    real = allm[allm["kind"] == "real"]
    null = allm[allm["kind"] == "null"]
    head_model = "MG_Soil_late"
    per_rows = []
    for met, sub_real in real[real["model"] == head_model].groupby("metabolite"):
        r2_real = float(sub_real["r2"].iloc[0])
        null_r2 = null[(null["model"] == head_model) &
                       (null["metabolite"] == met)]["r2"].values
        B = len(null_r2)
        p_perm = (1 + int(np.sum(null_r2 >= r2_real))) / (1 + B) if B > 0 else np.nan
        per_rows.append({"metabolite": met, "r2_real_MGSoil": r2_real,
                         "r2_null_mean_MGSoil": float(np.mean(null_r2)) if B else np.nan,
                         "n_perm": B, "p_perm": p_perm})
    per = pd.DataFrame(per_rows)

    for m, col in [("MG_only", "r2_real_MGonly"), ("Soil_only", "r2_real_Soilonly")]:
        tmp = real[real["model"] == m][["metabolite", "r2"]].rename(columns={"r2": col})
        per = per.merge(tmp, on="metabolite", how="left")

    mask = per["p_perm"].notna()
    per["p_fdr_bh"] = np.nan
    per["significant_fdr_005"] = False
    if mask.sum() > 0:
        rej, p_corr, _, _ = multipletests(per.loc[mask, "p_perm"].values,
                                          alpha=0.05, method="fdr_bh")
        per.loc[mask, "p_fdr_bh"] = p_corr
        per.loc[mask, "significant_fdr_005"] = rej

    per = per.sort_values("r2_real_MGSoil", ascending=False).reset_index(drop=True)
    per.to_csv(out_dir / "revalidation_per_metabolite.csv", index=False)

    def msummary(df, kind):
        g = (df.groupby("model")["r2"].agg(["mean", "median", "std", "min", "max"])
             .reset_index())
        g["kind"] = kind
        return g
    summary = pd.concat([msummary(real, "real"), msummary(null, "null")],
                        ignore_index=True)
    summary.to_csv(out_dir / "revalidation_summary.csv", index=False)

    n_sig = int(per["significant_fdr_005"].sum())
    head = {"n_metabolites": int(per.shape[0]),
            "mean_r2_MGSoil_real": float(real[real.model == head_model]["r2"].mean()),
            "median_r2_MGSoil_real": float(real[real.model == head_model]["r2"].median()),
            "mean_r2_MGonly_real": float(real[real.model == "MG_only"]["r2"].mean()),
            "mean_r2_Soilonly_real": float(real[real.model == "Soil_only"]["r2"].mean()),
            "mean_r2_MGSoil_null": float(null[null.model == head_model]["r2"].mean()) if len(null) else float("nan"),
            "n_significant_fdr_005": n_sig}
    with open(out_dir / "revalidation_headline.json", "w") as f:
        json.dump(head, f, indent=2)

    print("\n================ RESUME (47 metabolites) ================")
    print(json.dumps(head, indent=2))
    print("\nTop 10 metabolites :")
    cols = ["metabolite", "r2_real_MGSoil", "p_perm", "p_fdr_bh", "significant_fdr_005"]
    print(per[cols].head(10).to_string(index=False))
    print(f"\nMetabolites significatifs a FDR < 0.05 : {n_sig} / {per.shape[0]}")
    print("\nFichiers ecrits dans :", out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--task-id", type=int, default=None)
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--n-permutations", type=int, default=DEFAULT_N_PERMUTATIONS)
    ap.add_argument("--n-repeats", type=int, default=DEFAULT_N_REPEATS)
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    out_dir = root / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        aggregate_and_fdr(out_dir)
        return

    X, Y_all, metabolites, mg_cols, soil_cols = load_data(root)

    if args.task_id is None:
        print(f"{len(metabolites)} metabolites. Array a lancer : 0-{len(metabolites)-1}")
        print(f"MG={len(mg_cols)} features, Soil={len(soil_cols)} features")
        return

    if not (0 <= args.task_id < len(metabolites)):
        raise ValueError(f"task-id invalide {args.task_id} (0..{len(metabolites)-1})")

    met = metabolites[args.task_id]
    out_file = out_dir / f"metrics_task_{args.task_id:03d}.csv"
    print(f"[TASK {args.task_id}] {met} | n_repeats={args.n_repeats} | "
          f"n_perm={args.n_permutations}", flush=True)
    run_one_metabolite(X, Y_all, met, mg_cols, soil_cols,
                       n_repeats=args.n_repeats,
                       n_permutations=args.n_permutations,
                       out_file=out_file)
    print("[DONE]", out_file)


if __name__ == "__main__":
    main()
