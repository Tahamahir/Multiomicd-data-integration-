import pandas as pd
import numpy as np
from pathlib import Path

X_FILE = "X_model_with_id.csv"
Y_FILE = "Y_model_with_id.csv"
META_FILE = "metadata_model.csv"
Y_DICT_FILE = "Y_dictionary.csv"

ID_COL = "biosample_id_norm"

OUTDIR = Path("audit_reports")
OUTDIR.mkdir(exist_ok=True)

def safe_read(path):
    return pd.read_csv(path)

def pct(a, b):
    return 0.0 if b == 0 else 100.0 * a / b

def most_common_ratio(s: pd.Series) -> float:
    if len(s) == 0:
        return np.nan
    vc = s.value_counts(dropna=False)
    if len(vc) == 0:
        return np.nan
    return vc.iloc[0] / len(s)

# -----------------------------
# Load
# -----------------------------
X = safe_read(X_FILE)
Y = safe_read(Y_FILE)
META = safe_read(META_FILE)

try:
    YDICT = safe_read(Y_DICT_FILE)
except Exception:
    YDICT = None

for df in [X, Y, META]:
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()

if ID_COL not in X.columns or ID_COL not in Y.columns:
    raise ValueError(f"{ID_COL} doit être présent dans X_model_with_id.csv et Y_model_with_id.csv")

# -----------------------------
# Alignment audit
# -----------------------------
alignment_ok = X[ID_COL].equals(Y[ID_COL])

alignment_df = pd.DataFrame({
    "row_index": range(len(X)),
    "x_id": X[ID_COL],
    "y_id": Y[ID_COL],
})
alignment_df["same_id"] = alignment_df["x_id"] == alignment_df["y_id"]
alignment_df.to_csv(OUTDIR / "alignment_check.csv", index=False)

# -----------------------------
# Row-level audit
# -----------------------------
x_feat_cols = [c for c in X.columns if c != ID_COL]
y_feat_cols = [c for c in Y.columns if c != ID_COL]

row_audit = pd.DataFrame({
    ID_COL: X[ID_COL],
    "x_n_missing": X[x_feat_cols].isna().sum(axis=1),
    "x_pct_missing": X[x_feat_cols].isna().mean(axis=1) * 100,
    "y_n_nonzero": (Y[y_feat_cols] > 0).sum(axis=1),
    "y_n_zero": (Y[y_feat_cols] == 0).sum(axis=1),
    "y_pct_nonzero": (Y[y_feat_cols] > 0).mean(axis=1) * 100,
})

if ID_COL in META.columns:
    keep_meta = [c for c in ["site", "host_id", "genotype", "replicate"] if c in META.columns]
    if keep_meta:
        row_audit = row_audit.merge(META[[ID_COL] + keep_meta], on=ID_COL, how="left")

row_audit.to_csv(OUTDIR / "row_audit.csv", index=False)

# -----------------------------
# Column-level X audit
# -----------------------------
x_rows = []
for c in x_feat_cols:
    s = X[c]
    n = len(s)
    n_missing = s.isna().sum()
    n_non_missing = n - n_missing

    numeric = pd.api.types.is_numeric_dtype(s)
    if numeric:
        s_num = pd.to_numeric(s, errors="coerce")
        n_zero = (s_num.fillna(np.nan) == 0).sum()
        n_nonzero = (s_num.fillna(0) != 0).sum()
        variance = s_num.var(skipna=True)
        mean = s_num.mean(skipna=True)
        median = s_num.median(skipna=True)
    else:
        n_zero = np.nan
        n_nonzero = np.nan
        variance = np.nan
        mean = np.nan
        median = np.nan

    x_rows.append({
        "feature": c,
        "dtype": str(s.dtype),
        "n_missing": int(n_missing),
        "pct_missing": pct(n_missing, n),
        "n_non_missing": int(n_non_missing),
        "n_unique": int(s.nunique(dropna=False)),
        "most_common_ratio": most_common_ratio(s),
        "n_zero": n_zero,
        "pct_zero": pct(n_zero, n) if pd.notna(n_zero) else np.nan,
        "n_nonzero": n_nonzero,
        "pct_nonzero": pct(n_nonzero, n) if pd.notna(n_nonzero) else np.nan,
        "mean": mean,
        "median": median,
        "variance": variance,
    })

x_audit = pd.DataFrame(x_rows)

# flags de recommandation
x_audit["flag_missing_gt_40pct"] = x_audit["pct_missing"] > 40
x_audit["flag_zero_gt_95pct"] = x_audit["pct_zero"] > 95
x_audit["flag_near_constant"] = x_audit["most_common_ratio"] > 0.95
x_audit["flag_low_variance"] = x_audit["variance"].fillna(0) == 0

x_audit.to_csv(OUTDIR / "X_column_audit.csv", index=False)

# -----------------------------
# Column-level Y audit
# -----------------------------
y_rows = []
for c in y_feat_cols:
    s = pd.to_numeric(Y[c], errors="coerce")
    n = len(s)
    n_missing = s.isna().sum()
    n_zero = (s.fillna(np.nan) == 0).sum()
    n_nonzero = (s.fillna(0) != 0).sum()

    y_rows.append({
        "target": c,
        "n_missing": int(n_missing),
        "pct_missing": pct(n_missing, n),
        "n_zero": int(n_zero),
        "pct_zero": pct(n_zero, n),
        "n_nonzero": int(n_nonzero),
        "pct_nonzero": pct(n_nonzero, n),
        "n_samples_present": int(n_nonzero),
        "mean": s.mean(skipna=True),
        "median": s.median(skipna=True),
        "variance": s.var(skipna=True),
    })

y_audit = pd.DataFrame(y_rows)

# merge avec Y_dictionary si disponible

if YDICT is not None and "metabolite_key" in YDICT.columns:
    # éviter conflit avec les colonnes déjà calculées dans y_audit
    cols_to_drop = [c for c in ["n_samples_present"] if c in YDICT.columns]
    YDICT_merge = YDICT.drop(columns=cols_to_drop, errors="ignore").copy()

    y_audit = y_audit.merge(
        YDICT_merge,
        left_on="target",
        right_on="metabolite_key",
        how="left"
    )
# flags utiles
y_audit["flag_present_lt_10_samples"] = y_audit["n_samples_present"] < 10
y_audit["flag_present_lt_20_samples"] = y_audit["n_samples_present"] < 20
y_audit["flag_zero_gt_95pct"] = y_audit["pct_zero"] > 95

y_audit.to_csv(OUTDIR / "Y_column_audit.csv", index=False)

# -----------------------------
# Recommended subsets
# -----------------------------
# X: recommandations audit, sans suppression automatique du dataset original
x_recommended = x_audit[
    (~x_audit["flag_missing_gt_40pct"]) &
    (~x_audit["flag_zero_gt_95pct"]) &
    (~x_audit["flag_near_constant"]) &
    (~x_audit["flag_low_variance"])
].copy()
x_recommended.to_csv(OUTDIR / "X_recommended_features.csv", index=False)

# Y: cibles intéressantes pour premier modèle
# critère simple: présentes dans >=20 échantillons
y_recommended = y_audit[y_audit["n_samples_present"] >= 20].copy()
y_recommended.to_csv(OUTDIR / "Y_recommended_targets.csv", index=False)

# -----------------------------
# Global report
# -----------------------------
report_lines = []
report_lines.append("=== AUDIT MISSINGNESS & SPARSITY ===")
report_lines.append(f"Alignment X/Y OK: {alignment_ok}")
report_lines.append(f"N samples X: {len(X)}")
report_lines.append(f"N samples Y: {len(Y)}")
report_lines.append(f"N X features: {len(x_feat_cols)}")
report_lines.append(f"N Y targets: {len(y_feat_cols)}")
report_lines.append("")
report_lines.append("=== X SUMMARY ===")
report_lines.append(f"X features with >40% missing: {int(x_audit['flag_missing_gt_40pct'].sum())}")
report_lines.append(f"X features with >95% zeros: {int(x_audit['flag_zero_gt_95pct'].sum())}")
report_lines.append(f"X near-constant features: {int(x_audit['flag_near_constant'].sum())}")
report_lines.append(f"X recommended features after audit: {len(x_recommended)}")
report_lines.append("")
report_lines.append("=== Y SUMMARY ===")
report_lines.append(f"Y targets with <10 present samples: {int(y_audit['flag_present_lt_10_samples'].sum())}")
report_lines.append(f"Y targets with <20 present samples: {int(y_audit['flag_present_lt_20_samples'].sum())}")
report_lines.append(f"Y targets with >95% zeros: {int(y_audit['flag_zero_gt_95pct'].sum())}")
report_lines.append(f"Y recommended targets (>=20 present samples): {len(y_recommended)}")
report_lines.append("")
report_lines.append("=== ROW SUMMARY ===")
report_lines.append(f"Mean X % missing per sample: {row_audit['x_pct_missing'].mean():.4f}")
report_lines.append(f"Max X % missing per sample: {row_audit['x_pct_missing'].max():.4f}")
report_lines.append(f"Mean Y nonzero targets per sample: {row_audit['y_n_nonzero'].mean():.2f}")
report_lines.append(f"Min Y nonzero targets per sample: {row_audit['y_n_nonzero'].min()}")
report_lines.append(f"Max Y nonzero targets per sample: {row_audit['y_n_nonzero'].max()}")

report_path = OUTDIR / "audit_summary.txt"
report_path.write_text("\n".join(report_lines), encoding="utf-8")

print(f"Fichier créé : {OUTDIR / 'alignment_check.csv'}")
print(f"Fichier créé : {OUTDIR / 'row_audit.csv'}")
print(f"Fichier créé : {OUTDIR / 'X_column_audit.csv'}")
print(f"Fichier créé : {OUTDIR / 'Y_column_audit.csv'}")
print(f"Fichier créé : {OUTDIR / 'X_recommended_features.csv'}")
print(f"Fichier créé : {OUTDIR / 'Y_recommended_targets.csv'}")
print(f"Fichier créé : {OUTDIR / 'audit_summary.txt'}")
print("")
print(f"Alignment X/Y OK: {alignment_ok}")
print(f"N X features: {len(x_feat_cols)}")
print(f"N Y targets: {len(y_feat_cols)}")
print(f"X features with >40% missing: {int(x_audit['flag_missing_gt_40pct'].sum())}")
print(f"X features with >95% zeros: {int(x_audit['flag_zero_gt_95pct'].sum())}")
print(f"X near-constant features: {int(x_audit['flag_near_constant'].sum())}")
print(f"Y targets with <20 present samples: {int(y_audit['flag_present_lt_20_samples'].sum())}")
print(f"Y recommended targets (>=20 present samples): {len(y_recommended)}")
