from pathlib import Path
import pandas as pd
import numpy as np
import json


def find_sample_id_column(df: pd.DataFrame):
    candidates = [
        "sample_id_norm",
        "biosample_id_norm",
        "nmdc_biosample_id",
        "biosample_id",
        "sample_id",
        "id",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def global_zero_fraction(df: pd.DataFrame) -> float:
    if df.empty:
        return np.nan
    return float((df == 0).sum().sum() / (df.shape[0] * df.shape[1]))


def presence_fraction(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    return (df > 0).mean(axis=0)


def main():
    repo_root = Path(__file__).resolve().parents[2]

    x_path = repo_root / "10_analysis" / "outputs" / "phase2_preprocessing_fix" / "X_fixed_numeric.csv"
    y_path = repo_root / "10_analysis" / "outputs" / "phase1_audit" / "Y_aligned_phase1.csv"

    output_dir = repo_root / "10_analysis" / "outputs" / "phase2_preprocessing_fixed"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 2 FIXED - PREPARE ML MATRICES")
    print("=" * 70)
    print(f"X input   : {x_path}")
    print(f"Y input   : {y_path}")
    print(f"Output dir: {output_dir}")
    print()

    if not x_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {y_path}")

    X = pd.read_csv(x_path, low_memory=False)
    Y = pd.read_csv(y_path, low_memory=False)

    x_id_col = find_sample_id_column(X)
    y_id_col = find_sample_id_column(Y)

    if x_id_col is None:
        raise ValueError("Impossible de trouver la colonne ID dans X")
    if y_id_col is None:
        raise ValueError("Impossible de trouver la colonne ID dans Y")

    print(f"ID column in X: {x_id_col}")
    print(f"ID column in Y: {y_id_col}")
    print()

    sample_ids = pd.DataFrame({
        "sample_id": X[x_id_col].astype(str).str.strip()
    })

    y_ids = Y[y_id_col].astype(str).str.strip()
    same_order = bool((sample_ids["sample_id"].reset_index(drop=True) == y_ids.reset_index(drop=True)).all())

    print(f"Same order between X and Y: {same_order}")
    print()

    if not same_order:
        raise ValueError("X et Y ne sont pas dans le même ordre. Arrêt par sécurité.")

    X_num = X.select_dtypes(include=[np.number]).copy()
    Y_num = Y.select_dtypes(include=[np.number]).copy()

    print(f"X original shape        : {X.shape}")
    print(f"Y original shape        : {Y.shape}")
    print(f"X numeric-only shape    : {X_num.shape}")
    print(f"Y numeric-only shape    : {Y_num.shape}")
    print()

    x_presence = presence_fraction(X_num)
    y_presence = presence_fraction(Y_num)

    x_min_presence = 0.05
    y_min_presence = 0.10

    x_keep_cols = x_presence[x_presence >= x_min_presence].index.tolist()
    y_keep_cols = y_presence[y_presence >= y_min_presence].index.tolist()

    X_filt = X_num[x_keep_cols].copy()
    Y_filt = Y_num[y_keep_cols].copy()

    print(f"X columns kept after presence filter ({x_min_presence*100:.0f}%): {len(x_keep_cols)}")
    print(f"Y columns kept after presence filter ({y_min_presence*100:.0f}%): {len(y_keep_cols)}")
    print()

    x_var = X_filt.var()
    y_var = Y_filt.var()

    x_non_constant = x_var[x_var > 0].index.tolist()
    y_non_constant = y_var[y_var > 0].index.tolist()

    X_filt = X_filt[x_non_constant].copy()
    Y_filt = Y_filt[y_non_constant].copy()

    print(f"X shape after removing constant cols: {X_filt.shape}")
    print(f"Y shape after removing constant cols: {Y_filt.shape}")
    print()

    Y_log = np.log1p(Y_filt)

    summary = {
        "n_samples": int(len(sample_ids)),
        "x_original_total_columns": int(X.shape[1]),
        "y_original_total_columns": int(Y.shape[1]),
        "x_numeric_columns": int(X_num.shape[1]),
        "y_numeric_columns": int(Y_num.shape[1]),
        "x_columns_after_presence_filter": int(X_filt.shape[1]),
        "y_columns_after_presence_filter": int(Y_filt.shape[1]),
        "x_global_zero_fraction_before_filter": global_zero_fraction(X_num),
        "y_global_zero_fraction_before_filter": global_zero_fraction(Y_num),
        "x_global_zero_fraction_after_filter": global_zero_fraction(X_filt),
        "y_global_zero_fraction_after_filter": global_zero_fraction(Y_filt),
        "same_order_x_y": bool(same_order),
        "x_presence_threshold": x_min_presence,
        "y_presence_threshold": y_min_presence,
        "y_log_transform_applied": True,
    }

    sample_ids.to_csv(output_dir / "sample_ids.csv", index=False)

    X_num.to_csv(output_dir / "X_numeric_raw.csv", index=False)
    Y_num.to_csv(output_dir / "Y_numeric_raw.csv", index=False)

    X_filt.to_csv(output_dir / "X_ml_filtered.csv", index=False)
    Y_filt.to_csv(output_dir / "Y_ml_filtered_untransformed.csv", index=False)
    pd.DataFrame(Y_log, columns=Y_filt.columns).to_csv(output_dir / "Y_ml_filtered_log1p.csv", index=False)

    pd.DataFrame({"column": X_filt.columns}).to_csv(output_dir / "X_kept_columns.csv", index=False)
    pd.DataFrame({"column": Y_filt.columns}).to_csv(output_dir / "Y_kept_columns.csv", index=False)

    x_removed = [c for c in X_num.columns if c not in X_filt.columns]
    y_removed = [c for c in Y_num.columns if c not in Y_filt.columns]

    pd.DataFrame({"column": x_removed}).to_csv(output_dir / "X_removed_columns.csv", index=False)
    pd.DataFrame({"column": y_removed}).to_csv(output_dir / "Y_removed_columns.csv", index=False)

    x_presence.sort_values().reset_index().rename(
        columns={"index": "column", 0: "presence_fraction"}
    ).to_csv(output_dir / "X_presence_fraction.csv", index=False)

    y_presence.sort_values().reset_index().rename(
        columns={"index": "column", 0: "presence_fraction"}
    ).to_csv(output_dir / "Y_presence_fraction.csv", index=False)

    with open(output_dir / "phase2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    pd.DataFrame([summary]).to_csv(output_dir / "phase2_summary.csv", index=False)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Number of samples                    : {len(sample_ids)}")
    print(f"X numeric raw shape                  : {X_num.shape}")
    print(f"Y numeric raw shape                  : {Y_num.shape}")
    print(f"X filtered shape                     : {X_filt.shape}")
    print(f"Y filtered shape                     : {Y_filt.shape}")
    print(f"X zero fraction before filter        : {summary['x_global_zero_fraction_before_filter']:.4f}")
    print(f"Y zero fraction before filter        : {summary['y_global_zero_fraction_before_filter']:.4f}")
    print(f"X zero fraction after filter         : {summary['x_global_zero_fraction_after_filter']:.4f}")
    print(f"Y zero fraction after filter         : {summary['y_global_zero_fraction_after_filter']:.4f}")
    print(f"Y log1p applied                      : {summary['y_log_transform_applied']}")
    print()
    print("Main output files:")
    print(output_dir / "sample_ids.csv")
    print(output_dir / "X_ml_filtered.csv")
    print(output_dir / "Y_ml_filtered_untransformed.csv")
    print(output_dir / "Y_ml_filtered_log1p.csv")
    print()
    print("Phase 2 fixed completed successfully.")


if __name__ == "__main__":
    main()