from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# FIX NUMERIC-LIKE COLUMNS IN X
# ============================================================

def main():
    repo_root = Path(__file__).resolve().parents[2]

    x_path = repo_root / "10_analysis/outputs/phase1_audit/X_aligned_phase1.csv"
    output_dir = repo_root / "10_analysis/outputs/phase2_preprocessing_fix"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading X...")
    X = pd.read_csv(x_path, low_memory=False)

    print("Detecting object columns...")
    object_cols = X.select_dtypes(include=['object']).columns.tolist()

    print(f"Found {len(object_cols)} object columns")

    converted_cols = []
    failed_cols = []

    for col in object_cols:
        try:
            converted = pd.to_numeric(X[col], errors='coerce')

            # Si au moins 80% des valeurs sont convertibles → on garde
            non_nan_ratio = converted.notna().mean()

            if non_nan_ratio > 0.8:
                X[col] = converted
                converted_cols.append(col)
            else:
                failed_cols.append(col)

        except Exception:
            failed_cols.append(col)

    print(f"Converted columns: {len(converted_cols)}")
    print(f"Failed columns   : {len(failed_cols)}")

    pd.DataFrame({"converted_columns": converted_cols}).to_csv(
        output_dir / "converted_columns.csv", index=False
    )

    pd.DataFrame({"failed_columns": failed_cols}).to_csv(
        output_dir / "failed_columns.csv", index=False
    )

    X.to_csv(output_dir / "X_fixed_numeric.csv", index=False)

    print("Done. Output saved.")


if __name__ == "__main__":
    main()