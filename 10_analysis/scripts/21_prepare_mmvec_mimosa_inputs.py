from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr


# ============================================================
# PHASE 21 - MMVEC / MIMOSA2 PREPARATION + CO-OCCURRENCE CHECK
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


def detect_soil_columns(columns):
    prefixes = ["soil_", "chem__", "psize_", "moist_", "nitrif_", "denit_"]
    return [c for c in columns if any(str(c).lower().strip().startswith(p) for p in prefixes)]


def prevalence(df):
    return (df > 0).mean(axis=0)


def main():
    repo_root = Path(__file__).resolve().parents[2]

    x_path = repo_root / "10_analysis" / "outputs" / "phase3_soil_dedup" / "X_deduplicated.csv"
    y_log_path = repo_root / "10_analysis" / "outputs" / "phase2_preprocessing_fixed" / "Y_ml_filtered_log1p.csv"
    rel_path = repo_root / "10_analysis" / "outputs" / "phase17_final_best_model_pipeline" / "species_mb_relationships_interpretable_optimized.csv"

    output_dir = repo_root / "10_analysis" / "outputs" / "phase21_mmvec_mimosa_preparation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 21 - MMVEC / MIMOSA2 PREPARATION")
    print("=" * 70)

    for p in [x_path, y_log_path, rel_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    X = pd.read_csv(x_path, low_memory=False)
    Y_log = pd.read_csv(y_log_path, low_memory=False)
    rel = pd.read_csv(rel_path, low_memory=False)

    X = impute_X(X)

    soil_cols = detect_soil_columns(X.columns)
    mg_cols = [c for c in X.columns if c not in soil_cols]

    X_mg = X[mg_cols].copy()
    Y_original = np.expm1(Y_log)
    Y_original[Y_original < 0] = 0

    print(f"X total shape : {X.shape}")
    print(f"MG shape      : {X_mg.shape}")
    print(f"Y shape       : {Y_original.shape}")
    print(f"Relationships : {len(rel)}")
    print()

    # ------------------------------------------------------------
    # 1. Filtrage raisonnable pour MMvec / co-occurrence
    # ------------------------------------------------------------
    mg_prev = prevalence(X_mg)
    y_prev = prevalence(Y_original)

    X_mg_f = X_mg.loc[:, mg_prev >= 0.10].copy()
    Y_f = Y_original.loc[:, y_prev >= 0.10].copy()

    # Limiter la taille pour MMvec / analyses exploratoires
    max_mg = 500
    max_mb = 150

    mg_var = X_mg_f.var(axis=0).sort_values(ascending=False)
    mb_var = Y_f.var(axis=0).sort_values(ascending=False)

    X_mg_f = X_mg_f[mg_var.head(max_mg).index]
    Y_f = Y_f[mb_var.head(max_mb).index]

    print(f"Filtered MG shape : {X_mg_f.shape}")
    print(f"Filtered MB shape : {Y_f.shape}")

    # ------------------------------------------------------------
    # 2. Export format MMvec-like
    # MMvec attend généralement features x samples
    # ------------------------------------------------------------
    mmvec_microbes = X_mg_f.T.copy()
    mmvec_metabolites = Y_f.T.copy()

    mmvec_microbes.index.name = "feature_id"
    mmvec_metabolites.index.name = "feature_id"

    mmvec_microbes.to_csv(output_dir / "mmvec_microbes_features_by_samples.tsv", sep="\t")
    mmvec_metabolites.to_csv(output_dir / "mmvec_metabolites_features_by_samples.tsv", sep="\t")

    # Versions samples x features aussi utiles
    X_mg_f.to_csv(output_dir / "microbes_samples_by_features.csv", index=False)
    Y_f.to_csv(output_dir / "metabolites_samples_by_features.csv", index=False)

    # ------------------------------------------------------------
    # 3. Export MIMOSA2-like
    # Attention : MIMOSA2 demande souvent mapping taxons/fonctions/metabolites
    # ------------------------------------------------------------
    X_mg_f.to_csv(output_dir / "mimosa2_taxa_abundance_samples_by_taxa.csv", index=False)
    Y_f.to_csv(output_dir / "mimosa2_metabolites_samples_by_metabolites.csv", index=False)

    config_text = """# MIMOSA2 configuration template
# A COMPLETER selon les chemins et formats attendus par MIMOSA2.
# MIMOSA2 nécessite généralement :
# - microbiome/taxonomy abundance file
# - metabolomics file
# - reference database / metabolic potential
# - metabolite identifiers mappables

microbiome_file=mimosa2_taxa_abundance_samples_by_taxa.csv
metabolome_file=mimosa2_metabolites_samples_by_metabolites.csv

# TODO:
# reference_data_path=/path/to/mimosa2/reference/data
# metabolite_id_type=inchikey_or_hmdb_or_kegg
# taxonomy_id_type=species_or_genus
"""
    with open(output_dir / "mimosa2_configuration_template.txt", "w", encoding="utf-8") as f:
        f.write(config_text)

    # ------------------------------------------------------------
    # 4. Validation co-occurrence des relations déjà trouvées
    # ------------------------------------------------------------
    validation_rows = []

    for _, row in rel.iterrows():
        mg = row["mg_feature"]
        mb = row["metabolite"]

        if mg not in X_mg.columns or mb not in Y_original.columns:
            continue

        corr, pval = spearmanr(X_mg[mg].values, Y_original[mb].values)

        validation_rows.append({
            "metabolite": mb,
            "mg_feature": mg,
            "model_importance": row.get("importance", np.nan),
            "model_spearman_corr": row.get("spearman_corr", np.nan),
            "recomputed_spearman_corr": corr,
            "recomputed_spearman_pvalue": pval,
            "abs_recomputed_spearman": abs(corr) if not pd.isna(corr) else np.nan,
            "putative_role": row.get("putative_role", "unknown"),
            "confidence": row.get("confidence", "unknown"),
            "best_model": row.get("best_model", "unknown"),
            "cv_r2": row.get("cv_r2", np.nan),
        })

    validation_df = pd.DataFrame(validation_rows)
    validation_df = validation_df.sort_values("abs_recomputed_spearman", ascending=False)
    validation_df.to_csv(output_dir / "relationship_validation_spearman.csv", index=False)

    # ------------------------------------------------------------
    # 5. Top co-occurrence pairs MG-MB sur matrices filtrées
    # ------------------------------------------------------------
    co_rows = []

    mg_subset = X_mg_f.columns.tolist()
    mb_subset = Y_f.columns.tolist()

    for mb in mb_subset:
        y = Y_f[mb].values

        for mg in mg_subset:
            x = X_mg_f[mg].values
            corr, pval = spearmanr(x, y)

            if pd.isna(corr):
                continue

            co_rows.append({
                "metabolite": mb,
                "mg_feature": mg,
                "spearman_corr": corr,
                "spearman_pvalue": pval,
                "abs_spearman_corr": abs(corr),
                "direction": "positive" if corr > 0 else "negative"
            })

    co_df = pd.DataFrame(co_rows)
    co_df = co_df.sort_values("abs_spearman_corr", ascending=False)
    co_df.head(5000).to_csv(output_dir / "top_cooccurrence_pairs_spearman.csv", index=False)

    # ------------------------------------------------------------
    # 6. Overlap entre réseau ML et top co-occurrence
    # ------------------------------------------------------------
    ml_pairs = set(zip(validation_df["mg_feature"], validation_df["metabolite"]))
    top_co_pairs = set(zip(co_df.head(1000)["mg_feature"], co_df.head(1000)["metabolite"]))

    overlap = ml_pairs.intersection(top_co_pairs)

    overlap_summary = {
        "n_ml_relationship_pairs": len(ml_pairs),
        "n_top_cooccurrence_pairs": len(top_co_pairs),
        "n_overlap": len(overlap),
        "overlap_fraction_vs_ml": len(overlap) / len(ml_pairs) if len(ml_pairs) > 0 else np.nan,
        "overlap_fraction_vs_top_cooccurrence": len(overlap) / len(top_co_pairs) if len(top_co_pairs) > 0 else np.nan,
    }

    pd.DataFrame([overlap_summary]).to_csv(output_dir / "overlap_summary.csv", index=False)

    overlap_df = pd.DataFrame(list(overlap), columns=["mg_feature", "metabolite"])
    overlap_df.to_csv(output_dir / "overlap_ml_vs_cooccurrence_pairs.csv", index=False)

    # ------------------------------------------------------------
    # 7. README
    # ------------------------------------------------------------
    readme = f"""PHASE 21 OUTPUTS

Objectif:
Préparer les fichiers pour une analyse MMvec/MIMOSA2 et comparer les associations ML avec une co-occurrence Spearman.

Fichiers importants:
- mmvec_microbes_features_by_samples.tsv
- mmvec_metabolites_features_by_samples.tsv
- mimosa2_taxa_abundance_samples_by_taxa.csv
- mimosa2_metabolites_samples_by_metabolites.csv
- mimosa2_configuration_template.txt
- relationship_validation_spearman.csv
- top_cooccurrence_pairs_spearman.csv
- overlap_summary.csv

Attention:
- MMvec mesure des co-occurrences microbe-metabolite.
- MIMOSA2 vise une interprétation métabolique mécanistique, mais nécessite des mappings et des bases de référence.
- Les résultats ne remplacent pas le modèle prédictif RF/XGB/ET.
- Ils servent à valider/interpréter les relations MG-MB.
"""
    with open(output_dir / "README_phase21.txt", "w", encoding="utf-8") as f:
        f.write(readme)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Filtered MG features           : {X_mg_f.shape[1]}")
    print(f"Filtered metabolites           : {Y_f.shape[1]}")
    print(f"Validated ML relationships     : {len(validation_df)}")
    print(f"Top co-occurrence pairs        : {len(co_df)}")
    print(f"Overlap ML vs top1000 co-occ   : {overlap_summary['n_overlap']}")
    print(f"Overlap fraction vs ML         : {overlap_summary['overlap_fraction_vs_ml']:.4f}")
    print()
    print("Main outputs:")
    print(output_dir / "mmvec_microbes_features_by_samples.tsv")
    print(output_dir / "mmvec_metabolites_features_by_samples.tsv")
    print(output_dir / "mimosa2_configuration_template.txt")
    print(output_dir / "relationship_validation_spearman.csv")
    print(output_dir / "top_cooccurrence_pairs_spearman.csv")
    print(output_dir / "overlap_summary.csv")
    print()
    print("Phase 21 completed successfully.")


if __name__ == "__main__":
    main()