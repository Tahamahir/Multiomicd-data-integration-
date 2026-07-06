PHASE 21 OUTPUTS

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
