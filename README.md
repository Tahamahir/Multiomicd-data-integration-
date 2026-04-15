# PFE Multiomics Rhizosphere

Projet de PFE sur l’intégration multi-omics avec IA, centré sur la rhizosphère.

## Objectif
Prédire les profils métabolomiques à partir :
- des données métagénomiques
- des données de sol

## Données
Les données brutes lourdes ne sont pas versionnées dans Git.
Seuls les scripts, tables finales, dictionnaires et fichiers auditables sont suivis.

## Structure
- `05_tables/` : tables intermédiaires
- `07_ml_dataset/` : intégration X métagénomique
- `08_metabolomics_targets/` : construction de Y
- `09_final_ml/` : dataset final modélisation
- `10_model_ready/` : version figée et auditée