import pandas as pd

X_FILE = "X_model_with_id.csv"
Y_FILE = "Y_model_with_id.csv"

X_AUDIT_FILE = "audit_reports/X_recommended_features.csv"
Y_REC_FILE = "audit_reports/Y_recommended_targets.csv"

ID_COL = "biosample_id_norm"

X = pd.read_csv(X_FILE)
Y = pd.read_csv(Y_FILE)
X_audit = pd.read_csv(X_AUDIT_FILE)
Y_rec = pd.read_csv(Y_REC_FILE)

for df in [X, Y, X_audit, Y_rec]:
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()

# garder seulement les features X recommandées
x_keep = X_audit["feature"].tolist()
x_keep = [c for c in x_keep if c in X.columns and c != ID_COL]

X_reduced = X[[ID_COL] + x_keep].copy()

# garder seulement les targets Y recommandées
y_keep = Y_rec["target"].tolist()
y_keep = [c for c in y_keep if c in Y.columns and c != ID_COL]

Y_reduced = Y[[ID_COL] + y_keep].copy()

# tri explicite et audit d'alignement
X_reduced = X_reduced.sort_values(ID_COL).reset_index(drop=True)
Y_reduced = Y_reduced.sort_values(ID_COL).reset_index(drop=True)

if not X_reduced[ID_COL].equals(Y_reduced[ID_COL]):
    raise ValueError("X_reduced et Y_reduced ne sont pas alignés sur biosample_id_norm")

# sauvegarde versions avec id
X_reduced.to_csv("X_model_reduced_with_id.csv", index=False)
Y_reduced.to_csv("Y_model_recommended_with_id.csv", index=False)

# sauvegarde versions sans id
X_reduced.drop(columns=[ID_COL]).to_csv("X_model_reduced.csv", index=False)
Y_reduced.drop(columns=[ID_COL]).to_csv("Y_model_recommended.csv", index=False)

print("Fichier créé : X_model_reduced_with_id.csv")
print("Fichier créé : Y_model_recommended_with_id.csv")
print("Fichier créé : X_model_reduced.csv")
print("Fichier créé : Y_model_recommended.csv")
print(f"Lignes X : {X_reduced.shape[0]}")
print(f"Colonnes X avec id : {X_reduced.shape[1]}")
print(f"Lignes Y : {Y_reduced.shape[0]}")
print(f"Colonnes Y avec id : {Y_reduced.shape[1]}")
print(f"Nb features X retenues : {len(x_keep)}")
print(f"Nb targets Y retenues : {len(y_keep)}")