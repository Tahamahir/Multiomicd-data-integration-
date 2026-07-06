# figures_ch6.py — genere 3 figures pour le chapitre 6 (legendes en francais)
import numpy as np, pandas as pd, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import SparsePCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import warnings; warnings.filterwarnings("ignore")

# ---- donnees ----
X = pd.read_csv("X_deduplicated.csv", low_memory=False)
Y = pd.read_csv("Y_ml_filtered_log1p.csv", low_memory=False)
champ = pd.read_csv("T266_mi500_spca75_a10_w7_rf_a_metrics_per_metabolite.csv")
champ["metabolite"] = champ["metabolite"].astype(str)
metabolites = [m for m in champ["metabolite"].unique() if m in Y.columns]

SOIL_PREF = ("soil_","chem__","psize__","moist__","nitrif__","denit__")
soil = [c for c in X.columns if any(str(c).lower().startswith(p) for p in SOIL_PREF)]
mg   = [c for c in X.columns if c not in soil]
MI_K, NC, WMG, WS, NSPLIT = 500, 75, 0.7, 0.3, 5

def prep(Xtr, Xte, strat):
    imp = SimpleImputer(strategy="median") if strat=="median" else SimpleImputer(strategy="constant", fill_value=0)
    sc = StandardScaler()
    return sc.fit_transform(imp.fit_transform(Xtr)), sc.transform(imp.transform(Xte))
def mi_sel(Xtr, y, Xte):
    if Xtr.shape[1] <= MI_K: return Xtr, Xte
    s = np.nan_to_num(mutual_info_regression(Xtr, y, random_state=42))
    idx = np.argsort(s)[::-1][:MI_K]; return Xtr[:,idx], Xte[:,idx]
def spca(Xtr, Xte):
    n = min(NC, Xtr.shape[0]-1, Xtr.shape[1])
    if n < 1: return Xtr, Xte
    r = SparsePCA(n_components=n, alpha=1, random_state=42, n_jobs=-1, max_iter=100)
    return r.fit_transform(Xtr), r.transform(Xte)
def rf(seed):
    return RandomForestRegressor(n_estimators=800, min_samples_leaf=2, max_features="sqrt", random_state=seed, n_jobs=-1)

def oof_pred(y, seed):
    """predictions hors-fold MG+Soil pour un y donne"""
    cv = KFold(n_splits=NSPLIT, shuffle=True, random_state=seed)
    pmg = np.zeros(len(y)); ps = np.zeros(len(y))
    for fid,(tr,te) in enumerate(cv.split(X)):
        ytr = y[tr]
        a,b = prep(X.iloc[tr][mg], X.iloc[te][mg], "zero"); a,b = mi_sel(a,ytr,b); a,b = spca(a,b)
        m1 = rf(seed+fid); m1.fit(a,ytr); pmg[te] = m1.predict(b)
        c,d = prep(X.iloc[tr][soil], X.iloc[te][soil], "median")
        m2 = rf(1000+seed+fid); m2.fit(c,ytr); ps[te] = m2.predict(d)
    return WMG*pmg + WS*ps

best = champ.sort_values("r2", ascending=False)["metabolite"].iloc[0]
print("Meilleur metabolite:", best, flush=True)

# ================= FIGURE 1 : predit vs observe =================
print("Figure 1 : predit vs observe...", flush=True)
y = Y[best].values.astype(float)
pred = oof_pred(y, 42)
r2 = r2_score(y, pred)
plt.figure(figsize=(6,6))
plt.scatter(y, pred, alpha=0.6, edgecolor="k", linewidth=0.3, color="#2b7bba")
lims = [min(y.min(),pred.min()), max(y.max(),pred.max())]
plt.plot(lims, lims, "r--", label="y = x")
plt.xlabel("Intensité observée (log1p)"); plt.ylabel("Intensité prédite (log1p)")
plt.title(f"Prédiction vs observation — métabolite le mieux prédit\n$R^2$ = {r2:.3f}")
plt.legend(); plt.tight_layout()
plt.savefig("fig_predit_vs_observe.png", dpi=200); plt.close()

# ================= FIGURE 2 : distribution nulle + R2 reel =================
print("Figure 2 : distribution nulle (30 permutations)...", flush=True)
B = 30
null_r2 = [r2_score(np.random.default_rng(1000+b).permutation(y),
           oof_pred(np.random.default_rng(1000+b).permutation(y), 5000+7*b)) for b in range(B)]
# note: on recompute proprement le nul (y permute -> pipeline -> r2 vs y permute)
null_r2 = []
for b in range(B):
    yp = np.random.default_rng(1000+b).permutation(y)
    null_r2.append(r2_score(yp, oof_pred(yp, 5000+7*b)))
plt.figure(figsize=(7,5))
plt.hist(null_r2, bins=12, color="#bbbbbb", edgecolor="k", label="Modèles nuls (cibles permutées)")
plt.axvline(r2, color="red", linewidth=2, label=f"Modèle réel ($R^2$={r2:.3f})")
plt.xlabel("$R^2$ (validation croisée)"); plt.ylabel("Nombre de permutations")
plt.title("Test de permutation — métabolite le mieux prédit")
plt.legend(); plt.tight_layout()
plt.savefig("fig_permutation_nulle.png", dpi=200); plt.close()

# ================= FIGURE 3 : R2 reel vs nul pour les top metabolites =================
print("Figure 3 : reel vs nul (top 15)...", flush=True)
top = champ.sort_values("r2", ascending=False).head(15)
labels = [m.split("|")[-1][:16] for m in top["metabolite"]]  # nom court
r2_real = top["r2"].values
# nul moyen approx: on prend une permutation par metabolite (rapide) -> moyenne
r2_null = []
for met in top["metabolite"]:
    yy = Y[met].values.astype(float)
    vals = [r2_score(np.random.default_rng(1000+b).permutation(yy),
            oof_pred(np.random.default_rng(1000+b).permutation(yy), 5000+7*b)) for b in range(5)]
    r2_null.append(np.mean(vals))
xpos = np.arange(len(labels))
plt.figure(figsize=(10,6))
plt.bar(xpos-0.2, r2_real, 0.4, label="Modèle réel", color="#2b7bba")
plt.bar(xpos+0.2, r2_null, 0.4, label="Modèle nul", color="#bbbbbb")
plt.axhline(0, color="k", linewidth=0.6)
plt.xticks(xpos, labels, rotation=60, ha="right", fontsize=7)
plt.ylabel("$R^2$ (validation croisée)")
plt.title("Performance réelle vs modèle nul (15 métabolites les mieux prédits)")
plt.legend(); plt.tight_layout()
plt.savefig("fig_reel_vs_nul_top15.png", dpi=200); plt.close()

print("\nTermine. 3 fichiers PNG generes :")
print("  fig_predit_vs_observe.png")
print("  fig_permutation_nulle.png")
print("  fig_reel_vs_nul_top15.png")