# fig_continu.py — predit vs observe sur un metabolite CONTINU (peu de zeros)
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import SparsePCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import warnings; warnings.filterwarnings("ignore")

X = pd.read_csv("X_deduplicated.csv", low_memory=False)
Y = pd.read_csv("Y_ml_filtered_log1p.csv", low_memory=False)
champ = pd.read_csv("T266_mi500_spca75_a10_w7_rf_a_metrics_per_metabolite.csv")
champ["metabolite"] = champ["metabolite"].astype(str)

SOIL_PREF = ("soil_","chem__","psize__","moist__","nitrif__","denit__")
soil = [c for c in X.columns if any(str(c).lower().startswith(p) for p in SOIL_PREF)]
mg   = [c for c in X.columns if c not in soil]
MI_K, NC, WMG, WS, NSPLIT = 500, 75, 0.7, 0.3, 5

# ---- fraction de zeros par metabolite ----
rows=[]
for m in champ["metabolite"]:
    if m in Y.columns:
        zf = float((Y[m].values == 0).mean())
        r2 = float(champ.loc[champ.metabolite==m,"r2"].iloc[0])
        rows.append({"metabolite":m,"zero_frac":zf,"r2":r2})
info = pd.DataFrame(rows)

# ---- selection : peu de zeros ET bon R2 ----
for seuil in [0.10, 0.15, 0.20, 0.30]:
    cand = info[info.zero_frac <= seuil].sort_values("r2", ascending=False)
    if len(cand) > 0:
        best = cand.iloc[0]["metabolite"]; zf = cand.iloc[0]["zero_frac"]
        print(f"Metabolite continu choisi (zeros={zf:.0%}, seuil={seuil:.0%}): {best}", flush=True)
        break
else:
    best = info.sort_values("zero_frac").iloc[0]["metabolite"]
    print("Aucun metabolite tres continu, on prend le moins creux:", best, flush=True)

def prep(Xtr, Xte, strat):
    imp = SimpleImputer(strategy="median") if strat=="median" else SimpleImputer(strategy="constant", fill_value=0)
    sc = StandardScaler(); return sc.fit_transform(imp.fit_transform(Xtr)), sc.transform(imp.transform(Xte))
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

y = Y[best].values.astype(float)
cv = KFold(n_splits=NSPLIT, shuffle=True, random_state=42)
pmg = np.zeros(len(y)); ps = np.zeros(len(y))
for fid,(tr,te) in enumerate(cv.split(X)):
    ytr=y[tr]
    a,b=prep(X.iloc[tr][mg],X.iloc[te][mg],"zero"); a,b=mi_sel(a,ytr,b); a,b=spca(a,b)
    m1=rf(42+fid); m1.fit(a,ytr); pmg[te]=m1.predict(b)
    c,d=prep(X.iloc[tr][soil],X.iloc[te][soil],"median")
    m2=rf(1042+fid); m2.fit(c,ytr); ps[te]=m2.predict(d)
pred = WMG*pmg + WS*ps
r2 = r2_score(y, pred)

plt.figure(figsize=(6,6))
plt.scatter(y, pred, alpha=0.6, edgecolor="k", linewidth=0.3, color="#2b7bba")
lims=[min(y.min(),pred.min()), max(y.max(),pred.max())]
plt.plot(lims, lims, "r--", label="y = x")
plt.xlabel("Intensité observée (log1p)"); plt.ylabel("Intensité prédite (log1p)")
plt.title(f"Prédiction vs observation — métabolite continu\n$R^2$ = {r2:.3f}")
plt.legend(); plt.tight_layout()
plt.savefig("fig_predit_vs_observe_continu.png", dpi=200); plt.close()
print(f"OK -> fig_predit_vs_observe_continu.png  (R2={r2:.3f})", flush=True)