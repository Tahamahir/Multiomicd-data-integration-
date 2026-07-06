# nul_windows.py — nul + FDR sur les 47 metabolites, en local
import glob, numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import SparsePCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from statsmodels.stats.multitest import multipletests
import warnings; warnings.filterwarnings("ignore")

# ---- fichiers dans le dossier courant ----
X = pd.read_csv("X_deduplicated.csv", low_memory=False)
Y = pd.read_csv("Y_ml_filtered_log1p.csv", low_memory=False)
champ = pd.read_csv("T266_mi500_spca75_a10_w7_rf_a_metrics_per_metabolite.csv")
metabolites = [m for m in champ["metabolite"].astype(str).unique() if m in Y.columns]

# ---- config champion ----
SOIL_PREF = ("soil_","chem__","psize__","moist__","nitrif__","denit__")
soil = [c for c in X.columns if any(str(c).lower().startswith(p) for p in SOIL_PREF)]
mg   = [c for c in X.columns if c not in soil]
MI_K, NC, WMG, WS, NSPLIT = 500, 75, 0.7, 0.3, 5
B, TOPN = 30, 47            # 30 permutations, tous les metabolites
print(f"MG={len(mg)}  Soil={len(soil)}  metabolites={len(metabolites)
                                                     }", flush=True)

def prep(Xtr, Xte, strat):
    imp = SimpleImputer(strategy="median") if strat=="median" else SimpleImputer(strategy="constant", fill_value=0)
    sc = StandardScaler()
    return sc.fit_transform(imp.fit_transform(Xtr)), sc.transform(imp.transform(Xte))

def mi_sel(Xtr, y, Xte):
    if Xtr.shape[1] <= MI_K: return Xtr, Xte
    s = np.nan_to_num(mutual_info_regression(Xtr, y, random_state=42))
    idx = np.argsort(s)[::-1][:MI_K]
    return Xtr[:,idx], Xte[:,idx]

def spca(Xtr, Xte):
    n = min(NC, Xtr.shape[0]-1, Xtr.shape[1])
    if n < 1: return Xtr, Xte
    r = SparsePCA(n_components=n, alpha=1, random_state=42, n_jobs=-1, max_iter=100)
    return r.fit_transform(Xtr), r.transform(Xte)

def rf(seed):
    return RandomForestRegressor(n_estimators=800, min_samples_leaf=2, max_features="sqrt", random_state=seed, n_jobs=-1)

def r2_mgsoil(y, seed):
    cv = KFold(n_splits=NSPLIT, shuffle=True, random_state=seed)
    pmg = np.zeros(len(y)); ps = np.zeros(len(y))
    for fid,(tr,te) in enumerate(cv.split(X)):
        ytr = y[tr]
        a,b = prep(X.iloc[tr][mg], X.iloc[te][mg], "zero"); a,b = mi_sel(a,ytr,b); a,b = spca(a,b)
        mm = rf(seed+fid); mm.fit(a,ytr); pmg[te] = mm.predict(b)
        c,d = prep(X.iloc[tr][soil], X.iloc[te][soil], "median")
        ms = rf(1000+seed+fid); ms.fit(c,ytr); ps[te] = ms.predict(d)
    return r2_score(y, WMG*pmg + WS*ps)

real = champ.copy(); real["metabolite"]=real["metabolite"].astype(str)
top = real.sort_values("r2", ascending=False).head(TOPN)
rows=[]
for i,(_,r) in enumerate(top.iterrows()):
    met=r["metabolite"]; y=Y[met].values.astype(float)
    r2r=r2_mgsoil(y, 42)
    nulls=[r2_mgsoil(np.random.default_rng(1000+b).permutation(y), 5000+7*b) for b in range(B)]
    p=(1+int(np.sum(np.array(nulls)>=r2r)))/(1+B)
    rows.append({"metabolite":met,"r2_real":round(r2r,3),"r2_null_mean":round(float(np.mean(nulls)),3),"p_perm":round(p,4)})
    print(f"[{i+1}/{len(top)}] r2={r2r:.3f}  null={np.mean(nulls):.3f}  p={p:.3f}", flush=True)

per=pd.DataFrame(rows)
rej,pf,_,_=multipletests(per["p_perm"], alpha=0.05, method="fdr_bh")
per["p_fdr"]=pf.round(4); per["sig_fdr005"]=rej
per.to_csv("fdr_result_47.csv", index=False)
print("\n===== RESULTAT ====="); print(per.to_string(index=False))
print(f"\nSignificatifs FDR<0.05 : {int(rej.sum())}/{len(per)}")