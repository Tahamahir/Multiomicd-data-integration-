import glob, numpy as np, pandas as pd
from statsmodels.stats.multitest import multipletests
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('m','10_analysis/scripts/42b_revalidate_full_fdr.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

B = 20        # permutations
TOPN = 10     # nb de metabolites testes (les mieux predits)
root = Path('.').resolve()
X, Y_all, mets, mg, soil = m.load_data(root)

# R2 reel : recupere depuis les fichiers existants (partie 'real' valide)
real = pd.concat([pd.read_csv(f) for f in glob.glob(str(root/m.OUT_SUBDIR/'metrics_task_*.csv'))])
real = real[(real.kind=='real') & (real.model=='MG_Soil_late')].drop_duplicates('metabolite')
top = real.sort_values('r2', ascending=False).head(TOPN)
print(f"Nul local sur {len(top)} metabolites, B={B} permutations\n", flush=True)

rows=[]
for i,(_,r) in enumerate(top.iterrows()):
    met=r['metabolite']; r2_real=float(r['r2'])
    y=Y_all[met].values.astype(float)
    null_r2=[m.metrics_over_repeats(X, np.random.default_rng(1000+b).permutation(y),
             mg, soil, base_seed=5000+7*b, n_repeats=1)['MG_Soil_late']['r2'] for b in range(B)]
    p=(1+int(np.sum(np.array(null_r2)>=r2_real)))/(1+B)
    rows.append({'metabolite':met,'r2_real':round(r2_real,3),
                 'r2_null_mean':round(float(np.mean(null_r2)),3),'p_perm':round(p,4)})
    print(f"[{i+1}/{len(top)}] r2={r2_real:.3f}  null={np.mean(null_r2):.3f}  p={p:.3f}", flush=True)

per=pd.DataFrame(rows)
rej,pfdr,_,_=multipletests(per['p_perm'], alpha=0.05, method='fdr_bh')
per['p_fdr']=pfdr.round(4); per['sig_fdr_005']=rej
per.to_csv(str(root/m.OUT_SUBDIR/'fdr_top_local.csv'), index=False)
print("\n===== RESULTAT =====")
print(per.to_string(index=False))
print(f"\nSignificatifs FDR<0.05 : {int(rej.sum())}/{len(per)}", flush=True)
