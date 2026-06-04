from pathlib import Path
import pandas as pd
import numpy as np
import json
import time

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import SparsePCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


ROOT="."

OUT="10_analysis/outputs/phase28_nested_cv_light"

TOP=20

CANDIDATES=[
dict(name="T266",mi=500,comp=75,alpha=1.0,w=0.7),
dict(name="T271",mi=500,comp=75,alpha=1.0,w=0.8),
dict(name="T286",mi=500,comp=75,alpha=2.0,w=0.7),
]


def split_blocks(X):

    prefixes=[
    "soil_",
    "chem__",
    "psize__",
    "moist__",
    "nitrif__",
    "denit__"
    ]

    soil=[]

    for c in X.columns:

        s=str(c).lower()

        if any(s.startswith(p) for p in prefixes):

            soil.append(c)

    mg=[c for c in X.columns if c not in soil]

    return mg,soil


def load():

    X=pd.read_csv(
    "10_analysis/outputs/phase3_soil_dedup/X_deduplicated.csv"
    )

    Y=pd.read_csv(
    "10_analysis/outputs/phase2_preprocessing_fixed/Y_ml_filtered_log1p.csv"
    )

    best=pd.read_csv(
    "10_analysis/outputs/phase17_final_best_model_pipeline/best_model_per_metabolite_final.csv"
    )

    mets=[
    x
    for x
    in best
    .sort_values(
    "tuned_mean_r2",
    ascending=False
    )
    .metabolite
    .head(TOP)
    if x in Y.columns
    ]

    return X,Y[mets]


def fit_predict(
Xtr,
Xte,
ytr,
params
):

    imp=SimpleImputer()

    Xtr=imp.fit_transform(Xtr)

    Xte=imp.transform(Xte)

    sc=StandardScaler()

    Xtr=sc.fit_transform(Xtr)

    Xte=sc.transform(Xte)

    mi=mutual_info_regression(
    Xtr,
    ytr,
    random_state=42
    )

    idx=np.argsort(mi)[::-1][:params["mi"]]

    Xtr=Xtr[:,idx]

    Xte=Xte[:,idx]

    sp=SparsePCA(
    n_components=params["comp"],
    alpha=params["alpha"],
    random_state=42
    )

    Xtr=sp.fit_transform(Xtr)

    Xte=sp.transform(Xte)

    rf=RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
    )

    rf.fit(Xtr,ytr)

    return rf.predict(Xte)


def evaluate(
X,
Y,
params
):

    outer=KFold(
    3,
    shuffle=True,
    random_state=42
    )

    scores=[]

    for tr,te in outer.split(X):

        for m in Y.columns:

            y=Y[m].values

            pred=fit_predict(
            X.iloc[tr],
            X.iloc[te],
            y[tr],
            params
            )

            scores.append(
            r2_score(
            y[te],
            pred
            )
            )

    return np.mean(scores)


def main():

    t=time.time()

    X,Y=load()

    mg,soil=split_blocks(X)

    X=X[mg]

    rows=[]

    for c in CANDIDATES:

        print(c["name"])

        r=evaluate(
        X,
        Y,
        c
        )

        rows.append(
        {
        "pipeline":c["name"],
        "mean_r2":r
        }
        )

    df=pd.DataFrame(rows)

    df=df.sort_values(
    "mean_r2",
    ascending=False
    )

    Path(OUT).mkdir(
    parents=True,
    exist_ok=True
    )

    df.to_csv(
    f"{OUT}/summary.csv",
    index=False
    )

    print(df)

    print(
    "runtime",
    round(
    time.time()-t,
    1
    )
    )


if __name__=="__main__":

    main()
