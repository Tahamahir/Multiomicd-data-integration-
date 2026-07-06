from pathlib import Path
import json
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score

from sklearn.ensemble import (
RandomForestRegressor,
ExtraTreesRegressor
)

from sklearn.linear_model import ElasticNetCV

from sklearn.svm import SVR

from xgboost import XGBRegressor


ROOT=Path(".")

OUT=(
ROOT/
"10_analysis/outputs/phase31_best_model_per_mb"
)

OUT.mkdir(
parents=True,
exist_ok=True
)

MI_K=500

TOP=30


def load():

    X=pd.read_csv(
ROOT/
"10_analysis/outputs/phase3_soil_dedup/X_deduplicated.csv"
)

    Y=pd.read_csv(
ROOT/
"10_analysis/outputs/phase2_preprocessing_fixed/Y_ml_filtered_log1p.csv"
)

    metrics=pd.read_csv(
ROOT/
"10_analysis/outputs/phase26_tune_champion_late_sparsepca_rf/T266_mi500_spca75_a10_w7_rf_a_metrics_per_metabolite.csv"
)

    mets=(
metrics
.sort_values(
"r2",
ascending=False
)
.head(TOP)
["metabolite"]
.tolist()
)

    return X,Y[mets]


def split(X):

    soil=[]

    prefixes=[
"soil",
"ph",
"sand",
"moist",
"nh4",
"no3"
]

    for c in X.columns:

        if any(
p in str(c).lower()
for p
in prefixes
):

            soil.append(c)

    mg=[
c
for c
in X.columns
if c not in soil
]

    return mg,soil


def preprocess(X,y,mg):

    imp=SimpleImputer(
strategy="constant",
fill_value=0
)

    X=imp.fit_transform(
X[mg]
)

    sc=StandardScaler()

    X=sc.fit_transform(
X
)

    mi=mutual_info_regression(
X,
y,
random_state=42
)

    idx=np.argsort(
mi
)[::-1][:MI_K]

    return X[:,idx]


MODELS={

"RF":
RandomForestRegressor(
n_estimators=500,
max_features="sqrt",
min_samples_leaf=2,
random_state=42,
n_jobs=-1
),

"ET":
ExtraTreesRegressor(
n_estimators=500,
random_state=42,
n_jobs=-1
),

"XGB":
XGBRegressor(
n_estimators=300,
max_depth=4,
learning_rate=0.05,
subsample=0.8,
colsample_bytree=0.8,
n_jobs=-1
),

"ELASTIC":
ElasticNetCV(
cv=3,
random_state=42
),

"SVR":
SVR(
C=1,
epsilon=0.1
)

}


def evaluate(model,X,y):

    cv=RepeatedKFold(
n_splits=5,
n_repeats=10,
random_state=42
)

    pred=np.zeros(
len(y)
)

    for tr,te in cv.split(X):

        model.fit(
X[tr],
y[tr]
)

        pred[te]=model.predict(
X[te]
)

    return r2_score(
y,
pred
)


def main():

    start=time.time()

    X,Y=load()

    mg,soil=split(X)

    rows=[]

    for met in Y.columns:

        print()
        print(met)

        y=Y[met].values

        Xp=preprocess(
X,
y,
mg
)

        best=None

        best_r2=-999

        for name,model in MODELS.items():

            print(
name
)

            r2=evaluate(
model,
Xp,
y
)

            print(
round(
r2,
3
)
)

            rows.append({

"metabolite":
met,

"model":
name,

"r2":
r2

})

            if r2>best_r2:

                best_r2=r2

                best=name

        print(
"winner",
best
)

    df=pd.DataFrame(
rows
)

    df.to_csv(
OUT/
"all_results.csv",
index=False
)

    winners=(
df
.sort_values(
"r2",
ascending=False
)
.groupby(
"metabolite"
)
.first()
.reset_index()
)

    winners.to_csv(
OUT/
"best_model_per_metabolite.csv",
index=False
)

    summary={

"mean_r2":
float(
winners.r2.mean()
),

"median_r2":
float(
winners.r2.median()
),

"runtime_sec":
time.time()-start

}

    with open(
OUT/
"summary.json",
"w"
) as f:

        json.dump(
summary,
f,
indent=2
)

    print()
    print(summary)


if __name__=="__main__":

    main()
