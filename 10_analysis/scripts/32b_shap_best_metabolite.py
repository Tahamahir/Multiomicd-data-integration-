from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

ROOT = Path(".")
OUT = ROOT / "10_analysis/outputs/phase32_shap_best_metabolite"

OUT.mkdir(
parents=True,
exist_ok=True
)

FIG = OUT / "figures"

FIG.mkdir(
parents=True,
exist_ok=True
)

BEST = "C18_negative|IK:RGHHSNMVTDWUBI-UHFFFAOYSA-N"

MI_K = 500


def split_blocks(X):

    soil = [
        c for c in X.columns
        if (
            "soil" in str(c).lower()
            or "ph" in str(c).lower()
            or "nh4" in str(c).lower()
            or "no3" in str(c).lower()
            or "moist" in str(c).lower()
            or "sand" in str(c).lower()
        )
    ]

    mg = [
        c
        for c
        in X.columns
        if c not in soil
    ]

    return mg, soil


print("Loading")

X = pd.read_csv(
ROOT /
"10_analysis/outputs/phase3_soil_dedup/X_deduplicated.csv"
)

Y = pd.read_csv(
ROOT /
"10_analysis/outputs/phase2_preprocessing_fixed/Y_ml_filtered_log1p.csv"
)

y = Y[BEST].values

mg_cols, soil_cols = split_blocks(X)

print(
len(mg_cols),
len(soil_cols)
)

imp = SimpleImputer(
strategy="constant",
fill_value=0
)

Xmg = imp.fit_transform(
X[mg_cols]
)

sc = StandardScaler()

Xmg = sc.fit_transform(
Xmg
)

mi = mutual_info_regression(
Xmg,
y,
random_state=42
)

idx = np.argsort(
mi
)[::-1][:MI_K]

features = [
mg_cols[i]
for i
in idx
]

Xsel = Xmg[:, idx]

model = RandomForestRegressor(

n_estimators=500,

max_features="sqrt",

min_samples_leaf=2,

random_state=42,

n_jobs=-1

)

print("Training")

model.fit(
Xsel,
y
)

print("Computing SHAP")

explainer = shap.TreeExplainer(
model
)

sv = explainer.shap_values(
Xsel
)

Xdf = pd.DataFrame(
Xsel,
columns=features
)

# WATERFALL

sample = np.argmax(
y
)

plt.figure()

shap.plots.waterfall(

shap.Explanation(

values=sv[sample],

base_values=explainer.expected_value,

data=Xdf.iloc[sample],

feature_names=features

),

max_display=20,

show=False

)

plt.savefig(
FIG /
"waterfall_best_sample.png",
dpi=300,
bbox_inches="tight"
)

plt.close()

# TOP FEATURE

mean_abs = np.abs(
sv
).mean(
axis=0
)

best_idx = np.argmax(
mean_abs
)

best_feature = features[
best_idx
]

print(
"best feature",
best_feature
)

# DEPENDENCE

plt.figure()

shap.dependence_plot(

best_feature,

sv,

Xdf,

show=False

)

plt.savefig(

FIG /
"dependence_top_microbe.png",

dpi=300,

bbox_inches="tight"

)

plt.close()

# TRY PH

ph = None

for c in features:

    if "ph" in c.lower():

        ph = c

        break

if ph:

    plt.figure()

    shap.dependence_plot(

        ph,

        sv,

        Xdf,

        show=False

    )

    plt.savefig(

        FIG /
        "dependence_ph.png",

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

summary = {

"metabolite":
BEST,

"top_feature":
best_feature,

"sample":
int(sample)

}

pd.DataFrame(
[summary]
).to_csv(

OUT /
"summary.csv",

index=False

)

print()
print("Saved:")
print(FIG)
