from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUT="10_analysis/outputs/phase30_statistical_validation"
FIG=f"{OUT}/figures"

Path(FIG).mkdir(
parents=True,
exist_ok=True
)

matrix=pd.read_csv(
f"{OUT}/r2_matrix_metabolite_by_pipeline.csv",
index_col=0
)

summary=pd.read_csv(
f"{OUT}/pipeline_summary.csv"
)

top_pipelines=(
summary
.sort_values(
"mean_r2",
ascending=False
)
.head(10)
["pipeline"]
.tolist()
)

heat=matrix[
top_pipelines
]

heat["mean_r2"]=heat.mean(
axis=1
)

heat=(
heat
.sort_values(
"mean_r2",
ascending=False
)
.head(30)
)

heat=heat.drop(
columns=["mean_r2"]
)

rename_cols={}

for i,c in enumerate(
heat.columns
):
    rename_cols[c]=f"P{i+1}"

heat=heat.rename(
columns=rename_cols
)

sns.set_context(
"paper"
)

g=sns.clustermap(

heat,

cmap="viridis",

figsize=(12,12),

metric="euclidean",

method="average",

row_cluster=True,

col_cluster=True,

linewidths=0.1,

xticklabels=True,

yticklabels=True,

cbar_kws={

"label":"R²"

}

)

g.fig.suptitle(

"Clustered heatmap of top metabolites and pipelines",

y=1.02

)

g.savefig(

f"{FIG}/clustered_heatmap_top30_top10.png",

dpi=300,

bbox_inches="tight"

)

g.savefig(

f"{FIG}/clustered_heatmap_top30_top10.pdf",

bbox_inches="tight"

)

mapping=pd.DataFrame({

"Display":

list(rename_cols.values()),

"Pipeline":

list(rename_cols.keys())

})

mapping.to_csv(

f"{FIG}/pipeline_legend.csv",

index=False

)

print()
print("Saved:")
print(f"{FIG}/clustered_heatmap_top30_top10.png")
print(f"{FIG}/pipeline_legend.csv")
