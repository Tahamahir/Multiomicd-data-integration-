from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("10_analysis/outputs/phase30_statistical_validation")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

matrix_path = OUT / "r2_matrix_metabolite_by_pipeline.csv"

df = pd.read_csv(matrix_path, index_col=0)

summary = pd.read_csv(OUT / "pipeline_summary.csv")
top_pipelines = (
    summary
    .sort_values("mean_r2", ascending=False)
    .head(20)["pipeline"]
    .tolist()
)

heat = df[top_pipelines].copy()

# Trier les métabolites selon leur performance moyenne
heat["mean_r2"] = heat.mean(axis=1)
heat = heat.sort_values("mean_r2", ascending=False)
heat = heat.drop(columns=["mean_r2"])

plt.figure(figsize=(14, 10))
plt.imshow(heat.values, aspect="auto")
plt.colorbar(label="R²")

plt.xticks(
    range(len(heat.columns)),
    heat.columns,
    rotation=60,
    ha="right",
    fontsize=8
)

plt.yticks(
    range(len(heat.index)),
    heat.index,
    fontsize=7
)

plt.xlabel("Pipeline")
plt.ylabel("Metabolite")
plt.title("R² heatmap across metabolites and top-performing pipelines")

plt.tight_layout()

plt.savefig(FIG / "heatmap_r2_metabolites_top20_pipelines.png", dpi=300, bbox_inches="tight")
plt.savefig(FIG / "heatmap_r2_metabolites_top20_pipelines.pdf", bbox_inches="tight")

print("Saved:")
print(FIG / "heatmap_r2_metabolites_top20_pipelines.png")
print(FIG / "heatmap_r2_metabolites_top20_pipelines.pdf")
