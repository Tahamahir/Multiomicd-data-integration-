# plot_annotation_classes.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("annotation_class_summary.csv")

# garder l'ordre décroissant déjà généré
plt.figure(figsize=(10, 5))
plt.bar(df["classe_chimique"], df["nombre"])
plt.xticks(rotation=35, ha="right")
plt.ylabel("Nombre de métabolites")
plt.xlabel("Classe chimique")
plt.title("Répartition des métabolites annotés par classe chimique")
plt.tight_layout()

plt.savefig("fig_annotation_classes.png", dpi=300, bbox_inches="tight")
plt.savefig("fig_annotation_classes.pdf", bbox_inches="tight")
plt.close()

print("Figure générée : fig_annotation_classes.png")