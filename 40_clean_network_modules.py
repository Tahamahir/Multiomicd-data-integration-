#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 40: Clean Microbiome-Metabolome Interaction Network
Amélioration de la visualisation des modules et des hubs biologiques
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from community import community_louvain
import os

# ================================
# Paramètres
# ================================
INPUT_FILE = "10_analysis/outputs/phase40_biomarkers/feature_metabolite_links.csv"
OUTPUT_FOLDER = "10_analysis/outputs/phase40_modules"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ================================
# Chargement des données
# ================================
df_links = pd.read_csv(INPUT_FILE)

# Création du graphe bipartite
G = nx.Graph()
for _, row in df_links.iterrows():
    G.add_node(row['feature'], bipartite='feature')
    G.add_node(row['metabolite'], bipartite='metabolite')
    G.add_edge(row['feature'], row['metabolite'], weight=row.get('weight', 1.0))

# ================================
# Détection des modules (Louvain)
# ================================
partition = community_louvain.best_partition(G, resolution=1.0, weight='weight')

# Attribuer la couleur selon le module
modules = list(set(partition.values()))
colors = list(mcolors.TABLEAU_COLORS.values()) * (len(modules)//len(mcolors.TABLEAU_COLORS)+1)
module_color_map = {m: c for m, c in zip(modules, colors)}
node_colors = [module_color_map[partition[n]] for n in G.nodes()]

# ================================
# Taille des nSuds selon le nombre de connections (hub score)
# ================================
degree_dict = dict(G.degree())
node_sizes = [degree_dict[n]*50 for n in G.nodes()]  # Ajustez le facteur d'échelle

# ================================
# Dessin du graphe réseau annoté
# ================================
plt.figure(figsize=(18, 14))
pos = nx.spring_layout(G, k=0.4, seed=42)  # Layout visuel
nx.draw_networkx_edges(G, pos, alpha=0.3)
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')
plt.title("Microbiome-Metabolome Interaction Network (Final Clean Modules)", fontsize=18)
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "network_modules_final.png"), dpi=300)
plt.close()

# ================================
# Top hubs (features/metabolites avec le plus de connexions)
# ================================
hub_scores = pd.DataFrame.from_dict(degree_dict, orient='index', columns=['degree'])
hub_scores = hub_scores.sort_values('degree', ascending=False).head(20)
plt.figure(figsize=(14,6))
hub_scores.plot(kind='bar', legend=False, color='mediumseagreen')
plt.xticks(rotation=90)
plt.ylabel("Hub Score (degree)")
plt.title("Top Biological Hubs (Microbiome-Metabolome Network)", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "top_hubs_final.png"), dpi=300)
plt.close()

# ================================
# Sauvegarde du réseau en GEXF pour Cytoscape/Gephi
# ================================
nx.write_gexf(G, os.path.join(OUTPUT_FOLDER, "clean_network_modules_final.gexf"))

print(f"[INFO] Figures et réseau sauvegardés dans : {OUTPUT_FOLDER}")
