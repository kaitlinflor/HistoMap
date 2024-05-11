import numpy as np
import json
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering, Birch
import matplotlib.pyplot as plt
import pandas as pd
import os, sys
import umap
import pickle
import seaborn as sns
from matplotlib.colors import ListedColormap


print("--------------- commencer ---------------")

name = 'subset'
number_of_cells = 100

directory = '/home/kmf69/umap_analysis/FINAL_EMBEDDINGS'

# Get data
data_path = "/home/kmf69/umap_analysis/cluster_metrics/labeled_nuclei_final.csv"
# data_path = "/home/kmf69/umap_analysis/cluster_metrics/cluster_figures/nuclei_data.csv"
points = pd.read_csv(data_path)
# points = points.sample(n = number_of_cells)

points["Big Labels"] = points["Labels"].str[0]

print("GOT DATA AND EMBEDDINGS")

data = points[
    [
        "NuclearDensity",
        "Area",
        "Circularity",
        "Aspect Ratio",
    ]
].values


## Plot 3 clusters
print("PLOTTING 3 CLUSTERS")
colors = ["#663399", "#CC0000", "#FFCC00"]
colors_3 = sns.color_palette(colors)

features = ["NuclearDensity",
          "Area",
          "Circularity",
          "Aspect Ratio"]

for feature in features:
    plot = sns.boxplot(data=points, 
        x="Big Labels", 
        y=feature, 
        palette = colors_3,
        hue_order = ["0", "1", "2"],
        order = ["0", "1", "2"])
    plot.set(xlabel='Cluster')
    plt.savefig(f'three_{feature}.png')
    plt.close()

# Colors
cluster_0_colors = ["#30024a", "#652987", "#8d6aa1", "#d9d2e9"]
cluster_1_colors = ["#770000", "#de1d1d", "#b42727", "#f8b7b7"]
total_cluster_colors = cluster_1_colors + cluster_0_colors + ["#FFCC00"]
cluster_0_palette = sns.color_palette(cluster_0_colors)
cluster_1_palette = sns.color_palette(cluster_1_colors)
total_palette = sns.color_palette(total_cluster_colors)

for feature in features:
    plot = sns.boxplot(data=points, 
        x="Labels", 
        y=feature, 
        palette = total_palette,
        order = ["0a", "0b", "0c", "0d", "1a", "1b", "1c", "1d", "2"],
        hue_order = ["0a", "0b", "0c", "0d", "1a", "1b", "1c", "1d", "2"])
    plot.set(xlabel='Cluster')
    plt.savefig(f'nine_{feature}.png')
    plt.close()


for feature in features:
    plot = sns.catplot(data=points, x="Labels", y=feature, kind = "violin",
        palette = total_palette,
        order = ["0a", "0b", "0c", "0d", "1a", "1b", "1c", "1d", "2"],
        hue_order = ["0a", "0b", "0c", "0d", "1a", "1b", "1c", "1d", "2"],
        scale="width")
    plot.set(xlabel='Cluster')
    plt.savefig(f'nine_violinplot_{feature}.png')
    plt.close()

for feature in features:
    plot = sns.catplot(data=points,
        x="Big Labels", 
        y=feature, 
        kind = "violin",
        palette = colors_3,
        order = ["0", "1", "2"],
        hue_order = ["0", "1", "2"])
    plot.set(xlabel='Cluster')
    plt.savefig(f'three_violinplot_{feature}_{name}.png')
    plt.close()


# save_path = "/home/kmf69/umap_analysis/cluster_metrics"
# slide_points.to_csv(f"{save_path}/nuclei_with_clusters.csv")
# embedding = pd.DataFrame(embedding)
# embedding.to_csv(f"{save_path}/umap_embedding.csv")

print("----------------- fin -----------------")
