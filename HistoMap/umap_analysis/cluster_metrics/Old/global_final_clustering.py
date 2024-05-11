import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import os, sys
import umap
import umap.plot
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import seaborn as sns
import pickle
import json

c0_embedding_paths = r'/home/kmf69/umap_analysis/FINAL_EMBEDDINGS/cluster_0'
c1_embedding_paths = r'/home/kmf69/umap_analysis/FINAL_EMBEDDINGS/cluster_1'

print("GETTING EMBEDDINGS")
# Get PCA embedding
# Cluster 0
c0_pca_embedding_path = f'{c0_embedding_paths}/pca_pickle_fit.sav'
c0_pca_embedding = pickle.load(open(c0_pca_embedding_path, 'rb'))

# Cluster 1
c1_pca_embedding_path = f'{c1_embedding_paths}/pca_pickle_fit.sav'
c1_pca_embedding = pickle.load(open(c1_pca_embedding_path, 'rb'))

# Cluster 0
c0_umap_embedding_path = f'{c0_embedding_paths}/umap_pickle_fit_n_1000.sav'
c0_umap_embedding = pickle.load(open(c0_umap_embedding_path, 'rb'))

# Cluster 1
c1_umap_embedding_path = f'{c1_embedding_paths}/umap_pickle_fit_n_1000.sav'
c1_umap_embedding = pickle.load(open(c1_umap_embedding_path, 'rb'))

# Get clustering embedding
# Cluster 0
c0_birch = f'{c0_embedding_paths}/birch_clustering.sav'
c0_clusterer = pickle.load(open(c0_birch, 'rb'))

# Cluster 1
c1_birch = f'{c1_embedding_paths}/birch_clustering.sav'
c1_clusterer = pickle.load(open(c1_birch, 'rb'))

# Some plotting colors help 
cluster_0_colors = ["#30024a", "#652987", "#8d6aa1", "#d9d2e9"]
cluster_1_colors = ["#770000", "#de1d1d", "#b42727", "#f8b7b7"]
total_cluster_colors = cluster_0_colors + cluster_1_colors + ["#FFCC00"]
print(total_cluster_colors)
palette = sns.color_palette(total_cluster_colors)

# Readin in UMAP and slide points
clustered_nuclei_path = r'/home/kmf69/umap_analysis/cluster_metrics/nuclei_with_umap.csv'
nuclei_data = pd.read_csv(clustered_nuclei_path, sep = ',')
# nuclei_data = nuclei_data.sample(n = 100)

cluster_0 = nuclei_data[nuclei_data["Labels"] == 0].copy()
cluster_1 = nuclei_data[nuclei_data["Labels"] == 1].copy()

save_path = "/home/kmf69/umap_analysis/cluster_metrics/cluster_figures"

for i, cluster in enumerate([cluster_0, cluster_1]):
    pca_embedding = globals()[f'c{i}_pca_embedding']
    umap_embedding = globals()[f'c{i}_umap_embedding']
    birch_clusterer = globals()[f'c{i}_clusterer']

    data = cluster[
        [
            "NuclearDensity",
            "Area",
            "Circularity",
            "Aspect Ratio",
        ]
    ].values

    # PCA
    x = StandardScaler().fit_transform(data)
    principalComponents = pca_embedding.transform(x)
    pca_slide_params = pd.DataFrame(data = principalComponents)
    scaled_data = StandardScaler().fit_transform(pca_slide_params)

    # Transform with UMAP embedding
    print("DOING UMAP")
    slide_umap_embedding = umap_embedding.transform(scaled_data)

    # Cluster with Birch embedding
    pred = birch_clusterer.predict(slide_umap_embedding)

    if i == 0:
        cluster_0["Labels"] = np.array(pred)
        color_palette = sns.color_palette(cluster_0_colors)
    else:
        cluster_1["Labels"] = np.array(pred)
        color_palette = sns.color_palette(cluster_1_colors)
    
    slide_umap_embedding = pd.DataFrame(slide_umap_embedding)
    slide_umap_embedding = slide_umap_embedding.rename(columns={0: "umap1", 1: "umap2"})
    sns.scatterplot(x = 'umap1', 
        y = 'umap2', 
        data = slide_umap_embedding, 
        hue = pred, 
        s = 1, 
        palette = color_palette, 
        linewidth=0,
        legend="full")
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title(f'Cluster {i} UMAP', 
        fontsize=12, 
        loc = 'center', 
        wrap = True)
    cluster_save_path = f'{save_path}/cluster_{i}_UMAP_2.png'
    plt.savefig(cluster_save_path)
    plt.close()

for i, let in enumerate(['a', 'b', 'c', 'd']):
    cluster_0[cluster_0["Labels"] == i] = '0' + let
    cluster_1[cluster_1["Labels"] == i] = '1' + let

nuclei_data.loc[nuclei_data["Labels"] == 0, "Labels"] = cluster_0["Labels"]
nuclei_data.loc[nuclei_data["Labels"] == 2, "Labels"] = '2'
nuclei_data.loc[nuclei_data["Labels"] == 1, "Labels"] = cluster_1["Labels"]


cluster_order = ["0a", "0b", "0c", "0d", "1a", "1b", "1c", "1d", '2']

# Creating UMAP scatter plot
sns.set(rc={'figure.figsize':(10,8)})
sns.scatterplot(x='umap1', 
    y='umap2', 
    data=nuclei_data,
    hue="Labels",
    hue_order=cluster_order, 
    s=1, 
    palette=palette, 
    linewidth=0, 
    legend="full")
ax = plt.gca()
ax.xaxis.set_tick_params(labelbottom=False)
ax.yaxis.set_tick_params(labelleft=False)
ax.set_facecolor("white")
for spine in ax.spines.values():
    spine.set_edgecolor("black")
ax.set_xticks([])
ax.set_yticks([])
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title(f'UMAP', 
    fontsize=14, 
    loc = 'center', 
    wrap = True)
plt.savefig(f"{save_path}/all_clusters_total_umap_2.png")
plt.close()

# Save csvs
nuclei_data["Big Labels"] = nuclei_data["Labels"].str[0]
nuclei_data.to_csv(f'{save_path}/labeled_nuclei_final.csv')
