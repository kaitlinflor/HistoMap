import numpy as np
import json
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

nuclei = pd.read_csv("/home/kmf69/umap_analysis/cluster_metrics/nuclei_csv.csv")
nuclei = nuclei.sample(n = 300000).reset_index()

# Get PCA embedding
pca_embedding_path = "/home/kmf69/umap_analysis/FINAL_EMBEDDINGS/pca_fit_n_1000.sav"
pca_embedding = pickle.load(open(pca_embedding_path, 'rb'))

# Get UMAP embedding
umap_embedding_path = "/home/kmf69/umap_analysis/FINAL_EMBEDDINGS/umap_pickle_fit_n_1000.sav"
umap_embedding = pickle.load(open(umap_embedding_path, 'rb'))

# Get clustering embedding
birch = '/home/kmf69/umap_analysis/FINAL_EMBEDDINGS/birch_clustering_1000.sav'
clusterer = pickle.load(open(birch, 'rb'))
print(type(clusterer))


data = nuclei[
      [
          "NuclearDensity",
          "Area",
          "Circularity",
          "Aspect Ratio",
      ]
  ].values

# SHOULD CHANGE TO PCA FIT USED BEFORE
print("DOING PCA")
x = StandardScaler().fit_transform(data)
principalComponents = pca_embedding.transform(x)
pca_slide_params = pd.DataFrame(data = principalComponents)
scaled_data = StandardScaler().fit_transform(pca_slide_params)

# Transform with UMAP embedding
print("DOING UMAP")
slide_umap_embedding = umap_embedding.transform(scaled_data)
# Cluster with Birch embedding
pred = clusterer.predict(slide_umap_embedding)

## Plot UMAP distribution 
colors = ["#663399", "#CC0000", "#FFCC00"]
sns.set_palette(sns.color_palette(colors))

slide_umap_embedding = pd.DataFrame(slide_umap_embedding)
slide_umap_embedding = slide_umap_embedding.rename(columns={0: "umap1", 1: "umap2"})

print(slide_umap_embedding["umap1"])
print(slide_umap_embedding["umap2"])

nuclei["Labels"] = pred

nuclei["umap1"] = slide_umap_embedding["umap1"]
nuclei["umap2"] = slide_umap_embedding["umap2"]

print(nuclei["umap1"])
print(nuclei["umap2"])
nuclei.to_csv('nuclei_with_umap.csv')

# Creating a scatter plot
sns.scatterplot(x = 'umap1',
                y = 'umap2', 
                data = nuclei, 
                hue = "Labels", 
                s = 1, 
                palette = sns.color_palette(colors), 
                linewidth=0)
ax = plt.gca()
ax.xaxis.set_tick_params(labelbottom=False)
ax.yaxis.set_tick_params(labelleft=False)
ax.set_xticks([])
ax.set_yticks([])
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
print("PLOTTING UMAP")

plt.title(f'UMAP Projection', fontsize=24, loc = 'center', wrap = True)
plt.show()

plt.savefig("test_umap.png")
plt.close()