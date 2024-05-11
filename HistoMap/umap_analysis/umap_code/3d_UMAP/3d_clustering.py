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

print("--------------- commencer ---------------")

name = '3_clusters'

# Transform 
umap_transform_path = '/home/kmf69/umap_analysis/umap_code/3d_UMAP/umap_pickle_transform_3d.sav'
embedding = pickle.load(open(umap_transform_path, 'rb'))
print("GOT DATA")

nuclei = pd.read_csv("nuclei_csv_3d.csv")
embedding = pd.DataFrame(embedding)
embedding = embedding.sample(n = 200000)
sampled_indices = embedding.index
nuclei = nuclei.iloc[sampled_indices]
nuclei["UMAP1"] = embedding[0]
nuclei["UMAP2"] = embedding[1]
nuclei["UMAP3"] = embedding[2]

# Birch parameters
threshold = 0.1
cluster = 4
branching_factor = 200

print('--------------- BIRCH CLUSTERING COMMENCE ---------------')
model = Birch(threshold = threshold, branching_factor = branching_factor, n_clusters = cluster)
model.fit(embedding)
pred = model.predict(embedding)
print('--------------- BIRCH CLUSTERING FIN ---------------')

# Creating a scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(embedding[0], embedding[1], embedding[2], s=0.1, c=pred)
umap_plot_title = f"3D UMAP with Birch Clustering"
plt.title(umap_plot_title, fontsize=18)
plt.show()
plt.savefig(f"umap_3d_clustered_{cluster}.png")
plt.close()

nuclei["Label"] = pred
nuclei.to_csv(f"nuclei_with_clusters_3_{cluster}.csv")

pickle_path = f'/home/kmf69/umap_analysis/umap_code/3d_UMAP'
pickle_name = f"birch_clustering_1000_{name}.sav"
pickle.dump(model, open(f"{pickle_path}/{pickle_name}", 'wb'))

print("----------------- fin -----------------")
