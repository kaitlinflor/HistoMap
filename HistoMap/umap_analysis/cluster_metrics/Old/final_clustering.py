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

name = 'cluster_1'

directory = '/home/kmf69/project/total_slide_data/total_slide_data/raw_umap'

# TOTAL
# umap_transform_path = '/home/kmf69/umap_analysis/FINAL_EMBEDDINGS/umap_pickle_transform_n_1000.sav'

# CLUSTER 0
umap_transform_path = '/home/kmf69/umap_analysis/FINAL_EMBEDDINGS/cluster_1/umap_pickle_transform_n_1000.sav'
# CLUSTER 1
# umap_transform_path = '/home/kmf69/umap_analysis/FINAL_EMBEDDINGS/cluster_1/umap_pickle_transform_n_1000.sav'

embedding = pickle.load(open(umap_transform_path, 'rb'))

print("GOT DATA")

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
plt.scatter(embedding[:, 0], embedding[:, 1], c = pred, s = 0.1)
umap_title = f'birch_clustering_{name}.png'
umap_plot_title = f"Birch Clustering: thresh = {threshold}, #clusters = {cluster}, branching factor = {branching_factor}"
plt.title(umap_plot_title, loc = 'center', wrap = True)    
plt.savefig(umap_title)
plt.close()

pickle_path = f'/home/kmf69/umap_analysis/cluster_metrics'
pickle_name = f"birch_clustering_1000_{name}.sav"
pickle.dump(model, open(f"{pickle_path}/{pickle_name}", 'wb'))

print("----------------- fin -----------------")
