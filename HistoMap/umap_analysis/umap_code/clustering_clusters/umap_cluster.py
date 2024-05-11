import numpy as np
import json
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import os, sys
import umap
import pickle
import seaborn as sns

print("--------------- commencer ---------------")

directory = "/home/kmf69/umap_analysis/clustering_clusters/pickled_embeddings"


cluster_path = "/home/kmf69/umap_analysis/clustering_clusters/nuclei_with_clusters.csv"
clusters = pd.read_csv(cluster_path)

cluster_1 = clusters[clusters['Labels'] == 1]
print(len(cluster_1))

# Get Directory
# directory = '/home/kmf69/project/total_slide_data/'

# # Read in CSV and get nuclei above ND = 20
# data = pd.read_csv(f'{directory}all_slides_csv.csv', sep = ',')
# data = data[data["NuclearDensity"] > 20] 

data = cluster_1
data = data[
    [
        "NuclearDensity",
        "Area",
        "Circularity",
        "Aspect Ratio",
    ]
].values

pca = PCA(n_components = 3)
x = StandardScaler().fit_transform(data)
pca_fit = pca.fit(x)
principalComponents = pca.fit_transform(x)
pca_slide_params = pd.DataFrame(data = principalComponents)
scaled_data = StandardScaler().fit_transform(pca_slide_params)

print('DID PCA')

# Do UMAP
n_neighbors = [1000]
min_dist = 0

# Get reducer and embedding 
for n in n_neighbors:
    reducer = umap.UMAP(
    n_neighbors = n,
    min_dist = min_dist
    )
    # embedding = reducer.fit_transform(scaled_data)
    embedding_2 = reducer.fit(scaled_data)
    embedding = reducer.transform(scaled_data)

    print(f'DID UMAP ON {n} and {min_dist}')
    name = f"n_neighbors = {n}"

    # Add labels to csv
    params_title = f"UMAP: {name}"

    # Get label for parameters and umap plots
    umap_title = f'{directory}/umap_plot_n_{n}.png'
    umap_title = f'umap_n_{n}.png'

    print("PLOTTING")
    plt.scatter(embedding[:, 0], embedding[:, 1], s = 0.1)
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")

    umap_plot_title = f"Total UMAP: {name}" 

    plt.title(umap_plot_title, loc = 'center', wrap = True)    
    plt.savefig(umap_title)
    plt.close()

    print("PICKLING UMAP MODEL")
    pickle_name = f"{directory}/umap_pickle_transform_n_{n}.sav"
    print(pickle_name)
    pickle.dump(embedding, open(pickle_name, 'wb'))

    pickle_name = f"{directory}/umap_pickle_fit_n_{n}.sav"
    print(pickle_name)
    pickle.dump(embedding_2, open(pickle_name, 'wb'))

print("PICKLING PCA MODEL")
pickle_name = f"{directory}/pca_pickle_transform.sav"
print(pickle_name)
pickle.dump(principalComponents, open(pickle_name, 'wb'))

pickle_name = f"{directory}/umap_pca_fit.sav"
print(pickle_name)
pickle.dump(pca_fit, open(pickle_name, 'wb'))

print("----------------- fin -----------------")
