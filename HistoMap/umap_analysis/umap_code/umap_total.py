import numpy as np
import json
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import os, sys
import umap
import umap.plot 
import hdbscan
from hdbscan.flat import HDBSCAN_flat, approximate_predict_flat, membership_vector_flat, all_points_membership_vectors_flat
import pickle

# Active virtual env: 
#conda activate umap3

# Get Directory
directory = '/home/kmf69/project/total_slide_data/'

# Read in CSV
number_of_cells = 600000
data = pd.read_csv(f'{directory}all_slides_csv.csv', sep = ',')

# Randomize cells taken 
data = data.sample(n = number_of_cells)
data.to_csv("/home/kmf69/umap_analysis/cluster_metrics/nuclei_csv.csv")

data = data[
    [
        "NuclearDensity",
        "Area",
        "Circularity",
        "Aspect Ratio",
    ]
].values

# First do PCA on data
print("DOING PCA")

pca = PCA(n_components = 3)
x = StandardScaler().fit_transform(data)
pca_fit = pca.fit(x)
principalComponents = pca.fit_transform(x)
pca_slide_params = pd.DataFrame(data = principalComponents)
scaled_data = StandardScaler().fit_transform(pca_slide_params)

print('DID PCA')

# Do UMAP
n_neighbors = [1200]
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
    # labels = hdbscan.HDBSCAN().fit_predict(embedding)

    print(type(embedding))

    print(f'DID UMAP ON {n} and {min_dist}')

    name = f"n_neighbors = {n}"
    # Add labels to csv
    params_title = f"Total UMAP: {name}"

    # Get label for parameters and umap plots
    umap_title = f'{directory}raw_umap/umap_plot_n_{n}.png'
    umap_title = f'umap_n_{n}.png'

    print("PLOTTING")
    plt.scatter(embedding[:, 0], embedding[:, 1], s = 0.1)
    umap_plot_title = f"Total UMAP: {name}" 

    plt.title(umap_plot_title, loc = 'center', wrap = True)    
    plt.savefig(umap_title)
    plt.close()

    print("PICKLING UMAP MODEL")
    pickle_name = f"{directory}raw_umap/umap_pickle_transform_n_{n}.sav"
    print(pickle_name)
    pickle.dump(embedding, open(pickle_name, 'wb'))

    pickle_name = f"{directory}raw_umap/umap_pickle_fit_n_{n}.sav"
    print(pickle_name)
    pickle.dump(embedding_2, open(pickle_name, 'wb'))

print("PICKLING PCA MODEL")
pickle_name = f"{directory}raw_pca/pca_pickle_transform.sav"
print(pickle_name)
pickle.dump(principalComponents, open(pickle_name, 'wb'))

pickle_name = f"{directory}raw_pca/umap_pca_fit.sav"
print(pickle_name)
pickle.dump(pca_fit, open(pickle_name, 'wb'))