import numpy as np
import json
import sklearn
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
from mpl_toolkits import mplot3d

# %matplotlib qt

# Active virtual env: 
#conda activate umap3

# Get Directory
# Get Directory
directory = '/home/kmf69/project/total_slide_data/'

# Read in CSV
number_of_cells = 600000
data = pd.read_csv(f'{directory}all_slides_csv.csv', sep = ',')

# Randomize cells taken 
data = data.sample(n = number_of_cells)
data.to_csv("/home/kmf69/umap_analysis/umap_code/nuclei_csv_3d.csv")

# Randomize cells taken 
data = data.sample(n = number_of_cells)
data = data[
    [
        "NuclearDensity",
        "Area",
        "Circularity",
        "Aspect Ratio",
    ]
].values


# Do UMAP
n_neighbors = 1000
min_dist = 0
# Get reducer and embedding 
reducer = umap.UMAP(
    n_neighbors = n_neighbors,
    min_dist = min_dist,
    n_components = 3
)

pca = PCA(n_components = 3)
x = StandardScaler().fit_transform(data)
pca_fit = pca.fit(x)
principalComponents = pca.fit_transform(x)
pca_slide_params = pd.DataFrame(data = principalComponents)
scaled_data = StandardScaler().fit_transform(pca_slide_params)

embedding_2 = reducer.fit(scaled_data)
embedding = reducer.transform(scaled_data)

## SAVE EMBEDDINGS

directory = '/home/kmf69/umap_analysis/umap_code'
print("PICKLING UMAP MODEL")
pickle_name = f"{directory}/umap_pickle_transform_3d.sav"
print(pickle_name)
pickle.dump(embedding, open(pickle_name, 'wb'))

pickle_name = f"{directory}/umap_pickle_fit_3d.sav"
print(pickle_name)
pickle.dump(embedding_2, open(pickle_name, 'wb'))


# PLOTTING FIGURE
print(f'DID UMAP ON {n_neighbors} and {min_dist}')
name = f"n_neighbors = {n_neighbors}"
params_title = f"Total UMAP: {name}"
umap_title = 'umap_3d.png'

umap_plot_title = f"Total UMAP: {name}" 
plt.title(umap_plot_title, loc = 'center', wrap = True)    

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(embedding[:,0], embedding[:,1], embedding[:,2], s=0.1)
plt.title(umap_plot_title, fontsize=18)
plt.show()
plt.savefig("umap_3d.png")
# pickle.dump(fig, open('umap_3d.fig.pickle', 'wb'))

