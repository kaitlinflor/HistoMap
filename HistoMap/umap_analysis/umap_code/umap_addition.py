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

import hdbscan
from hdbscan.flat import HDBSCAN_flat, approximate_predict_flat, membership_vector_flat, all_points_membership_vectors_flat
import pickle


# Active virtual env: 
#conda activate umap3

# Set random seed
np.random.seed(42)


# Get Directory
directory = 'slides/'

# Number of cells from each slide
number_of_cells = 100000

print("GETTING SLIDES")
# make list of all the slides
slides = [folder for folder in os.listdir(directory)]
slide_paths = [os.path.join(directory, slide) for slide in slides]

# Get PCA embedding
pca_embedding_path = "/home/kmf69/umap_analysis/total_slide_data/raw_pca/umap_pca_fit.sav"
pca_embedding = pickle.load(open(pca_embedding_path, 'rb'))

# Get UMAP embedding
umap_embedding_path = "/home/kmf69/umap_analysis/total_slide_data/raw_umap/umap_pickle_fit_n_1000.sav"
umap_embedding = pickle.load(open(umap_embedding_path, 'rb'))


def umap_on_slide(slide_num):  
  slide_points = pd.read_csv(slide_paths[slide_num] + '/combined_total_nucdens.csv', sep = ',')
  slide_points = slide_points.sample(n = number_of_cells)

  data = slide_points[
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
  
  print("PLOTTING UMAP")
  plt.scatter(
      slide_umap_embedding[:, 0],
      slide_umap_embedding[:, 1], s = 0.2, cmap = 'Spectral')
  plt.title(f'UMAP projection of {slides[slide_num]}', fontsize=24, loc = 'center', wrap = True)
  plt.show()

  umap_save_dir = f"/home/kmf69/umap_analysis/umap_embedding_imgs/umap_embedding_{slide_num}_1000.png"
  plt.savefig(umap_save_dir)
  plt.close()

  pickle_name = f"/home/kmf69/umap_analysis/umap_embeddings/umap_pickle_{slide_num}_1000.sav"
  pickle.dump(slide_umap_embedding, open(pickle_name, 'wb'))

if __name__ == '__main__':
  arg1 = int(sys.argv[1])
  umap_on_slide(arg1)