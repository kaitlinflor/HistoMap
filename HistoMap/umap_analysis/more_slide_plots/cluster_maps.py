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

# Active virtual env: 
#conda activate umap3

PATH = r'/home/kmf69/project/processed_slides/csv_9_clusters'

slides = [slide for slide in os.listdir(PATH)] # list of all slides
slide_paths = [os.path.join(PATH, slide) for slide in slides] # list of slide paths
slide_names = [slide.split(".csv")[0] for slide in slides] 
print(slides)
print(slide_paths)
print(slide_names)

def plot_clusters(slide_num):
  slide_points = pd.read_csv(slide_paths[slide_num], sep = ',')
#   slide_points = slide_points.sample(n = 1000)

  cluster_0 = ['0a', '0b', '0c', '0d']
  cluster_1 = ['1a', '1b', '1c', '1d']
  cluster_2 = ['2']

  cluster_orders = [cluster_0, cluster_1, cluster_2]

  cluster_0_points =  slide_points[slide_points['Labels'].isin(cluster_0)]
  cluster_1_points =  slide_points[slide_points['Labels'].isin(cluster_1)]
  cluster_2_points =  slide_points[slide_points['Labels'].isin(cluster_2)]

#   Shaded Colors
#   cluster_0_colors = ["#30024a", "#652987", "#8d6aa1", "#d9d2e9"]
#   cluster_1_colors = ["#770000", "#de1d1d", "#b42727", "#f8b7b7"]
#   cluster_2_colors = ["#FFCC00"]

# Different colors
  cluster_0_colors = ["#0f5bd6", "#0fa842", "#a81e0f", "#f0e40e"]
  cluster_1_colors = cluster_0_colors
#   cluster_1_colors = ["#770000", "#de1d1d", "#b42727", "#f8b7b7"]
  cluster_2_colors = ["#FFCC00"]

  cluster_points = [cluster_0_points, cluster_1_points, cluster_2_points]
  cluster_colors = [cluster_0_colors, cluster_1_colors, cluster_2_colors]


  for i, cluster in enumerate(cluster_points):

    # Replot groups onto centers
    sns.set(rc={'figure.figsize':(10,8)})
    sns.scatterplot(x='x', 
        y='y', 
        data=cluster, 
        hue='Labels',  
        hue_order=cluster_orders[i], 
        palette=sns.color_palette(cluster_colors[i]), 
        s=1, 
        linewidth=0, 
        legend="full")
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
    plt.xlabel("")
    plt.ylabel("")
    remap_save_dir = f"/home/kmf69/umap_analysis/more_slide_plots/cluster_{i}/"
    remap_title = remap_save_dir + f'{slide_names[slide_num]}_remapped.png'
    remap_plot_title = f"{slide_names[slide_num]} Remapped"
    plt.title(remap_plot_title, loc = "center", wrap = True)
    plt.savefig(remap_title)
    plt.close()

if __name__ == '__main__':
  arg1 = int(sys.argv[1])
  plot_clusters(arg1)