import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import os, sys
import seaborn as sns
import pickle
import json

# Active virtual env: 
#conda activate umap3

PATH = '/home/kmf69/project/processed_slides/csv_3_clusters'
slides = [slide for slide in os.listdir(PATH)] # list of all slides
slide_paths = [os.path.join(PATH, slide) for slide in slides] # list of slide paths
slide_names = [slide.split(".csv")[0] for slide in slides] 
print(slides)
print(slide_paths)
print(slide_names)

def size_threshold(slide_num, size):
  slide_points = pd.read_csv(slide_paths[slide_num], sep=',')
  slide_points = slide_points.sample(n=200000)

  slide_points_small = slide_points.copy()[slide_points["Area"] < size]
  slide_points_big = slide_points.copy()[slide_points["Area"] > size]
  slide_points_small.loc[:, 'Size'] = f'< {size}'
  slide_points_big.loc[:, 'Size'] = f'> {size}'
  slide_points = pd.concat([slide_points_small, slide_points_big])

  cluster_0 = [0]
  cluster_1 = [1]
  cluster_2 = [2]

  cluster_orders = [cluster_0, cluster_1, cluster_2]

  cluster_0_points = slide_points[slide_points['Labels'].isin(cluster_0)]
  cluster_1_points = slide_points[slide_points['Labels'].isin(cluster_1)]
  cluster_2_points = slide_points[slide_points['Labels'].isin(cluster_2)]

  # Different colors
  cluster_0_colors = ["#663399"]
  cluster_1_colors = ["#CC0000"]
  cluster_2_colors = ["#FFCC00"]

  cluster_points = [cluster_0_points, cluster_1_points, cluster_2_points]
  cluster_colors = [cluster_0_colors, cluster_1_colors, cluster_2_colors]

  for i, cluster in enumerate(cluster_points):

    # Replot groups onto centers
    sns.set(rc={'figure.figsize':(10,8)})
    sns.scatterplot(x='x', 
                    y='y', 
                    data=cluster, 
                    hue='Size',  
                    palette=sns.color_palette(["#00a2ff", "#ff0000"]), 
                    hue_order=[f'> {size}', f'< {size}'],
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
    remap_save_dir = f"/home/kmf69/umap_analysis/size_threshold/plots/cluster_{i}/"
    remap_title = remap_save_dir + f'{slide_names[slide_num]}_remapped_cluster{i}.png'
    remap_plot_title = f"{slide_names[slide_num]} Remapped, Cluster {i}"
    plt.title(remap_plot_title, loc="center", wrap=True)
    plt.savefig(remap_title)
    plt.close()

if __name__ == '__main__':
  size_threshold(2, 200)


# def plot_clusters(slide_num):
#   slide_points = pd.read_csv(slide_paths[slide_num], sep = ',')
#   slide_points = slide_points.sample(n = 50000)

#   sizes = [150]
#   for k, size in enumerate(sizes):
#     slide_points_small = slide_points[slide_points["Area"] < sizes[k]]
#     slide_points_big = slide_points[slide_points["Area"] > sizes[k]]
    
#     comparers = ['>', '<']
#     names = ["big", "small"]

    
#     for j, slide_points in enumerate([slide_points_big, slide_points_small]):
#         cluster_0 = [0]
#         cluster_1 = [1]
#         cluster_2 = [2]

#         cluster_orders = [cluster_0, cluster_1, cluster_2]

#         cluster_0_points =  slide_points[slide_points['Labels'].isin(cluster_0)]
#         cluster_1_points =  slide_points[slide_points['Labels'].isin(cluster_1)]
#         cluster_2_points =  slide_points[slide_points['Labels'].isin(cluster_2)]

#         # # Different colors

#         cluster_0_colors = ["#663399"]
#         cluster_1_colors = ["#CC0000"]
#         cluster_2_colors = ["#FFCC00"]

#         cluster_points = [cluster_0_points, cluster_1_points, cluster_2_points]
#         cluster_colors = [cluster_0_colors, cluster_1_colors, cluster_2_colors]


#         for i, cluster in enumerate(cluster_points):

#             # Replot groups onto centers
#             sns.set(rc={'figure.figsize':(10,8)})
#             sns.scatterplot(x='x', 
#                 y='y', 
#                 data=cluster, 
#                 hue='Labels',  
#                 hue_order=cluster_orders[i], 
#                 palette=sns.color_palette(cluster_colors[i]), 
#                 s=1, 
#                 linewidth=0, 
#                 legend="full")
#             ax = plt.gca()
#             ax.xaxis.set_tick_params(labelbottom=False)
#             ax.yaxis.set_tick_params(labelleft=False)
#             ax.set_facecolor("white")
#             ax.set_xticks([])
#             ax.set_yticks([])
#             for spine in ax.spines.values():
#                 spine.set_edgecolor("black")
#             plt.xlabel("")
#             plt.ylabel("")
#             remap_save_dir = f"/home/kmf69/umap_analysis/size_threshold/plots/cluster_{i}/"
#             remap_title = remap_save_dir + f'{slide_names[slide_num]}_remapped_{names[j]}_{sizes[k]}.png'
#             remap_plot_title = f"{slide_names[slide_num]} Remapped, Cluster {i}, Area {comparers[j]} {sizes[k]}"
#             plt.title(remap_plot_title, loc = "center", wrap = True)
#             plt.savefig(remap_title)
#             plt.close()

# if __name__ == '__main__':
    # for i in range(12):
    # plot_clusters(1)
#   arg1 = int(sys.argv[1])