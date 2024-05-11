import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

nuclei = pd.read_csv("labeled_nuclei_final.csv")
nuclei = nuclei.sample(n=4000)

# PAIR PLOTS
nuclei_1 = nuclei[
    ["Area",
    "NuclearDensity",
    # "Perimeter",
    "Aspect Ratio",
    "Circularity",
    "Big Labels"
    ]
]

sns.pairplot(nuclei_1, 
    kind="kde",
    plot_kws = {"s":3, "linewidth":0})
plt.savefig("pairplot_gen.png")

exit()

colors = ["#663399", "#CC0000", "#FFCC00"]
palette = sns.color_palette(colors)
sns.pairplot(nuclei_1, 
    hue = "Big Labels",
    hue_order = [0, 1, 2],
    palette = palette,
    plot_kws = {"s":3, "linewidth":0})
plt.savefig("pairplot_3.png")


nuclei_2 = nuclei[
    ["Area",
    "NuclearDensity",
    # "Perimeter",
    "Aspect Ratio",
    "Circularity",
    "Labels"
    ]
]

cluster_0_colors = ["#30024a", "#652987", "#8d6aa1", "#d9d2e9"]
cluster_1_colors = ["#770000", "#de1d1d", "#b42727", "#f8b7b7"]
total_cluster_colors = cluster_0_colors + cluster_1_colors + ["#FFCC00"]
print(total_cluster_colors)
palette = sns.color_palette(total_cluster_colors)
sns.pairplot(nuclei_2, 
    hue = "Labels",
    palette = palette,
    hue_order = ["0a", "0b", "0c", "0d", "1a", "1b", "1c", "1d", "2"],
    plot_kws = {"s":3, "linewidth":0})
plt.savefig("pairplot_9.png")


cluster0 = nuclei[nuclei["Big Labels"] == 0]
cluster1 = nuclei[nuclei["Big Labels"] == 1]

orders = [["0a", "0b", "0c", "0d"], ["1a", "1b", "1c", "1d"]]
colors = [cluster_0_colors, cluster_1_colors]

for i, cluster in enumerate([cluster0, cluster1]):
    palette = sns.color_palette(colors[i])
    sns.pairplot(nuclei_2, 
        hue = "Labels",
        palette = palette,
        hue_order = orders[i],
        plot_kws = {"s":3, "linewidth":0})
    plt.savefig(f"pairplot_cluster_{i}.png")


## Other stats
# cluster_2a = nuclei[nuclei["Big Labels"] == 2]
# cluster_2b = nuclei[nuclei["Labels"] == '2']


# mean_area_2a = cluster_2a["Area"].mean()
# # print(cluster_2b["Area"])
# # print(sum(cluster_2b["Area"].isnan()))
# mean_area_2b = cluster_2b["Area"].mean()

# print(f"Mean 2 Big Labels: {mean_area_2a}, Mean 2 Labels: {mean_area_2b}")

# # Some stats
# features = ["Area",
#     "NuclearDensity",
#     "Aspect Ratio",
#     "Circularity",
#     ]

# # Group by the group column and calculate the mean and variance for each feature
# group_stats_3 = nuclei.groupby('Big Labels')[features].agg(['mean'])
# group_stats_3.to_csv("means_3.csv")
# print(group_stats_3)

# group_stats_9 = nuclei.groupby('Labels')[features].agg(['mean'])
# group_stats_9.to_csv("means_9.csv")

# print(group_stats_9)
