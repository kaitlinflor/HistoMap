import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

nuclei = pd.read_csv('labeled_nuclei_final.csv')
# nuclei = nuclei.sample(n=3000)

features = [
    "Area",
    "NuclearDensity",
    "Aspect Ratio",
    "Circularity"
]


palette = sns.dark_palette("#69d", as_cmap=True, reverse = True)

for feature in features:
    nuclei[feature] = pd.to_numeric(nuclei[feature])
    # Creating UMAP scatter plot
    fig, ax = plt.subplots()
    sns.set(rc={'figure.figsize':(10,8)})
    scatter = sns.scatterplot(x='umap1', 
        y='umap2', 
        data=nuclei,
        hue=feature,
        # size="Area",
        s=1, 
        linewidth=0,
        legend = False,
        palette = palette
        )

    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title(f'UMAP by {feature}', 
        fontsize=14, 
        loc = 'center', 
        wrap = True)

    # plt.subplots_adjust(left=0.1, right=0.5, top=0.95, bottom=0.1)

    norm = plt.Normalize(nuclei[feature].min(), nuclei[feature].max())
    sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
    sm.set_array([])

    cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, anchor = (1.0, 0.5), label='Area')

    plt.savefig(f"{feature}_UMAP.png")
    plt.close()

# cluster_order = ["0a", "0b", "0c", "0d", "1a", "1b", "1c", "1d", '2']

# cluster_0_colors = ["#30024a", "#652987", "#8d6aa1", "#d9d2e9"]
# cluster_1_colors = ["#770000", "#de1d1d", "#b42727", "#f8b7b7"]
# total_cluster_colors = cluster_0_colors + cluster_1_colors + ["#FFCC00"]
# print(total_cluster_colors)
# palette = sns.color_palette(total_cluster_colors)

# print(embedding["Labels"])
# # Creating UMAP scatter plot
# sns.set(rc={'figure.figsize':(10,8)})
# sns.scatterplot(x='umap1', 
#     y='umap2', 
#     data=embedding,
#     hue="Labels",
#     hue_order=cluster_order, 
#     s=1, 
#     palette=palette, 
#     linewidth=0, 
#     legend="full")
# ax = plt.gca()
# ax.xaxis.set_tick_params(labelbottom=False)
# ax.yaxis.set_tick_params(labelleft=False)
# ax.set_facecolor("white")
# for spine in ax.spines.values():
#     spine.set_edgecolor("black")
# ax.set_xticks([])
# ax.set_yticks([])
# plt.xlabel("UMAP 1")
# plt.ylabel("UMAP 2")
# plt.title(f'UMAP', 
#     fontsize=14, 
#     loc = 'center', 
#     wrap = True)
# plt.savefig(f"test_big_plot.png")
# plt.close()
