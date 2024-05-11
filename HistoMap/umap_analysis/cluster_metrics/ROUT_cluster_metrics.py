import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('labeled_nuclei_final.csv')

# Define variables of interest
variables_of_interest = ['Area', 'NuclearDensity', 'Perimeter', 'Circularity']

# Define groups of interest
big_labels = [0, 1, 2]
labels = ['0a', '0b', '0c', '0d', '1a', '1b', '1c', '1d', '2']

# Create a subset of the data for each group
group_subsets = {}
for label in labels:
    group_subsets[label] = df.loc[df['Labels'] == label, variables_of_interest]

# Remove outliers using the interquartile range (IQR) method
def remove_outliers(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return df[((df > lower_bound) & (df < upper_bound)).all(axis=1)]

group_subsets_no_outliers = {}
for label in labels:
    group_subsets_no_outliers[label] = remove_outliers(group_subsets[label])

# Compare group distributions
for variable in variables_of_interest:
    sns.boxplot(data=pd.DataFrame([group_subsets_no_outliers[label][variable] for label in labels]), 
                x=[int(label[0]) for label in labels],
                hue=[label[1:] if len(label) > 1 else ' ' for label in labels])
    sns.stripplot(data=pd.DataFrame([group_subsets_no_outliers[label][variable] for label in labels]), 
                  x=[int(label[0]) for label in labels],
                  hue=[label[1:] if len(label) > 1 else ' ' for label in labels],
                  jitter=True, dodge=True)
    plt.xlabel('Big Label')
    plt.ylabel(variable)
    plt.title(f'{variable} by Big Label and Label')
    plt.legend(title='Subgroup', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(f"{variable}_by_feature.png")

# Explore unique characteristics of each group
unique_characteristics = {}
for label in labels:
    group_mean = group_subsets_no_outliers[label].mean()
    other_groups_mean = pd.concat([group_subsets_no_outliers[other_label] for other_label in labels if other_label != label]).mean()
    unique_characteristics[label] = group_mean[(group_mean > other_groups_mean) & (group_mean.notnull())]

for label in labels:
    print(f'Unique characteristics of {label}:')
    print(unique_characteristics[label])
    print('\n')





# import scipy.stats as stats




# # Perform a Kruskal-Wallis H test to determine if there are significant differences between the groups
# # in any of the features
# kwh_results = {}
# for col in ['Area', 'NuclearDensity', 'Perimeter', 'Circularity']:
#     kwh_results[col] = stats.kruskal(*[df[col][df['Labels'] == label] for label in df['Labels'].unique()])

# # Print the results
# for col in kwh_results:
#     print(f"Kruskal-Wallis H test for {col}: H={kwh_results[col][0]:.3f}, p={kwh_results[col][1]:.3g}")

# # Perform pairwise Wilcoxon rank-sum tests to determine which groups have significant differences in each feature
# for col in ['Area', 'NuclearDensity', 'Perimeter', 'Circularity']:
#     for i, label1 in enumerate(df['Labels'].unique()):
#         for j, label2 in enumerate(df['Labels'].unique()):
#             if i >= j:
#                 continue
#             label1_data = df[col][df['Labels'] == label1]
#             label2_data = df[col][df['Labels'] == label2]
#             p_value = stats.ranksums(label1_data, label2_data)[1]
#             if p_value < 0.05:
#                 print(f"Wilcoxon rank-sum test for {col}, {label1} vs {label2}: p={p_value:.3g}")