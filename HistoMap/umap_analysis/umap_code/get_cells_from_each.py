import pandas as pd
import numpy as np
import os, sys

# Active virtual env: 
#conda activate umap

# Set random seed
np.random.seed(42)

# Get Directory
directory = 'slides/'

# Number of cells from each slide
number_of_cells = 90000

print("GETTING SLIDES")
# make list of all the slides
slides = [folder for folder in os.listdir(directory)]
slide_paths = [os.path.join(directory, slide) for slide in slides]


def get_slide(slide_num):
  os.mkdir(f'total_slide_data/slides/{slides[slide_num]}')
  df_temp = pd.read_csv(slide_paths[slide_num] + '/combined_total_nucdens.csv', sep = ',')
  df = df_temp.sample(n = number_of_cells)
  df.to_csv(f'total_slide_data/slides/{slides[slide_num]}/data.csv')

if __name__ == '__main__':
  arg1 = int(sys.argv[1])
  get_slide(arg1)