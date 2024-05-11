import pandas as pd
import numpy as np
import os, sys
import umap
import hdbscan

# Active virtual env: 
#conda activate umap

# Get Directory
directory = 'total_slide_data/slides'

print("GETTING SLIDES")
# make list of all the slides
slides = [folder for folder in os.listdir(directory)]
slide_paths = [os.path.join(directory, slide) for slide in slides]

print(slide_paths)
df = []
for slide in slide_paths:
  print("APPENDING")
  df.append(pd.read_csv(f'{slide}/data.csv'))

data = pd.concat(df)
data.to_csv('total_slide_data/all_slides_csv.csv')
