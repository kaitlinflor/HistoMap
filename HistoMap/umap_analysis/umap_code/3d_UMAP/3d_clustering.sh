#!/bin/bash

#SBATCH --job-name=3d_clustering
#SBATCH --time=12:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2
#SBATCH --mem-per-cpu=20G

#SBATCH --mail-type=ALL

module load miniconda
source activate umap3

python 3d_clustering.py
