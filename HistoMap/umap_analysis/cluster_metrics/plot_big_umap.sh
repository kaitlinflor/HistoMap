#!/bin/bash

#SBATCH --job-name=plot_big_umap
#SBATCH --time=12:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=6
#SBATCH --mem-per-cpu=40G

#SBATCH --mail-type=ALL

module load miniconda
conda init bash
source activate umap3

python plot_big_umap.py