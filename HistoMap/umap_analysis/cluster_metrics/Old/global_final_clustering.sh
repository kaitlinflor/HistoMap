#!/bin/bash

#SBATCH --job-name=global_final_clustering
#SBATCH --time=12:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=6
#SBATCH --mem-per-cpu=40G

#SBATCH --mail-type=ALL

module load miniconda
conda init bash
source activate umap3

python global_final_clustering.py