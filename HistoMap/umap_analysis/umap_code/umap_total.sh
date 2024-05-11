#!/bin/bash

#SBATCH --job-name=umap_total
#SBATCH --time=12:00:00
#SBATCH --partition=pi_mak
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=6
#SBATCH --mem-per-cpu=40G

#SBATCH --mail-type=ALL

module load miniconda
conda init bash
source activate umap3

python umap_total.py