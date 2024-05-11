#!/bin/bash

#SBATCH --job-name=umap_total
#SBATCH --time=2-00:00:00
#SBATCH --partition=pi_mak
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2
#SBATCH --mem-per-cpu=20G

#SBATCH --mail-type=ALL

module load miniconda
conda init bash
source activate umap3

python all_slides_csv.py