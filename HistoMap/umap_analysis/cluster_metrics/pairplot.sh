#!/bin/bash

#SBATCH --job-name=pairplot
#SBATCH --time=12:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=6
#SBATCH --mem-per-cpu=40G
#SBATCH --partition=pi_mak


#SBATCH --mail-type=ALL

module load miniconda
conda init bash
source activate umap3

python pairplot.py