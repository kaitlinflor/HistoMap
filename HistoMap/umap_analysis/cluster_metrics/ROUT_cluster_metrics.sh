#!/bin/bash

#SBATCH --job-name=rout_cluster_metrics
#SBATCH --time=12:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=6
#SBATCH --mem-per-cpu=40G

#SBATCH --mail-type=ALL

module load miniconda
conda init bash
source activate umap3

python ROUT_cluster_metrics.py