#!/bin/bash

#SBATCH --job-name=cluster_maps
#SBATCH --time=12:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2
#SBATCH --mem-per-cpu=20G


#SBATCH --mail-type=ALL
#SBATCH --array=0-18

module load miniconda
source activate umap3

python cluster_maps.py $SLURM_ARRAY_TASK_ID
