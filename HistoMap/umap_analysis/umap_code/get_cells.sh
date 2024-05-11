#!/bin/bash

#SBATCH --job-name=umap_get_cells
#SBATCH --time=2-00:00:00
#SBATCH --partition=pi_mak
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2
#SBATCH --mem-per-cpu=20G

#SBATCH --mail-type=ALL
#SBATCH --array=0-9

module load miniconda
source activate umap3

python get_cells_from_each.py $SLURM_ARRAY_TASK_ID
