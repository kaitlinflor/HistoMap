#!/bin/bash

#SBATCH --job-name=size_thresh
#SBATCH --time=12:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=6
#SBATCH --mem-per-cpu=40G


#SBATCH --mail-type=ALL
#SBATCH --array=0-1

module load miniconda
source activate umap3

python size_thresh.py $SLURM_ARRAY_TASK_ID