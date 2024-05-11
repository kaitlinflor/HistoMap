#!/bin/bash
#SBATCH --job-name=histomap
#SBATCH --time=36:00:00
#SBATCH --partition=pi_mak
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem-per-cpu=20G
#SBATCH --mail-type=all 
#SBATCH --mail-user=kaitlin.flores@yale.edu
#SBATCH --array=0-200:7

module load miniconda
conda activate stardist

python3 histo_map.py $SLURM_ARRAY_TASK_ID S12-32465_22
mem_bytes=$(</sys/fs/cgroup/memory/slurm/uid_${SLURM_JOB_UID}/job_${SLURM_JOB_ID}/memory.limit_in_bytes)
mem_gbytes=$(( $mem_bytes / 1024 **3 ))

echo "Starting at $(date)"
echo "Job submitted to the ${SLURM_JOB_PARTITION} partition, the default partition on ${SLURM_CLUSTER_NAME}"
echo "Job name: ${SLURM_JOB_NAME}, Job ID: ${SLURM_JOB_ID}"
echo "  I have ${SLURM_CPUS_ON_NODE} CPUs and ${mem_gbytes} GiB of RAM on compute node $(hostname)"