#!/bin/bash
#SBATCH --job-name=sbmobjective
#SBATCH --output=logs/out_%A_%a.out
#SBATCH --error=logs/err_%A_%a.err
#SBATCH --mem=2.5G
#SBATCH --time=05:00:00
#SBATCH --array=0-49
#SBATCH --cpus-per-task=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sbmenv

python sbm.py $SLURM_ARRAY_TASK_ID

