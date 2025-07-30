#!/bin/bash
#SBATCH --job-name=sbmobjective
#SBATCH --output=logs/out_%A_%a.out
#SBATCH --error=logs/err_%A_%a.err
#SBATCH --mem=20GB
#SBATCH --time=20:00:00
#SBATCH --array=0-49
#SBATCH --cpus-per-task=8

source ./sbmenv/bin/activate

python hidden_markov_graph.py $SLURM_ARRAY_TASK_ID
#python sbm.py $SLURM_ARRAY_TASK_ID

