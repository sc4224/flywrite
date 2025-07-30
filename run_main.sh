#!/bin/bash
#SBATCH --job-name=skopt_sbm
#SBATCH --output=logs/main_%j.out       # STDOUT log file (%j = job ID)
#SBATCH --error=logs/main_%j.err        # STDERR log file
#SBATCH --time=20:00:00                 # Max run time (adjust as needed)
#SBATCH --mem=4GB                        # Memory allocation
#SBATCH --cpus-per-task=1              # Not parallel, just the coordinator

# Load conda or your Python environment
source ./sbmenv/bin/activate

# Create log/output directories if they don't exist
# mkdir -p logs configs results
mkdir -p logs credible_interval_results

# Run your controller script
python slurm_sbm.py

