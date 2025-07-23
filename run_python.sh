#!/bin/bash
#SBATCH --job-name=descriptive_plotting
#SBATCH --account=project_2010938
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --output=descriptive_plotting.out

module load python-data/3.10-24.04

source myenv/bin/activate

time python descriptive_plotting.py

sacct -j $SLURM_JOB_ID --format=JobID,MaxRSS,Elapsed,ReqMem,State

echo "Job ended at: $(date)"
