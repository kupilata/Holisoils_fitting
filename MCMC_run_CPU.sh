#!/bin/bash
#SBATCH --job-name=multiCPU_opt
#SBATCH --account=project_2010938
#SBATCH --partition=small
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=22
#SBATCH --mem-per-cpu=4000

# set the number of threads based on --cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge
module load pytorch

srun python3 MCMC_puhti_CPU.py --data_path=$LOCAL_SCRATCH/cifar-10-batches-py