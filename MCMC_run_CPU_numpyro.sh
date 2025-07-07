#!/bin/bash
#SBATCH --job-name=multiCPU_opt
#SBATCH --account=project_2010938
#SBATCH --partition=small
#SBATCH --time 65:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --mem-per-cpu=15000
#SBATCH --output=mcmc_output.log
#SBATCH --error=mcmc_error.log

# Activate the virtual environment
source venv_mine/bin/activate

# set the number of threads based on --cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge

srun python3 MCMC_local_CPU_numpyro_MAPinit_trench_missingimpute.py --data_path=$LOCAL_SCRATCH/cifar-10-batches-py