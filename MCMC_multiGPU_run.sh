#!/bin/bash
#SBATCH --account=project_2010938
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=280
#SBATCH --gres=gpu:v100:4,nvme:10

module purge
module load pytorch

srun python3 MCMC_puhti_multi.py --data_path=$LOCAL_SCRATCH/cifar-10-batches-py