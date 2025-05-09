#!/bin/bash
#SBATCH --account=project_2010938
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=15
#SBATCH --gres=gpu:v100:1,nvme:10

module purge
module load pytorch

srun python3 test_puhti.py --data_path=$LOCAL_SCRATCH/cifar-10-batches-py