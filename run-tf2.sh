#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=test
#SBATCH --gres=gpu:v100:1
#SBATCH --time=0:15:00
#SBATCH --mem=64G
#SBATCH --account=project_2000859

module purge
module load tensorflow/2.0.0
module list

# export DATADIR=/scratch/project_2000859/demo/extracted

set -xv
srun python3 $*