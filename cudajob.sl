#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=00:01:00
#SBATCH --output=cudajob.out
#SBATCH --error=cudajob.err
#SBATCH --gres=gpu

module load cuda/10.1.243

echo "Matrix Multiplication"
./myblas


