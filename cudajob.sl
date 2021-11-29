#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=00:01:00
#SBATCH --output=out/cudajob.out
#SBATCH --error=out/cudajob.err
#SBATCH --gres=gpu

module load cuda/10.1.243

echo "GPU Implementation"
./bin/gpu-impl

module purge
module load gcc/8.3.0
module load python/3.9.2

python src/torch-test.py