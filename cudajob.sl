#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=00:05:00
#SBATCH --output=out/cudajob.out
#SBATCH --error=out/cudajob.err
#SBATCH --gres=gpu

module purge
module load gcc/8.3.0
module load cuda/10.2.89
module load python/3.9.2

pip install --user torch==1.8.2+cu102 torchvision==0.9.2+cu102 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

make clean
make
echo "Pytorch generation"
python3 ./src/generate.py
echo "CUDA implementation"
./bin/cuda-impl