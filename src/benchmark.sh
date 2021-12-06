#! bash
make clean
make
echo "Pytorch generation"
python3 ./src/generate.py
echo "CUDA implementation"
./bin/cuda-impl