# ee599-project-cuda-impl

## Environment Requirments
### Running Platform
Computing cluster with Slurm enabled. This project requires CUDA GPU support. Ideally, the GPU architecture should be Volta or higher (Volta/Turing/Ampere).

### Modules
- Python 3.6 or higher
- PyTorch LTS (1.8.2)
- CUDA 10.2 or higher
- gcc 8.3 or higher

## Project Structures
### ./src
./src contains the core deconvolution module of this project. It also has a PyTorch testing data generation script.
### ./inc
./inc contains header files and a header-only csv reader library.
### ./data
If ./data does not exist, it will be created by generate.py script. It contains our testing data in a csv formate.
### ./out
It is recommended to create the ./out directory in the first place to avoid unexpected Slurm issues. However, in either way, cudajob.sl will create the directory at the run time. ./out contains the output and the error file from Slurm.
### ./bin, ./obj
Binary files generated by the makefile.

## Running Instructions
### Make and Run
At first, use the `make` commend to compile our project. Then use `sbatch cudajob.sl` to run the binaries.

Outputs will be stored under ./out/cudajob.out
### Create different test samples
One may change the dimensional parameters *(N,H,W,C,M,KH,KW,SH,SW,PH,PW)* in generate.py and test.cu to create different testing samples.
