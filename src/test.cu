#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <cstdio>
#include <cstdlib>
// #include <curand.h>
#include "trans-conv.h"

// void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
//      curandGenerator_t prng;
//      curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
//      curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
//      curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
// }

void print_matrix(float *A, int nr_rows_A, int nr_cols_A) 
{
    for(int i = 0; i < nr_rows_A; ++i){
        for(int j = 0; j < nr_cols_A; ++j){
                std::cout << A[j * nr_rows_A + i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(void) {
    int i, j;
    int rows_input, cols_input, rows_kernel, cols_kernel, rows_output, cols_output;

    //for simplicity we define input and kernel both to be 2*2 matrices
    rows_input = cols_input = 3;
    rows_kernel = cols_kernel = 3;
    rows_output = rows_input + rows_kernel - 1;
    cols_output = cols_input + cols_kernel - 1;
 
    // allocate host memory
    float *host_input = (float *)malloc(rows_input * cols_input * sizeof(float));
    float *host_kernel = (float *)malloc(rows_kernel * cols_kernel * sizeof(float));
    float *host_output = (float *)malloc(rows_output * cols_output * sizeof(float));

    // initialize host matrices (do we need to initialize output?)
    int counter = 1.0;
    for(i=0; i<rows_input; i++){
        for(j=0; j<cols_input; j++){
            host_input[j * rows_input + i] = counter;
            counter += 1.0;
        }
    }

    for(i=0; i<rows_kernel; i++){
        for(j=0; j<cols_kernel; j++){
            host_kernel[j * rows_kernel + i] = counter;
            counter += 1.0;
        }
    }
    std::cout << "input =" << std::endl;
    print_matrix(host_input, rows_input, cols_input);

    std::cout << "kernel =" << std::endl;
    print_matrix(host_kernel, rows_kernel, cols_kernel);

    // try to test to trans_conv
    trans_conv(host_input, host_kernel, host_output, rows_input, cols_input, 1, 1, rows_kernel);

    // try gemm
    // gemm(host_input, host_kernel, host_output, 3, 3, 3);

    std::cout << "output =" << std::endl;
    print_matrix(host_output, rows_output, cols_output);

    // Free CPU memory
    free(host_input);
    free(host_kernel);
    free(host_output);
 
    return 0;
}