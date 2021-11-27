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

void print_col_matrix(float *A, int rows, int cols) 
{
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
                std::cout << A[j * rows + i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void print_row_matrix(float *A, int rows, int cols) 
{
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
                std::cout << A[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void init_col_major(float *A, int rows, int cols, float counter)
{
    for(int i=0; i<rows; ++i){
        for(int j=0; j<cols; ++j){
            A[j * rows + i] = counter;
            counter += 1.0;
        }
    }
}

void init_row_major(float *A, int rows, int cols, float counter)
{
    for(int i=0; i<rows; ++i){
        for(int j=0; j<cols; ++j){
            A[i*cols + j] = counter;
            counter += 1.0;
        }
    }
}

int main(void) {
    int rows_input, cols_input, rows_kernel, cols_kernel, rows_output, cols_output;

    //for simplicity we define input and kernel both to be 2*2 matrices
    rows_input = cols_input = 2;
    rows_kernel = cols_kernel = 2;
    rows_output = rows_input + rows_kernel - 1;
    cols_output = cols_input + cols_kernel - 1;
 
    // allocate host memory
    float *host_input = (float *)malloc(rows_input * cols_input * sizeof(float));
    float *host_kernel = (float *)malloc(rows_kernel * cols_kernel * sizeof(float));
    float *host_output = (float *)malloc(rows_output * cols_output * sizeof(float));

    // initialize host matrices in column major
    // init_col_major

    // initialize host matrices in row major
    init_row_major(host_input, rows_input, cols_input, 1.0);
    init_row_major(host_kernel, rows_kernel, cols_kernel, 5.0);

    std::cout << "input =" << std::endl;
    print_row_matrix(host_input, rows_input, cols_input);

    std::cout << "kernel =" << std::endl;
    print_row_matrix(host_kernel, rows_kernel, cols_kernel);

    // try to test to trans_conv
    trans_conv(host_input, host_kernel, host_output, rows_input, cols_input, 1, 1, rows_kernel);

    // try gemm
    // gemm(host_input, host_kernel, host_output, 3, 3, 3);

    std::cout << "output =" << std::endl;
    print_row_matrix(host_output, rows_output, cols_output);

    // Free CPU memory
    free(host_input);
    free(host_kernel);
    free(host_output);
 
    return 0;
}