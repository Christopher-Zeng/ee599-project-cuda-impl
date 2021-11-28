#include <iostream>
#include <cstdlib>
#include <time.h>
#include "trans-conv.h"
#include "test.h"

void print_matrix(float *mat, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            std::cout << mat[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void init_matrix(float *mat, int rows, int cols, float seed)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            mat[i * cols + j] = i * cols + j + seed;
        }
    }
}

int main(void)
{
    int rows_input, cols_input, rows_kernel, cols_kernel, rows_output, cols_output;

    //for simplicity we define input and kernel both to be 2*2 matrices
    rows_input = cols_input = 2;
    rows_kernel = cols_kernel = 2;
    rows_output = rows_input + rows_kernel - 1;
    cols_output = cols_input + cols_kernel - 1;

    // allocate host memory
    float *host_input = (float *)malloc(rows_input * cols_input * sizeof(float));
    float *host_kernel = (float *)malloc(rows_kernel * cols_kernel * sizeof(float));
    float *host_conv_output = (float *)malloc(rows_output * cols_output * sizeof(float));

    // initialize host matrices in row major
    init_matrix(host_input, rows_input, cols_input, 1.0);
    init_matrix(host_kernel, rows_kernel, cols_kernel, 5.0);

    std::cout << "input =" << std::endl;
    print_matrix(host_input, rows_input, cols_input);

    std::cout << "kernel =" << std::endl;
    print_matrix(host_kernel, rows_kernel, cols_kernel);

    // try to test to trans_conv
    trans_conv(host_input, host_kernel, host_conv_output, rows_input, cols_input, 1, 1, rows_kernel);

    std::cout << "conv output =" << std::endl;
    print_matrix(host_conv_output, rows_output, cols_output);

    // Free CPU memory
    free(host_input);
    free(host_kernel);
    free(host_conv_output);

    return 0;
}