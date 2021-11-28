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
    // Arguements for trans_conv
    int H = 3;
    int W = 4;
    int C = 2;
    int M = 2;
    int KH = 3;
    int KW = 2;
    int SH = 2;
    int SW = 1;
    int PH = 2;
    int PW = 1;
    int OH = SH * (H - 1) + KH - 2 * PH;
    int OW = SW * (W - 1) + KW - 2 * PW;
    float *input = (float *)malloc(H * W * C * sizeof(float));
    float *kernel = (float *)malloc(C * M * KH * KW * sizeof(float));
    float *output = (float *)malloc(OH * OW * M * sizeof(float));
    init_matrix(input, H * W, C, 1.0);
    init_matrix(kernel, C * M, KH * KW, 1.0);

    std::cout << "input =" << std::endl;
    print_matrix(input, H * W, C);

    std::cout << "kernel =" << std::endl;
    print_matrix(kernel, C * M, KH * KW);

    // try to test to trans_conv
    trans_conv(input, kernel, output, H, W, C, M, KH, KW, SH, SW, PH, PW);

    std::cout << "conv output =" << std::endl;
    print_matrix(output, OH * OW, M);

    // Free CPU memory
    free(input);
    free(kernel);
    free(output);

    return 0;
}