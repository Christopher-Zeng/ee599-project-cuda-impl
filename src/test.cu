#include <iostream>
#include <iterator>
#include <cstdlib>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <utility>
#include "trans-conv.h"
#include "test.h"
#include "csv_parser.hpp"

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

void read_matrix(std::string filename, float *matrix)
{
    csv::CSVReader kernelReader = csv::CSVReader(filename);

    int matrixPos = 0;
    for (auto &row : kernelReader)
    {
        for (auto &field : row)
        {
            matrix[matrixPos++] = field.get<float>();
        }
    }
}

int main(void)
{
    const int N = 64;
    const int H = 32;
    const int W = 32;
    const int C = 256;
    const int M = 256;
    const int KH = 7;
    const int KW = 7;
    const int SH = 3;
    const int SW = 3;
    const int PH = 3;
    const int PW = 3;
    const int OH = SH * (H - 1) + KH - 2 * PH;
    const int OW = SW * (W - 1) + KW - 2 * PW;

    struct timespec start, stop;
    double avg_execution_time = 0;

    float *kernel = (float *)malloc(C * M * KH * KW * sizeof(float));
    float *input = (float *)malloc(H * W * C * sizeof(float));
    float *output = (float *)malloc(OH * OW * M * sizeof(float));
    double *execution_times = (double *)malloc(N * sizeof(double));

    read_matrix("./data/kernel.csv", kernel);

    for (int n = 0; n < N; ++n)
    {
        printf("Epoch %d.\n", n);
        read_matrix("./data/input_" + std::to_string(n) + ".csv", input);
        clock_gettime(CLOCK_REALTIME, &start);
        trans_conv(input, kernel, output, H, W, C, M, KH, KW, SH, SW, PH, PW);
        clock_gettime(CLOCK_REALTIME, &stop);
        execution_times[n] = (double)(stop.tv_sec - start.tv_sec) * 1e6 + (double)(stop.tv_nsec - start.tv_nsec) / 1e3;
    }

    for (int n = 0; n < N; ++n)
    {
        avg_execution_time += execution_times[n];
    }
    avg_execution_time /= N;

    printf("CUDA: Average execution time per sample is %f ms.\n", avg_execution_time);
    // 1074MiB
    // CUDA: Average execution time per sample is 50304.213891 ms.

    // Free CPU memory
    free(input);
    free(kernel);
    free(output);

    return 0;
}
