#include <iostream>
#include <iterator>
#include <cstdlib>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
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
    const int W = 64;
    const int C = 256;
    const int M = C;
    const int KH = 7;
    const int KW = KH;
    const int SH = 7;
    const int SW = SH;
    const int PH = SH;
    const int PW = PH;
    const int OH = SH * (H - 1) + KH - 2 * PH;
    const int OW = SW * (W - 1) + KW - 2 * PW;

    struct timespec start, stop;
    double avg_execution_time = 0;
    double avg_mean_square_errors = 0;
    double avg_mean_absolute_percentage_errors = 0;

    float *kernel = (float *)malloc(C * M * KH * KW * sizeof(float));
    float *input = (float *)malloc(H * W * C * sizeof(float));
    float *output = (float *)malloc(OH * OW * M * sizeof(float));
    float *golden = (float *)malloc(OH * OW * M * sizeof(float));
    double *execution_times = (double *)malloc(N * sizeof(double));
    double *mean_square_errors = (double *)malloc(N * sizeof(double));
    double *mean_absolute_percentage_errors = (double *)malloc(N * sizeof(double));

    read_matrix("./data/kernel.csv", kernel);

    for (int n = 0; n < N; ++n)
    {
        printf("Epoch %d.\n", n);
        read_matrix("./data/input_" + std::to_string(n) + ".csv", input);

        clock_gettime(CLOCK_REALTIME, &start);
        trans_conv(input, kernel, output, H, W, C, M, KH, KW, SH, SW, PH, PW);
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_REALTIME, &stop);

        execution_times[n] = (double)(stop.tv_sec - start.tv_sec) * 1e6 + (double)(stop.tv_nsec - start.tv_nsec) / 1e3;

        read_matrix("./data/output_" + std::to_string(n) + ".csv", golden);
        mean_square_errors[n] = 0;
        mean_absolute_percentage_errors[n] = 0;
        for (int pos = 0; pos < OH * OW * M; ++pos)
        {
            mean_square_errors[n] += pow(output[pos] - golden[pos], 2);
            mean_absolute_percentage_errors[n] += abs(output[pos] - golden[pos]) / golden[pos];
        }
        mean_square_errors[n] /= OH * OW * M;
        mean_absolute_percentage_errors[n] /= OH * OW * M;
    }

    for (int n = 0; n < N; ++n)
    {
        avg_execution_time += execution_times[n];
        avg_mean_square_errors += mean_square_errors[n];
        avg_mean_absolute_percentage_errors += mean_absolute_percentage_errors[n];
    }
    avg_execution_time /= N;
    avg_mean_square_errors /= N;
    avg_mean_absolute_percentage_errors /= N;

    printf("CUDA: Average execution time per sample is %f ms.\n", avg_execution_time);
    printf("CUDA: Average MSE per sample is %f\n", avg_mean_square_errors);
    printf("CUDA: Average MAPE per sample is %f\n", avg_mean_absolute_percentage_errors);
    // 586MiB
    // CUDA: Average execution time per sample is 49870.893953 ms.

    // Free CPU memory
    free(input);
    free(kernel);
    free(output);
    free(golden);
    free(execution_times);
    free(mean_square_errors);
    free(mean_absolute_percentage_errors);

    return 0;
}
