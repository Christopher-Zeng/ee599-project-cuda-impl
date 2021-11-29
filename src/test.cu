#include <iostream>
#include <iterator>
#include <cstdlib>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <utility>
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

void read_data(std::string path, int *dim, float *input, float *kernel, float *goldenOutput)
{
    std::ifstream myFile(path);
    if (!myFile.is_open())
        throw std::runtime_error("Could not open file");

    dim = (int *)malloc(10 * sizeof(int));
    read_file<int>(myFile, dim);

    input = (float *)malloc(dim[0] * dim[1] * dim[2] * sizeof(float));
    read_file<float>(myFile, input);
    kernel = (float *)malloc(dim[3] * dim[4] * dim[5] * sizeof(float));
    read_file<float>(myFile, kernel);
    // goldenOutput = (float *)malloc(dim[0] * dim[1] * dim[3] sizeof(float));
    // read_file<float>(myFile, dim);
}

template <typename T>
void read_file(std::ifstream &fs, T *buffer)
{
    std::string line;
    int val, index;
    index = 0;
    std::getline(fs, line);
    std::stringstream ss(line);
    while (ss >> val)
    {
        std::cout << val << std::endl;
        buffer[index] = val;
        if (ss.peek() == ',')
            ss.ignore();
        index++;
    }
}

int main(void)
{
    int *dim;
    float *input;
    float *kernel;
    float *goldenOutput;

    read_data("/data/dim.csv", dim, input, kernel, goldenOutput);

    int H = dim[0];
    int W = dim[1];
    int C = dim[2];
    int M = dim[3];
    int KH = dim[4];
    int KW = dim[5];
    int SH = dim[6];
    int SW = dim[7];
    int PH = dim[8];
    int PW = dim[9];

    // Arguements for trans_conv
    int OH = SH * (H - 1) + KH - 2 * PH;
    int OW = SW * (W - 1) + KW - 2 * PW;
    float *output = (float *)malloc(OH * OW * M * sizeof(float));

    struct timespec start, stop;
    double time;
    if (clock_gettime(CLOCK_REALTIME, &start) == -1)
    {
        perror("clock gettime");
    }

    // try to test to trans_conv
    trans_conv(input, kernel, output, H, W, C, M, KH, KW, SH, SW, PH, PW);

    if (clock_gettime(CLOCK_REALTIME, &stop) == -1)
    {
        perror("clock gettime");
    }
    time = (stop.tv_sec - start.tv_sec) * 1e9 + (double)(stop.tv_nsec - start.tv_nsec);
    printf("time is %f ms\n", time / 1e3);

    std::cout << "conv output =" << std::endl;
    // print_matrix(output, OH * OW, M);

    // Free CPU memory
    free(input);
    free(kernel);
    free(output);

    return 0;
}
