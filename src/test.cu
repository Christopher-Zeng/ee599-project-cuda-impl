#include <iostream>
#include <iterator>
#include <cstdlib>
#include <time.h>
#include <fstream>
#include <sstream>
#include <utility>
#include <stdexcept>
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
    int *dim = (int *) malloc (5*sizeof(int));

    std::ifstream myFile("data/dim.csv");
    if(!myFile.is_open()) throw std::runtime_error("Could not open file");

    std::string line;
    int dim_val, index;
    index = 0;
    std::getline(myFile, line);
    std::stringstream ss(line);
    while(ss >> dim_val){
        std::cout << dim_val << std::endl;
        dim[index] = dim_val;
        if(ss.peek() == ',') ss.ignore();
        index ++;
    }

    int H = dim[0];
    int W = dim[1];
    int C = dim[2];
    int M = dim[3];
    int K = dim[4];

    float *host_input = (float *)malloc(H * W * C * sizeof(float));
    float host_val;
    index = 0;
    std::getline(myFile, line);
    std::stringstream s1(line);
    while(s1 >> host_val){
        host_input[index] = host_val;
        if(s1.peek() == ',') s1.ignore();
        index ++;
    }

    float *host_kernel = (float *)malloc(C * M * K * K * sizeof(float));
    index = 0;
    std::getline(myFile, line);
    std::stringstream s2(line);
    while(s2 >> host_val){
        host_kernel[index] = host_val;
        if(s2.peek() == ',') s2.ignore();
        index ++;
    }
    
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
