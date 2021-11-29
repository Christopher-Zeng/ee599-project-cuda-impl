#include <iostream>
#include <iterator>
#include <cstdlib>
#include <stdlib.h>
#include <stdio.h>
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
    int KH, KW, SH, SW, PH, PW;
    KH = KW = K;
    SH = SW = 1;
    PH = PW = 0;
    int OH = SH * (H - 1) + KH - 2 * PH;
    int OW = SW * (W - 1) + KW - 2 * PW;
    float *output = (float *)malloc(OH * OW * M * sizeof(float));

    struct timespec start, stop; 
    double time;
    if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}

    // try to test to trans_conv
    trans_conv(host_input, host_kernel, output, H, W, C, M, KH, KW, SH, SW, PH, PW);

    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}	  
	time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
	printf("time is %f ns\n", time*1e9);	

    std::cout << "conv output =" << std::endl;
    print_matrix(output, OH * OW, M);

    // Free CPU memory
    free(host_input);
    free(host_kernel);
    free(output);

    return 0;
}
