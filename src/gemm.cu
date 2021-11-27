#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include "trans-conv.h"

void blas_gemm(const float *A, const float *B, float *C, const int m, const int k, const int n)
{
    // define cublasSgemm parameters
    int lda = m, ldb = k, ldc = m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    // create handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // call gemm
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

    // destory handle
    cublasDestroy(handle);
}

void row2col(float *A, float *target, int rows, int cols)
{
    for(int i=0; i<rows; ++i){
        for(int j=0; j<cols; ++j){
            target[j * rows + i] = A[i * cols + j];
        }
    }
}

void col2row(float *A, float *target, int rows, int cols)
{
    for(int i=0; i<rows; ++i){
        for(int j=0; j<cols; ++j){
            target[i * cols + j] = A[j * rows + i];
        }
    }
}

void gemm(float *opera, float *operb, float *res, int H, int W, int K){
    // define input and output dimensions
    int rows_A, cols_A, rows_B, cols_B, rows_C, cols_C;
    rows_A = rows_C = H;
    cols_A = rows_B = K;
    cols_B = cols_C = W;

    float *col_major_A = (float *)malloc(rows_A * cols_A * sizeof(float));
    float *col_major_B = (float *)malloc(rows_B * cols_B * sizeof(float));
    float *col_major_C = (float *)malloc(rows_C * cols_C * sizeof(float));

    row2col(opera, col_major_A, rows_A, cols_A);
    row2col(operb, col_major_B, rows_B, cols_B);

    // allocate device memories
    float *device_A, *device_B, *device_C;
    cudaMalloc(&device_A, rows_A * cols_A * sizeof(float));
    cudaMalloc(&device_B, rows_B * cols_B * sizeof(float));
    cudaMalloc(&device_C, rows_C * cols_C * sizeof(float));

    // set the values of device matrices
    cublasStatus_t status;
    status = cublasSetMatrix(rows_A, cols_A, sizeof(float), col_major_A, rows_A, device_A, rows_A);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        throw EXIT_FAILURE;
    }

    status = cublasSetMatrix(rows_B, cols_B, sizeof(float), col_major_B, rows_B, device_B, rows_B);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        throw EXIT_FAILURE;
    }

    // Multiply A and B on GPU
    blas_gemm(device_A, device_B, device_C, rows_A, cols_A, cols_B);

    // Copy the result on host memory
    cudaMemcpy(col_major_C, device_C, rows_C * cols_C * sizeof(float), cudaMemcpyDeviceToHost);

    col2row(col_major_C, res, rows_C, cols_C);

    //Free GPU memory
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
}