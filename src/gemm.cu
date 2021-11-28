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

void gemm(float *opera, float *operb, float *res, int H, int W, int K){
    // define input and output dimensions
    int rows_A, cols_A, rows_B, cols_B;
    rows_A = H;
    cols_A = rows_B = K;
    cols_B = W;

    // allocate device memories
    float *device_A, *device_B;
    cudaMalloc(&device_A, rows_A * cols_A * sizeof(float));
    cudaMalloc(&device_B, rows_B * cols_B * sizeof(float));

    // set the values of device matrices
    cublasStatus_t status;
    status = cublasSetMatrix(rows_A, cols_A, sizeof(float), opera, rows_A, device_A, rows_A);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        throw EXIT_FAILURE;
    }

    status = cublasSetMatrix(rows_B, cols_B, sizeof(float), operb, rows_B, device_B, rows_B);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        throw EXIT_FAILURE;
    }

    // Multiply A and B on GPU
    blas_gemm(device_B, device_A, res, rows_B, cols_B, cols_A);

    // Copy the result on host memory
    // cudaMemcpy(C, res, rows_C * cols_C * sizeof(float), cudaMemcpyDeviceToHost);

    //Free GPU memory
    cudaFree(device_A);
    cudaFree(device_B);
}