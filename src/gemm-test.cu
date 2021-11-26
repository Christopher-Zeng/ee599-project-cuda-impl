#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <cublas_v2.h>
#include <time.h>
#include <cstdio>
#include <cstdlib>
#include <curand.h>

#define size 3
#define BLOCK_SIZE 3

void blas_gemm(const float *A, const float *B, float *C, const int m, const int k, const int n) {
    int lda=m,ldb=k,ldc=m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    cublasHandle_t handle;
    cublasCreate(&handle); 
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    cublasDestroy(handle);
}

void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
     curandGenerator_t prng;
     curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
     curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
     curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

void print_matrix(const float *A, int nr_rows_A, int nr_cols_A) {

    for(int i = 0; i < nr_rows_A; ++i){
       for(int j = 0; j < nr_cols_A; ++j){
            std::cout << A[j * nr_rows_A + i] << " ";
       }
      std::cout << std::endl;
    }
   std::cout << std::endl;
}

int main() {
    int i;
    int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
    nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = size;
 
    // allocate host memory
    float *host_A = (float *)malloc(nr_rows_A * nr_cols_A * sizeof(float));
    float *host_B = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
    float *host_C = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));
    cublasStatus_t status;

    // initialize host memory
    for(i=0; i<size*size; i++){
        host_A[i] = 1.0;
        host_B[i] = 2.0;
        host_C[i] = 3.0;
    }

    // allocate device memory
    float *device_A, *device_B, *device_C;
    cudaMalloc(&device_A,nr_rows_A * nr_cols_A * sizeof(float));
    cudaMalloc(&device_B,nr_rows_B * nr_cols_B * sizeof(float));
    cudaMalloc(&device_C,nr_rows_C * nr_cols_C * sizeof(float));

    // copy host to memory, or instead we can fill rand on device
    // GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);
    // GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);
    status = cublasSetMatrix(BLOCK_SIZE, BLOCK_SIZE, sizeof(float), host_A, BLOCK_SIZE, device_A, BLOCK_SIZE);
    if (status != CUBLAS_STATUS_SUCCESS) 
    {
        return EXIT_FAILURE;
    }

    status = cublasSetMatrix(BLOCK_SIZE, BLOCK_SIZE, sizeof(float), host_B, BLOCK_SIZE, device_B, BLOCK_SIZE);
    if (status != CUBLAS_STATUS_SUCCESS) 
    {
        return EXIT_FAILURE;
    }
 
    // Multiply A and B on GPU
    blas_gemm(device_A, device_B, device_C, nr_rows_A, nr_cols_A, nr_cols_B);
 
    // Copy (and print) the result on host memory
    cudaMemcpy(host_C,device_C,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);
    std::cout << "C =" << std::endl;
    print_matrix(host_C, nr_rows_C, nr_cols_C);
 
    //Free GPU memory
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);

    // Free CPU memory
    free(host_A);
    free(host_B);
    free(host_C);
 
    return 0;
}