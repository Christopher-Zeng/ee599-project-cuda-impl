#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <time.h>
#include <cstdio>
#include <cstdlib>

#define size 1024
#define grid_size 64
#define block_size 16

void gpuCublasMmul(cublasHandle_t &handle, const float *a, const float *b, float *c, 
                    const int m, const int k, const int n) 
{
    int lda=m,ldb=k,ldc=m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

int main()
{
    int x, y;
    float *a = (float *)malloc(sizeof(float) * size * size);
    float *b = (float *)malloc(sizeof(float) * size * size);
    float *c = (float *)malloc(sizeof(float) * size * size);

    // notice blas is in column major order. But it doesn't matter in this case.
    for (y = 0; y < size; ++y)
    {
        for (x = 0; x < size; ++x)
        {
            a[x + y * size] = 1.0;
            b[x + y * size] = 2.0;
            c[x + y * size] = 0.0;
        }
    }

    float *vram_a, *vram_b, *vram_c;
    cudaMalloc(&vram_a, sizeof(float) * size * size);
    cudaMalloc(&vram_b, sizeof(float) * size * size);
    cudaMalloc(&vram_c, sizeof(float) * size * size);

    struct timespec start, stop;
    double time;

    cudaMemcpy(a, vram_a, sizeof(float) * size * size, cudaMemcpyHostToDevice);
    cudaMemcpy(b, vram_b, sizeof(float) * size * size, cudaMemcpyHostToDevice);

    if (clock_gettime(CLOCK_REALTIME, &start) == -1)
    {
        perror("clock gettime");
    }
    // matrix_multiply<<<dimGrid, dimBlock>>>(vram_a, vram_b, vram_c);

    // cublas gemm
    cublasHandle_t handle;
    cublasCreate(&handle);
    gpuCublasMmul(handle, vram_a, vram_b, vram_c, size, size, size);

    cudaMemcpy(c, vram_c, sizeof(float)*size*size, cudaMemcpyDeviceToHost);

    if (clock_gettime(CLOCK_REALTIME, &stop) == -1)
    {
        perror("clock gettime");
    }
    time = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / 1e9;
    printf("time is %f ns\n", time * 1e9);

    cudaFree(vram_a);
    cudaFree(vram_b);
    cudaFree(vram_c);

    printf("the value of c[451][451] is %f.\n", c[451 + 451 * size]);

    free(a);
    free(b);
    free(c);

    return 0;
}
