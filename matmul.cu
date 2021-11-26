#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <time.h>

#define size 1024
#define grid_size 64
#define block_size 16

__global__ void matrix_multiply(int *a, int *b, int *c)
{
    int my_x, my_y, pos;
    my_x = blockIdx.x * blockDim.x + threadIdx.x;
    my_y = blockIdx.y * blockDim.y + threadIdx.y;
    c[my_x + my_y * size] = 0;
    for (pos = 0; pos < size; ++pos)
    {
        c[my_x + my_y * size] += a[pos + my_y * size] * b[my_x + pos * size];
    }
}

int main()
{
    int x, y;
    int *a = (int *)malloc(sizeof(int) * size * size);
    int *b = (int *)malloc(sizeof(int) * size * size);
    int *c = (int *)malloc(sizeof(int) * size * size);

    for (y = 0; y < size; ++y)
    {
        for (x = 0; x < size; ++x)
        {
            a[x + y * size] = 1;
            b[x + y * size] = 2;
            c[x + y * size] = 0;
        }
    }

    int *vram_a, *vram_b, *vram_c;
    cudaMalloc((void **)&vram_a, sizeof(int) * size * size);
    cudaMalloc((void **)&vram_b, sizeof(int) * size * size);
    cudaMalloc((void **)&vram_c, sizeof(int) * size * size);

    struct timespec start, stop;
    double time;

    cudaMemcpy(vram_a, a, sizeof(int) * size * size, cudaMemcpyHostToDevice);
    cudaMemcpy(vram_b, b, sizeof(int) * size * size, cudaMemcpyHostToDevice);

    dim3 dimGrid(grid_size, grid_size);
    dim3 dimBlock(block_size, block_size);

    if (clock_gettime(CLOCK_REALTIME, &start) == -1)
    {
        perror("clock gettime");
    }
    matrix_multiply<<<dimGrid, dimBlock>>>(vram_a, vram_b, vram_c);
    cudaMemcpy(c, vram_c, sizeof(int) * size * size, cudaMemcpyDeviceToHost);

    if (clock_gettime(CLOCK_REALTIME, &stop) == -1)
    {
        perror("clock gettime");
    }
    time = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / 1e9;
    printf("time is %f ns\n", time * 1e9);

    cudaFree(vram_a);
    cudaFree(vram_b);
    cudaFree(vram_c);

    printf("the value of c[451][451] is %d.\n", c[451 + 451 * size]);

    free(a);
    free(b);
    free(c);

    return 0;
}
