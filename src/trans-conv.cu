#include <stdlib.h>
#include "trans-conv.h"

__global__ void gema_kernel(float *opera, float *operb)
{
    opera[threadIdx.x * blockDim.y + threadIdx.y] += operb[threadIdx.x * blockDim.y + threadIdx.y];
}

void gema(float *opera, float *operb, int H, int W)
{
    float *vramOpera, *vramOperb;
    cudaMalloc((void **)&vramOpera, H * W * sizeof(float));
    cudaMalloc((void **)&vramOperb, H * W * sizeof(float));
    cudaMemcpy(vramOpera, opera, H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vramOperb, operb, H * W * sizeof(float), cudaMemcpyHostToDevice);
    dim3 dimGrid(1);
    dim3 dimBlock(H, W);
    gema_kernel<<<dimGrid, dimBlock>>>(vramOpera, vramOperb);
    cudaMemcpy(opera, vramOpera, H * W * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(vramOpera);
    cudaFree(vramOperb);
}

void trans_conv(float *input, float *kernel, float *output, int H, int W, int C, int M, int K)
{
    // patch: the patch tensor to be merged back together. Should be patch [H][W][M][K][K] serialized array.
    float *patch;

    patch = (float *)malloc(H * W * M * K * K * sizeof(float));

    // Perform GEMM to get the patch matrix.
    gemm(input, kernel, patch, H * W, M * K * K, C);

    // Perform shift-add to convert the patch matrix to result matrix.
    shift_add(patch, output, H, W, M, K);

    free(patch);
}

/*
    For H * W patches, each of M * K * K size, 
    this function perform shift_add over the rows of the patches.
    The result should be M * H patch rows, each of K * (W+K-1) size. 
*/
__global__ void shift_add_rows(float *patch, float *patchRows, int W)
{
    // Regain tensor indexes;
    int h = blockIdx.x;
    int m = threadIdx.x;
    int x = threadIdx.y;
    int y = threadIdx.z;
    int w = 0;

    int H = gridDim.x;
    int M = blockDim.x;
    int K = blockDim.y;

    // Utilized on-chip cache to speed up.
    for (w = 0; w < W; ++w)
    {
        patchRows[m * H * K * (W + K - 1) +
                  h * K * (W + K - 1) +
                  x * (W + K - 1) +
                  y + w] +=
            patch[h * W * M * K * K +
                  w * M * K * K +
                  m * K * K +
                  x * K +
                  y];
    }
}

/*
    For M * H patch rows, each of K * (W+K-1) size, 
    this function perform shift_add over the columns of the patch rows.
    The result should be M * (H+K-1) * (W+K-1) size.
*/
__global__ void shift_add_cols(float *patchRows, float *output, int H)
{
    // Regain tensor indexes.
    int m = blockIdx.x;
    int w = threadIdx.x;
    int x = threadIdx.y;
    int h = 0;

    int W = blockDim.x;
    int K = blockDim.y;

    for (h = 0; h < H; ++h)
    {
        output[m * (H + K - 1) * W +
               (h + x) * W +
               w] +=
            patchRows[m * H * K * W +
                      h * K * W +
                      x * W +
                      w];
    }
}

void shift_add(float *patch, float *output, int H, int W, int M, int K)
{
    // Device Memory
    float *vramPatch, *vramPatchRows, *vramOutput;

    cudaMalloc((void **)&vramPatch, H * W * M * K * K * sizeof(float));
    cudaMalloc((void **)&vramPatchRows, M * H * K * (W + K - 1) * sizeof(float));
    cudaMemcpy(vramPatch, patch, H * W * M * K * K * sizeof(float), cudaMemcpyHostToDevice);
    dim3 dimGridShiftAddRows(H);
    dim3 dimBlockShiftAddRows(M, K, K);
    shift_add_rows<<<dimGridShiftAddRows, dimBlockShiftAddRows>>>(vramPatch, vramPatchRows, W);
    cudaFree(vramPatch);

    cudaMalloc((void **)&vramOutput, M * (H + K - 1) * (W + K - 1) * sizeof(float));
    dim3 dimGridShiftAddCol(M);
    dim3 dimBlockShiftAddCol((W + K - 1), K);
    shift_add_cols<<<dimGridShiftAddCol, dimBlockShiftAddCol>>>(vramPatchRows, vramOutput, H);
    cudaFree(vramPatchRows);
    cudaMemcpy(output, vramOutput, M * (H + K - 1) * (W + K - 1) * sizeof(float), cudaMemcpyDeviceToHost);
}