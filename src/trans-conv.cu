#include <stdlib.h>
#include <iostream>
#include "trans-conv.h"
#include "test.h"

void trans_conv(float *input, float *kernel, float *output, int H, int W, int C, int M, int K)
{

    // Memory transfer
    float *vramInput, *vramKernel;
    cudaMalloc(&vramInput, H * W * C * sizeof(float));
    cudaMalloc(&vramKernel, C * M * K * K * sizeof(float));
    cudaMemcpy(vramInput, input, H * W * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vramKernel, kernel, C * M * K * K * sizeof(float), cudaMemcpyHostToDevice);

    // vramPatch: the patch tensor to be merged back together. Should be patch [H][W][M][K][K] serialized array.
    float *vramPatch;
    cudaMalloc(&vramPatch, H * W * M * K * K * sizeof(float));

    // Perform GEMM to get the patch matrix.
    gemm(input, kernel, vramPatch, H * W, M * K * K, C);

    // Memory management
    cudaFree(vramInput);
    cudaFree(vramKernel);

    // DEBUG CODE
    float *patch = (float *)malloc(H * W * M * K * K * sizeof(float));
    cudaMemcpy(patch, vramPatch, H * W * M * K * K * sizeof(float), cudaMemcpyDeviceToHost);
    print_matrix(patch, H * W, M * K * K);
    free(patch);

    // vramOutput: the output tensor. Should be output[M][H+K-1][W+K-1] serialized array.
    float *vramOutput;
    cudaMalloc(&vramOutput, M * H * W * sizeof(float));

    // Perform shift-add to convert the patch matrix to result matrix.
    shift_add(vramPatch, vramOutput, H, W, M, K);

    // Memory management
    cudaFree(vramPatch);

    // Memory transfer
    cudaMemcpy(output, vramOutput, M * H * W * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(vramOutput);
}

/*
    For H * W patches, each of M * K * K size, 
    this function perform shift_add over the rows of the patches.
    The result should be M * H patch rows, each of (W+K-1) * K size. 
    Alway called with (H) grid, (M, K, K) block.
*/
__global__ void shift_add_rows(float *patch, float *rowPatch, int W)
{
    // Regain tensor indexes;
    int h = blockIdx.x;
    int m = threadIdx.x;
    int k2 = threadIdx.y;
    int k1 = threadIdx.z;
    int w;

    int H = gridDim.x;
    int M = blockDim.x;
    int K = blockDim.y;

    // Don't remove the parenthesis if you want to understand the code after one minute.
    // rowPatch (M, H, (W+K-1), K) accessed as rowPatch[m, h, w + k1, k2]
    int rowPatchStride = K;
    int rowPatchOffset = k2 + k1 * (K) + h * ((W + K - 1) * K) + m * (H * (W + K - 1) * K);
    // patch (H, W, M, K, K) accessed as patch[h, w, m, k2, k1]
    int patchStride = M * K * K;
    int patchOffset = k1 + k2 * (K) + m * (K * K) + h * (W * M * K * K);

    if (k1 == 0)
    {
        for (w = 0; w < W + K - 1; ++w)
        {
            rowPatch[w * rowPatchStride + rowPatchOffset] = 0;
        }
    }
    __syncthreads();

    /*
    Notice that the memory access pattern on rowPatch is not efficient yet. 
    Relies on device optimization with shared cached.
    */
    for (w = 0; w < W; ++w)
    {
        // equivilent to rowPatch[m, h, w + k1, k2] += patch[h, w, m, k2, k1];
        rowPatch[w * rowPatchStride + rowPatchOffset] +=
            patch[w * patchStride + patchOffset];
        __syncthreads();
    }
}

/*
    For M * H patch rows, each of K * (W+K-1) size, 
    this function perform shift_add over the columns of the patch rows.
    The result should be M * (H+K-1) * (W+K-1) size.
    Alway called with (M) grid, (W + K - 1) block.
*/
__global__ void shift_add_cols(float *rowPatch, float *output, int H, int K)
{
    // Regain tensor indexes.
    int m = blockIdx.x;
    int w = threadIdx.x;
    int h, k2;

    int Wp = blockDim.x;

    // Don't remove the parenthesis if you want to understand the code after one minute.
    // output (M, (H+K-1), (W+K-1)) accessed as output[m, h + k2, w]
    int outputStride = Wp;
    int outputOffset = w + m * ((H + K - 1) * Wp);
    // rowPatch (M, H, (W+K-1), K) accessed as rowPatch[m, h, w, k2]
    int rowPatchStride = K * Wp;
    int rowPatchOffset = w * (K) + m * ((H + K - 1) * Wp * K);

    for (h = 0; h < (H + K - 1); ++h)
    {
        output[h * outputStride + outputOffset] = 0;
    }

    for (h = 0; h < H; ++h)
    {
        for (k2 = 0; k2 < K; ++k2)
        {
            // equivilent to output[m, h + k2, w] += rowPatch[m, h, w, k2];
            output[(h + k2) * outputStride + outputOffset] +=
                rowPatch[k2 + h * rowPatchStride + rowPatchOffset];
        }
    }
}

void shift_add(float *vramPatch, float *vramOutput, int H, int W, int M, int K)
{
    // Device Memory
    float *vramRowPatch;

    // DEBUG CODE
    float *patch = (float *)malloc(H * W * M * K * K * sizeof(float));
    cudaMemcpy(patch, vramPatch, H * W * M * K * K * sizeof(float), cudaMemcpyDeviceToHost);
    print_matrix(patch, H * W, M * K * K);
    free(patch);

    cudaMalloc((void **)&vramRowPatch, M * H * K * (W + K - 1) * sizeof(float));
    dim3 dimGridShiftAddRows(H);
    dim3 dimBlockShiftAddRows(M, K, K);
    shift_add_rows<<<dimGridShiftAddRows, dimBlockShiftAddRows>>>(vramPatch, vramRowPatch, W);
    cudaFree(vramPatch);

    // DEBUG CODE
    float *rowPatch = (float *)malloc(M * H * (W + K - 1) * K * sizeof(float));
    cudaMemcpy(rowPatch, vramRowPatch, M * H * (W + K - 1) * K * sizeof(float), cudaMemcpyDeviceToHost);
    print_matrix(rowPatch, M * H, (W + K - 1) * K);
    free(rowPatch);

    dim3 dimGridShiftAddCol(M);
    dim3 dimBlockShiftAddCol((W + K - 1));
    shift_add_cols<<<dimGridShiftAddCol, dimBlockShiftAddCol>>>(vramRowPatch, vramOutput, H, K);
    cudaFree(vramRowPatch);
}