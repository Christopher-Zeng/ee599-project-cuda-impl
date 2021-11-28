#include <stdlib.h>
#include <iostream>
#include <cublas_v2.h>
#include "trans-conv.h"
#include "test.h"

void trans_conv(
    float *input, float *kernel, float *output,
    const int H, const int W, const int C, const int M, const int K)
{
    // Memory transfer
    float *vramInput, *vramKernel;
    cudaMalloc(&vramInput, H * W * C * sizeof(float));
    cudaMalloc(&vramKernel, C * M * K * K * sizeof(float));
    cudaMemcpy(vramInput, input, H * W * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vramKernel, kernel, C * M * K * K * sizeof(float), cudaMemcpyHostToDevice);

    /*
    vramPatch: the patch tensor to be merged back together. 
    Should be vramPatch[H][W][M][K][K] serialized array.
    */
    float *vramPatch;
    cudaMalloc(&vramPatch, H * W * M * K * K * sizeof(float));
    // Perform GEMM to get the patch matrix.
    gemm(vramInput, vramKernel, vramPatch, H * W, M * K * K, C);
    // Memory recycle
    cudaFree(vramInput);
    cudaFree(vramKernel);

    // DEBUG CODE
    float *patch = (float *)malloc(H * W * M * K * K * sizeof(float));
    cudaMemcpy(patch, vramPatch, H * W * M * K * K * sizeof(float), cudaMemcpyDeviceToHost);
    print_matrix(patch, H * W, M * K * K);
    free(patch);

    /*
    vramRowPatch: The patch tensor of rows.
    M * H patch rows, each of (W+K-1) * K size.
    */
    float *vramRowPatch;
    cudaMalloc((void **)&vramRowPatch, M * H * K * (W + K - 1) * sizeof(float));
    // Perform shift add over the row-axis of the patches.
    dim3 dimGridShiftAddRows(H);
    dim3 dimBlockShiftAddRows(M, K, K);
    shift_add_rows<<<dimGridShiftAddRows, dimBlockShiftAddRows>>>(vramPatch, vramRowPatch, W);
    // Memory recycle
    cudaFree(vramPatch);

    // DEBUG CODE
    float *rowPatch = (float *)malloc(M * H * (W + K - 1) * K * sizeof(float));
    cudaMemcpy(rowPatch, vramRowPatch, M * H * (W + K - 1) * K * sizeof(float), cudaMemcpyDeviceToHost);
    print_matrix(rowPatch, M * H, (W + K - 1) * K);
    free(rowPatch);

    /*
    vramOutput: the output tensor. 
    Should be output[M][H+K-1][W+K-1] serialized array.
    */
    float *vramOutput = NULL;
    cudaMalloc(&vramOutput, M * (H + K - 1) * (W + K - 1) * sizeof(float));
    // Perform shift_add over the columns of the patch rows.
    dim3 dimGridShiftAddCol(M);
    dim3 dimBlockShiftAddCol((W + K - 1));
    shift_add_cols<<<dimGridShiftAddCol, dimBlockShiftAddCol>>>(vramRowPatch, vramOutput, H, K);
    // Memory recycle
    cudaFree(vramRowPatch);

    // Memory transfer
    cudaMemcpy(output, vramOutput, M * (H + K - 1) * (W + K - 1) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(vramOutput);
}

void gemm(
    const float *vramOpera, const float *vramOperb, float *vramRes,
    const int H, const int W, const int K)
{
    // BLAS GEMM arguements
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1;
    const float beta = 0;

    /*
    Notice that althought BLAS routines assume column-major layout for matrices, 
    the change of layout is equivalent to matrix transpose.
    With the simple math fact that (B^T * A^T) = (AB)^T, 
    it is evident that we can call this BLAS routine with row-major layout matrices,
    and still get the correct row-major layout result,
    if we just exchange the operands. 
    */
    cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        W, H, K,
        &alpha,
        vramOperb, W,
        vramOpera, K,
        &beta,
        vramRes, W);

    // destory handle
    cublasDestroy(handle);
}

/*
    For H * W patches, each of M * K * K size, 
    this function perform shift_add over the rows of the patches.
    The result should be M * H patch rows, each of (W+K-1) * K size. 
    Alway called with (H) grid, (M, K, K) block.
*/
__global__ void shift_add_rows(
    const float *vramPatch, float *vramRowPatch, int W)
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
            vramRowPatch[w * rowPatchStride + rowPatchOffset] = 0;
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
        vramRowPatch[w * rowPatchStride + rowPatchOffset] +=
            vramPatch[w * patchStride + patchOffset];
        __syncthreads();
    }
}

/*
    For M * H patch rows, each of K * (W+K-1) size, 
    this function perform shift_add over the columns of the patch rows.
    The result should be M * (H+K-1) * (W+K-1) size.
    Alway called with (M) grid, (W + K - 1) block.
*/
__global__ void shift_add_cols(
    const float *rowPatch, float *vramOutput,
    const int H, const int K)
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
        vramOutput[h * outputStride + outputOffset] = 0;
    }

    for (h = 0; h < H; ++h)
    {
        for (k2 = 0; k2 < K; ++k2)
        {
            // equivilent to output[m, h + k2, w] += rowPatch[m, h, w, k2];
            vramOutput[(h + k2) * outputStride + outputOffset] +=
                rowPatch[k2 + h * rowPatchStride + rowPatchOffset];
        }
    }
}
