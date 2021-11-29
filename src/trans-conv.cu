#include <stdlib.h>
#include <iostream>
#include <cublas_v2.h>
#include "trans-conv.h"
#include "test.h"

void trans_conv(
    float *input, float *kernel, float *output,
    const int H, const int W, const int C,
    const int M, const int KH, const int KW,
    const int SH, const int SW,
    const int PH, const int PW)
{
    // Output size
    int OH = SH * (H - 1) + KH - 2 * PH;
    int OW = SW * (W - 1) + KW - 2 * PW;

    // Memory transfer
    /*
    vramInput: 
    the input tensor to be feed into GEMM.
    Should be vramInput[H][W][C] serialized array.
    vramKernel: 
    the kernel tensor to be feed into GEMM.
    Should be vramKernel[C][M][KH][KW] serialized array
    */
    float *vramInput, *vramKernel;
    int inputSize = H * W * C;
    int kernelSize = C * M * KH * KW;
    cudaMalloc(&vramInput, inputSize * sizeof(float));
    cudaMalloc(&vramKernel, kernelSize * sizeof(float));
    cudaMemcpy(vramInput, input, inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vramKernel, kernel, kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    /*
    vramPatch: 
    the patch tensor to be merged back together. 
    Should be vramPatch[H][W][M][KH][KW] serialized array.
    */
    float *vramPatch;
    int patchSize = H * W * M * KH * KW;
    cudaMalloc(&vramPatch, patchSize * sizeof(float));
    cudaMemset(vramPatch, 0, patchSize);
    // Perform GEMM to get the patch matrix.
    gemm(vramInput, vramKernel, vramPatch, H * W, M * KH * KW, C);
    // Memory recycle
    cudaFree(vramInput);
    cudaFree(vramKernel);

    // DEBUG CODE
    float *patch = (float *)malloc(patchSize * sizeof(float));
    cudaMemcpy(patch, vramPatch, patchSize * sizeof(float), cudaMemcpyDeviceToHost);
    print_matrix(patch, H * W * M, KH * KW);
    free(patch);

    /*
    vramRowPatch: 
    the patch tensor that have already gone through the W dimension shift add,
    and to be feed into H dimension shift add.
    H patch rows, each of OW * M * KH size.
    */
    float *vramRowPatch;
    int rowPatchSize = H * OW * M * KH;
    cudaMalloc((void **)&vramRowPatch, rowPatchSize * sizeof(float));
    cudaMemset(vramRowPatch, 0, rowPatchSize);
    // Perform shift add over the row-axis of the patches.
    dim3 dimGridShiftAddRows(H, M);
    dim3 dimBlockShiftAddRows(KH, KW);
    shift_add_rows<<<dimGridShiftAddRows, dimBlockShiftAddRows>>>(vramPatch, vramRowPatch, W, SW, PW);
    // Memory recycle
    cudaFree(vramPatch);

    // DEBUG CODE
    float *rowPatch = (float *)malloc(rowPatchSize * sizeof(float));
    cudaMemcpy(rowPatch, vramRowPatch, rowPatchSize * sizeof(float), cudaMemcpyDeviceToHost);
    print_matrix(rowPatch, H * OW, M * KH);
    free(rowPatch);

    /*
    vramOutput: 
    the output tensor. 
    Should be output[OH][OW][M] serialized array.
    */
    float *vramOutput;
    int outputSize = OH * OW * M;
    cudaMalloc(&vramOutput, outputSize * sizeof(float));
    cudaMemset(vramOutput, 0, outputSize);
    // Perform shift_add over the columns of the patch rows.
    dim3 dimGridShiftAddCol(OW, M);
    dim3 dimBlockShiftAddCol(KH);
    shift_add_cols<<<dimGridShiftAddCol, dimBlockShiftAddCol>>>(vramRowPatch, vramOutput, H, SH, PH);
    // Memory recycle
    cudaFree(vramRowPatch);

    // Memory transfer
    cudaMemcpy(output, vramOutput, outputSize * sizeof(float), cudaMemcpyDeviceToHost);
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

__global__ void shift_add_rows(
    const float *vramPatch, float *vramRowPatch,
    int W, const int SW, const int PW)
{
    // Regain tensor dimensions;
    int H = gridDim.x;
    int M = gridDim.y;
    int KH = blockDim.x;
    int KW = blockDim.y;
    int OW = SW * (W - 1) + KW - 2 * PW;
    // Regain tensor indexes;
    int h = blockIdx.x;
    int m = blockIdx.y;
    int kh = threadIdx.x;
    int kw = threadIdx.y;
    // The index to be iterated over.
    int w, ow;

    // Don't remove the parenthesis if you want to understand the code after one minute.
    // rowPatch (H, OW, M, KH) accessed as rowPatch[h, ow, m, kh]
    int rowPatchStride = M * KH;
    int rowPatchOffset = (((h)*OW + 0) * M + m) * KH + kh;
    // patch (H, W, M, KH, KW) accessed as patch[h, w, m, kh, kw]
    int patchStride = M * KH * KW;
    int patchOffset = ((((h)*OW + 0) * M + m) * KH + kh) * KW + kw;

    /*
    Notice that the memory access pattern on rowPatch is not efficient yet. 
    Relies on device optimization with shared cached.
    */
    for (w = 0; w < W; ++w)
    {
        ow = SW * w + kw - PW;
        // equivilent to rowPatch[h, SW*w + kw - PW, m, kh] += patch[h, w, m, kh, kw];
        if (ow > -1 && ow < OW)
        {
            vramRowPatch[ow * rowPatchStride + rowPatchOffset] +=
                vramPatch[w * patchStride + patchOffset];
        }
        __syncthreads();
    }
}

__global__ void shift_add_cols(
    const float *rowPatch, float *vramOutput,
    int H, const int SH, const int PH)
{
    // Regain tensor dimensions;
    int OW = gridDim.x;
    int M = gridDim.y;
    int KH = blockDim.x;
    int OH = SH * (H - 1) + KH - 2 * PH;
    // Regain tensor indexes;
    int ow = blockIdx.x;
    int m = blockIdx.y;
    int kh = threadIdx.x;
    // The index to be iterated over.
    int h, oh;

    // Don't remove the parenthesis if you want to understand the code after one minute.
    // output (OH, OW, M) accessed as output[oh, ow, m]
    int outputStride = OW * M;
    int outputOffset = ((0) * OH + ow) * OW + m;
    // rowPatch (H, OW, M, KH) accessed as rowPatch[h, ow, m, kh]
    int rowPatchStride = OW * M * KH;
    int rowPatchOffset = (((0) * H + ow) * OW + m) * M + kh;

    for (h = 0; h < H; ++h)
    {
        oh = SH * h + kh - PH;
        if (oh > -1 && oh < OH)
        {
            // equivilent to output[m, S*h + k2, w] += rowPatch[m, h, w, k2];
            vramOutput[oh * outputStride + outputOffset] +=
                rowPatch[h * rowPatchStride + rowPatchOffset];
        }
    }
}
