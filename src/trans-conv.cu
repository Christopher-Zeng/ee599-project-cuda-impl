#include <stdlib.h>
#include "trans-conv.h"

/*
    Perform general matrix multiplication as operc = opera @ operb.
    opera: the operand A. Should be opera[H][K] serialized array.
    opera: the operand B. Should be operb[K][W] serialized array.
    res: the result C. Should be operc[H][W] serialized array.
*/
bool gemm(float *opera, float *operb, float *res, int H, int W, int K);

/*
    Perform general matrix addition as opera += operb.
    opera: the operand A. Should be opera[H][W] serialized array.
    opera: the operand B. Should be operb[H][W] serialized array.
*/
bool gema(float *opera, float *operb, int H, int W);

/*
    input: the input tensor to be convolved. Should be input[H][W][C] serialized array.
    kernel: the kernel tensor. Should be kernel[C][M][K][K] serialized array. 
    output: the result output tensor. Should be result[M][H][W] serialized array. 
*/
bool trans_conv(float *input, float *kernel, float *output, int H, int W, int C, int M, int K)
{
    // patch: the patch tensor to be merged back together. Should be patch [H][W][M][K][K] serialized array.
    float *patch;

    patch = (float *)malloc(H * W * M * K * K * sizeof(float));

    // Perform GEMM to get the patch matrix.
    gemm(input, kernel, patch, H * W, M * K * K, C);

    // Perform shift-add to convert the patch matrix to result matrix.
    shift_add(patch, output, H, W, M, K);

    return true;
}

/*
    patch: the patch tensor to be merged back together. Should be patch [H][W][M][K][K] serialized array.
    output: the result output tensor. Should be result[M][H][W] serialized array. 
*/
bool shift_add(float *patch, float *output, int H, int W, int M, int K)
{

    // Local variables
    // the starting y of the current patch
    int h;
    // the starting x of the current patch
    int w;
    // the starting z of the current patch
    int m;

    for (h = 0; h < H; ++h)
    {
        for (w = 0; w < W; ++w)
        {
            for (m = 0; m < M; ++m)
            {
                gema(
                    output + m * H * W + h * W + w,
                    patch + h * W * M * K * K + w * M * K * K + m * K * K,
                    K, K
                    );
            }
        }
    }

    return true;
}