/*
    input: the input tensor to be convolved. Should be input[H][W][C] serialized array.
    kernel: the kernel tensor. Should be kernel[C][M][K][K] serialized array. 
    output: the result output tensor. Should be output[M][H+K-1][W+K-1] serialized array. 
*/
void trans_conv(
    float *input, float *kernel, float *output,
    const int H, const int W, const int C, const int M, const int K);
/*
    Perform general matrix multiplication as operc = opera @ operb.
    vramOpera: the operand A. Should be opera[H][K] serialized array on device.
    vramOperb: the operand B. Should be operb[K][W] serialized array on device.
    vramRes: the result C. Should be operc[H][W] serialized array on device.
*/
void gemm(
    const float *vramOpera, const float *vramOperb, float *vramRes,
    const int H, const int W, const int K);
/*
    For H * W patches, each of M * K * K size, 
    this function perform shift_add over the rows of the patches.
    The result should be M * H patch rows, each of (W+K-1) * K size. 
    Alway called with (H) grid, (M, K, K) block.
*/
__global__ void shift_add_rows(
    const float *vramPatch, float *vramRowPatch, int W);
/*
    For M * H patch rows, each of K * (W+K-1) size, 
    this function perform shift_add over the columns of the patch rows.
    The result should be M * (H+K-1) * (W+K-1) size.
    Alway called with (M) grid, (W + K - 1) block.
*/
__global__ void shift_add_cols(
    const float *rowPatch, float *vramOutput,
    const int H, const int K);
