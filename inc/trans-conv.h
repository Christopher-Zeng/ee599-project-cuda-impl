/*
    Perform transposed convolution with M sets of C-channel KH-by-KW kernels,
    on an input of C-channel H-by-W images, 
    by strides of SH over the H dimension and SW over the W dimension,
    and paddings of PH over the H dimension, and PW over the W dimension.
    Due to the definition of transposed convolution, 
    SH, PH <= KH, SW, PW <= 
    The result should be M sets of OH-by-OW images, 
    with OH = SH * (H-1) + KH - 2PH,
    and OW = SW * (W-1) + KW - 2PW.
    input: the input tensor to be convolved. Should be input[H][W][C] serialized array.
    kernel: the kernel tensor. Should be kernel[C][M][KH][KW] serialized array. 
    output: the result output tensor. Should be output[OH][OW][W] serialized array. 
*/
void trans_conv(
    float *input, float *kernel, float *output,
    const int H, const int W, const int C,
    const int M, const int KH, const int KW,
    const int SH, const int SW,
    const int PH, const int PW);
/*
    Perform general matrix multiplication as operc = opera @ operb.
    vramOpera: the operand A. Should be opera[H][K] serialized array on device.
    vramOperb: the operand B. Should be operb[K][W] serialized array on device.
    vramRes: the result C. Should be operc[H][W] serialized array on device.
*/
void gemm(
    const float *vramOpera, const float *vramOperb, float *vramRes,
    const int H, const int W, const int KW);
/*
    For H * W patches, each of M * KH * KW size, 
    this function perform shift add over the W dimension of the patches,
    by strides of SW and paddings of PW.
    The result should be H patch rows, each of OW * M * KH size, 
    where OW = SW * (W-1) + KW - 2PW.
    Alway called with (H, M) grid, (KH, KW) block.
*/
__global__ void shift_add_rows(
    const float *vramPatch, float *vramRowPatch,
    int W, const int SW, const int PW);
/*
    For H patch rows, each of OW * M * KH size,, 
    this function perform shift add over H dimension of the patch rows,
    by strides of SH and paddings of PH.
    The result should be OH * OW * M size,
    where OH = SH * (H-1) + KH - 2PH
    Alway called with (OW, M) grid, (KH) block.
*/
__global__ void shift_add_cols(
    const float *vramRowPatch, float *vramOutput,
    int H, const int SH, const int PH);
