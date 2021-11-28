/*
    input: the input tensor to be convolved. Should be input[H][W][C] serialized array.
    kernel: the kernel tensor. Should be kernel[C][M][K][K] serialized array. 
    output: the result output tensor. Should be output[M][H+K-1][W+K-1] serialized array. 
*/
void trans_conv(float *input, float *kernel, float *output, int H, int W, int C, int M, int K);
/*
    Perform general matrix multiplication as operc = opera @ operb.
    vramOpera: the operand A. Should be opera[H][K] serialized array on device.
    vramOperb: the operand B. Should be operb[K][W] serialized array on device.
    vramRes: the result C. Should be operc[H][W] serialized array on device.
*/
void gemm(float *vramOpera, float *vramOperb, float *vramRes, int H, int W, int K);
/*
    vramPatch: the patch tensor to be merged back together. Should be patch [H][W][M][K][K] serialized array on device.
    vramOutput: the result output tensor. Should be output[M][H+K-1][W+K-1] serialized array on device. 
*/
void shift_add(float *vramPatch, float *vramOutput, int H, int W, int M, int K);
