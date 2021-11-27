/*
    Perform general matrix multiplication as operc = opera @ operb.
    opera: the operand A. Should be opera[H][K] serialized array.
    opera: the operand B. Should be operb[K][W] serialized array.
    res: the result C. Should be operc[H][W] serialized array.
*/
void gemm(float *opera, float *operb, float *res, int H, int W, int K);
/*
    Perform general matrix addition as opera += operb.
    opera: the operand A. Should be opera[H][W] serialized array.
    opera: the operand B. Should be operb[H][W] serialized array.
*/
void gema(float *opera, float *operb, int H, int W);
/*
    input: the input tensor to be convolved. Should be input[H][W][C] serialized array.
    kernel: the kernel tensor. Should be kernel[C][M][K][K] serialized array. 
    output: the result output tensor. Should be output[M][H+K-1][W+K-1] serialized array. 
*/
void trans_conv(float *input, float *kernel, float *output, int H, int W, int C, int M, int K);
/*
    patch: the patch tensor to be merged back together. Should be patch [H][W][M][K][K] serialized array.
    output: the result output tensor. Should be output[M][H+K-1][W+K-1] serialized array. 
*/
void shift_add(float *patch, float *output, int H, int W, int M, int K);
