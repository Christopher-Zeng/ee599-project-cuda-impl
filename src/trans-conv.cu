bool gemm(float *opera, float *operb, float *operc, int H, int W, int K);
bool gema(float *opera, float *operb, float *operc, int H, int W);

/*
    input: the input tensor to be convolved. Should be input[H][W][C] serialized array.
    kernel: the kernel tensor. Should be kernel[C][M][K][K] serialized array. 
    output: the result output tensor. Should be result[M][H][W] serialized array. 
*/
bool trans_conv(float *input, float *kernel, float *output, int H, int W, int C, int M, int K);
/*
    patch: the patch tensor to be merged back together. Should be patch [H][W][M][K][K] serialized array.
    output: the result output tensor. Should be result[M][H][W] serialized array. 
*/
bool shift_add(float *patch, float *output, int H, int W, int M, int K)