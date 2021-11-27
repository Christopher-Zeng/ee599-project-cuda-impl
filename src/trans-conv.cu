#include <stdlib.h>

bool gemm(float *opera, float *operb, float *res, int H, int W, int K);

__global__ void gema_kernel(float *opera, float *operb)
{
    opera[threadIdx.x * blockDim.y + threadIdx.y] += operb[threadIdx.x * blockDim.y + threadIdx.y];
}

bool gema(float *opera, float *operb, int H, int W)
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

bool trans_conv(float *input, float *kernel, float *output, int H, int W, int C, int M, int K)
{
    // patch: the patch tensor to be merged back together. Should be patch [H][W][M][K][K] serialized array.
    float *patch;

    patch = (float *)malloc(H * W * M * K * K * sizeof(float));

    // Perform GEMM to get the patch matrix.
    gemm(input, kernel, patch, H * W, M * K * K, C);

    // Perform shift-add to convert the patch matrix to result matrix.
    shift_add(patch, output, H, W, M, K);

    free(patch);

    return output;
}

bool shift_add(float *patch, float *output, int H, int W, int M, int K)
{

    // Local variables
    // the starting y, x, z of the current patch
    int h, w, m;

    for (h = 0; h < H; ++h)
    {
        for (w = 0; w < W; ++w)
        {
            for (m = 0; m < M; ++m)
            {
                gema(
                    output + m * H * W + h * W + w,
                    patch + h * W * M * K * K + w * M * K * K + m * K * K,
                    K, K);
            }
        }
    }
}