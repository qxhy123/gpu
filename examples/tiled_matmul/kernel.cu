// examples/tiled_matmul/kernel.cu
extern "C" __global__ void tile_matmul(const float* A, const float* B, float* C) {
    __shared__ float sA[16*16], sB[16*16];
    int col = threadIdx.x, row = threadIdx.y;
    sA[row*16 + col] = A[row*16 + col];
    sB[row*16 + col] = B[row*16 + col];
    __syncthreads();
    float acc = 0.0f;
    for (int k = 0; k < 16; ++k)
        acc += sA[row*16 + k] * sB[k*16 + col];
    C[row*16 + col] = acc;
}
