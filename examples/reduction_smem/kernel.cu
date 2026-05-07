// examples/reduction_smem/kernel.cu — for cross-reference
extern "C" __global__ void reduce32(const int* A, int* OUT) {
    __shared__ int s[32];
    int tid = threadIdx.x;
    s[tid] = A[tid]; __syncthreads();
    for (int off = 16; off > 0; off >>= 1) {
        if (tid < off) s[tid] += s[tid + off];
        __syncthreads();
    }
    if (tid == 0) *OUT = s[0];
}
