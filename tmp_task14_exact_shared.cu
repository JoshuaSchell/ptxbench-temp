#include <cuda_runtime.h>

extern "C" __global__ void exact_shared_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ out,
    int ncols,
    int nrows,
    float scale
) {
    constexpr int TILE_K = 256;
    __shared__ float x_tile[TILE_K];
    __shared__ float smem[256];

    const int batch_row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* x_row = x + static_cast<long long>(batch_row) * ncols;

    float total = 0.0f;
    for (int h = tid; h < nrows; h += blockDim.x) {
        const float* w_row = w + static_cast<long long>(h) * ncols;
        float sum = 0.0f;
        for (int k0 = 0; k0 < ncols; k0 += TILE_K) {
            x_tile[tid] = x_row[k0 + tid];
            __syncthreads();
#pragma unroll 1
            for (int kk = 0; kk < TILE_K; ++kk) {
                sum = fmaf(x_tile[kk], w_row[k0 + kk], sum);
            }
            __syncthreads();
        }
        total += sum;
    }

    smem[tid] = total;
    __syncthreads();

    for (int offset = 128; offset > 0; offset >>= 1) {
        if (tid < offset) {
            smem[tid] += smem[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[batch_row] = smem[0] * scale;
    }
}
