#include <cuda_runtime.h>

extern "C" __global__ void exact_tile_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ out,
    int ncols,
    int nrows,
    float scale
) {
    constexpr int WARPS = 8;
    constexpr int TILE_K = 256;
    __shared__ float x_tile[TILE_K];
    __shared__ float warp_totals[WARPS];

    const int batch_row = blockIdx.x;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * 32 + lane;

    const float* x_row = x + static_cast<long long>(batch_row) * ncols;
    float warp_total = 0.0f;

    for (int h = warp; h < nrows; h += WARPS) {
        const float* w_row = w + static_cast<long long>(h) * ncols;
        float sum = 0.0f;

        for (int k0 = 0; k0 < ncols; k0 += TILE_K) {
            x_tile[tid] = x_row[k0 + tid];
            __syncthreads();

#pragma unroll
            for (int kk = lane; kk < TILE_K; kk += 32) {
                sum = fmaf(x_tile[kk], w_row[k0 + kk], sum);
            }
            __syncthreads();
        }

#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }
        if (lane == 0) {
            warp_total += sum;
        }
    }

    if (lane == 0) {
        warp_totals[warp] = warp_total;
    }
    __syncthreads();

    if (warp == 0) {
        float total = lane < WARPS ? warp_totals[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(0xffffffffu, total, offset);
        }
        if (lane == 0) {
            out[batch_row] = total * scale;
        }
    }
}
