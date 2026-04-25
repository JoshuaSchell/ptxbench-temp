#include <cuda_runtime.h>

extern "C" __global__ void exact_f64_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ out,
    int ncols,
    int nrows,
    float scale
) {
    __shared__ double smem[1024];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const float* x_row = x + static_cast<long long>(row) * ncols;

    double total = 0.0;
    for (int h = tid; h < nrows; h += stride) {
        const float* w_row = w + static_cast<long long>(h) * ncols;
        double sum = 0.0;
#pragma unroll 1
        for (int k = 0; k < ncols; ++k) {
            sum += static_cast<double>(x_row[k]) * static_cast<double>(w_row[k]);
        }
        total += sum;
    }

    smem[tid] = total;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            smem[tid] += smem[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[row] = static_cast<float>(smem[0] * static_cast<double>(scale));
    }
}
