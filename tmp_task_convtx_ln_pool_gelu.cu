#include <cuda_runtime.h>
#include <math_constants.h>

extern "C" __global__ void ln_width64_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    const float add_scalar,
    const unsigned int rows
) {
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= rows) {
        return;
    }

    const unsigned int base = row << 6;
    const float v0 = x[base + lane] + add_scalar;
    const float v1 = x[base + lane + 32] + add_scalar;

    float sum = v0 + v1;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }

    const float mean = __shfl_sync(0xffffffffu, sum * (1.0f / 64.0f), 0);
    const float d0 = v0 - mean;
    const float d1 = v1 - mean;
    float sq = d0 * d0 + d1 * d1;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sq += __shfl_down_sync(0xffffffffu, sq, offset);
    }
    const float var = __shfl_sync(0xffffffffu, sq, 0) * (1.0f / 64.0f) + 1.0e-5f;
    const float inv_std = 1.0f / sqrtf(var);

    out[base + lane] = d0 * inv_std * gamma[lane] + beta[lane];
    out[base + lane + 32] = d1 * inv_std * gamma[lane + 32] + beta[lane + 32];
}

__device__ __forceinline__ float gelu_exact_like(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
}

extern "C" __global__ void pool_gelu_kernel(
    const float* __restrict__ x,
    float* __restrict__ out
) {
    const unsigned int idx = static_cast<unsigned int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned int total = 32u * 64u * 16u * 32u * 32u;
    if (idx >= total) {
        return;
    }

    unsigned int t = idx;
    const unsigned int ow = t & 31u;
    t >>= 5;
    const unsigned int oh = t & 31u;
    t >>= 5;
    const unsigned int od = t & 15u;
    t >>= 4;
    const unsigned int c = t & 63u;
    const unsigned int n = t >> 6;

    const unsigned int in_w0 = ow << 1;
    const unsigned int in_h0 = oh << 1;
    const unsigned int in_d0 = od << 1;
    const unsigned int base = (((n * 64u + c) * 32u + in_d0) * 64u + in_h0) * 64u + in_w0;
    const unsigned int dhw = 64u * 64u;
    const unsigned int hw = 64u;

    float acc = 0.0f;
    acc += x[base];
    acc += x[base + 1u];
    acc += x[base + hw];
    acc += x[base + hw + 1u];
    acc += x[base + dhw];
    acc += x[base + dhw + 1u];
    acc += x[base + dhw + hw];
    acc += x[base + dhw + hw + 1u];
    out[idx] = gelu_exact_like(acc * 0.125f);
}
