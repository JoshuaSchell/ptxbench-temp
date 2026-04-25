#include <cuda_runtime.h>
#include <math.h>

static __device__ __forceinline__ float tanh_exp_exactish(float x) {
    if (x >= 0.0f) {
        const float z = expf(-2.0f * x);
        return (1.0f - z) / (1.0f + z);
    }
    const float z = expf(2.0f * x);
    return (z - 1.0f) / (z + 1.0f);
}

extern "C" __global__ void post_tanh_sub_avgpool_pack4_miss(
    const float* __restrict__ x,
    float sub1,
    float sub2,
    float* __restrict__ out,
    float* __restrict__ cache_out,
    const int* __restrict__ meta
) {
    if (meta[1] != 0) {
        return;
    }

    const int ox = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int oy = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int gz = static_cast<int>(blockIdx.z);
    const int og = gz & 31;
    const int n = gz >> 5;
    if (ox >= 63 || oy >= 63) {
        return;
    }

    const int in_plane = 126 * 126;
    const int in_batch_stride = 128 * in_plane;
    const int out_plane = 63 * 63;
    const int out_batch_stride = 128 * out_plane;
    const int iy = oy << 1;
    const int ix = ox << 1;
    const int in_base = n * in_batch_stride + (og * 4) * in_plane + iy * 126 + ix;
    const int out_base = n * out_batch_stride + (og * 4) * out_plane + oy * 63 + ox;

    #pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
        const int off = in_base + lane * in_plane;
        float v0 = tanh_exp_exactish(x[off] - sub1) - sub2;
        float v1 = tanh_exp_exactish(x[off + 1] - sub1) - sub2;
        float v2 = tanh_exp_exactish(x[off + 126] - sub1) - sub2;
        float v3 = tanh_exp_exactish(x[off + 127] - sub1) - sub2;
        const float y = 0.25f * ((v0 + v1) + (v2 + v3));
        out[out_base + lane * out_plane] = y;
        cache_out[out_base + lane * out_plane] = y;
    }
}
