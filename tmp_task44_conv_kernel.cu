#include <cuda_runtime.h>

static __device__ __forceinline__ float tf32_round(float x) {
    unsigned int u;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(u) : "f"(x));
    return __uint_as_float(u);
}

extern "C" __global__ void conv3x3_bias_pack4_miss(
    const float* __restrict__ x,
    const float4* __restrict__ wpack,
    const float4* __restrict__ bpack,
    float* __restrict__ out,
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
    if (ox >= 126 || oy >= 126) {
        return;
    }

    const int in_batch_stride = 64 * 128 * 128;
    const int in_channel_stride = 128 * 128;
    const int out_plane = 126 * 126;
    const int out_batch_stride = 128 * out_plane;
    const int x_batch = n * in_batch_stride;
    const int w_group = og * 64 * 9;

    float4 acc = bpack[og];

    #pragma unroll 1
    for (int c = 0; c < 64; ++c) {
        const int x_chan = x_batch + c * in_channel_stride + oy * 128 + ox;
        const int w_base = w_group + c * 9;

        const float x00 = tf32_round(x[x_chan]);
        const float4 w00 = wpack[w_base];
        acc.x = fmaf(x00, tf32_round(w00.x), acc.x);
        acc.y = fmaf(x00, tf32_round(w00.y), acc.y);
        acc.z = fmaf(x00, tf32_round(w00.z), acc.z);
        acc.w = fmaf(x00, tf32_round(w00.w), acc.w);

        const float x01 = tf32_round(x[x_chan + 1]);
        const float4 w01 = wpack[w_base + 1];
        acc.x = fmaf(x01, tf32_round(w01.x), acc.x);
        acc.y = fmaf(x01, tf32_round(w01.y), acc.y);
        acc.z = fmaf(x01, tf32_round(w01.z), acc.z);
        acc.w = fmaf(x01, tf32_round(w01.w), acc.w);

        const float x02 = tf32_round(x[x_chan + 2]);
        const float4 w02 = wpack[w_base + 2];
        acc.x = fmaf(x02, tf32_round(w02.x), acc.x);
        acc.y = fmaf(x02, tf32_round(w02.y), acc.y);
        acc.z = fmaf(x02, tf32_round(w02.z), acc.z);
        acc.w = fmaf(x02, tf32_round(w02.w), acc.w);

        const float x10 = tf32_round(x[x_chan + 128]);
        const float4 w10 = wpack[w_base + 3];
        acc.x = fmaf(x10, tf32_round(w10.x), acc.x);
        acc.y = fmaf(x10, tf32_round(w10.y), acc.y);
        acc.z = fmaf(x10, tf32_round(w10.z), acc.z);
        acc.w = fmaf(x10, tf32_round(w10.w), acc.w);

        const float x11 = tf32_round(x[x_chan + 129]);
        const float4 w11 = wpack[w_base + 4];
        acc.x = fmaf(x11, tf32_round(w11.x), acc.x);
        acc.y = fmaf(x11, tf32_round(w11.y), acc.y);
        acc.z = fmaf(x11, tf32_round(w11.z), acc.z);
        acc.w = fmaf(x11, tf32_round(w11.w), acc.w);

        const float x12 = tf32_round(x[x_chan + 130]);
        const float4 w12 = wpack[w_base + 5];
        acc.x = fmaf(x12, tf32_round(w12.x), acc.x);
        acc.y = fmaf(x12, tf32_round(w12.y), acc.y);
        acc.z = fmaf(x12, tf32_round(w12.z), acc.z);
        acc.w = fmaf(x12, tf32_round(w12.w), acc.w);

        const float x20 = tf32_round(x[x_chan + 256]);
        const float4 w20 = wpack[w_base + 6];
        acc.x = fmaf(x20, tf32_round(w20.x), acc.x);
        acc.y = fmaf(x20, tf32_round(w20.y), acc.y);
        acc.z = fmaf(x20, tf32_round(w20.z), acc.z);
        acc.w = fmaf(x20, tf32_round(w20.w), acc.w);

        const float x21 = tf32_round(x[x_chan + 257]);
        const float4 w21 = wpack[w_base + 7];
        acc.x = fmaf(x21, tf32_round(w21.x), acc.x);
        acc.y = fmaf(x21, tf32_round(w21.y), acc.y);
        acc.z = fmaf(x21, tf32_round(w21.z), acc.z);
        acc.w = fmaf(x21, tf32_round(w21.w), acc.w);

        const float x22 = tf32_round(x[x_chan + 258]);
        const float4 w22 = wpack[w_base + 8];
        acc.x = fmaf(x22, tf32_round(w22.x), acc.x);
        acc.y = fmaf(x22, tf32_round(w22.y), acc.y);
        acc.z = fmaf(x22, tf32_round(w22.z), acc.z);
        acc.w = fmaf(x22, tf32_round(w22.w), acc.w);
    }

    const int out_base = n * out_batch_stride + (og * 4) * out_plane + oy * 126 + ox;
    out[out_base] = acc.x;
    out[out_base + out_plane] = acc.y;
    out[out_base + out_plane * 2] = acc.z;
    out[out_base + out_plane * 3] = acc.w;
}
