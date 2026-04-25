#include <cuda_runtime.h>
#include <math.h>

static __device__ __forceinline__ float tf32_round(float x) {
    unsigned int u;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(u) : "f"(x));
    return __uint_as_float(u);
}

static __device__ __forceinline__ float mish_scalar(float x) {
    float sp;
    if (x > 20.0f) {
        sp = x;
    } else if (x < -20.0f) {
        sp = expf(x);
    } else {
        sp = log1pf(expf(x));
    }
    return x * tanhf(sp);
}

static __device__ __forceinline__ float hswish_sub(float x, float sub) {
    x -= sub;
    float t = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
    return x * (t * (1.0f / 6.0f));
}

extern "C" __global__ void conv3x3_bias_tf32_pack4(
    const float* __restrict__ x,
    const float4* __restrict__ wpack,
    const float4* __restrict__ bpack,
    float* __restrict__ out
) {
    const int ox = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int oy = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    const int gz = (int)blockIdx.z;
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

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    for (int c = 0; c < 64; ++c) {
        const int x_chan = x_batch + c * in_channel_stride + oy * 128 + ox;
        const int w_base = w_group + c * 9;

        float x00 = tf32_round(x[x_chan]);
        float4 w00 = wpack[w_base];
        acc0 = fmaf(x00, tf32_round(w00.x), acc0);
        acc1 = fmaf(x00, tf32_round(w00.y), acc1);
        acc2 = fmaf(x00, tf32_round(w00.z), acc2);
        acc3 = fmaf(x00, tf32_round(w00.w), acc3);

        float x01 = tf32_round(x[x_chan + 1]);
        float4 w01 = wpack[w_base + 1];
        acc0 = fmaf(x01, tf32_round(w01.x), acc0);
        acc1 = fmaf(x01, tf32_round(w01.y), acc1);
        acc2 = fmaf(x01, tf32_round(w01.z), acc2);
        acc3 = fmaf(x01, tf32_round(w01.w), acc3);

        float x02 = tf32_round(x[x_chan + 2]);
        float4 w02 = wpack[w_base + 2];
        acc0 = fmaf(x02, tf32_round(w02.x), acc0);
        acc1 = fmaf(x02, tf32_round(w02.y), acc1);
        acc2 = fmaf(x02, tf32_round(w02.z), acc2);
        acc3 = fmaf(x02, tf32_round(w02.w), acc3);

        float x10 = tf32_round(x[x_chan + 128]);
        float4 w10 = wpack[w_base + 3];
        acc0 = fmaf(x10, tf32_round(w10.x), acc0);
        acc1 = fmaf(x10, tf32_round(w10.y), acc1);
        acc2 = fmaf(x10, tf32_round(w10.z), acc2);
        acc3 = fmaf(x10, tf32_round(w10.w), acc3);

        float x11 = tf32_round(x[x_chan + 129]);
        float4 w11 = wpack[w_base + 4];
        acc0 = fmaf(x11, tf32_round(w11.x), acc0);
        acc1 = fmaf(x11, tf32_round(w11.y), acc1);
        acc2 = fmaf(x11, tf32_round(w11.z), acc2);
        acc3 = fmaf(x11, tf32_round(w11.w), acc3);

        float x12 = tf32_round(x[x_chan + 130]);
        float4 w12 = wpack[w_base + 5];
        acc0 = fmaf(x12, tf32_round(w12.x), acc0);
        acc1 = fmaf(x12, tf32_round(w12.y), acc1);
        acc2 = fmaf(x12, tf32_round(w12.z), acc2);
        acc3 = fmaf(x12, tf32_round(w12.w), acc3);

        float x20 = tf32_round(x[x_chan + 256]);
        float4 w20 = wpack[w_base + 6];
        acc0 = fmaf(x20, tf32_round(w20.x), acc0);
        acc1 = fmaf(x20, tf32_round(w20.y), acc1);
        acc2 = fmaf(x20, tf32_round(w20.z), acc2);
        acc3 = fmaf(x20, tf32_round(w20.w), acc3);

        float x21 = tf32_round(x[x_chan + 257]);
        float4 w21 = wpack[w_base + 7];
        acc0 = fmaf(x21, tf32_round(w21.x), acc0);
        acc1 = fmaf(x21, tf32_round(w21.y), acc1);
        acc2 = fmaf(x21, tf32_round(w21.z), acc2);
        acc3 = fmaf(x21, tf32_round(w21.w), acc3);

        float x22 = tf32_round(x[x_chan + 258]);
        float4 w22 = wpack[w_base + 8];
        acc0 = fmaf(x22, tf32_round(w22.x), acc0);
        acc1 = fmaf(x22, tf32_round(w22.y), acc1);
        acc2 = fmaf(x22, tf32_round(w22.z), acc2);
        acc3 = fmaf(x22, tf32_round(w22.w), acc3);
    }

    float4 b = bpack[og];
    acc0 += b.x;
    acc1 += b.y;
    acc2 += b.z;
    acc3 += b.w;

    const int out_base = n * out_batch_stride + (og * 4) * out_plane + oy * 126 + ox;
    out[out_base] = acc0;
    out[out_base + out_plane] = acc1;
    out[out_base + out_plane * 2] = acc2;
    out[out_base + out_plane * 3] = acc3;
}

extern "C" __global__ void post_pool_hswish_mish_pack4(
    const float* __restrict__ x,
    float sub,
    float* __restrict__ out
) {
    const int ox = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int oy = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    const int gz = (int)blockIdx.z;
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

    for (int lane = 0; lane < 4; ++lane) {
        const int off = in_base + lane * in_plane;
        float m0 = hswish_sub(x[off], sub);
        float m1 = hswish_sub(x[off + 1], sub);
        float m2 = hswish_sub(x[off + 126], sub);
        float m3 = hswish_sub(x[off + 127], sub);
        float v = fmaxf(fmaxf(m0, m1), fmaxf(m2, m3));
        out[out_base + lane * out_plane] = mish_scalar(v);
    }
}
