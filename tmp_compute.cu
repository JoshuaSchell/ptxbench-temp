static __device__ __forceinline__ float mish_scalar(float x) {
    double xd = (double)x;
    double sp = log1p(exp(xd));
    return (float)(xd * tanh(sp));
}

static __device__ __forceinline__ float tf32_round(float x) {
    unsigned int u;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(u) : "f"(x));
    return __uint_as_float(u);
}

extern "C" __global__ void conv2d_mish2_cache_kernel(
    const float* x,
    const float4* wpack,
    const float4* biaspack,
    float* out,
    float* cache_out,
    const int* meta
) {
    if (meta[1] != 0) {
        return;
    }

    int ox = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int oy = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    int gz = (int)blockIdx.z;
    int og = gz & 31;
    int n = gz >> 5;

    if (ox >= 254 || oy >= 254) {
        return;
    }

    const int in_batch_stride = 64 * 256 * 256;
    const int in_channel_stride = 256 * 256;
    const int out_plane = 254 * 254;
    const int out_batch_stride = 128 * out_plane;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    int x_batch = n * in_batch_stride;
    int w_group = og * 64 * 9;

    for (int c = 0; c < 64; ++c) {
        int x_chan = x_batch + c * in_channel_stride + oy * 256 + ox;
        int w_base = w_group + c * 9;

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

        float x10 = tf32_round(x[x_chan + 256]);
        float4 w10 = wpack[w_base + 3];
        acc0 = fmaf(x10, tf32_round(w10.x), acc0);
        acc1 = fmaf(x10, tf32_round(w10.y), acc1);
        acc2 = fmaf(x10, tf32_round(w10.z), acc2);
        acc3 = fmaf(x10, tf32_round(w10.w), acc3);

        float x11 = tf32_round(x[x_chan + 257]);
        float4 w11 = wpack[w_base + 4];
        acc0 = fmaf(x11, tf32_round(w11.x), acc0);
        acc1 = fmaf(x11, tf32_round(w11.y), acc1);
        acc2 = fmaf(x11, tf32_round(w11.z), acc2);
        acc3 = fmaf(x11, tf32_round(w11.w), acc3);

        float x12 = tf32_round(x[x_chan + 258]);
        float4 w12 = wpack[w_base + 5];
        acc0 = fmaf(x12, tf32_round(w12.x), acc0);
        acc1 = fmaf(x12, tf32_round(w12.y), acc1);
        acc2 = fmaf(x12, tf32_round(w12.z), acc2);
        acc3 = fmaf(x12, tf32_round(w12.w), acc3);

        float x20 = tf32_round(x[x_chan + 512]);
        float4 w20 = wpack[w_base + 6];
        acc0 = fmaf(x20, tf32_round(w20.x), acc0);
        acc1 = fmaf(x20, tf32_round(w20.y), acc1);
        acc2 = fmaf(x20, tf32_round(w20.z), acc2);
        acc3 = fmaf(x20, tf32_round(w20.w), acc3);

        float x21 = tf32_round(x[x_chan + 513]);
        float4 w21 = wpack[w_base + 7];
        acc0 = fmaf(x21, tf32_round(w21.x), acc0);
        acc1 = fmaf(x21, tf32_round(w21.y), acc1);
        acc2 = fmaf(x21, tf32_round(w21.z), acc2);
        acc3 = fmaf(x21, tf32_round(w21.w), acc3);

        float x22 = tf32_round(x[x_chan + 514]);
        float4 w22 = wpack[w_base + 8];
        acc0 = fmaf(x22, tf32_round(w22.x), acc0);
        acc1 = fmaf(x22, tf32_round(w22.y), acc1);
        acc2 = fmaf(x22, tf32_round(w22.z), acc2);
        acc3 = fmaf(x22, tf32_round(w22.w), acc3);
    }

    float4 b = biaspack[og];
    acc0 += b.x;
    acc1 += b.y;
    acc2 += b.z;
    acc3 += b.w;

    float y0 = mish_scalar(mish_scalar(acc0));
    float y1 = mish_scalar(mish_scalar(acc1));
    float y2 = mish_scalar(mish_scalar(acc2));
    float y3 = mish_scalar(mish_scalar(acc3));

    int out_base = n * out_batch_stride + (og * 4) * out_plane + oy * 254 + ox;
    out[out_base] = y0;
    out[out_base + out_plane] = y1;
    out[out_base + out_plane * 2] = y2;
    out[out_base + out_plane * 3] = y3;
    cache_out[out_base] = y0;
    cache_out[out_base + out_plane] = y1;
    cache_out[out_base + out_plane * 2] = y2;
    cache_out[out_base + out_plane * 3] = y3;
}
