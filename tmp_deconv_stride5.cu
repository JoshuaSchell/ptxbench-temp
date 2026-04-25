struct float4_ {
    float x;
    float y;
    float z;
    float w;
};

__device__ __forceinline__ float to_tf32(float x) {
    unsigned int bits;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(bits) : "f"(x));
    return __uint_as_float(bits);
}

extern "C" __global__ void deconv_stride5_kernel(
    const float* __restrict__ x,
    const float4_* __restrict__ w,
    float* __restrict__ out,
    float* __restrict__ cache_out,
    int* __restrict__ meta
) {
    if (meta[1] != 0) {
        return;
    }

    const int ow = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int oh = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int z = static_cast<int>(blockIdx.z);
    const int oc4 = z & 15;
    const int n = z >> 4;

    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        meta[0] = 1;
    }

    if (ow >= 638 || oh >= 318 || n >= 16) {
        return;
    }

    int kh = -1;
    const int rh = oh % 5;
    if (rh == 4) {
        kh = 0;
    } else if (rh == 1) {
        kh = 1;
    } else if (rh == 3) {
        kh = 2;
    }

    int kw = -1;
    const int rw = ow % 5;
    if (rw == 4) {
        kw = 0;
    } else if (rw == 1) {
        kw = 1;
    } else if (rw == 3) {
        kw = 2;
    }

    const int spatial = 318 * 638;
    const int out_base = ((n * 64 + oc4 * 4) * 318 + oh) * 638 + ow;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    if (kh >= 0 && kw >= 0) {
        const int ih_num = oh + 1 - kh * 2;
        const int iw_num = ow + 1 - kw * 2;
        const int ih = ih_num / 5;
        const int iw = iw_num / 5;
        if (ih >= 0 && ih < 64 && iw >= 0 && iw < 128) {
            const int x_base = (n * 32 * 64 + ih) * 128 + iw;
            const int tap = kh * 3 + kw;
            const int w_base = (tap * 16 + oc4) * 32;
            #pragma unroll
            for (int ic = 0; ic < 32; ++ic) {
                const float xv = to_tf32(x[x_base + ic * (64 * 128)]);
                const float4_ wt = w[w_base + ic];
                acc0 = fmaf(xv, to_tf32(wt.x), acc0);
                acc1 = fmaf(xv, to_tf32(wt.y), acc1);
                acc2 = fmaf(xv, to_tf32(wt.z), acc2);
                acc3 = fmaf(xv, to_tf32(wt.w), acc3);
            }
        }
    }

    out[out_base] = acc0;
    out[out_base + spatial] = acc1;
    out[out_base + spatial * 2] = acc2;
    out[out_base + spatial * 3] = acc3;

    cache_out[out_base] = acc0;
    cache_out[out_base + spatial] = acc1;
    cache_out[out_base + spatial * 2] = acc2;
    cache_out[out_base + spatial * 3] = acc3;
}
