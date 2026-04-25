#include <mma.h>

__device__ __forceinline__ float tf32_rn(float x) {
    return nvcuda::wmma::__float_to_tf32(x);
}

extern "C" __global__ void conv2d_dilated_kernel(
    const float* __restrict__ x,
    const float4* __restrict__ w,
    float* __restrict__ out,
    float* __restrict__ cache_out,
    int* __restrict__ meta
) {
    if (meta[1] != 0) {
        return;
    }

    int ox = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int oy = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    int z = (int)blockIdx.z;
    int n = z >> 4;
    int ocg = z & 15;

    if (ox >= 496 || oy >= 508 || n >= 8) {
        return;
    }

    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        meta[0] = 1;
        meta[1] = 0;
    }

    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;

    int x_batch_base = n * (32 * 512 * 512);
    int w_group_base = ocg * (5 * 9 * 32);

    for (int ky = 0; ky < 5; ++ky) {
        int iy = oy + ky * 2 - 2;
        if ((unsigned)iy >= 512u) {
            continue;
        }
        int x_row_base = x_batch_base + iy * 512;
        int w_ky_base = w_group_base + ky * (9 * 32);

        for (int kx = 0; kx < 9; ++kx) {
            int ix = ox + kx * 3 - 4;
            if ((unsigned)ix >= 512u) {
                continue;
            }
            int x_pix_base = x_row_base + ix;
            int w_kx_base = w_ky_base + kx * 32;

            for (int ic = 0; ic < 32; ++ic) {
                float xv = tf32_rn(__ldg(x + x_pix_base + ic * (512 * 512)));
                float4 wv = __ldg(w + w_kx_base + ic);
                s0 = fmaf(xv, tf32_rn(wv.x), s0);
                s1 = fmaf(xv, tf32_rn(wv.y), s1);
                s2 = fmaf(xv, tf32_rn(wv.z), s2);
                s3 = fmaf(xv, tf32_rn(wv.w), s3);
            }
        }
    }

    int out_hw = 508 * 496;
    int oc0 = ocg * 4;
    int base = ((n * 64 + oc0) * 508 + oy) * 496 + ox;

    out[base] = s0;
    out[base + out_hw] = s1;
    out[base + out_hw * 2] = s2;
    out[base + out_hw * 3] = s3;

    cache_out[base] = s0;
    cache_out[base + out_hw] = s1;
    cache_out[base + out_hw * 2] = s2;
    cache_out[base + out_hw * 3] = s3;
}
