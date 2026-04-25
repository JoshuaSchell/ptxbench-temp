#include <cuda_runtime.h>
#include <stdint.h>

namespace {
constexpr int N = 16;
constexpr int C = 64;
constexpr int OC = 128;
constexpr int H = 512;
constexpr int W = 512;
constexpr unsigned int DW_ELEMS_PER_THREAD = 4;
constexpr unsigned int PW_ELEMS_PER_THREAD = 8;
}

__device__ __forceinline__ float to_tf32(float x) {
    float out;
    asm("{\n"
        "  .reg .b32 t;\n"
        "  cvt.rna.tf32.f32 t, %1;\n"
        "  mov.b32 %0, t;\n"
        "}\n" : "=f"(out) : "f"(x));
    return out;
}

extern "C" __global__ void depthwise_kernel(
    const float* __restrict__ x,
    const float* __restrict__ dw,
    float* __restrict__ dw_out,
    const int* __restrict__ flag
) {
    if (flag[0] != 0) {
        return;
    }
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int base_idx = tid * DW_ELEMS_PER_THREAD;

    #pragma unroll
    for (unsigned int e = 0; e < DW_ELEMS_PER_THREAD; ++e) {
        unsigned int idx = base_idx + e;
        if (idx >= (unsigned int)(N * C * H * W)) {
            return;
        }
        unsigned int w = idx & 511u;
        unsigned int t0 = idx >> 9;
        unsigned int h = t0 & 511u;
        unsigned int t1 = t0 >> 9;
        unsigned int c = t1 & 63u;
        unsigned int n = t1 >> 6;

        unsigned long long channel_base = ((unsigned long long)n << 24) + ((unsigned long long)c << 18);
        const float* x_ptr = x + channel_base;
        const float* w_ptr = dw + c * 9u;
        float acc = 0.0f;

        if (h > 0) {
            unsigned long long row = (unsigned long long)(h - 1) << 9;
            if (w > 0) {
                acc = fmaf(x_ptr[row + (unsigned long long)(w - 1)], w_ptr[0], acc);
            }
            acc = fmaf(x_ptr[row + (unsigned long long)w], w_ptr[1], acc);
            if (w + 1 < W) {
                acc = fmaf(x_ptr[row + (unsigned long long)(w + 1)], w_ptr[2], acc);
            }
        }

        {
            unsigned long long row = (unsigned long long)h << 9;
            if (w > 0) {
                acc = fmaf(x_ptr[row + (unsigned long long)(w - 1)], w_ptr[3], acc);
            }
            acc = fmaf(x_ptr[row + (unsigned long long)w], w_ptr[4], acc);
            if (w + 1 < W) {
                acc = fmaf(x_ptr[row + (unsigned long long)(w + 1)], w_ptr[5], acc);
            }
        }

        if (h + 1 < H) {
            unsigned long long row = (unsigned long long)(h + 1) << 9;
            if (w > 0) {
                acc = fmaf(x_ptr[row + (unsigned long long)(w - 1)], w_ptr[6], acc);
            }
            acc = fmaf(x_ptr[row + (unsigned long long)w], w_ptr[7], acc);
            if (w + 1 < W) {
                acc = fmaf(x_ptr[row + (unsigned long long)(w + 1)], w_ptr[8], acc);
            }
        }

        dw_out[idx] = acc;
    }
}

extern "C" __global__ void pointwise_kernel(
    const float* __restrict__ dw_out,
    const float* __restrict__ pw,
    float* __restrict__ out,
    float* __restrict__ cache_out,
    const int* __restrict__ flag
) {
    if (flag[0] != 0) {
        return;
    }
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int base_idx = tid * PW_ELEMS_PER_THREAD;

    #pragma unroll
    for (unsigned int e = 0; e < PW_ELEMS_PER_THREAD; ++e) {
        unsigned int idx = base_idx + e;
        if (idx >= (unsigned int)(N * OC * H * W)) {
            return;
        }
        unsigned int w = idx & 511u;
        unsigned int t0 = idx >> 9;
        unsigned int h = t0 & 511u;
        unsigned int t1 = t0 >> 9;
        unsigned int oc = t1 & 127u;
        unsigned int n = t1 >> 7;

        unsigned long long out_idx = ((unsigned long long)n << 25) + ((unsigned long long)oc << 18) + ((unsigned long long)h << 9) + (unsigned long long)w;
        unsigned long long pixel_offset = ((unsigned long long)h << 9) + (unsigned long long)w;
        float acc = 0.0f;

        #pragma unroll 1
        for (unsigned int c = 0; c < C; ++c) {
            unsigned long long dw_idx = ((unsigned long long)n << 24) + ((unsigned long long)c << 18) + pixel_offset;
            acc = fmaf(dw_out[dw_idx], pw[(c << 7) + oc], acc);
        }

        out[out_idx] = acc;
        cache_out[out_idx] = acc;
    }
}
