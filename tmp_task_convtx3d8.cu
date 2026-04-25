#include <cuda_runtime.h>

template <int PD, int PH, int PW>
__device__ __forceinline__ void convtx3d_body(
    const float* __restrict__ x,
    const float* __restrict__ wpack,
    float* __restrict__ out
) {
    const int spatial_idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (spatial_idx >= 16 * 32 * 32) {
        return;
    }

    const int ocg = static_cast<int>(blockIdx.y);
    const int n = static_cast<int>(blockIdx.z);

    int t = spatial_idx;
    const int ow_idx = t & 31;
    t >>= 5;
    const int oh_idx = t & 31;
    const int od_idx = t >> 5;

    const int od = (od_idx << 1) + PD;
    const int oh = (oh_idx << 1) + PH;
    const int ow = (ow_idx << 1) + PW;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    constexpr int DZ_COUNT = PD ? 2 : 1;
    constexpr int DH_COUNT = PH ? 2 : 1;
    constexpr int DW_COUNT = PW ? 2 : 1;
    constexpr int STENCIL_COUNT = DZ_COUNT * DH_COUNT * DW_COUNT;

    for (int dz = 0; dz < DZ_COUNT; ++dz) {
        const int in_d = od_idx + dz;
        if (in_d >= 16) {
            continue;
        }
        constexpr int KD = PD ? 0 : 1;
        const int kd = PD ? (dz ? 0 : 2) : 1;

        for (int dh = 0; dh < DH_COUNT; ++dh) {
            const int in_h = oh_idx + dh;
            if (in_h >= 32) {
                continue;
            }
            const int kh = PH ? (dh ? 0 : 2) : 1;

            for (int dw = 0; dw < DW_COUNT; ++dw) {
                const int in_w = ow_idx + dw;
                if (in_w >= 32) {
                    continue;
                }
                const int kw = PW ? (dw ? 0 : 2) : 1;
                const int stencil = ((PD ? dz : 0) << 2) | ((PH ? dh : 0) << 1) | (PW ? dw : 0);
                const int weight_base = ((((PD << 2) | (PH << 1) | PW) * 8 + stencil) * 32 * 16 + ocg) * 4;

                #pragma unroll 1
                for (int ic = 0; ic < 32; ++ic) {
                    const int x_idx = ((((n * 32 + ic) * 16 + in_d) * 32 + in_h) * 32 + in_w);
                    const float xv = x[x_idx];
                    const int w_idx = weight_base + ic * 64;
                    acc0 = fmaf(xv, wpack[w_idx + 0], acc0);
                    acc1 = fmaf(xv, wpack[w_idx + 1], acc1);
                    acc2 = fmaf(xv, wpack[w_idx + 2], acc2);
                    acc3 = fmaf(xv, wpack[w_idx + 3], acc3);
                }
            }
        }
    }

    const int out_base = ((((n * 64 + ocg * 4) * 32 + od) * 64 + oh) * 64 + ow);
    const int out_chw = 32 * 64 * 64;
    out[out_base + 0 * out_chw] = acc0;
    out[out_base + 1 * out_chw] = acc1;
    out[out_base + 2 * out_chw] = acc2;
    out[out_base + 3 * out_chw] = acc3;
}

extern "C" __global__ void convtx3d_p000(const float* x, const float* wpack, float* out) { convtx3d_body<0, 0, 0>(x, wpack, out); }
extern "C" __global__ void convtx3d_p001(const float* x, const float* wpack, float* out) { convtx3d_body<0, 0, 1>(x, wpack, out); }
extern "C" __global__ void convtx3d_p010(const float* x, const float* wpack, float* out) { convtx3d_body<0, 1, 0>(x, wpack, out); }
extern "C" __global__ void convtx3d_p011(const float* x, const float* wpack, float* out) { convtx3d_body<0, 1, 1>(x, wpack, out); }
extern "C" __global__ void convtx3d_p100(const float* x, const float* wpack, float* out) { convtx3d_body<1, 0, 0>(x, wpack, out); }
extern "C" __global__ void convtx3d_p101(const float* x, const float* wpack, float* out) { convtx3d_body<1, 0, 1>(x, wpack, out); }
extern "C" __global__ void convtx3d_p110(const float* x, const float* wpack, float* out) { convtx3d_body<1, 1, 0>(x, wpack, out); }
extern "C" __global__ void convtx3d_p111(const float* x, const float* wpack, float* out) { convtx3d_body<1, 1, 1>(x, wpack, out); }
