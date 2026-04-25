extern "C" __device__ __forceinline__ void fma8(float x, const float* __restrict__ w, float& a0, float& a1, float& a2, float& a3, float& a4, float& a5, float& a6, float& a7) {
    a0 = fmaf(x, w[0], a0);
    a1 = fmaf(x, w[1], a1);
    a2 = fmaf(x, w[2], a2);
    a3 = fmaf(x, w[3], a3);
    a4 = fmaf(x, w[4], a4);
    a5 = fmaf(x, w[5], a5);
    a6 = fmaf(x, w[6], a6);
    a7 = fmaf(x, w[7], a7);
}

extern "C" __global__ void convtx3d_groupvec_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ out,
    unsigned int spatial_count,
    unsigned int w_count,
    unsigned int h_count,
    unsigned int x_w,
    unsigned int x_hw,
    unsigned int x_chw,
    unsigned int out_w,
    unsigned int out_hw,
    unsigned int out_chw,
    unsigned int pattern
) {
    unsigned int idx = static_cast<unsigned int>(blockIdx.x) * static_cast<unsigned int>(blockDim.x) + static_cast<unsigned int>(threadIdx.x);
    if (idx >= spatial_count) {
        return;
    }

    unsigned int g = static_cast<unsigned int>(blockIdx.y);
    unsigned int n = static_cast<unsigned int>(blockIdx.z);
    unsigned int pd = pattern >> 2;
    unsigned int ph = (pattern >> 1) & 1U;
    unsigned int pw = pattern & 1U;

    unsigned int ow_idx = idx % w_count;
    unsigned int tmp = idx / w_count;
    unsigned int oh_idx = tmp % h_count;
    unsigned int od_idx = tmp / h_count;

    unsigned int ow = (ow_idx << 1) + pw;
    unsigned int oh = (oh_idx << 1) + ph;
    unsigned int od = (od_idx << 1) + pd;

    unsigned int x_group_base = (n * 32U + g * 8U) * x_chw;
    unsigned int out_group_base = (n * 32U + g * 8U) * out_chw;
    unsigned int out_offset = out_group_base + od * out_hw + oh * out_w + ow;

    float a0 = 0.0f;
    float a1 = 0.0f;
    float a2 = 0.0f;
    float a3 = 0.0f;
    float a4 = 0.0f;
    float a5 = 0.0f;
    float a6 = 0.0f;
    float a7 = 0.0f;

    unsigned int dz_count = pd ? 2U : 1U;
    unsigned int dh_count = ph ? 2U : 1U;
    unsigned int dw_count = pw ? 2U : 1U;
    unsigned int stencil = 0U;

    #pragma unroll
    for (unsigned int dz = 0; dz < 2U; ++dz) {
        if (dz >= dz_count) {
            continue;
        }
        unsigned int id = od_idx + (pd ? dz : 0U);
        #pragma unroll
        for (unsigned int dh = 0; dh < 2U; ++dh) {
            if (dh >= dh_count) {
                continue;
            }
            unsigned int ih = oh_idx + (ph ? dh : 0U);
            #pragma unroll
            for (unsigned int dw = 0; dw < 2U; ++dw) {
                if (dw >= dw_count) {
                    continue;
                }
                unsigned int iw = ow_idx + (pw ? dw : 0U);
                const float* x_ptr = x + x_group_base + id * x_hw + ih * x_w + iw;
                const float* w_ptr = w + (((pattern * 4U + g) * 8U + stencil) * 64U);
                #pragma unroll
                for (unsigned int ic = 0; ic < 8U; ++ic) {
                    float xv = x_ptr[ic * x_chw];
                    fma8(xv, w_ptr + ic * 8U, a0, a1, a2, a3, a4, a5, a6, a7);
                }
                ++stencil;
            }
        }
    }

    out[out_offset + 0U * out_chw] = a0;
    out[out_offset + 1U * out_chw] = a1;
    out[out_offset + 2U * out_chw] = a2;
    out[out_offset + 3U * out_chw] = a3;
    out[out_offset + 4U * out_chw] = a4;
    out[out_offset + 5U * out_chw] = a5;
    out[out_offset + 6U * out_chw] = a6;
    out[out_offset + 7U * out_chw] = a7;
}
