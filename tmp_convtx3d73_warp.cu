extern "C" __global__ void convtx3d_warp_kernel(
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
    unsigned int lane = static_cast<unsigned int>(threadIdx.x) & 31U;
    unsigned int warp_in_block = static_cast<unsigned int>(threadIdx.x) >> 5;
    unsigned int spatial_idx = static_cast<unsigned int>(blockIdx.x) * (static_cast<unsigned int>(blockDim.x) >> 5) + warp_in_block;
    if (spatial_idx >= spatial_count) {
        return;
    }

    unsigned int n = static_cast<unsigned int>(blockIdx.z);
    unsigned int pd = pattern >> 2;
    unsigned int ph = (pattern >> 1) & 1U;
    unsigned int pw = pattern & 1U;

    unsigned int ow_idx = spatial_idx % w_count;
    unsigned int tmp = spatial_idx / w_count;
    unsigned int oh_idx = tmp % h_count;
    unsigned int od_idx = tmp / h_count;

    unsigned int ow = (ow_idx << 1) + pw;
    unsigned int oh = (oh_idx << 1) + ph;
    unsigned int od = (od_idx << 1) + pd;

    unsigned int x_base_n = n * 32U * x_chw;
    unsigned int out_base_n = n * 32U * out_chw;
    unsigned int out_offset = out_base_n + lane * out_chw + od * out_hw + oh * out_w + ow;

    float acc = 0.0f;
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
                const float* x_ptr = x + x_base_n + lane * x_chw + id * x_hw + ih * x_w + iw;
                float x_lane = *x_ptr;
                const float* w_ptr = w + ((pattern * 8U + stencil) * 1024U) + lane;
                #pragma unroll
                for (unsigned int ic = 0; ic < 32U; ++ic) {
                    float x_val = __shfl_sync(0xffffffffU, x_lane, static_cast<int>(ic));
                    acc = fmaf(x_val, w_ptr[ic * 32U], acc);
                }
                ++stencil;
            }
        }
    }

    out[out_offset] = acc;
}
