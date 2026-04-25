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

template<int PD, int PH, int PW>
__device__ __forceinline__ void run_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ out,
    int spatial_count,
    int w_count,
    int h_count,
    int x_w,
    int x_hw,
    int x_chw,
    int out_w,
    int out_hw,
    int out_chw,
    int pattern
) {
    int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (idx >= spatial_count) {
        return;
    }

    int g = static_cast<int>(blockIdx.y);
    int n = static_cast<int>(blockIdx.z);

    int ow_idx = idx % w_count;
    int tmp = idx / w_count;
    int oh_idx = tmp % h_count;
    int od_idx = tmp / h_count;

    int ow = (ow_idx << 1) + PW;
    int oh = (oh_idx << 1) + PH;
    int od = (od_idx << 1) + PD;

    int x_group_base = (n * 32 + g * 8) * x_chw;
    int out_group_base = (n * 32 + g * 8) * out_chw;
    int out_offset = out_group_base + od * out_hw + oh * out_w + ow;

    float a0 = 0.0f;
    float a1 = 0.0f;
    float a2 = 0.0f;
    float a3 = 0.0f;
    float a4 = 0.0f;
    float a5 = 0.0f;
    float a6 = 0.0f;
    float a7 = 0.0f;

    int s = 0;

    #pragma unroll
    for (int dz = 0; dz < (PD ? 2 : 1); ++dz) {
        int id = od_idx + (PD ? dz : 0);
        #pragma unroll
        for (int dh = 0; dh < (PH ? 2 : 1); ++dh) {
            int ih = oh_idx + (PH ? dh : 0);
            #pragma unroll
            for (int dw = 0; dw < (PW ? 2 : 1); ++dw) {
                int iw = ow_idx + (PW ? dw : 0);
                const float* x_ptr = x + x_group_base + id * x_hw + ih * x_w + iw;
                const float* w_ptr = w + (((pattern * 4 + g) * 8 + s) * 64);
                #pragma unroll
                for (int ic = 0; ic < 8; ++ic) {
                    float xv = x_ptr[ic * x_chw];
                    fma8(xv, w_ptr + ic * 8, a0, a1, a2, a3, a4, a5, a6, a7);
                }
                ++s;
            }
        }
    }

    out[out_offset + 0 * out_chw] = a0;
    out[out_offset + 1 * out_chw] = a1;
    out[out_offset + 2 * out_chw] = a2;
    out[out_offset + 3 * out_chw] = a3;
    out[out_offset + 4 * out_chw] = a4;
    out[out_offset + 5 * out_chw] = a5;
    out[out_offset + 6 * out_chw] = a6;
    out[out_offset + 7 * out_chw] = a7;
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
    switch (pattern) {
        case 0: run_kernel<0, 0, 0>(x, w, out, static_cast<int>(spatial_count), static_cast<int>(w_count), static_cast<int>(h_count), static_cast<int>(x_w), static_cast<int>(x_hw), static_cast<int>(x_chw), static_cast<int>(out_w), static_cast<int>(out_hw), static_cast<int>(out_chw), 0); break;
        case 1: run_kernel<0, 0, 1>(x, w, out, static_cast<int>(spatial_count), static_cast<int>(w_count), static_cast<int>(h_count), static_cast<int>(x_w), static_cast<int>(x_hw), static_cast<int>(x_chw), static_cast<int>(out_w), static_cast<int>(out_hw), static_cast<int>(out_chw), 1); break;
        case 2: run_kernel<0, 1, 0>(x, w, out, static_cast<int>(spatial_count), static_cast<int>(w_count), static_cast<int>(h_count), static_cast<int>(x_w), static_cast<int>(x_hw), static_cast<int>(x_chw), static_cast<int>(out_w), static_cast<int>(out_hw), static_cast<int>(out_chw), 2); break;
        case 3: run_kernel<0, 1, 1>(x, w, out, static_cast<int>(spatial_count), static_cast<int>(w_count), static_cast<int>(h_count), static_cast<int>(x_w), static_cast<int>(x_hw), static_cast<int>(x_chw), static_cast<int>(out_w), static_cast<int>(out_hw), static_cast<int>(out_chw), 3); break;
        case 4: run_kernel<1, 0, 0>(x, w, out, static_cast<int>(spatial_count), static_cast<int>(w_count), static_cast<int>(h_count), static_cast<int>(x_w), static_cast<int>(x_hw), static_cast<int>(x_chw), static_cast<int>(out_w), static_cast<int>(out_hw), static_cast<int>(out_chw), 4); break;
        case 5: run_kernel<1, 0, 1>(x, w, out, static_cast<int>(spatial_count), static_cast<int>(w_count), static_cast<int>(h_count), static_cast<int>(x_w), static_cast<int>(x_hw), static_cast<int>(x_chw), static_cast<int>(out_w), static_cast<int>(out_hw), static_cast<int>(out_chw), 5); break;
        case 6: run_kernel<1, 1, 0>(x, w, out, static_cast<int>(spatial_count), static_cast<int>(w_count), static_cast<int>(h_count), static_cast<int>(x_w), static_cast<int>(x_hw), static_cast<int>(x_chw), static_cast<int>(out_w), static_cast<int>(out_hw), static_cast<int>(out_chw), 6); break;
        default: run_kernel<1, 1, 1>(x, w, out, static_cast<int>(spatial_count), static_cast<int>(w_count), static_cast<int>(h_count), static_cast<int>(x_w), static_cast<int>(x_hw), static_cast<int>(x_chw), static_cast<int>(out_w), static_cast<int>(out_hw), static_cast<int>(out_chw), 7); break;
    }
}
