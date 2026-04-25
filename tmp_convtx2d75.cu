extern "C" __global__ void zero4_kernel(float* out, unsigned int n_vec4) {
    unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n_vec4) {
        return;
    }
    reinterpret_cast<float4*>(out)[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

template <int PATTERN>
__device__ __forceinline__ float load_broadcast(const float* ptr, bool valid, unsigned int lane, unsigned int warp_lane) {
    float value = 0.0f;
    if ((lane == 0) && valid) {
        value = __ldg(ptr);
    }
    return __shfl_sync(0xffffffffu, value, warp_lane & 16, 32);
}

template <int PATTERN>
__device__ __forceinline__ void deconv75_kernel_body(const float* __restrict__ x, const float* __restrict__ wp, float* __restrict__ out) {
    constexpr int SUBWARP = 16;
    constexpr int SUBWARPS_PER_BLOCK = 8;
    constexpr int N_IN = 16;
    constexpr int C_IN = 32;
    constexpr int C_OUT = 64;
    constexpr int GROUPS = 4;
    constexpr int IC_PER_G = 8;
    constexpr int OC_PER_G = 16;
    constexpr int H_IN = 128;
    constexpr int W_IN = 256;
    constexpr int H_OUT = 257;
    constexpr int W_OUT = 766;

    unsigned int tid = threadIdx.x;
    unsigned int lane = tid & 15;
    unsigned int warp_lane = tid & 31;
    unsigned int sub = tid >> 4;
    unsigned int spatial_idx = blockIdx.x * SUBWARPS_PER_BLOCK + sub;
    unsigned int g = blockIdx.y;
    unsigned int n = blockIdx.z;

    unsigned int width_count = PATTERN == 0 ? 256 : 255;
    unsigned int spatial_count = 128 * width_count;
    if (spatial_idx >= spatial_count) {
        return;
    }

    unsigned int ph = spatial_idx / width_count;
    unsigned int pw = spatial_idx - ph * width_count;
    unsigned int oh = ph * 2 + 1;
    unsigned int ow = pw * 3 + PATTERN;
    unsigned int oc = g * OC_PER_G + lane;
    unsigned int ic_base = g * IC_PER_G;

    const float* wp_base = wp + (((PATTERN * GROUPS + g) * 6) * IC_PER_G * OC_PER_G);

    float acc = 0.0f;

    #pragma unroll
    for (int ic = 0; ic < IC_PER_G; ++ic) {
        unsigned int x_c = ic_base + ic;
        unsigned int x_base = ((n * C_IN + x_c) * H_IN) * W_IN;
        const float* w_lane = wp_base + ic * OC_PER_G + lane;

        bool v0 = ph < 127;
        bool v1 = true;
        bool v2 = ph > 0;

        unsigned int row_p1 = ph + 1;
        unsigned int row_0 = ph;
        unsigned int row_m1 = ph - 1;

        if constexpr (PATTERN == 0) {
            float x0 = load_broadcast<PATTERN>(x + x_base + row_p1 * W_IN + pw, v0, lane, warp_lane);
            float x1 = load_broadcast<PATTERN>(x + x_base + row_0 * W_IN + pw, v1, lane, warp_lane);
            float x2 = load_broadcast<PATTERN>(x + x_base + row_m1 * W_IN + pw, v2, lane, warp_lane);

            acc = fmaf(x0, __ldg(w_lane + 0 * IC_PER_G * OC_PER_G), acc);
            acc = fmaf(x1, __ldg(w_lane + 1 * IC_PER_G * OC_PER_G), acc);
            acc = fmaf(x2, __ldg(w_lane + 2 * IC_PER_G * OC_PER_G), acc);
        } else {
            unsigned int col_p1 = pw + 1;
            unsigned int col_0 = pw;
            float x0 = load_broadcast<PATTERN>(x + x_base + row_p1 * W_IN + col_p1, v0, lane, warp_lane);
            float x1 = load_broadcast<PATTERN>(x + x_base + row_p1 * W_IN + col_0, v0, lane, warp_lane);
            float x2 = load_broadcast<PATTERN>(x + x_base + row_0 * W_IN + col_p1, v1, lane, warp_lane);
            float x3 = load_broadcast<PATTERN>(x + x_base + row_0 * W_IN + col_0, v1, lane, warp_lane);
            float x4 = load_broadcast<PATTERN>(x + x_base + row_m1 * W_IN + col_p1, v2, lane, warp_lane);
            float x5 = load_broadcast<PATTERN>(x + x_base + row_m1 * W_IN + col_0, v2, lane, warp_lane);

            acc = fmaf(x0, __ldg(w_lane + 0 * IC_PER_G * OC_PER_G), acc);
            acc = fmaf(x1, __ldg(w_lane + 1 * IC_PER_G * OC_PER_G), acc);
            acc = fmaf(x2, __ldg(w_lane + 2 * IC_PER_G * OC_PER_G), acc);
            acc = fmaf(x3, __ldg(w_lane + 3 * IC_PER_G * OC_PER_G), acc);
            acc = fmaf(x4, __ldg(w_lane + 4 * IC_PER_G * OC_PER_G), acc);
            acc = fmaf(x5, __ldg(w_lane + 5 * IC_PER_G * OC_PER_G), acc);
        }
    }

    unsigned int out_idx = (((n * C_OUT + oc) * H_OUT + oh) * W_OUT + ow);
    out[out_idx] = acc;
}

extern "C" __global__ void deconv75_p0(const float* __restrict__ x, const float* __restrict__ wp, float* __restrict__ out) {
    deconv75_kernel_body<0>(x, wp, out);
}

extern "C" __global__ void deconv75_p1(const float* __restrict__ x, const float* __restrict__ wp, float* __restrict__ out) {
    deconv75_kernel_body<1>(x, wp, out);
}

extern "C" __global__ void deconv75_p2(const float* __restrict__ x, const float* __restrict__ wp, float* __restrict__ out) {
    deconv75_kernel_body<2>(x, wp, out);
}
