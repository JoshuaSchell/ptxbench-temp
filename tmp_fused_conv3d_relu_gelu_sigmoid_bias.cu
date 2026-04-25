extern "C" __device__ __forceinline__ float gelu_pos_approx(float x) {
    const float c0 = 0.7978845608028654f;
    const float c1 = 0.044715f;
    const float x2 = x * x;
    const float u = c0 * (x + c1 * x * x2);
    const float e = __expf(-2.0f * u);
    const float t = (1.0f - e) / (1.0f + e);
    return 0.5f * x * (1.0f + t);
}

extern "C" __device__ __forceinline__ float sigmoid_pos_approx(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

extern "C" __global__ void fused_conv3d_relu_gelu_sigmoid_bias_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ conv_bias,
    const float* __restrict__ out_bias,
    float* __restrict__ out
) {
    constexpr int OW_TILE = 8;
    constexpr int OH_TILE = 4;
    constexpr int OD_TILE = 4;
    constexpr int XTW = OW_TILE + 2;
    constexpr int XTH = OH_TILE + 2;
    constexpr int XTD = OD_TILE + 2;
    constexpr int THREADS = OW_TILE * OH_TILE * OD_TILE;
    constexpr int SMEM_X = XTD * XTH * XTW * 8;
    constexpr int SMEM_W = 3 * 3 * 3 * 8 * 8;
    constexpr int IN_DHW = 32 * 64 * 64;
    constexpr int IN_CHW = 32 * 64 * 64;
    constexpr int IN_HW = 64 * 64;
    constexpr int OUT_DHW = 30 * 62 * 62;
    constexpr int OUT_HW = 62 * 62;

    __shared__ float sx[SMEM_X];
    __shared__ float sw[SMEM_W];

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int tz = static_cast<int>(threadIdx.z);
    const int tid = tx + ty * OW_TILE + tz * (OW_TILE * OH_TILE);

    int z = static_cast<int>(blockIdx.z);
    const int oc_group = z & 3;
    z >>= 2;
    const int od_tile = z & 7;
    const int n = z >> 3;

    const int od_base = od_tile * OD_TILE;
    const int oh_base = static_cast<int>(blockIdx.y) * OH_TILE;
    const int ow_base = static_cast<int>(blockIdx.x) * OW_TILE;

    const int x_sample_base = n * 8 * IN_CHW;
    for (int idx = tid; idx < SMEM_X; idx += THREADS) {
        int t = idx;
        const int c = t & 7;
        t >>= 3;
        const int iw = t % XTW;
        t /= XTW;
        const int ih = t % XTH;
        const int id = t / XTH;

        const int gd = od_base + id;
        const int gh = oh_base + ih;
        const int gw = ow_base + iw;

        float v = 0.0f;
        if (gd < 32 && gh < 64 && gw < 64) {
            const int x_idx = x_sample_base + c * IN_CHW + gd * IN_HW + gh * 64 + gw;
            v = x[x_idx];
        }
        sx[idx] = v;
    }

    const int w_group_base = oc_group * SMEM_W;
    for (int idx = tid; idx < SMEM_W; idx += THREADS) {
        sw[idx] = w[w_group_base + idx];
    }
    __syncthreads();

    const int od = od_base + tz;
    const int oh = oh_base + ty;
    const int ow = ow_base + tx;
    if (od >= 30 || oh >= 62 || ow >= 62) {
        return;
    }

    float a0 = conv_bias[oc_group * 8 + 0];
    float a1 = conv_bias[oc_group * 8 + 1];
    float a2 = conv_bias[oc_group * 8 + 2];
    float a3 = conv_bias[oc_group * 8 + 3];
    float a4 = conv_bias[oc_group * 8 + 4];
    float a5 = conv_bias[oc_group * 8 + 5];
    float a6 = conv_bias[oc_group * 8 + 6];
    float a7 = conv_bias[oc_group * 8 + 7];

    const int sx_base = ((tz * XTH + ty) * XTW + tx) * 8;
    #pragma unroll
    for (int kd = 0; kd < 3; ++kd) {
        #pragma unroll
        for (int kh = 0; kh < 3; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < 3; ++kw) {
                const int xk_base = sx_base + ((kd * XTH + kh) * XTW + kw) * 8;
                const int wk_base = ((kd * 3 + kh) * 3 + kw) * 64;
                #pragma unroll
                for (int ic = 0; ic < 8; ++ic) {
                    const float xv = sx[xk_base + ic];
                    const int wic = wk_base + ic * 8;
                    a0 = fmaf(xv, sw[wic + 0], a0);
                    a1 = fmaf(xv, sw[wic + 1], a1);
                    a2 = fmaf(xv, sw[wic + 2], a2);
                    a3 = fmaf(xv, sw[wic + 3], a3);
                    a4 = fmaf(xv, sw[wic + 4], a4);
                    a5 = fmaf(xv, sw[wic + 5], a5);
                    a6 = fmaf(xv, sw[wic + 6], a6);
                    a7 = fmaf(xv, sw[wic + 7], a7);
                }
            }
        }
    }

    a0 = a0 > 0.0f ? a0 : 0.0f;
    a1 = a1 > 0.0f ? a1 : 0.0f;
    a2 = a2 > 0.0f ? a2 : 0.0f;
    a3 = a3 > 0.0f ? a3 : 0.0f;
    a4 = a4 > 0.0f ? a4 : 0.0f;
    a5 = a5 > 0.0f ? a5 : 0.0f;
    a6 = a6 > 0.0f ? a6 : 0.0f;
    a7 = a7 > 0.0f ? a7 : 0.0f;

    a0 = sigmoid_pos_approx(gelu_pos_approx(a0)) + out_bias[oc_group * 8 + 0];
    a1 = sigmoid_pos_approx(gelu_pos_approx(a1)) + out_bias[oc_group * 8 + 1];
    a2 = sigmoid_pos_approx(gelu_pos_approx(a2)) + out_bias[oc_group * 8 + 2];
    a3 = sigmoid_pos_approx(gelu_pos_approx(a3)) + out_bias[oc_group * 8 + 3];
    a4 = sigmoid_pos_approx(gelu_pos_approx(a4)) + out_bias[oc_group * 8 + 4];
    a5 = sigmoid_pos_approx(gelu_pos_approx(a5)) + out_bias[oc_group * 8 + 5];
    a6 = sigmoid_pos_approx(gelu_pos_approx(a6)) + out_bias[oc_group * 8 + 6];
    a7 = sigmoid_pos_approx(gelu_pos_approx(a7)) + out_bias[oc_group * 8 + 7];

    const int out_base = n * 32 * OUT_DHW + od * OUT_HW + oh * 62 + ow;
    out[out_base + 0 * OUT_DHW + oc_group * 8 * OUT_DHW] = a0;
    out[out_base + 1 * OUT_DHW + oc_group * 8 * OUT_DHW] = a1;
    out[out_base + 2 * OUT_DHW + oc_group * 8 * OUT_DHW] = a2;
    out[out_base + 3 * OUT_DHW + oc_group * 8 * OUT_DHW] = a3;
    out[out_base + 4 * OUT_DHW + oc_group * 8 * OUT_DHW] = a4;
    out[out_base + 5 * OUT_DHW + oc_group * 8 * OUT_DHW] = a5;
    out[out_base + 6 * OUT_DHW + oc_group * 8 * OUT_DHW] = a6;
    out[out_base + 7 * OUT_DHW + oc_group * 8 * OUT_DHW] = a7;
}
