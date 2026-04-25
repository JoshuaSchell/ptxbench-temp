__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

template <int PARITY>
__device__ __forceinline__ int out_count_dim() {
    return PARITY ? 15 : 16;
}

template <>
__device__ __forceinline__ int out_count_dim<0>() { return 16; }

template <>
__device__ __forceinline__ int out_count_dim<1>() { return 15; }

template <int PD, int PH, int PW>
__device__ __forceinline__ void deconv_case(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out) {
    constexpr int ODN = PD ? 15 : 16;
    constexpr int OHN = PH ? 31 : 32;
    constexpr int OWN = PW ? 31 : 32;
    constexpr int POS_COUNT = ODN * OHN * OWN;

    unsigned pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= POS_COUNT) {
        return;
    }

    unsigned tmp = pos;
    unsigned ow_i = tmp % OWN;
    tmp /= OWN;
    unsigned oh_i = tmp % OHN;
    unsigned od_i = tmp / OHN;

    unsigned od = od_i * 2 + PD;
    unsigned oh = oh_i * 2 + PH;
    unsigned owv = ow_i * 2 + PW;
    unsigned oc = blockIdx.y;
    unsigned n = blockIdx.z;

    unsigned id0 = od >> 1;
    unsigned ih0 = oh >> 1;
    unsigned iw0 = owv >> 1;
    unsigned id1 = (od + 1) >> 1;
    unsigned ih1 = (oh + 1) >> 1;
    unsigned iw1 = (owv + 1) >> 1;

    float acc = 0.0f;

    #pragma unroll
    for (int ic = 0; ic < 16; ++ic) {
        const float* w_base = w + (((ic * 32 + oc) * 3) * 3) * 3;
        if constexpr (PD == 0) {
            const float* x_d0 = x + ((((n * 16 + ic) * 16 + id0) * 32) * 32);
            if constexpr (PH == 0) {
                const float* x_h0 = x_d0 + (ih0 * 32);
                if constexpr (PW == 0) {
                    acc = fmaf(x_h0[iw0], w_base[(1 * 3 + 1) * 3 + 1], acc);
                } else {
                    acc = fmaf(x_h0[iw1], w_base[(1 * 3 + 1) * 3 + 0], acc);
                    acc = fmaf(x_h0[iw0], w_base[(1 * 3 + 1) * 3 + 2], acc);
                }
            } else {
                const float* x_h1 = x_d0 + (ih1 * 32);
                const float* x_h0 = x_d0 + (ih0 * 32);
                if constexpr (PW == 0) {
                    acc = fmaf(x_h1[iw0], w_base[(1 * 3 + 0) * 3 + 1], acc);
                    acc = fmaf(x_h0[iw0], w_base[(1 * 3 + 2) * 3 + 1], acc);
                } else {
                    acc = fmaf(x_h1[iw1], w_base[(1 * 3 + 0) * 3 + 0], acc);
                    acc = fmaf(x_h1[iw0], w_base[(1 * 3 + 0) * 3 + 2], acc);
                    acc = fmaf(x_h0[iw1], w_base[(1 * 3 + 2) * 3 + 0], acc);
                    acc = fmaf(x_h0[iw0], w_base[(1 * 3 + 2) * 3 + 2], acc);
                }
            }
        } else {
            const float* x_d1 = x + ((((n * 16 + ic) * 16 + id1) * 32) * 32);
            const float* x_d0 = x + ((((n * 16 + ic) * 16 + id0) * 32) * 32);
            if constexpr (PH == 0) {
                const float* x10 = x_d1 + (ih0 * 32);
                const float* x00 = x_d0 + (ih0 * 32);
                if constexpr (PW == 0) {
                    acc = fmaf(x10[iw0], w_base[(0 * 3 + 1) * 3 + 1], acc);
                    acc = fmaf(x00[iw0], w_base[(2 * 3 + 1) * 3 + 1], acc);
                } else {
                    acc = fmaf(x10[iw1], w_base[(0 * 3 + 1) * 3 + 0], acc);
                    acc = fmaf(x10[iw0], w_base[(0 * 3 + 1) * 3 + 2], acc);
                    acc = fmaf(x00[iw1], w_base[(2 * 3 + 1) * 3 + 0], acc);
                    acc = fmaf(x00[iw0], w_base[(2 * 3 + 1) * 3 + 2], acc);
                }
            } else {
                const float* x11 = x_d1 + (ih1 * 32);
                const float* x10 = x_d1 + (ih0 * 32);
                const float* x01 = x_d0 + (ih1 * 32);
                const float* x00 = x_d0 + (ih0 * 32);
                if constexpr (PW == 0) {
                    acc = fmaf(x11[iw0], w_base[(0 * 3 + 0) * 3 + 1], acc);
                    acc = fmaf(x10[iw0], w_base[(0 * 3 + 2) * 3 + 1], acc);
                    acc = fmaf(x01[iw0], w_base[(2 * 3 + 0) * 3 + 1], acc);
                    acc = fmaf(x00[iw0], w_base[(2 * 3 + 2) * 3 + 1], acc);
                } else {
                    acc = fmaf(x11[iw1], w_base[(0 * 3 + 0) * 3 + 0], acc);
                    acc = fmaf(x11[iw0], w_base[(0 * 3 + 0) * 3 + 2], acc);
                    acc = fmaf(x10[iw1], w_base[(0 * 3 + 2) * 3 + 0], acc);
                    acc = fmaf(x10[iw0], w_base[(0 * 3 + 2) * 3 + 2], acc);
                    acc = fmaf(x01[iw1], w_base[(2 * 3 + 0) * 3 + 0], acc);
                    acc = fmaf(x01[iw0], w_base[(2 * 3 + 0) * 3 + 2], acc);
                    acc = fmaf(x00[iw1], w_base[(2 * 3 + 2) * 3 + 0], acc);
                    acc = fmaf(x00[iw0], w_base[(2 * 3 + 2) * 3 + 2], acc);
                }
            }
        }
    }

    unsigned out_index = (((n * 32 + oc) * 31 + od) * 63 + oh) * 63 + owv;
    out[out_index] = acc;
}

extern "C" __global__ void model_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ out,
    float* __restrict__ sample_means,
    float* __restrict__ channel_sums,
    float* __restrict__ channel_sumsq,
    float* __restrict__ scales,
    const float* __restrict__ bn_weight,
    unsigned op
) {
    if (op == 0) { deconv_case<0, 0, 0>(x, w, out); return; }
    if (op == 1) { deconv_case<0, 0, 1>(x, w, out); return; }
    if (op == 2) { deconv_case<0, 1, 0>(x, w, out); return; }
    if (op == 3) { deconv_case<0, 1, 1>(x, w, out); return; }
    if (op == 4) { deconv_case<1, 0, 0>(x, w, out); return; }
    if (op == 5) { deconv_case<1, 0, 1>(x, w, out); return; }
    if (op == 6) { deconv_case<1, 1, 0>(x, w, out); return; }
    if (op == 7) { deconv_case<1, 1, 1>(x, w, out); return; }

    if (op == 8) {
        unsigned c = threadIdx.x;
        if (c < 32) {
            channel_sums[c] = 0.0f;
            channel_sumsq[c] = 0.0f;
        }
        return;
    }

    if (op == 9) {
        constexpr unsigned SPATIAL = 123039;
        unsigned nc = blockIdx.x;
        if (nc >= 512) {
            return;
        }
        const float* ptr = out + ((unsigned long long)nc) * SPATIAL;
        float sum = 0.0f;
        float sumsq = 0.0f;
        for (unsigned i = threadIdx.x; i < SPATIAL; i += blockDim.x) {
            float v = ptr[i];
            sum += v;
            sumsq = fmaf(v, v, sumsq);
        }
        sum = warp_reduce_sum(sum);
        sumsq = warp_reduce_sum(sumsq);
        __shared__ float shared_sum[8];
        __shared__ float shared_sumsq[8];
        unsigned lane = threadIdx.x & 31;
        unsigned warp = threadIdx.x >> 5;
        if (lane == 0) {
            shared_sum[warp] = sum;
            shared_sumsq[warp] = sumsq;
        }
        __syncthreads();
        if (warp == 0) {
            float block_sum = lane < 8 ? shared_sum[lane] : 0.0f;
            float block_sumsq = lane < 8 ? shared_sumsq[lane] : 0.0f;
            block_sum = warp_reduce_sum(block_sum);
            block_sumsq = warp_reduce_sum(block_sumsq);
            if (lane == 0) {
                sample_means[nc] = block_sum * (1.0f / 123039.0f);
                unsigned c = nc & 31u;
                atomicAdd(channel_sums + c, block_sum);
                atomicAdd(channel_sumsq + c, block_sumsq);
            }
        }
        return;
    }

    if (op == 10) {
        unsigned c = threadIdx.x;
        if (c < 32) {
            float mean = channel_sums[c] * (1.0f / 1968624.0f);
            float var = channel_sumsq[c] * (1.0f / 1968624.0f) - mean * mean;
            var = var > 0.0f ? var : 0.0f;
            scales[c] = bn_weight[c] / sqrtf(var + 1.0e-5f);
        }
        return;
    }

    if (op == 11) {
        constexpr unsigned SPATIAL = 123039;
        constexpr unsigned TOTAL = 62995968;
        unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned stride = gridDim.x * blockDim.x;
        for (unsigned i = idx; i < TOTAL; i += stride) {
            unsigned nc = i / SPATIAL;
            unsigned c = nc & 31u;
            float v = out[i];
            out[i] = (v - sample_means[nc]) * scales[c];
        }
    }
}
