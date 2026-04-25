extern "C" __global__ void final_lse_kernel(
    const float* __restrict__ sums,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ extra_bias,
    float* __restrict__ out,
    int B,
    int H,
    int W,
    int K
) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    if (b >= B || tid >= 128) return;

    const int IC = 64;
    const int OC = 128;
    const int KK = 9;
    float denom = (float)((H + K - 1) * (W + K - 1));

    float acc = 0.0f;
    const float* sums_b = sums + (size_t)b * IC;

    #pragma unroll
    for (int ic = 0; ic < IC; ++ic) {
        const float* wptr = weight + (((size_t)ic * OC + tid) * KK);
        float ksum = 0.0f;
        #pragma unroll
        for (int k = 0; k < KK; ++k) {
            ksum += wptr[k];
        }
        acc = fmaf(sums_b[ic], ksum, acc);
    }
    float v = acc / denom + conv_bias[tid] + extra_bias[tid];

    __shared__ float values[128];
    values[tid] = v;
    __syncthreads();

    for (int stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other = values[tid + stride];
            values[tid] = values[tid] > other ? values[tid] : other;
        }
        __syncthreads();
    }
    float vmax = values[0];

    values[tid] = expf(v - vmax);
    __syncthreads();

    for (int stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) {
            values[tid] += values[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[b] = (vmax + logf(values[0])) * 10.0f;
    }
}
