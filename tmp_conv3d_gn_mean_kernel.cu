#include <cuda_runtime.h>

extern "C" __global__ void conv_hswish_channel_sums_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ sums,
    float* __restrict__ sqs,
    unsigned int batch
) {
    const unsigned int bc = (unsigned int)blockIdx.x;
    const unsigned int oc = bc & 15u;
    const unsigned int n = bc >> 4;
    if (n >= batch) {
        return;
    }

    __shared__ float w_sh[192];
    __shared__ float red_sum[256];
    __shared__ float red_sq[256];

    const unsigned int tid = (unsigned int)threadIdx.x;
    if (tid < 192u) {
        w_sh[tid] = w[oc * 192u + tid];
    }
    __syncthreads();

    const float bias = b[oc];
    const unsigned int spatial = 13u * 29u * 29u;

    double local_sum = 0.0;
    double local_sq = 0.0;

    for (unsigned int idx = tid; idx < spatial; idx += blockDim.x) {
        const unsigned int od = idx / (29u * 29u);
        const unsigned int rem = idx - od * (29u * 29u);
        const unsigned int oh = rem / 29u;
        const unsigned int ow = rem - oh * 29u;

        float acc = bias;

        #pragma unroll
        for (unsigned int ic = 0; ic < 3u; ++ic) {
            #pragma unroll
            for (unsigned int kd = 0; kd < 4u; ++kd) {
                #pragma unroll
                for (unsigned int kh = 0; kh < 4u; ++kh) {
                    #pragma unroll
                    for (unsigned int kw = 0; kw < 4u; ++kw) {
                        const unsigned int x_index =
                            (((((n * 3u + ic) * 16u + (od + kd)) * 32u + (oh + kh)) * 32u) + (ow + kw));
                        const unsigned int w_index = (((ic * 4u + kd) * 4u + kh) * 4u + kw);
                        acc = fmaf(x[x_index], w_sh[w_index], acc);
                    }
                }
            }
        }

        const float t = fminf(fmaxf(acc + 3.0f, 0.0f), 6.0f) * (1.0f / 6.0f);
        const float y = acc * t;
        local_sum += (double)y;
        local_sq += (double)y * (double)y;
    }

    red_sum[tid] = (float)local_sum;
    red_sq[tid] = (float)local_sq;
    __syncthreads();

    for (unsigned int offset = blockDim.x >> 1; offset > 32u; offset >>= 1) {
        if (tid < offset) {
            red_sum[tid] += red_sum[tid + offset];
            red_sq[tid] += red_sq[tid + offset];
        }
        __syncthreads();
    }

    if (tid < 32u) {
        volatile float* v_sum = red_sum;
        volatile float* v_sq = red_sq;
        v_sum[tid] += v_sum[tid + 32u];
        v_sq[tid] += v_sq[tid + 32u];
        v_sum[tid] += v_sum[tid + 16u];
        v_sq[tid] += v_sq[tid + 16u];
        v_sum[tid] += v_sum[tid + 8u];
        v_sq[tid] += v_sq[tid + 8u];
        v_sum[tid] += v_sum[tid + 4u];
        v_sq[tid] += v_sq[tid + 4u];
        v_sum[tid] += v_sum[tid + 2u];
        v_sq[tid] += v_sq[tid + 2u];
        v_sum[tid] += v_sum[tid + 1u];
        v_sq[tid] += v_sq[tid + 1u];
    }

    if (tid == 0u) {
        sums[bc] = red_sum[0];
        sqs[bc] = red_sq[0];
    }
}

extern "C" __global__ void finalize_groupnorm_mean_kernel(
    const float* __restrict__ sums,
    const float* __restrict__ sqs,
    const float* __restrict__ gn_weight,
    const float* __restrict__ gn_bias,
    float* __restrict__ out,
    unsigned int batch
) {
    const unsigned int n = (unsigned int)blockIdx.x;
    const unsigned int c = (unsigned int)threadIdx.x;
    if (n >= batch || c >= 16u) {
        return;
    }

    const unsigned int base = n * 16u;
    const unsigned int gbase = base + ((c >> 2) << 2);
    const float gsum = sums[gbase] + sums[gbase + 1u] + sums[gbase + 2u] + sums[gbase + 3u];
    const float gsq = sqs[gbase] + sqs[gbase + 1u] + sqs[gbase + 2u] + sqs[gbase + 3u];
    const float inv_count = 1.0f / (4.0f * 13.0f * 29.0f * 29.0f);
    const float mean = gsum * inv_count;
    const float var = fmaxf(gsq * inv_count - mean * mean, 0.0f);
    const float invstd = rsqrtf(var + 1.0e-5f);
    const float ch_mean = sums[base + c] * (1.0f / (13.0f * 29.0f * 29.0f));
    out[base + c] = (ch_mean - mean) * invstd * gn_weight[c] + gn_bias[c];
}
