#include <cuda_runtime.h>
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
