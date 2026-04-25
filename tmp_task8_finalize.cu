extern "C" __global__ void finalize_out(
    const float* __restrict__ accum,
    const float* __restrict__ post_bias_sum,
    float* __restrict__ out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 128) {
        out[idx] = accum[idx] * (0.5f / 6727.0f) + post_bias_sum[0];
    }
}
