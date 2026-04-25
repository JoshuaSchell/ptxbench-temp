extern "C" __device__ __forceinline__ float gelu_fast(float x) {
    const float inv_sqrt2 = 0.7071067811865475f;
    const float p = 0.3275911f;
    const float a1 = 0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 = 1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 = 1.061405429f;

    float z = x * inv_sqrt2;
    float az = fabsf(z);
    float t = 1.0f / (1.0f + p * az);
    float poly = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t;
    float erf_approx = copysignf(1.0f - poly * __expf(-az * az), z);
    return 0.5f * x * (1.0f + erf_approx);
}

extern "C" __global__ void fused_ln_gelu_scale_inplace(
    float* x,
    float scale,
    unsigned int rows
) {
    unsigned int tid = threadIdx.x;
    unsigned int lane = tid & 31u;
    unsigned int warp = tid >> 5;
    unsigned int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= rows) {
        return;
    }

    float* row_ptr = x + (((unsigned long long)row) << 6);
    float v0 = row_ptr[lane];
    float v1 = row_ptr[lane + 32];
    float sum = v0 + v1;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    float mean = __shfl_sync(0xffffffffu, sum * (1.0f / 64.0f), 0);

    float d0 = v0 - mean;
    float d1 = v1 - mean;
    float var = d0 * d0 + d1 * d1;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        var += __shfl_down_sync(0xffffffffu, var, offset);
    }
    float inv_std = rsqrtf(__shfl_sync(0xffffffffu, var * (1.0f / 64.0f) + 1.0e-5f, 0));

    row_ptr[lane] = gelu_fast(d0 * inv_std) * scale;
    row_ptr[lane + 32] = gelu_fast(d1 * inv_std) * scale;
}
