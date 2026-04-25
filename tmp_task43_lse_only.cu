extern "C" __global__ void lse_relu_kernel(
    const float* pooled,
    float* out,
    const int* meta,
    unsigned int n
) {
    if (meta[1]) {
        return;
    }
    unsigned int idx = blockIdx.x;
    if (threadIdx.x != 0 || idx >= n) {
        return;
    }

    unsigned int t = idx;
    unsigned int ow = t & 63;
    t >>= 6;
    unsigned int oh = t & 63;
    t >>= 6;
    unsigned int od = t & 15;
    unsigned int b = t >> 4;

    unsigned int spatial = ((od * 64) + oh) * 64 + ow;
    float maxv = -3.402823466e38f;
    for (int co = 0; co < 64; ++co) {
        float v = pooled[((((b * 64U) + (unsigned int)co) * 16U) * 64U * 64U) + spatial];
        if (v > maxv) {
            maxv = v;
        }
    }

    float sum = 0.0f;
    for (int co = 0; co < 64; ++co) {
        float v = pooled[((((b * 64U) + (unsigned int)co) * 16U) * 64U * 64U) + spatial];
        sum += expf(v - maxv);
    }
    float y = maxv + logf(sum);
    out[idx] = y > 0.0f ? y : 0.0f;
}
