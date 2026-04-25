extern "C" __global__ void clear_channel_stats(float* channel_sums, float* channel_sumsq) {
    unsigned c = threadIdx.x;
    if (c < 32) {
        channel_sums[c] = 0.0f;
        channel_sumsq[c] = 0.0f;
    }
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

extern "C" __global__ void reduce_stats(
    const float* __restrict__ x,
    float* __restrict__ sample_means,
    float* __restrict__ channel_sums,
    float* __restrict__ channel_sumsq
) {
    constexpr unsigned SPATIAL = 123039;
    unsigned nc = blockIdx.x;
    if (nc >= 512) {
        return;
    }

    const float* ptr = x + ((unsigned long long)nc) * SPATIAL;
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
}

extern "C" __global__ void finalize_scales(
    const float* __restrict__ channel_sums,
    const float* __restrict__ channel_sumsq,
    const float* __restrict__ weight,
    float* __restrict__ scales
) {
    unsigned c = threadIdx.x;
    if (c >= 32) {
        return;
    }
    float mean = channel_sums[c] * (1.0f / 1968624.0f);
    float var = channel_sumsq[c] * (1.0f / 1968624.0f) - mean * mean;
    var = var > 0.0f ? var : 0.0f;
    scales[c] = weight[c] / sqrtf(var + 1.0e-5f);
}

extern "C" __global__ void apply_scale_center(
    float* __restrict__ x,
    const float* __restrict__ sample_means,
    const float* __restrict__ scales
) {
    constexpr unsigned SPATIAL = 123039;
    constexpr unsigned TOTAL = 62995968;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned stride = gridDim.x * blockDim.x;
    for (unsigned i = idx; i < TOTAL; i += stride) {
        unsigned nc = i / SPATIAL;
        unsigned c = nc & 31u;
        float v = x[i];
        x[i] = (v - sample_means[nc]) * scales[c];
    }
}
