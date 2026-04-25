__device__ __forceinline__ float clamp01(float x) {
    x = x < 0.0f ? 0.0f : x;
    return x > 1.0f ? 1.0f : x;
}

__device__ __forceinline__ float warp_reduce_max(float value) {
    value = max(value, __shfl_down_sync(0xffffffff, value, 16));
    value = max(value, __shfl_down_sync(0xffffffff, value, 8));
    value = max(value, __shfl_down_sync(0xffffffff, value, 4));
    value = max(value, __shfl_down_sync(0xffffffff, value, 2));
    value = max(value, __shfl_down_sync(0xffffffff, value, 1));
    return value;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
    value += __shfl_down_sync(0xffffffff, value, 16);
    value += __shfl_down_sync(0xffffffff, value, 8);
    value += __shfl_down_sync(0xffffffff, value, 4);
    value += __shfl_down_sync(0xffffffff, value, 2);
    value += __shfl_down_sync(0xffffffff, value, 1);
    return value;
}

template <int THREADS>
__device__ __forceinline__ float block_reduce_max(float value) {
    __shared__ float shared[THREADS / 32];
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp = threadIdx.x >> 5;
    value = warp_reduce_max(value);
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < (THREADS / 32) ? shared[lane] : 0.0f;
    if (warp == 0) {
        value = warp_reduce_max(value);
    }
    return value;
}

template <int THREADS>
__device__ __forceinline__ float block_reduce_sum(float value) {
    __shared__ float shared[THREADS / 32];
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp = threadIdx.x >> 5;
    value = warp_reduce_sum(value);
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < (THREADS / 32) ? shared[lane] : 0.0f;
    if (warp == 0) {
        value = warp_reduce_sum(value);
    }
    return value;
}

extern "C" __global__ __launch_bounds__(256, 2) void reduce_clamp_softmax_stats_kernel(
    const float* __restrict__ x,
    float* __restrict__ stats,
    unsigned int rows
) {
    const unsigned int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    constexpr unsigned int kSpatial = 131072;
    const float* row_ptr = x + (static_cast<unsigned long long>(row) << 17);

    float local_max = 0.0f;
    for (unsigned int idx = threadIdx.x; idx < kSpatial; idx += 256) {
        local_max = max(local_max, clamp01(row_ptr[idx]));
    }

    const float row_max = block_reduce_max<256>(local_max);
    __shared__ float shared_max;
    if (threadIdx.x == 0) {
        shared_max = row_max;
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (unsigned int idx = threadIdx.x; idx < kSpatial; idx += 256) {
        local_sum += __expf(clamp01(row_ptr[idx]) - shared_max);
    }

    const float row_sum = block_reduce_sum<256>(local_sum);
    if (threadIdx.x == 0) {
        const unsigned long long base = static_cast<unsigned long long>(row) << 1;
        stats[base] = shared_max;
        stats[base + 1] = 1.0f / row_sum;
    }
}

extern "C" __global__ __launch_bounds__(256, 4) void apply_clamp_softmax_inplace_kernel(
    float* __restrict__ x,
    const float* __restrict__ stats,
    unsigned int total_vec4
) {
    const unsigned int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_index >= total_vec4) {
        return;
    }

    const unsigned int row = vec_index >> 15;
    const unsigned long long stats_base = static_cast<unsigned long long>(row) << 1;
    const float row_max = stats[stats_base];
    const float row_inv_sum = stats[stats_base + 1];

    float4 values = reinterpret_cast<float4*>(x)[vec_index];
    values.x = __expf(clamp01(values.x) - row_max) * row_inv_sum;
    values.y = __expf(clamp01(values.y) - row_max) * row_inv_sum;
    values.z = __expf(clamp01(values.z) - row_max) * row_inv_sum;
    values.w = __expf(clamp01(values.w) - row_max) * row_inv_sum;
    reinterpret_cast<float4*>(x)[vec_index] = values;
}
