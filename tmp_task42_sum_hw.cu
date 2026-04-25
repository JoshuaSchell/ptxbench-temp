extern "C" __device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

extern "C" __global__ void sum_hw_kernel(
    const float* __restrict__ x,
    float* __restrict__ sums,
    int B,
    int C,
    int H,
    int W
) {
    int bc = blockIdx.x;
    int tid = threadIdx.x;
    if (bc >= B * C) return;

    int plane = H * W;
    int n4 = plane >> 2;
    const float4* plane4 = reinterpret_cast<const float4*>(x + (size_t)bc * (size_t)plane);
    float acc = 0.0f;

    for (int i = tid; i < n4; i += blockDim.x) {
        float4 v = plane4[i];
        acc += v.x + v.y + v.z + v.w;
    }

    int rem_start = n4 << 2;
    for (int i = rem_start + tid; i < plane; i += blockDim.x) {
        acc += x[(size_t)bc * (size_t)plane + (size_t)i];
    }

    acc = warp_sum(acc);

    __shared__ float warp_buf[8];
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) warp_buf[warp] = acc;
    __syncthreads();

    if (warp == 0) {
        float total = (lane < (blockDim.x >> 5)) ? warp_buf[lane] : 0.0f;
        total = warp_sum(total);
        if (lane == 0) sums[bc] = total;
    }
}
