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

    float ex = expf(v - vmax);
    values[tid] = ex;
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
