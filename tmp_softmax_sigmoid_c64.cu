extern "C" __global__ __launch_bounds__(256) void softmax_sigmoid_c64_inplace(
    float* __restrict__ x,
    int rows,
    int spatial
)
{
    __shared__ float warp_max[8];
    __shared__ float row_max[4];
    __shared__ float warp_sum[8];
    __shared__ float row_sum[4];

    const int tid = (int)threadIdx.x;
    const int group = tid >> 6;
    const int lane64 = tid & 63;
    const int lane32 = tid & 31;
    const int warp = tid >> 5;
    const int row = ((int)blockIdx.x << 2) + group;

    bool active = row < rows;
    long long idx = 0;
    float v = -3.402823466e38f;
    if (active) {
        const int n = row / spatial;
        const int s = row - n * spatial;
        idx = (((long long)n * 64LL) + (long long)lane64) * (long long)spatial + (long long)s;
        v = x[idx];
    }

    float local_max = v;
    local_max = fmaxf(local_max, __shfl_down_sync(0xffffffffu, local_max, 16));
    local_max = fmaxf(local_max, __shfl_down_sync(0xffffffffu, local_max, 8));
    local_max = fmaxf(local_max, __shfl_down_sync(0xffffffffu, local_max, 4));
    local_max = fmaxf(local_max, __shfl_down_sync(0xffffffffu, local_max, 2));
    local_max = fmaxf(local_max, __shfl_down_sync(0xffffffffu, local_max, 1));
    if (lane32 == 0) {
        warp_max[warp] = local_max;
    }
    __syncthreads();

    if (lane64 == 0) {
        const int warp0 = group << 1;
        row_max[group] = fmaxf(warp_max[warp0], warp_max[warp0 + 1]);
    }
    __syncthreads();

    const float maxv = row_max[group];
    float e = active ? __expf(v - maxv) : 0.0f;
    float local_sum = e;
    local_sum += __shfl_down_sync(0xffffffffu, local_sum, 16);
    local_sum += __shfl_down_sync(0xffffffffu, local_sum, 8);
    local_sum += __shfl_down_sync(0xffffffffu, local_sum, 4);
    local_sum += __shfl_down_sync(0xffffffffu, local_sum, 2);
    local_sum += __shfl_down_sync(0xffffffffu, local_sum, 1);
    if (lane32 == 0) {
        warp_sum[warp] = local_sum;
    }
    __syncthreads();

    if (lane64 == 0) {
        const int warp0 = group << 1;
        row_sum[group] = warp_sum[warp0] + warp_sum[warp0 + 1];
    }
    __syncthreads();

    if (active) {
        const float p = e / row_sum[group];
        x[idx] = 1.0f / (1.0f + __expf(-p));
    }
}
