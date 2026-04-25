extern "C" __global__ __launch_bounds__(256) void softmax_sigmoid_c64_warp_inplace(
    float* __restrict__ x,
    int rows,
    int spatial
)
{
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int warps_per_block = 8;
    const int row = ((int)blockIdx.x * warps_per_block) + warp;
    if (row >= rows) {
        return;
    }

    const int n = row / spatial;
    const int s = row - n * spatial;
    const long long stride = (long long)spatial;
    const long long base = ((long long)n * 64LL) * stride + (long long)s;
    const long long idx0 = base + ((long long)lane) * stride;
    const long long idx1 = idx0 + 32LL * stride;

    const float v0 = x[idx0];
    const float v1 = x[idx1];

    float maxv = fmaxf(v0, v1);
    maxv = fmaxf(maxv, __shfl_down_sync(0xffffffffu, maxv, 16));
    maxv = fmaxf(maxv, __shfl_down_sync(0xffffffffu, maxv, 8));
    maxv = fmaxf(maxv, __shfl_down_sync(0xffffffffu, maxv, 4));
    maxv = fmaxf(maxv, __shfl_down_sync(0xffffffffu, maxv, 2));
    maxv = fmaxf(maxv, __shfl_down_sync(0xffffffffu, maxv, 1));
    maxv = __shfl_sync(0xffffffffu, maxv, 0);

    const float e0 = __expf(v0 - maxv);
    const float e1 = __expf(v1 - maxv);
    float sum = e0 + e1;
    sum += __shfl_down_sync(0xffffffffu, sum, 16);
    sum += __shfl_down_sync(0xffffffffu, sum, 8);
    sum += __shfl_down_sync(0xffffffffu, sum, 4);
    sum += __shfl_down_sync(0xffffffffu, sum, 2);
    sum += __shfl_down_sync(0xffffffffu, sum, 1);
    sum = __shfl_sync(0xffffffffu, sum, 0);

    const float p0 = e0 / sum;
    const float p1 = e1 / sum;
    x[idx0] = 1.0f / (1.0f + __expf(-p0));
    x[idx1] = 1.0f / (1.0f + __expf(-p1));
}
