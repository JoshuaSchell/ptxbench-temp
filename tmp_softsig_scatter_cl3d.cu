extern "C" __global__ __launch_bounds__(256) void softsig_scatter_cl3d_kernel(
    const float* __restrict__ packed,
    float* __restrict__ out,
    int rows,
    int w2,
    int h2,
    int d2
)
{
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row = ((int)blockIdx.x << 3) + warp;
    if (row >= rows) {
        return;
    }

    const int hw2 = h2 * w2;
    const int dhw2 = d2 * hw2;
    const int n = row / dhw2;
    int rem = row - n * dhw2;
    const int od = rem / hw2;
    rem -= od * hw2;
    const int oh = rem / w2;
    const int ow = rem - oh * w2;

    const int phase = ((od & 1) << 2) | ((oh & 1) << 1) | (ow & 1);
    const int id = od >> 1;
    const int ih = oh >> 1;
    const int iw = ow >> 1;
    const int h = h2 >> 1;
    const int w = w2 >> 1;
    const long long low_row = ((((long long)n * (long long)(d2 >> 1) + (long long)id) * (long long)h + (long long)ih) * (long long)w + (long long)iw);
    const long long packed_base = (low_row << 9) + ((long long)phase << 6);
    const long long out_base = ((long long)row) << 6;

    const float v0 = packed[packed_base + (long long)lane];
    const float v1 = packed[packed_base + (long long)lane + 32LL];

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
    out[out_base + (long long)lane] = 1.0f / (1.0f + __expf(-p0));
    out[out_base + (long long)lane + 32LL] = 1.0f / (1.0f + __expf(-p1));
}
