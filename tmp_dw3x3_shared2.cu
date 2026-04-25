extern "C" __global__ __launch_bounds__(256)
void depthwise3x3_shared2_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ out
) {
    constexpr int C = 64;
    constexpr int IH = 512;
    constexpr int IW = 512;
    constexpr int OH = 510;
    constexpr int OW = 510;
    constexpr int TILE_W = 64;
    constexpr int TILE_H = 8;
    constexpr int THREAD_W = 32;
    constexpr int THREAD_H = 8;
    constexpr int TILE_PW = TILE_W + 2;
    constexpr int TILE_PH = TILE_H + 2;

    __shared__ float smem[TILE_PW * TILE_PH + 9];
    float* tile = smem;
    float* wk = smem + TILE_PW * TILE_PH;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int oz = blockIdx.z;
    const int c = oz & (C - 1);
    const int n = oz >> 6;
    const int ox0 = blockIdx.x * TILE_W + tx;
    const int ox1 = ox0 + THREAD_W;
    const int oy = blockIdx.y * TILE_H + ty;

    for (int i = tid; i < 9; i += 256) {
        wk[i] = w[c * 9 + i];
    }

    for (int i = tid; i < TILE_PW * TILE_PH; i += 256) {
        const int lx = i % TILE_PW;
        const int ly = i / TILE_PW;
        const int gx = blockIdx.x * TILE_W + lx;
        const int gy = blockIdx.y * TILE_H + ly;
        float v = 0.0f;
        if (gx < IW && gy < IH) {
            const int idx = (((n * C + c) * IH + gy) * IW + gx);
            v = x[idx];
        }
        tile[i] = v;
    }

    __syncthreads();

    if (oy < OH) {
        const int base0 = ty * TILE_PW + tx;
        const int base1 = base0 + THREAD_W;
        if (ox0 < OW) {
            float acc0 = 0.0f;
            acc0 = __fmaf_rn(wk[0], tile[base0], acc0);
            acc0 = __fmaf_rn(wk[1], tile[base0 + 1], acc0);
            acc0 = __fmaf_rn(wk[2], tile[base0 + 2], acc0);
            acc0 = __fmaf_rn(wk[3], tile[base0 + TILE_PW], acc0);
            acc0 = __fmaf_rn(wk[4], tile[base0 + TILE_PW + 1], acc0);
            acc0 = __fmaf_rn(wk[5], tile[base0 + TILE_PW + 2], acc0);
            acc0 = __fmaf_rn(wk[6], tile[base0 + 2 * TILE_PW], acc0);
            acc0 = __fmaf_rn(wk[7], tile[base0 + 2 * TILE_PW + 1], acc0);
            acc0 = __fmaf_rn(wk[8], tile[base0 + 2 * TILE_PW + 2], acc0);
            const int out_idx0 = (((n * C + c) * OH + oy) * OW + ox0);
            out[out_idx0] = acc0;
        }
        if (ox1 < OW) {
            float acc1 = 0.0f;
            acc1 = __fmaf_rn(wk[0], tile[base1], acc1);
            acc1 = __fmaf_rn(wk[1], tile[base1 + 1], acc1);
            acc1 = __fmaf_rn(wk[2], tile[base1 + 2], acc1);
            acc1 = __fmaf_rn(wk[3], tile[base1 + TILE_PW], acc1);
            acc1 = __fmaf_rn(wk[4], tile[base1 + TILE_PW + 1], acc1);
            acc1 = __fmaf_rn(wk[5], tile[base1 + TILE_PW + 2], acc1);
            acc1 = __fmaf_rn(wk[6], tile[base1 + 2 * TILE_PW], acc1);
            acc1 = __fmaf_rn(wk[7], tile[base1 + 2 * TILE_PW + 1], acc1);
            acc1 = __fmaf_rn(wk[8], tile[base1 + 2 * TILE_PW + 2], acc1);
            const int out_idx1 = (((n * C + c) * OH + oy) * OW + ox1);
            out[out_idx1] = acc1;
        }
    }
}
