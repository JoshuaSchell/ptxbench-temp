extern "C" __global__ __launch_bounds__(256)
void depthwise3x3_shared_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ out
) {
    constexpr int C = 64;
    constexpr int IH = 512;
    constexpr int IW = 512;
    constexpr int OH = 510;
    constexpr int OW = 510;
    constexpr int TILE_W = 32;
    constexpr int TILE_H = 8;
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
    const int ox = blockIdx.x * TILE_W + tx;
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

    if (ox < OW && oy < OH) {
        const int base = ty * TILE_PW + tx;
        float acc = 0.0f;
        acc = __fmaf_rn(wk[0], tile[base], acc);
        acc = __fmaf_rn(wk[1], tile[base + 1], acc);
        acc = __fmaf_rn(wk[2], tile[base + 2], acc);
        acc = __fmaf_rn(wk[3], tile[base + TILE_PW], acc);
        acc = __fmaf_rn(wk[4], tile[base + TILE_PW + 1], acc);
        acc = __fmaf_rn(wk[5], tile[base + TILE_PW + 2], acc);
        acc = __fmaf_rn(wk[6], tile[base + 2 * TILE_PW], acc);
        acc = __fmaf_rn(wk[7], tile[base + 2 * TILE_PW + 1], acc);
        acc = __fmaf_rn(wk[8], tile[base + 2 * TILE_PW + 2], acc);
        const int out_idx = (((n * C + c) * OH + oy) * OW + ox);
        out[out_idx] = acc;
    }
}
