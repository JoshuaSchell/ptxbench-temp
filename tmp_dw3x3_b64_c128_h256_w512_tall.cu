extern "C" __global__ __launch_bounds__(256)
void depthwise3x3_b64_c128_h256_w512_tall_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ out
) {
    constexpr int C = 128;
    constexpr int IH = 256;
    constexpr int IW = 512;
    constexpr int OH = 254;
    constexpr int OW = 510;
    constexpr int TILE_W = 64;
    constexpr int TILE_H = 16;
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
    const int n = oz >> 7;
    const int ox0 = blockIdx.x * TILE_W + tx;
    const int ox1 = ox0 + THREAD_W;
    const int oy0 = blockIdx.y * TILE_H + ty;
    const int oy1 = oy0 + THREAD_H;

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

    const int base00 = ty * TILE_PW + tx;
    const int base01 = base00 + THREAD_W;
    const int base10 = base00 + THREAD_H * TILE_PW;
    const int base11 = base10 + THREAD_W;

    if (oy0 < OH) {
        if (ox0 < OW) {
            float acc00 = 0.0f;
            acc00 = __fmaf_rn(wk[0], tile[base00], acc00);
            acc00 = __fmaf_rn(wk[1], tile[base00 + 1], acc00);
            acc00 = __fmaf_rn(wk[2], tile[base00 + 2], acc00);
            acc00 = __fmaf_rn(wk[3], tile[base00 + TILE_PW], acc00);
            acc00 = __fmaf_rn(wk[4], tile[base00 + TILE_PW + 1], acc00);
            acc00 = __fmaf_rn(wk[5], tile[base00 + TILE_PW + 2], acc00);
            acc00 = __fmaf_rn(wk[6], tile[base00 + 2 * TILE_PW], acc00);
            acc00 = __fmaf_rn(wk[7], tile[base00 + 2 * TILE_PW + 1], acc00);
            acc00 = __fmaf_rn(wk[8], tile[base00 + 2 * TILE_PW + 2], acc00);
            const int out_idx00 = (((n * C + c) * OH + oy0) * OW + ox0);
            out[out_idx00] = acc00;
        }
        if (ox1 < OW) {
            float acc01 = 0.0f;
            acc01 = __fmaf_rn(wk[0], tile[base01], acc01);
            acc01 = __fmaf_rn(wk[1], tile[base01 + 1], acc01);
            acc01 = __fmaf_rn(wk[2], tile[base01 + 2], acc01);
            acc01 = __fmaf_rn(wk[3], tile[base01 + TILE_PW], acc01);
            acc01 = __fmaf_rn(wk[4], tile[base01 + TILE_PW + 1], acc01);
            acc01 = __fmaf_rn(wk[5], tile[base01 + TILE_PW + 2], acc01);
            acc01 = __fmaf_rn(wk[6], tile[base01 + 2 * TILE_PW], acc01);
            acc01 = __fmaf_rn(wk[7], tile[base01 + 2 * TILE_PW + 1], acc01);
            acc01 = __fmaf_rn(wk[8], tile[base01 + 2 * TILE_PW + 2], acc01);
            const int out_idx01 = (((n * C + c) * OH + oy0) * OW + ox1);
            out[out_idx01] = acc01;
        }
    }

    if (oy1 < OH) {
        if (ox0 < OW) {
            float acc10 = 0.0f;
            acc10 = __fmaf_rn(wk[0], tile[base10], acc10);
            acc10 = __fmaf_rn(wk[1], tile[base10 + 1], acc10);
            acc10 = __fmaf_rn(wk[2], tile[base10 + 2], acc10);
            acc10 = __fmaf_rn(wk[3], tile[base10 + TILE_PW], acc10);
            acc10 = __fmaf_rn(wk[4], tile[base10 + TILE_PW + 1], acc10);
            acc10 = __fmaf_rn(wk[5], tile[base10 + TILE_PW + 2], acc10);
            acc10 = __fmaf_rn(wk[6], tile[base10 + 2 * TILE_PW], acc10);
            acc10 = __fmaf_rn(wk[7], tile[base10 + 2 * TILE_PW + 1], acc10);
            acc10 = __fmaf_rn(wk[8], tile[base10 + 2 * TILE_PW + 2], acc10);
            const int out_idx10 = (((n * C + c) * OH + oy1) * OW + ox0);
            out[out_idx10] = acc10;
        }
        if (ox1 < OW) {
            float acc11 = 0.0f;
            acc11 = __fmaf_rn(wk[0], tile[base11], acc11);
            acc11 = __fmaf_rn(wk[1], tile[base11 + 1], acc11);
            acc11 = __fmaf_rn(wk[2], tile[base11 + 2], acc11);
            acc11 = __fmaf_rn(wk[3], tile[base11 + TILE_PW], acc11);
            acc11 = __fmaf_rn(wk[4], tile[base11 + TILE_PW + 1], acc11);
            acc11 = __fmaf_rn(wk[5], tile[base11 + TILE_PW + 2], acc11);
            acc11 = __fmaf_rn(wk[6], tile[base11 + 2 * TILE_PW], acc11);
            acc11 = __fmaf_rn(wk[7], tile[base11 + 2 * TILE_PW + 1], acc11);
            acc11 = __fmaf_rn(wk[8], tile[base11 + 2 * TILE_PW + 2], acc11);
            const int out_idx11 = (((n * C + c) * OH + oy1) * OW + ox1);
            out[out_idx11] = acc11;
        }
    }
}
