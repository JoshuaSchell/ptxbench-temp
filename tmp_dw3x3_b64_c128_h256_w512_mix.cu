extern "C" __global__ __launch_bounds__(256)
void depthwise3x3_b64_c128_h256_w512_mix_kernel(
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
    constexpr int OUTS_X = 2;
    constexpr int OUTS_Y = 2;
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
    const int ox0 = blockIdx.x * TILE_W + tx * OUTS_X;
    const int ox1 = ox0 + 1;
    const int oy0 = blockIdx.y * TILE_H + ty;
    const int oy1 = oy0 + 8;

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

    const float w0 = wk[0];
    const float w1 = wk[1];
    const float w2 = wk[2];
    const float w3 = wk[3];
    const float w4 = wk[4];
    const float w5 = wk[5];
    const float w6 = wk[6];
    const float w7 = wk[7];
    const float w8 = wk[8];

    const int base0 = ty * TILE_PW + tx * OUTS_X;
    const int row01 = base0 + TILE_PW;
    const int row02 = row01 + TILE_PW;
    const float a00 = tile[base0];
    const float a01 = tile[base0 + 1];
    const float a02 = tile[base0 + 2];
    const float a03 = tile[base0 + 3];
    const float a10 = tile[row01];
    const float a11 = tile[row01 + 1];
    const float a12 = tile[row01 + 2];
    const float a13 = tile[row01 + 3];
    const float a20 = tile[row02];
    const float a21 = tile[row02 + 1];
    const float a22 = tile[row02 + 2];
    const float a23 = tile[row02 + 3];

    if (oy0 < OH) {
        if (ox0 < OW) {
            float acc00 = 0.0f;
            acc00 = __fmaf_rn(w0, a00, acc00);
            acc00 = __fmaf_rn(w1, a01, acc00);
            acc00 = __fmaf_rn(w2, a02, acc00);
            acc00 = __fmaf_rn(w3, a10, acc00);
            acc00 = __fmaf_rn(w4, a11, acc00);
            acc00 = __fmaf_rn(w5, a12, acc00);
            acc00 = __fmaf_rn(w6, a20, acc00);
            acc00 = __fmaf_rn(w7, a21, acc00);
            acc00 = __fmaf_rn(w8, a22, acc00);
            const int out_idx00 = (((n * C + c) * OH + oy0) * OW + ox0);
            out[out_idx00] = acc00;
        }
        if (ox1 < OW) {
            float acc01 = 0.0f;
            acc01 = __fmaf_rn(w0, a01, acc01);
            acc01 = __fmaf_rn(w1, a02, acc01);
            acc01 = __fmaf_rn(w2, a03, acc01);
            acc01 = __fmaf_rn(w3, a11, acc01);
            acc01 = __fmaf_rn(w4, a12, acc01);
            acc01 = __fmaf_rn(w5, a13, acc01);
            acc01 = __fmaf_rn(w6, a21, acc01);
            acc01 = __fmaf_rn(w7, a22, acc01);
            acc01 = __fmaf_rn(w8, a23, acc01);
            const int out_idx01 = (((n * C + c) * OH + oy0) * OW + ox1);
            out[out_idx01] = acc01;
        }
    }

    const int base1 = base0 + 8 * TILE_PW;
    const int row11 = base1 + TILE_PW;
    const int row12 = row11 + TILE_PW;
    const float b00 = tile[base1];
    const float b01 = tile[base1 + 1];
    const float b02 = tile[base1 + 2];
    const float b03 = tile[base1 + 3];
    const float b10 = tile[row11];
    const float b11 = tile[row11 + 1];
    const float b12 = tile[row11 + 2];
    const float b13 = tile[row11 + 3];
    const float b20 = tile[row12];
    const float b21 = tile[row12 + 1];
    const float b22 = tile[row12 + 2];
    const float b23 = tile[row12 + 3];

    if (oy1 < OH) {
        if (ox0 < OW) {
            float acc10 = 0.0f;
            acc10 = __fmaf_rn(w0, b00, acc10);
            acc10 = __fmaf_rn(w1, b01, acc10);
            acc10 = __fmaf_rn(w2, b02, acc10);
            acc10 = __fmaf_rn(w3, b10, acc10);
            acc10 = __fmaf_rn(w4, b11, acc10);
            acc10 = __fmaf_rn(w5, b12, acc10);
            acc10 = __fmaf_rn(w6, b20, acc10);
            acc10 = __fmaf_rn(w7, b21, acc10);
            acc10 = __fmaf_rn(w8, b22, acc10);
            const int out_idx10 = (((n * C + c) * OH + oy1) * OW + ox0);
            out[out_idx10] = acc10;
        }
        if (ox1 < OW) {
            float acc11 = 0.0f;
            acc11 = __fmaf_rn(w0, b01, acc11);
            acc11 = __fmaf_rn(w1, b02, acc11);
            acc11 = __fmaf_rn(w2, b03, acc11);
            acc11 = __fmaf_rn(w3, b11, acc11);
            acc11 = __fmaf_rn(w4, b12, acc11);
            acc11 = __fmaf_rn(w5, b13, acc11);
            acc11 = __fmaf_rn(w6, b21, acc11);
            acc11 = __fmaf_rn(w7, b22, acc11);
            acc11 = __fmaf_rn(w8, b23, acc11);
            const int out_idx11 = (((n * C + c) * OH + oy1) * OW + ox1);
            out[out_idx11] = acc11;
        }
    }
}
