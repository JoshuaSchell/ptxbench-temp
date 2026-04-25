extern "C" __global__ __launch_bounds__(256)
void depthwise3x3_b64_c128_h256_w512_adj4_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ out
) {
    constexpr int C = 128;
    constexpr int IH = 256;
    constexpr int IW = 512;
    constexpr int OH = 254;
    constexpr int OW = 510;
    constexpr int TILE_W = 128;
    constexpr int TILE_H = 8;
    constexpr int OUTS_PER_THREAD = 4;
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
    const int ox = blockIdx.x * TILE_W + tx * OUTS_PER_THREAD;
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
        const int base = ty * TILE_PW + tx * OUTS_PER_THREAD;
        const float r00 = tile[base];
        const float r01 = tile[base + 1];
        const float r02 = tile[base + 2];
        const float r03 = tile[base + 3];
        const float r04 = tile[base + 4];
        const float r05 = tile[base + 5];
        const int row1 = base + TILE_PW;
        const float r10 = tile[row1];
        const float r11 = tile[row1 + 1];
        const float r12 = tile[row1 + 2];
        const float r13 = tile[row1 + 3];
        const float r14 = tile[row1 + 4];
        const float r15 = tile[row1 + 5];
        const int row2 = row1 + TILE_PW;
        const float r20 = tile[row2];
        const float r21 = tile[row2 + 1];
        const float r22 = tile[row2 + 2];
        const float r23 = tile[row2 + 3];
        const float r24 = tile[row2 + 4];
        const float r25 = tile[row2 + 5];

        const float w0 = wk[0];
        const float w1 = wk[1];
        const float w2 = wk[2];
        const float w3 = wk[3];
        const float w4 = wk[4];
        const float w5 = wk[5];
        const float w6 = wk[6];
        const float w7 = wk[7];
        const float w8 = wk[8];

        if (ox < OW) {
            float acc0 = 0.0f;
            acc0 = __fmaf_rn(w0, r00, acc0);
            acc0 = __fmaf_rn(w1, r01, acc0);
            acc0 = __fmaf_rn(w2, r02, acc0);
            acc0 = __fmaf_rn(w3, r10, acc0);
            acc0 = __fmaf_rn(w4, r11, acc0);
            acc0 = __fmaf_rn(w5, r12, acc0);
            acc0 = __fmaf_rn(w6, r20, acc0);
            acc0 = __fmaf_rn(w7, r21, acc0);
            acc0 = __fmaf_rn(w8, r22, acc0);
            const int out_idx0 = (((n * C + c) * OH + oy) * OW + ox);
            out[out_idx0] = acc0;
        }
        if (ox + 1 < OW) {
            float acc1 = 0.0f;
            acc1 = __fmaf_rn(w0, r01, acc1);
            acc1 = __fmaf_rn(w1, r02, acc1);
            acc1 = __fmaf_rn(w2, r03, acc1);
            acc1 = __fmaf_rn(w3, r11, acc1);
            acc1 = __fmaf_rn(w4, r12, acc1);
            acc1 = __fmaf_rn(w5, r13, acc1);
            acc1 = __fmaf_rn(w6, r21, acc1);
            acc1 = __fmaf_rn(w7, r22, acc1);
            acc1 = __fmaf_rn(w8, r23, acc1);
            const int out_idx1 = (((n * C + c) * OH + oy) * OW + ox + 1);
            out[out_idx1] = acc1;
        }
        if (ox + 2 < OW) {
            float acc2 = 0.0f;
            acc2 = __fmaf_rn(w0, r02, acc2);
            acc2 = __fmaf_rn(w1, r03, acc2);
            acc2 = __fmaf_rn(w2, r04, acc2);
            acc2 = __fmaf_rn(w3, r12, acc2);
            acc2 = __fmaf_rn(w4, r13, acc2);
            acc2 = __fmaf_rn(w5, r14, acc2);
            acc2 = __fmaf_rn(w6, r22, acc2);
            acc2 = __fmaf_rn(w7, r23, acc2);
            acc2 = __fmaf_rn(w8, r24, acc2);
            const int out_idx2 = (((n * C + c) * OH + oy) * OW + ox + 2);
            out[out_idx2] = acc2;
        }
        if (ox + 3 < OW) {
            float acc3 = 0.0f;
            acc3 = __fmaf_rn(w0, r03, acc3);
            acc3 = __fmaf_rn(w1, r04, acc3);
            acc3 = __fmaf_rn(w2, r05, acc3);
            acc3 = __fmaf_rn(w3, r13, acc3);
            acc3 = __fmaf_rn(w4, r14, acc3);
            acc3 = __fmaf_rn(w5, r15, acc3);
            acc3 = __fmaf_rn(w6, r23, acc3);
            acc3 = __fmaf_rn(w7, r24, acc3);
            acc3 = __fmaf_rn(w8, r25, acc3);
            const int out_idx3 = (((n * C + c) * OH + oy) * OW + ox + 3);
            out[out_idx3] = acc3;
        }
    }
}
