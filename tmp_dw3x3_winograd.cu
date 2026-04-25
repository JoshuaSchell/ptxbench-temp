extern "C" __global__ __launch_bounds__(64)
void depthwise3x3_winograd_kernel(
    const float* __restrict__ x,
    const float* __restrict__ uw,
    float* __restrict__ out
) {
    constexpr int C = 128;
    constexpr int IH = 256;
    constexpr int IW = 512;
    constexpr int OH = 254;
    constexpr int OW = 510;
    constexpr int TILES_X = 8;
    constexpr int TILES_Y = 8;
    constexpr int OUT_W = TILES_X * 2;
    constexpr int OUT_H = TILES_Y * 2;
    constexpr int IN_W = OUT_W + 2;
    constexpr int IN_H = OUT_H + 2;

    __shared__ float smem[IN_W * IN_H + 16];
    float* tile = smem;
    float* w = smem + IN_W * IN_H;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int oz = blockIdx.z;
    const int c = oz & (C - 1);
    const int n = oz >> 7;

    for (int i = tid; i < 16; i += 64) {
        w[i] = uw[c * 16 + i];
    }

    for (int i = tid; i < IN_W * IN_H; i += 64) {
        const int lx = i % IN_W;
        const int ly = i / IN_W;
        const int gx = blockIdx.x * OUT_W + lx;
        const int gy = blockIdx.y * OUT_H + ly;
        float v = 0.0f;
        if (gx < IW && gy < IH) {
            const int idx = (((n * C + c) * IH + gy) * IW + gx);
            v = x[idx];
        }
        tile[i] = v;
    }

    __syncthreads();

    const int ox = blockIdx.x * OUT_W + tx * 2;
    const int oy = blockIdx.y * OUT_H + ty * 2;
    if (ox >= OW || oy >= OH) {
        return;
    }

    const int base = (ty * 2) * IN_W + tx * 2;
    const float d00 = tile[base];
    const float d01 = tile[base + 1];
    const float d02 = tile[base + 2];
    const float d03 = tile[base + 3];
    const float d10 = tile[base + IN_W];
    const float d11 = tile[base + IN_W + 1];
    const float d12 = tile[base + IN_W + 2];
    const float d13 = tile[base + IN_W + 3];
    const float d20 = tile[base + 2 * IN_W];
    const float d21 = tile[base + 2 * IN_W + 1];
    const float d22 = tile[base + 2 * IN_W + 2];
    const float d23 = tile[base + 2 * IN_W + 3];
    const float d30 = tile[base + 3 * IN_W];
    const float d31 = tile[base + 3 * IN_W + 1];
    const float d32 = tile[base + 3 * IN_W + 2];
    const float d33 = tile[base + 3 * IN_W + 3];

    const float m00 = d00 - d20;
    const float m01 = d01 - d21;
    const float m02 = d02 - d22;
    const float m03 = d03 - d23;
    const float m10 = d10 + d20;
    const float m11 = d11 + d21;
    const float m12 = d12 + d22;
    const float m13 = d13 + d23;
    const float m20 = d20 - d10;
    const float m21 = d21 - d11;
    const float m22 = d22 - d12;
    const float m23 = d23 - d13;
    const float m30 = d10 - d30;
    const float m31 = d11 - d31;
    const float m32 = d12 - d32;
    const float m33 = d13 - d33;

    const float v00 = m00 - m02;
    const float v01 = m01 + m02;
    const float v02 = m02 - m01;
    const float v03 = m01 - m03;
    const float v10 = m10 - m12;
    const float v11 = m11 + m12;
    const float v12 = m12 - m11;
    const float v13 = m11 - m13;
    const float v20 = m20 - m22;
    const float v21 = m21 + m22;
    const float v22 = m22 - m21;
    const float v23 = m21 - m23;
    const float v30 = m30 - m32;
    const float v31 = m31 + m32;
    const float v32 = m32 - m31;
    const float v33 = m31 - m33;

    const float z00 = v00 * w[0];
    const float z01 = v01 * w[1];
    const float z02 = v02 * w[2];
    const float z03 = v03 * w[3];
    const float z10 = v10 * w[4];
    const float z11 = v11 * w[5];
    const float z12 = v12 * w[6];
    const float z13 = v13 * w[7];
    const float z20 = v20 * w[8];
    const float z21 = v21 * w[9];
    const float z22 = v22 * w[10];
    const float z23 = v23 * w[11];
    const float z30 = v30 * w[12];
    const float z31 = v31 * w[13];
    const float z32 = v32 * w[14];
    const float z33 = v33 * w[15];

    const float t00 = z00 + z10 + z20;
    const float t01 = z01 + z11 + z21;
    const float t02 = z02 + z12 + z22;
    const float t03 = z03 + z13 + z23;
    const float t10 = z10 - z20 - z30;
    const float t11 = z11 - z21 - z31;
    const float t12 = z12 - z22 - z32;
    const float t13 = z13 - z23 - z33;

    const float y00 = t00 + t01 + t02;
    const float y01 = t01 - t02 - t03;
    const float y10 = t10 + t11 + t12;
    const float y11 = t11 - t12 - t13;

    const int out_base = (((n * C + c) * OH + oy) * OW + ox);
    out[out_base] = y00;
    out[out_base + 1] = y01;
    out[out_base + OW] = y10;
    out[out_base + OW + 1] = y11;
}
