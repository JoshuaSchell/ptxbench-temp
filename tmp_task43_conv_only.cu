extern "C" __global__ void convpool3d_kernel(
    const float* x,
    const float* w,
    const float* bias,
    float* pooled,
    const int* meta,
    unsigned int n
) {
    if (meta[1]) {
        return;
    }
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    unsigned int t = idx;
    unsigned int ow = t & 63;
    t >>= 6;
    unsigned int oh = t & 63;
    t >>= 6;
    unsigned int od = t & 15;
    t >>= 4;
    unsigned int co = t & 63;
    unsigned int b = t >> 6;

    int base_d = (int)(od << 1);
    int base_h = (int)(oh << 1);
    int base_w = (int)(ow << 1);
    float maxv = -3.402823466e38f;

    #pragma unroll
    for (int pz = 0; pz < 2; ++pz) {
        int cd = base_d + pz;
        #pragma unroll
        for (int py = 0; py < 2; ++py) {
            int ch = base_h + py;
            #pragma unroll
            for (int px = 0; px < 2; ++px) {
                int cw = base_w + px;
                float acc = bias[co];
                for (int ci = 0; ci < 32; ++ci) {
                    int x_ci_base = ((((int)b * 32 + ci) * 32) * 128) * 128;
                    int w_ci_base = ((((int)co * 32 + ci) * 3) * 3) * 3;
                    #pragma unroll
                    for (int kz = 0; kz < 3; ++kz) {
                        int iz = cd + kz - 1;
                        if ((unsigned int)iz >= 32U) {
                            continue;
                        }
                        #pragma unroll
                        for (int ky = 0; ky < 3; ++ky) {
                            int iy = ch + ky - 1;
                            if ((unsigned int)iy >= 128U) {
                                continue;
                            }
                            #pragma unroll
                            for (int kx = 0; kx < 3; ++kx) {
                                int ix = cw + kx - 1;
                                if ((unsigned int)ix >= 128U) {
                                    continue;
                                }
                                int x_idx = x_ci_base + ((iz * 128 + iy) * 128 + ix);
                                int w_idx = w_ci_base + ((kz * 3 + ky) * 3 + kx);
                                acc = fmaf(x[x_idx], w[w_idx], acc);
                            }
                        }
                    }
                }
                if (acc > maxv) {
                    maxv = acc;
                }
            }
        }
    }

    pooled[idx] = maxv;
}
