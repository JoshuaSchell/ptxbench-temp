extern "C" __global__ void fused_convtx3d_sum_residual_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ conv_bias,
    const float* __restrict__ bias_term,
    float* __restrict__ out,
    unsigned int n_elem
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (; idx < n_elem; idx += stride) {
        unsigned int ow = idx & 63;
        unsigned int oh = (idx >> 6) & 63;
        unsigned int od = (idx >> 12) & 31;
        unsigned int oc = (idx >> 17) & 63;
        unsigned int n = idx >> 23;

        unsigned int iw0 = ow >> 1;
        unsigned int ih0 = oh >> 1;
        unsigned int id0 = od >> 1;

        unsigned int kw0 = (ow & 1) ? 2 : 1;
        unsigned int kh0 = (oh & 1) ? 2 : 1;
        unsigned int kd0 = (od & 1) ? 2 : 1;

        unsigned int iw1 = iw0 + 1;
        unsigned int ih1 = ih0 + 1;
        unsigned int id1 = id0 + 1;

        unsigned int wn = (ow & 1) && iw1 < 32 ? 2 : 1;
        unsigned int hn = (oh & 1) && ih1 < 32 ? 2 : 1;
        unsigned int dn = (od & 1) && id1 < 16 ? 2 : 1;

        float acc = conv_bias[oc];

        unsigned int base_n = n << 19;

#pragma unroll
        for (unsigned int dz = 0; dz < 2; ++dz) {
            if (dz >= dn) {
                continue;
            }
            unsigned int iz = dz ? id1 : id0;
            unsigned int kd = dz ? 0 : kd0;

#pragma unroll
            for (unsigned int dy = 0; dy < 2; ++dy) {
                if (dy >= hn) {
                    continue;
                }
                unsigned int iy = dy ? ih1 : ih0;
                unsigned int kh = dy ? 0 : kh0;

#pragma unroll
                for (unsigned int dx = 0; dx < 2; ++dx) {
                    if (dx >= wn) {
                        continue;
                    }
                    unsigned int ix = dx ? iw1 : iw0;
                    unsigned int kw = dx ? 0 : kw0;

                    unsigned int w_base = (((kd * 3 + kh) * 3 + kw) * 64 + oc) << 5;
                    unsigned int x_base = base_n + (iz << 10) + (iy << 5) + ix;

#pragma unroll
                    for (unsigned int ic = 0; ic < 32; ++ic) {
                        acc = fmaf(x[x_base + (ic << 14)], w[w_base + ic], acc);
                    }
                }
            }
        }

        float b = bias_term[oc];
        out[idx] = fmaf(acc, acc + acc, acc * b);
    }
}
