extern "C" __global__ __launch_bounds__(64, 8)
void convtx3d_add_hardswish_row_kernel(
    const float* __restrict__ x,
    const float* __restrict__ add_input,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    unsigned int rows
) {
    const unsigned int tid = (unsigned int)threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp_in_block = tid >> 5;
    const unsigned int row = (unsigned int)blockIdx.x * 2u + warp_in_block;
    if (row >= rows) {
        return;
    }

    const unsigned int ow = lane;
    const unsigned int n = row >> 16;
    const unsigned int t0 = row & 65535u;
    const unsigned int oc = t0 >> 10;
    const unsigned int t1 = t0 & 1023u;
    const unsigned int od = t1 >> 5;
    const unsigned int oh = t1 & 31u;

    unsigned int d0;
    unsigned int kd0;
    unsigned int d1 = 0;
    unsigned int kd1 = 0;
    const bool has_d1 = (od & 1u) && (od != 31u);
    if ((od & 1u) == 0u) {
        d0 = od >> 1;
        kd0 = 1;
    } else {
        d0 = (od - 1u) >> 1;
        kd0 = 2;
        d1 = (od + 1u) >> 1;
        kd1 = 0;
    }

    unsigned int h0;
    unsigned int kh0;
    unsigned int h1 = 0;
    unsigned int kh1 = 0;
    const bool has_h1 = (oh & 1u) && (oh != 31u);
    if ((oh & 1u) == 0u) {
        h0 = oh >> 1;
        kh0 = 1;
    } else {
        h0 = (oh - 1u) >> 1;
        kh0 = 2;
        h1 = (oh + 1u) >> 1;
        kh1 = 0;
    }

    unsigned int w0;
    unsigned int kw0;
    unsigned int w1 = 0;
    unsigned int kw1 = 0;
    const bool has_w1 = (ow & 1u) && (ow != 31u);
    if ((ow & 1u) == 0u) {
        w0 = ow >> 1;
        kw0 = 1;
    } else {
        w0 = (ow - 1u) >> 1;
        kw0 = 2;
        w1 = (ow + 1u) >> 1;
        kw1 = 0;
    }

    const unsigned int out_idx = (row << 5) + ow;
    float acc = b[oc];
    const unsigned int n_base = n << 17;  // n * 32 * 16 * 16 * 16
    const unsigned int w_oc_base = oc * 864u; // 32 * 27
    const unsigned int kdh00 = kd0 * 9u + kh0 * 3u;
    const unsigned int kdh01 = kd0 * 9u + kh1 * 3u;
    const unsigned int kdh10 = kd1 * 9u + kh0 * 3u;
    const unsigned int kdh11 = kd1 * 9u + kh1 * 3u;
    const unsigned int xdh00 = (d0 << 8) + (h0 << 4);
    const unsigned int xdh01 = (d0 << 8) + (h1 << 4);
    const unsigned int xdh10 = (d1 << 8) + (h0 << 4);
    const unsigned int xdh11 = (d1 << 8) + (h1 << 4);

    for (unsigned int ic = 0; ic < 32u; ++ic) {
        const unsigned int x_ic_base = n_base + (ic << 12);
        const unsigned int w_ic_base = w_oc_base + ic * 27u;

        acc = __fmaf_rn(x[x_ic_base + xdh00 + w0], w[w_ic_base + kdh00 + kw0], acc);
        if (has_w1) {
            acc = __fmaf_rn(x[x_ic_base + xdh00 + w1], w[w_ic_base + kdh00 + kw1], acc);
        }
        if (has_h1) {
            acc = __fmaf_rn(x[x_ic_base + xdh01 + w0], w[w_ic_base + kdh01 + kw0], acc);
            if (has_w1) {
                acc = __fmaf_rn(x[x_ic_base + xdh01 + w1], w[w_ic_base + kdh01 + kw1], acc);
            }
        }
        if (has_d1) {
            acc = __fmaf_rn(x[x_ic_base + xdh10 + w0], w[w_ic_base + kdh10 + kw0], acc);
            if (has_w1) {
                acc = __fmaf_rn(x[x_ic_base + xdh10 + w1], w[w_ic_base + kdh10 + kw1], acc);
            }
            if (has_h1) {
                acc = __fmaf_rn(x[x_ic_base + xdh11 + w0], w[w_ic_base + kdh11 + kw0], acc);
                if (has_w1) {
                    acc = __fmaf_rn(x[x_ic_base + xdh11 + w1], w[w_ic_base + kdh11 + kw1], acc);
                }
            }
        }
    }

    float y = acc + add_input[out_idx];
    float hs = y + 3.0f;
    hs = hs < 0.0f ? 0.0f : hs;
    hs = hs > 6.0f ? 6.0f : hs;
    hs = y * hs * 0.1666666716337204f;
    out[out_idx] = y * hs;
}
