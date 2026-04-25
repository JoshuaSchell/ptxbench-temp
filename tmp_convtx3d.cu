extern "C" __global__ void convtx3d_gather_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ out)
{
    const int ow = (int)blockIdx.x * 32 + (int)threadIdx.x;
    const int oc = (int)threadIdx.y;
    const int yh = (int)blockIdx.y;
    const int zg = (int)blockIdx.z;

    if (ow >= 96 || oc >= 8) {
        return;
    }

    const int od = yh / 48;
    const int oh = yh - od * 48;
    const int n = zg >> 2;
    const int g = zg & 3;

    float acc = 0.0f;

    int d_idx0 = 0;
    int d_idx1 = 0;
    int d_w0 = 0;
    int d_w1 = 0;
    int d_count = 0;
    if ((od & 1) == 0) {
        d_idx0 = od >> 1;
        d_w0 = 1;
        d_count = 1;
    } else {
        d_idx0 = (od + 1) >> 1;
        d_w0 = 0;
        d_idx1 = (od - 1) >> 1;
        d_w1 = 2;
        d_count = 2;
    }

    int h_idx0 = 0;
    int h_idx1 = 0;
    int h_idx2 = 0;
    int h_w0 = 0;
    int h_w1 = 0;
    int h_w2 = 0;
    int h_count = 0;
    if ((oh & 1) == 0) {
        h_idx0 = (oh + 2) >> 1;
        h_w0 = 0;
        h_idx1 = oh >> 1;
        h_w1 = 2;
        h_idx2 = (oh - 2) >> 1;
        h_w2 = 4;
        h_count = 3;
    } else {
        h_idx0 = (oh + 1) >> 1;
        h_w0 = 1;
        h_idx1 = (oh - 1) >> 1;
        h_w1 = 3;
        h_count = 2;
    }

    int w_idx0 = 0;
    int w_idx1 = 0;
    int w_idx2 = 0;
    int w_idx3 = 0;
    int w_w0 = 0;
    int w_w1 = 0;
    int w_w2 = 0;
    int w_w3 = 0;
    int w_count = 0;
    if ((ow & 1) == 0) {
        w_idx0 = (ow + 2) >> 1;
        w_w0 = 1;
        w_idx1 = ow >> 1;
        w_w1 = 3;
        w_idx2 = (ow - 2) >> 1;
        w_w2 = 5;
        w_count = 3;
    } else {
        w_idx0 = (ow + 3) >> 1;
        w_w0 = 0;
        w_idx1 = (ow + 1) >> 1;
        w_w1 = 2;
        w_idx2 = (ow - 1) >> 1;
        w_w2 = 4;
        w_idx3 = (ow - 3) >> 1;
        w_w3 = 6;
        w_count = 4;
    }

    const int x_batch_base = n * 442368 + g * 110592;
    const int w_group_base = g * 6720 + oc * 840;
    const int out_idx = (((n * 32 + g * 8 + oc) * 24 + od) * 48 + oh) * 96 + ow;

    #pragma unroll
    for (int ic = 0; ic < 8; ++ic) {
        const int x_chan_base = x_batch_base + ic * 13824;
        const int w_ic_base = w_group_base + ic * 105;

        #pragma unroll
        for (int di = 0; di < 2; ++di) {
            if (di >= d_count) {
                continue;
            }
            const int id = (di == 0) ? d_idx0 : d_idx1;
            const int kd = (di == 0) ? d_w0 : d_w1;
            if ((unsigned)id >= 12u) {
                continue;
            }
            const int x_d_base = x_chan_base + id * 1152;
            const int w_d_base = w_ic_base + kd * 35;

            #pragma unroll
            for (int hi = 0; hi < 3; ++hi) {
                if (hi >= h_count) {
                    continue;
                }
                const int ih = hi == 0 ? h_idx0 : (hi == 1 ? h_idx1 : h_idx2);
                const int kh = hi == 0 ? h_w0 : (hi == 1 ? h_w1 : h_w2);
                if ((unsigned)ih >= 24u) {
                    continue;
                }
                const int x_h_base = x_d_base + ih * 48;
                const int w_h_base = w_d_base + kh * 7;

                #pragma unroll
                for (int wi = 0; wi < 4; ++wi) {
                    if (wi >= w_count) {
                        continue;
                    }
                    const int iw = wi == 0 ? w_idx0 : (wi == 1 ? w_idx1 : (wi == 2 ? w_idx2 : w_idx3));
                    const int kw = wi == 0 ? w_w0 : (wi == 1 ? w_w1 : (wi == 2 ? w_w2 : w_w3));
                    if ((unsigned)iw >= 48u) {
                        continue;
                    }
                    acc += x[x_h_base + iw] * w[w_h_base + kw];
                }
            }
        }
    }

    out[out_idx] = acc;
}
