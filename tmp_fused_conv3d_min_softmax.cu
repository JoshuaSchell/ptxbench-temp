extern "C" __global__ void fused_conv3d_min_softmax_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int n
) {
    const int hw = blockIdx.x;
    const int batch = blockIdx.y;
    const int lane = threadIdx.x;

    const int oh = hw / 30;
    const int ow = hw - oh * 30;
    const int oc = lane;

    float v = 3.402823466e38f;
    if (oc < 24) {
        #pragma unroll
        for (int od = 0; od < 22; ++od) {
            float acc = b[oc];
            #pragma unroll
            for (int ic = 0; ic < 3; ++ic) {
                #pragma unroll
                for (int kz = 0; kz < 3; ++kz) {
                    #pragma unroll
                    for (int ky = 0; ky < 3; ++ky) {
                        #pragma unroll
                        for (int kx = 0; kx < 3; ++kx) {
                            const int in_off =
                                ((((batch * 3 + ic) * 24 + (od + kz)) * 32 + (oh + ky)) * 32 + (ow + kx));
                            const int w_off =
                                ((((oc * 3 + ic) * 3 + kz) * 3 + ky) * 3 + kx);
                            acc = fmaf(x[in_off], w[w_off], acc);
                        }
                    }
                }
            }
            v = v < acc ? v : acc;
        }
    } else {
        v = -3.402823466e38f;
    }

    unsigned mask = 0xffffffffu;
    float vmax = v;
    vmax = fmaxf(vmax, __shfl_down_sync(mask, vmax, 16));
    vmax = fmaxf(vmax, __shfl_down_sync(mask, vmax, 8));
    vmax = fmaxf(vmax, __shfl_down_sync(mask, vmax, 4));
    vmax = fmaxf(vmax, __shfl_down_sync(mask, vmax, 2));
    vmax = fmaxf(vmax, __shfl_down_sync(mask, vmax, 1));
    vmax = __shfl_sync(mask, vmax, 0);

    float e = oc < 24 ? expf(v - vmax) : 0.0f;
    float esum = e;
    esum += __shfl_down_sync(mask, esum, 16);
    esum += __shfl_down_sync(mask, esum, 8);
    esum += __shfl_down_sync(mask, esum, 4);
    esum += __shfl_down_sync(mask, esum, 2);
    esum += __shfl_down_sync(mask, esum, 1);
    esum = __shfl_sync(mask, esum, 0);

    if (oc < 24) {
        const int out_off = (((batch * 24 + oc) * 30 + oh) * 30 + ow);
        out[out_off] = e / esum;
    }
}
