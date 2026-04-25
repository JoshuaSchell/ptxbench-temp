extern "C" __global__ void fused_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out
) {
    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;
    const int ow2 = block_id % 7;
    const int t0 = block_id / 7;
    const int oh2 = t0 % 7;
    const int t1 = t0 / 7;
    const int od2 = t1 % 3;
    const int n = t1 / 3;

    const int d0 = od2 * 4;
    const int h0 = oh2 * 4;
    const int w0 = ow2 * 4;

    __shared__ float sx[648];
    __shared__ float sw[1296];
    __shared__ float sb[16];
    __shared__ float sprobs[1024];

    for (int i = tid; i < 648; i += 64) {
        const int c = i / 216;
        const int rem = i - c * 216;
        const int d = rem / 36;
        const int rem2 = rem - d * 36;
        const int h = rem2 / 6;
        const int ww = rem2 - h * 6;
        const int x_idx = (((n * 3 + c) * 16 + (d0 + d)) * 32 + (h0 + h)) * 32 + (w0 + ww);
        sx[i] = x[x_idx];
    }
    for (int i = tid; i < 1296; i += 64) {
        sw[i] = w[i];
    }
    for (int i = tid; i < 16; i += 64) {
        sb[i] = b[i];
    }
    __syncthreads();

    if (tid < 64) {
        const int pd = tid >> 4;
        const int rem = tid & 15;
        const int ph = rem >> 2;
        const int pw = rem & 3;
        float acc[16];
        #pragma unroll
        for (int oc = 0; oc < 16; ++oc) {
            acc[oc] = sb[oc];
        }
        #pragma unroll
        for (int ic = 0; ic < 3; ++ic) {
            #pragma unroll
            for (int kd = 0; kd < 3; ++kd) {
                #pragma unroll
                for (int kh = 0; kh < 3; ++kh) {
                    #pragma unroll
                    for (int kw = 0; kw < 3; ++kw) {
                        const int xv = ((ic * 6 + (pd + kd)) * 6 + (ph + kh)) * 6 + (pw + kw);
                        const float xval = sx[xv];
                        const int wbase = (((ic * 3 + kd) * 3 + kh) * 3 + kw);
                        #pragma unroll
                        for (int oc = 0; oc < 16; ++oc) {
                            acc[oc] = fmaf(xval, sw[oc * 81 + wbase], acc[oc]);
                        }
                    }
                }
            }
        }

        float vmax = acc[0];
        #pragma unroll
        for (int oc = 1; oc < 16; ++oc) {
            vmax = vmax > acc[oc] ? vmax : acc[oc];
        }
        float esum = 0.0f;
        float probs[16];
        #pragma unroll
        for (int oc = 0; oc < 16; ++oc) {
            const float p = expf(acc[oc] - vmax);
            probs[oc] = p;
            esum += p;
        }
        const float inv = 1.0f / esum;
        #pragma unroll
        for (int oc = 0; oc < 16; ++oc) {
            sprobs[oc * 64 + tid] = probs[oc] * inv;
        }
    }
    __syncthreads();

    if (tid < 16) {
        float vmax = sprobs[tid * 64];
        #pragma unroll
        for (int i = 1; i < 64; ++i) {
            const float v = sprobs[tid * 64 + i];
            vmax = vmax > v ? vmax : v;
        }
        const int out_idx = ((((n * 16 + tid) * 3 + od2) * 7 + oh2) * 7 + ow2);
        out[out_idx] = vmax;
    }
}
