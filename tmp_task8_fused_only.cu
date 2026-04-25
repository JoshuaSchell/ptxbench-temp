#include <cuda_runtime.h>
extern "C" __global__ void fused_convpool_accum(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ conv_b,
    float* __restrict__ accum
) {
    const int tid = threadIdx.x;
    const int oc = tid >> 3;
    const int ic = tid & 7;
    const int bid = blockIdx.x;

    const int pw = bid % 31;
    int t = bid / 31;
    const int ph = t % 31;
    t /= 31;
    const int pd = t % 7;
    const int n = t / 7;

    __shared__ float sx[512];
    __shared__ float sw[3456];
    __shared__ float sb[16];
    __shared__ float sch[16];

    const int d0 = pd * 2;
    const int h0 = ph * 2;
    const int w0 = pw * 2;

    for (int idx = tid; idx < 512; idx += 128) {
        const int c = idx >> 6;
        const int rem = idx & 63;
        const int dz = rem >> 4;
        const int rem2 = rem & 15;
        const int dy = rem2 >> 2;
        const int dx = rem2 & 3;
        const int x_idx = (((((n * 8 + c) * 16) + (d0 + dz)) * 64 + (h0 + dy)) * 64 + (w0 + dx));
        sx[idx] = x[x_idx];
    }
    for (int idx = tid; idx < 3456; idx += 128) {
        sw[idx] = w[idx];
    }
    if (tid < 16) {
        sb[tid] = conv_b[tid];
    }
    __syncthreads();

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;

    const int x_base = ic << 6;
    const int w_base = ((oc << 3) + ic) * 27;

    #pragma unroll
    for (int kd = 0; kd < 3; ++kd) {
        #pragma unroll
        for (int kh = 0; kh < 3; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < 3; ++kw) {
                const float wv = sw[w_base + ((kd * 3 + kh) * 3 + kw)];
                const int off000 = x_base + ((kd * 4 + kh) * 4 + kw);
                const int off001 = off000 + 1;
                const int off010 = off000 + 4;
                const int off011 = off010 + 1;
                const int off100 = off000 + 16;
                const int off101 = off100 + 1;
                const int off110 = off100 + 4;
                const int off111 = off110 + 1;
                s0 = fmaf(sx[off000], wv, s0);
                s1 = fmaf(sx[off001], wv, s1);
                s2 = fmaf(sx[off010], wv, s2);
                s3 = fmaf(sx[off011], wv, s3);
                s4 = fmaf(sx[off100], wv, s4);
                s5 = fmaf(sx[off101], wv, s5);
                s6 = fmaf(sx[off110], wv, s6);
                s7 = fmaf(sx[off111], wv, s7);
            }
        }
    }

    #pragma unroll
    for (int delta = 4; delta > 0; delta >>= 1) {
        s0 += __shfl_down_sync(0xffffffffu, s0, delta, 8);
        s1 += __shfl_down_sync(0xffffffffu, s1, delta, 8);
        s2 += __shfl_down_sync(0xffffffffu, s2, delta, 8);
        s3 += __shfl_down_sync(0xffffffffu, s3, delta, 8);
        s4 += __shfl_down_sync(0xffffffffu, s4, delta, 8);
        s5 += __shfl_down_sync(0xffffffffu, s5, delta, 8);
        s6 += __shfl_down_sync(0xffffffffu, s6, delta, 8);
        s7 += __shfl_down_sync(0xffffffffu, s7, delta, 8);
    }

    if (ic == 0) {
        const float bias = sb[oc];
        float vmax = s0 + bias;
        vmax = fmaxf(vmax, s1 + bias);
        vmax = fmaxf(vmax, s2 + bias);
        vmax = fmaxf(vmax, s3 + bias);
        vmax = fmaxf(vmax, s4 + bias);
        vmax = fmaxf(vmax, s5 + bias);
        vmax = fmaxf(vmax, s6 + bias);
        vmax = fmaxf(vmax, s7 + bias);
        sch[oc] = vmax;
    }
    __syncthreads();

    if (tid == 0) {
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            sum += sch[i];
        }
        atomicAdd(accum + n, sum);
    }
}
