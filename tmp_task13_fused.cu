#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void reduce_depth_sum(
    const float* __restrict__ x,
    float* __restrict__ sum,
    int total
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total) {
        return;
    }
    int tmp = idx;
    int w = tmp & 127;
    tmp >>= 7;
    int h = tmp & 127;
    tmp >>= 7;
    int c = tmp & 15;
    int n = tmp >> 4;

    int base = (((n * 16 + c) * 32) * 128 + h) * 128 + w;
    float acc = 0.0f;
    #pragma unroll
    for (int d = 0; d < 32; ++d) {
        acc += x[base + d * 16384];
    }
    sum[idx] = acc;
}

extern "C" __global__ void fused_out(
    const float* __restrict__ x,
    const float* __restrict__ sum,
    const float* __restrict__ wsum,
    const float* __restrict__ wfront,
    const float* __restrict__ wback,
    const float* __restrict__ bias,
    float* __restrict__ out
) {
    int ow = (int)blockIdx.x;
    int oh = (int)blockIdx.y;
    int n = (int)blockIdx.z;
    int oc = (int)threadIdx.x;

    __shared__ float patch[432];
    __shared__ float logits[64];
    __shared__ float reduce_buf[64];

    for (int i = oc; i < 432; i += 64) {
        int src = i / 144;
        int rem = i - src * 144;
        int c = rem / 9;
        int rem2 = rem - c * 9;
        int ky = rem2 / 3;
        int kx = rem2 - ky * 3;
        int ih = oh + ky - 1;
        int iw = ow + kx - 1;
        float v = 0.0f;
        if ((unsigned)ih < 128u && (unsigned)iw < 128u) {
            if (src == 0) {
                int idx = (((n * 16 + c) * 128 + ih) * 128 + iw);
                v = sum[idx];
            } else if (src == 1) {
                int idx = ((((n * 16 + c) * 32) * 128 + ih) * 128 + iw);
                v = x[idx];
            } else {
                int idx = ((((n * 16 + c) * 32 + 31) * 128 + ih) * 128 + iw);
                v = x[idx];
            }
        }
        patch[i] = v;
    }
    __syncthreads();

    float acc = bias[oc];
    const float* wsum_oc = wsum + oc * 144;
    const float* wfront_oc = wfront + oc * 144;
    const float* wback_oc = wback + oc * 144;
    #pragma unroll
    for (int i = 0; i < 144; ++i) {
        acc += patch[i] * wsum_oc[i];
        acc -= patch[144 + i] * wfront_oc[i];
        acc -= patch[288 + i] * wback_oc[i];
    }
    acc *= 0.03125f;
    acc += bias[oc] * 0.96875f;
    logits[oc] = acc;
    reduce_buf[oc] = acc;
    __syncthreads();

    for (int stride = 32; stride > 0; stride >>= 1) {
        if (oc < stride) {
            float a = reduce_buf[oc];
            float b = reduce_buf[oc + stride];
            reduce_buf[oc] = a > b ? a : b;
        }
        __syncthreads();
    }
    float vmax = reduce_buf[0];
    float ev = expf(logits[oc] - vmax);
    reduce_buf[oc] = ev;
    __syncthreads();

    for (int stride = 32; stride > 0; stride >>= 1) {
        if (oc < stride) {
            reduce_buf[oc] += reduce_buf[oc + stride];
        }
        __syncthreads();
    }

    float prob = ev / reduce_buf[0];
    int out_idx = (((n * 64 + oc) * 128 + oh) * 128 + ow);
    out[out_idx] = tanhf(prob) * 2.0f;
}

extern "C" __global__ void fused_logits(
    const float* __restrict__ x,
    const float* __restrict__ sum,
    const float* __restrict__ wsum,
    const float* __restrict__ wfront,
    const float* __restrict__ wback,
    const float* __restrict__ bias,
    float* __restrict__ out
) {
    int ow = (int)blockIdx.x;
    int oh = (int)blockIdx.y;
    int n = (int)blockIdx.z;
    int oc = (int)threadIdx.x;

    __shared__ float patch[432];

    for (int i = oc; i < 432; i += 64) {
        int src = i / 144;
        int rem = i - src * 144;
        int c = rem / 9;
        int rem2 = rem - c * 9;
        int ky = rem2 / 3;
        int kx = rem2 - ky * 3;
        int ih = oh + ky - 1;
        int iw = ow + kx - 1;
        float v = 0.0f;
        if ((unsigned)ih < 128u && (unsigned)iw < 128u) {
            if (src == 0) {
                int idx = (((n * 16 + c) * 128 + ih) * 128 + iw);
                v = sum[idx];
            } else if (src == 1) {
                int idx = ((((n * 16 + c) * 32) * 128 + ih) * 128 + iw);
                v = x[idx];
            } else {
                int idx = ((((n * 16 + c) * 32 + 31) * 128 + ih) * 128 + iw);
                v = x[idx];
            }
        }
        patch[i] = v;
    }
    __syncthreads();

    float conv = 0.0f;
    const float* wsum_oc = wsum + oc * 144;
    const float* wfront_oc = wfront + oc * 144;
    const float* wback_oc = wback + oc * 144;
    #pragma unroll
    for (int i = 0; i < 144; ++i) {
        conv += patch[i] * wsum_oc[i];
        conv -= patch[144 + i] * wfront_oc[i];
        conv -= patch[288 + i] * wback_oc[i];
    }
    float acc = bias[oc] + conv * 0.03125f;
    int out_idx = (((n * 64 + oc) * 128 + oh) * 128 + ow);
    out[out_idx] = acc;
}

extern "C" __global__ void softmax_tanh_scale(
    const float* __restrict__ logits,
    float* __restrict__ out,
    int total_pixels
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_pixels) {
        return;
    }
    int spatial = idx & 16383;
    int n = idx >> 14;
    int oh = spatial >> 7;
    int ow = spatial & 127;

    int base = (((n * 64) * 128 + oh) * 128 + ow);
    float vmax = logits[base];
    #pragma unroll
    for (int oc = 1; oc < 64; ++oc) {
        float v = logits[base + oc * 16384];
        vmax = v > vmax ? v : vmax;
    }
    float esum = 0.0f;
    float probs[64];
    #pragma unroll
    for (int oc = 0; oc < 64; ++oc) {
        float p = expf(logits[base + oc * 16384] - vmax);
        probs[oc] = p;
        esum += p;
    }
    float inv = 1.0f / esum;
    #pragma unroll
    for (int oc = 0; oc < 64; ++oc) {
        out[base + oc * 16384] = tanhf(probs[oc] * inv) * 2.0f;
    }
}
