#include <cuda_runtime.h>
#include <math_constants.h>

extern "C" __global__ void convpool3d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ pooled,
    const int* __restrict__ meta,
    unsigned int n
) {
    if (meta[1] != 0) {
        return;
    }

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    unsigned int t = idx;
    unsigned int ow = t & 63u;
    t >>= 6;
    unsigned int oh = t & 63u;
    t >>= 6;
    unsigned int od = t & 15u;
    t >>= 4;
    unsigned int oc = t & 63u;
    unsigned int b = t >> 6;

    unsigned int z0 = od << 1;
    unsigned int y0 = oh << 1;
    unsigned int x0 = ow << 1;

    float best = -CUDART_INF_F;

    #pragma unroll
    for (int pd = 0; pd < 2; ++pd) {
        int oz = static_cast<int>(z0) + pd;
        #pragma unroll
        for (int ph = 0; ph < 2; ++ph) {
            int oy = static_cast<int>(y0) + ph;
            #pragma unroll
            for (int pw = 0; pw < 2; ++pw) {
                int ox = static_cast<int>(x0) + pw;
                float sum = bias[oc];
                for (int ic = 0; ic < 32; ++ic) {
                    const float* x_base = x + (((b * 32 + ic) * 32) << 14);
                    const float* w_base = w + ((oc * 32 + ic) * 27);
                    #pragma unroll
                    for (int kz = 0; kz < 3; ++kz) {
                        int iz = oz + kz - 1;
                        if (static_cast<unsigned int>(iz) >= 32u) {
                            continue;
                        }
                        #pragma unroll
                        for (int ky = 0; ky < 3; ++ky) {
                            int iy = oy + ky - 1;
                            if (static_cast<unsigned int>(iy) >= 128u) {
                                continue;
                            }
                            #pragma unroll
                            for (int kx = 0; kx < 3; ++kx) {
                                int ix = ox + kx - 1;
                                if (static_cast<unsigned int>(ix) >= 128u) {
                                    continue;
                                }
                                float xv = x_base[(iz << 14) + (iy << 7) + ix];
                                float wv = w_base[(kz * 9) + (ky * 3) + kx];
                                sum = fmaf(xv, wv, sum);
                            }
                        }
                    }
                }
                best = fmaxf(best, sum);
            }
        }
    }

    pooled[idx] = best;
}

extern "C" __global__ void lse_relu_kernel(
    const float* __restrict__ pooled,
    float* __restrict__ out,
    const int* __restrict__ meta,
    unsigned int n
) {
    if (meta[1] != 0) {
        return;
    }

    unsigned int out_idx = blockIdx.x;
    unsigned int tid = threadIdx.x;
    if (out_idx >= n || tid >= 64) {
        return;
    }

    unsigned int t = out_idx;
    unsigned int ow = t & 63u;
    t >>= 6;
    unsigned int oh = t & 63u;
    t >>= 6;
    unsigned int od = t & 15u;
    unsigned int b = t >> 4;

    __shared__ float buf[64];
    __shared__ double dbuf[64];
    unsigned int pooled_idx = (((b * 64 + tid) * 16 + od) * 64 + oh) * 64 + ow;
    float v = pooled[pooled_idx];
    buf[tid] = v;
    __syncthreads();

    if (tid < 32) buf[tid] = fmaxf(buf[tid], buf[tid + 32]);
    __syncthreads();
    if (tid < 16) buf[tid] = fmaxf(buf[tid], buf[tid + 16]);
    __syncthreads();
    if (tid < 8) buf[tid] = fmaxf(buf[tid], buf[tid + 8]);
    __syncthreads();
    if (tid < 4) buf[tid] = fmaxf(buf[tid], buf[tid + 4]);
    __syncthreads();
    if (tid < 2) buf[tid] = fmaxf(buf[tid], buf[tid + 2]);
    __syncthreads();
    if (tid < 1) buf[tid] = fmaxf(buf[tid], buf[tid + 1]);
    __syncthreads();

    float m = buf[0];
    dbuf[tid] = exp(static_cast<double>(v) - static_cast<double>(m));
    __syncthreads();

    if (tid < 32) dbuf[tid] += dbuf[tid + 32];
    __syncthreads();
    if (tid < 16) dbuf[tid] += dbuf[tid + 16];
    __syncthreads();
    if (tid < 8) dbuf[tid] += dbuf[tid + 8];
    __syncthreads();
    if (tid < 4) dbuf[tid] += dbuf[tid + 4];
    __syncthreads();
    if (tid < 2) dbuf[tid] += dbuf[tid + 2];
    __syncthreads();
    if (tid < 1) dbuf[tid] += dbuf[tid + 1];
    __syncthreads();
    if (tid < 1) {
        double y = log(dbuf[0]) + static_cast<double>(m);
        out[out_idx] = static_cast<float>(y > 0.0 ? y : 0.0);
    }
}
