#include <cuda_runtime.h>
#include <math.h>

static __device__ __forceinline__ float mish_tanh(float x) {
    float sp = fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
    float m = x * tanhf(sp);
    return tanhf(m);
}

extern "C" __global__ void fused_mish_tanh_inplace(float* x, unsigned int n4) {
    unsigned int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx4 >= n4) return;
    float4 v = reinterpret_cast<float4*>(x)[idx4];
    v.x = mish_tanh(v.x);
    v.y = mish_tanh(v.y);
    v.z = mish_tanh(v.z);
    v.w = mish_tanh(v.w);
    reinterpret_cast<float4*>(x)[idx4] = v;
}
