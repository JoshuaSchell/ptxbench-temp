extern "C" __global__ void copy4_hit_kernel(const int4* src, int4* dst, const int* meta, unsigned int n) {
    if (!meta[1]) {
        return;
    }
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}
