extern "C" __global__ void copy4_miss_kernel(const int4* src, int4* dst, const int* meta, int n4) {
    if (meta[1] != 0) {
        return;
    }
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < n4) {
        dst[idx] = src[idx];
    }
}
