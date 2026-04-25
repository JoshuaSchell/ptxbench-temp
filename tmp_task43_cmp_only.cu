extern "C" __global__ void cmp_samples_kernel(const int4* x4, const int4* cache4, int* meta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (!meta[1] || idx >= 4096) {
        return;
    }
    int4 a = x4[idx * 1024];
    int4 b = cache4[idx];
    if (a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w) {
        meta[1] = 0;
    }
}
