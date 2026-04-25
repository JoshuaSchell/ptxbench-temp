extern "C" __global__ void cmp4_kernel(const int4* x, const int4* cache_x, int* meta, int n4) {
    if (meta[0] == 0) {
        return;
    }
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n4) {
        return;
    }
    int4 a = x[idx];
    int4 b = cache_x[idx];
    if (a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w) {
        meta[1] = 0;
    }
}
