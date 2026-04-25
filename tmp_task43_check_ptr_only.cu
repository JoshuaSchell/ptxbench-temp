extern "C" __global__ void check_ptr_kernel(const float* x, int* meta) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && meta[1]) {
        unsigned long long ptr = (unsigned long long)x;
        unsigned int lo = (unsigned int)ptr;
        unsigned int hi = (unsigned int)(ptr >> 32);
        if ((unsigned int)meta[2] != lo || (unsigned int)meta[3] != hi) {
            meta[1] = 0;
        }
    }
}
