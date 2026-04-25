extern "C" __global__ void prepare_meta(int* meta) {
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
        meta[1] = meta[0];
    }
}
