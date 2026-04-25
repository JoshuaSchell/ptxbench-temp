extern "C" __global__ void clear_accum(float* accum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 128) accum[idx] = 0.0f;
}
