extern "C" __global__ void fused_linear_lrelu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w_t,
    const float* __restrict__ bias,
    float* __restrict__ out
) {
    constexpr int M = 1024;
    constexpr int N = 8192;
    constexpr int K = 8192;
    constexpr int BM = 16;
    constexpr int BN = 16;
    constexpr int BK = 16;
    constexpr float NEGATIVE_SLOPE = 0.1f;

    __shared__ float a_tile[BM][BK];
    __shared__ float b_tile[BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * BM + ty;
    const int col = blockIdx.x * BN + tx;
    const int linear_tid = ty * BN + tx;

    float acc = 0.0f;
    if (col < N) {
        acc = bias[col];
    }

    #pragma unroll 1
    for (int k0 = 0; k0 < K; k0 += BK) {
        const int a_row = linear_tid / BK;
        const int a_col = linear_tid - a_row * BK;
        a_tile[a_row][a_col] = x[(blockIdx.y * BM + a_row) * K + (k0 + a_col)];

        const int b_row = linear_tid / BN;
        const int b_col = linear_tid - b_row * BN;
        b_tile[b_row][b_col] = w_t[(k0 + b_row) * N + (blockIdx.x * BN + b_col)];

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            acc = fmaf(a_tile[ty][kk], b_tile[kk][tx], acc);
        }

        __syncthreads();
    }

    float y = acc;
    if (y < 0.0f) {
        y *= NEGATIVE_SLOPE;
    }
    out[row * N + col] = y;
}
