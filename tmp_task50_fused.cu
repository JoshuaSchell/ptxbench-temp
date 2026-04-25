extern "C" __global__ __launch_bounds__(128, 4) void fused_conv_pool_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    unsigned int sites,
    unsigned int in_d,
    unsigned int in_h,
    unsigned int in_w
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int site = blockIdx.x * blockDim.x + tid;
    const unsigned int oc = blockIdx.y;
    const unsigned int n = blockIdx.z;

    __shared__ float sh_w[24];
    __shared__ float sh_b[1];

    if (tid < 24) {
        sh_w[tid] = w[oc * 24 + tid];
    }
    if (tid == 24) {
        sh_b[0] = b[oc];
    }
    __syncthreads();

    if (site >= sites) {
        return;
    }

    const unsigned int out_w = in_w - 1;
    const unsigned int out_h = in_h - 1;
    const unsigned int tmp = site / out_w;
    const unsigned int ow = site - tmp * out_w;
    const unsigned int od = tmp / out_h;
    const unsigned int oh = tmp - od * out_h;

    const unsigned int in_hw = in_h * in_w;
    const unsigned int in_chw = in_d * in_hw;
    const unsigned long long batch_base = (unsigned long long)n * 3ull * (unsigned long long)in_chw;
    const unsigned long long base = batch_base + (unsigned long long)od * in_hw + (unsigned long long)oh * in_w + ow;

    float acc = sh_b[0];

    #pragma unroll
    for (unsigned int ic = 0; ic < 3; ++ic) {
        const float* xp = x + base + (unsigned long long)ic * in_chw;
        const float* wp = sh_w + ic * 8;

        const float x000 = xp[0];
        const float x001 = xp[1];
        const float x010 = xp[in_w];
        const float x011 = xp[in_w + 1];
        const float x100 = xp[in_hw];
        const float x101 = xp[in_hw + 1];
        const float x110 = xp[in_hw + in_w];
        const float x111 = xp[in_hw + in_w + 1];

        acc = fmaf(x000, wp[0], acc);
        acc = fmaf(x001, wp[1], acc);
        acc = fmaf(x010, wp[2], acc);
        acc = fmaf(x011, wp[3], acc);
        acc = fmaf(x100, wp[4], acc);
        acc = fmaf(x101, wp[5], acc);
        acc = fmaf(x110, wp[6], acc);
        acc = fmaf(x111, wp[7], acc);
    }

    const unsigned long long out_index = (((unsigned long long)n * 16ull) + oc) * (unsigned long long)sites + site;
    out[out_index] = acc;
}
