import hashlib
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")

__device__ __forceinline__ double warp_sum(double v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

template <int BLOCK>
__device__ __forceinline__ void block_reduce_pair(double& a, double& b) {
    __shared__ double shared_a[BLOCK / 32];
    __shared__ double shared_b[BLOCK / 32];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    a = warp_sum(a);
    b = warp_sum(b);
    if (lane == 0) {
        shared_a[wid] = a;
        shared_b[wid] = b;
    }
    __syncthreads();
    a = (threadIdx.x < BLOCK / 32) ? shared_a[lane] : 0.0;
    b = (threadIdx.x < BLOCK / 32) ? shared_b[lane] : 0.0;
    if (wid == 0) {
        a = warp_sum(a);
        b = warp_sum(b);
    }
}

template <int BLOCK>
__global__ void stats_kernel(
    const float* __restrict__ x,
    double* __restrict__ partial_sum,
    double* __restrict__ partial_sq,
    int rows,
    int n,
    int blocks_per_row
) {
    const int global_block = blockIdx.x;
    const int row = global_block / blocks_per_row;
    const int block_in_row = global_block - row * blocks_per_row;
    if (row >= rows) {
        return;
    }

    const int tid = threadIdx.x;
    const float* row_ptr = x + static_cast<long long>(row) * n;
    const int n4 = n >> 2;
    const float4* row4 = reinterpret_cast<const float4*>(row_ptr);
    double sum = 0.0;
    double sq = 0.0;

    for (int idx4 = block_in_row * BLOCK + tid; idx4 < n4; idx4 += blocks_per_row * BLOCK) {
        const float4 v = row4[idx4];
        const double x0 = static_cast<double>(v.x);
        const double x1 = static_cast<double>(v.y);
        const double x2 = static_cast<double>(v.z);
        const double x3 = static_cast<double>(v.w);
        sum += x0 + x1 + x2 + x3;
        sq += x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
    }

    const int tail_start = n4 << 2;
    for (int i = tail_start + block_in_row * BLOCK + tid; i < n; i += blocks_per_row * BLOCK) {
        const double v = static_cast<double>(row_ptr[i]);
        sum += v;
        sq += v * v;
    }

    block_reduce_pair<BLOCK>(sum, sq);
    if (tid == 0) {
        partial_sum[global_block] = sum;
        partial_sq[global_block] = sq;
    }
}

template <int BLOCK>
__global__ void finalize_kernel(
    const double* __restrict__ partial_sum,
    const double* __restrict__ partial_sq,
    float* __restrict__ mean,
    float* __restrict__ invstd,
    int rows,
    int blocks_per_row,
    int n,
    float eps
) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    double sum = 0.0;
    double sq = 0.0;
    for (int i = threadIdx.x; i < blocks_per_row; i += BLOCK) {
        const int idx = row * blocks_per_row + i;
        sum += partial_sum[idx];
        sq += partial_sq[idx];
    }

    block_reduce_pair<BLOCK>(sum, sq);
    if (threadIdx.x == 0) {
        const double m = sum / static_cast<double>(n);
        double var = sq / static_cast<double>(n) - m * m;
        if (var < 0.0) {
            var = 0.0;
        }
        mean[row] = static_cast<float>(m);
        invstd[row] = static_cast<float>(1.0 / sqrt(var + static_cast<double>(eps)));
    }
}

template <int BLOCK>
__global__ void apply_kernel(
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ invstd,
    float* __restrict__ y,
    int rows,
    int n,
    int blocks_per_row
) {
    const int global_block = blockIdx.x;
    const int row = global_block / blocks_per_row;
    const int block_in_row = global_block - row * blocks_per_row;
    if (row >= rows) {
        return;
    }

    const int tid = threadIdx.x;
    const float m = mean[row];
    const float is = invstd[row];
    const float* row_ptr = x + static_cast<long long>(row) * n;
    float* out_ptr = y + static_cast<long long>(row) * n;
    const int n4 = n >> 2;
    const float4* row4 = reinterpret_cast<const float4*>(row_ptr);
    float4* out4 = reinterpret_cast<float4*>(out_ptr);

    for (int idx4 = block_in_row * BLOCK + tid; idx4 < n4; idx4 += blocks_per_row * BLOCK) {
        const float4 v = row4[idx4];
        float4 o;
        o.x = (v.x - m) * is;
        o.y = (v.y - m) * is;
        o.z = (v.z - m) * is;
        o.w = (v.w - m) * is;
        out4[idx4] = o;
    }

    const int tail_start = n4 << 2;
    for (int i = tail_start + block_in_row * BLOCK + tid; i < n; i += blocks_per_row * BLOCK) {
        out_ptr[i] = (row_ptr[i] - m) * is;
    }
}

torch::Tensor layer_norm_cuda(torch::Tensor x, int64_t n, double eps) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    CHECK_FLOAT(x);
    TORCH_CHECK(n > 0, "normalized element count must be positive");
    TORCH_CHECK(n <= 2147483647LL, "normalized element count is too large");
    TORCH_CHECK(x.numel() % n == 0, "input shape is incompatible with normalized_shape");

    const int rows = static_cast<int>(x.numel() / n);
    const int n_int = static_cast<int>(n);
    auto out = torch::empty_like(x);
    auto partial_opts = x.options().dtype(torch::kFloat64);
    auto stats_opts = x.options().dtype(torch::kFloat32);

    const int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int blocks_by_size = std::max(1, std::min(128, (n_int + 32767) / 32768));
    const int blocks_by_sm = std::max(1, std::min(128, (sm_count * 8 + rows - 1) / rows));
    const int blocks_per_row = std::max(blocks_by_size, blocks_by_sm);
    const int total_blocks = rows * blocks_per_row;

    auto partial_sum = torch::empty({total_blocks}, partial_opts);
    auto partial_sq = torch::empty({total_blocks}, partial_opts);
    auto mean = torch::empty({rows}, stats_opts);
    auto invstd = torch::empty({rows}, stats_opts);

    constexpr int BLOCK = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    stats_kernel<BLOCK><<<total_blocks, BLOCK, 0, stream>>>(
        x.data_ptr<float>(),
        partial_sum.data_ptr<double>(),
        partial_sq.data_ptr<double>(),
        rows,
        n_int,
        blocks_per_row
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    finalize_kernel<BLOCK><<<rows, BLOCK, 0, stream>>>(
        partial_sum.data_ptr<double>(),
        partial_sq.data_ptr<double>(),
        mean.data_ptr<float>(),
        invstd.data_ptr<float>(),
        rows,
        blocks_per_row,
        n_int,
        static_cast<float>(eps)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    apply_kernel<BLOCK><<<total_blocks, BLOCK, 0, stream>>>(
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        invstd.data_ptr<float>(),
        out.data_ptr<float>(),
        rows,
        n_int,
        blocks_per_row
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""


_EXT = load_inline(
    name=f"ptxbench_layernorm_{hashlib.md5(CUDA_SRC.encode()).hexdigest()[:16]}",
    cpp_sources="torch::Tensor layer_norm_cuda(torch::Tensor x, int64_t n, double eps);",
    cuda_sources=CUDA_SRC,
    functions=["layer_norm_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "-gencode=arch=compute_89,code=sm_89"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(int(v) for v in normalized_shape)
        n = 1
        for v in self.normalized_shape:
            n *= v
        self.n = n
        self.eps = 1e-5

    def forward(self, x):
        return _EXT.layer_norm_cuda(x, self.n, self.eps)
