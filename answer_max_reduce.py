import hashlib

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <float.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")

constexpr int DIM1_THREADS = 128;
constexpr int REDUCE_THREADS = 256;

__inline__ __device__ float warp_reduce_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    }
    return v;
}

__global__ __launch_bounds__(DIM1_THREADS, 2) void max_dim1_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int batch,
    int d1,
    int d2) {
    const int col = blockIdx.x * DIM1_THREADS + threadIdx.x;
    const int b = blockIdx.y;
    if (b >= batch || col >= d2) {
        return;
    }

    const long long base = (static_cast<long long>(b) * d1 * d2) + col;
    float vmax = -FLT_MAX;
    #pragma unroll 1
    for (int i = 0; i < d1; ++i) {
        vmax = fmaxf(vmax, __ldg(x + base + static_cast<long long>(i) * d2));
    }
    out[static_cast<long long>(b) * d2 + col] = vmax;
}

__global__ __launch_bounds__(REDUCE_THREADS, 2) void max_dim2_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int batch,
    int d1,
    int d2) {
    const int row = blockIdx.x;
    const int b = blockIdx.y;
    if (b >= batch || row >= d1) {
        return;
    }

    const int tid = threadIdx.x;
    const long long row_base = (static_cast<long long>(b) * d1 + row) * d2;
    float vmax = -FLT_MAX;
    for (int col = tid; col < d2; col += REDUCE_THREADS) {
        vmax = fmaxf(vmax, __ldg(x + row_base + col));
    }

    vmax = warp_reduce_max(vmax);

    __shared__ float warp_max[REDUCE_THREADS / 32];
    if ((tid & 31) == 0) {
        warp_max[tid >> 5] = vmax;
    }
    __syncthreads();

    if (tid < 32) {
        float block_max = tid < (REDUCE_THREADS / 32) ? warp_max[tid] : -FLT_MAX;
        block_max = warp_reduce_max(block_max);
        if (tid == 0) {
            out[static_cast<long long>(b) * d1 + row] = block_max;
        }
    }
}

__global__ __launch_bounds__(DIM1_THREADS, 2) void max_dim0_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int batch,
    int d1,
    int d2) {
    const int col = blockIdx.x * DIM1_THREADS + threadIdx.x;
    const int row = blockIdx.y;
    if (row >= d1 || col >= d2) {
        return;
    }

    const long long offset = static_cast<long long>(row) * d2 + col;
    const long long plane = static_cast<long long>(d1) * d2;
    float vmax = -FLT_MAX;
    for (int b = 0; b < batch; ++b) {
        vmax = fmaxf(vmax, __ldg(x + static_cast<long long>(b) * plane + offset));
    }
    out[offset] = vmax;
}

torch::Tensor max_reduce_cuda(torch::Tensor x, int64_t dim) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    CHECK_FLOAT(x);
    TORCH_CHECK(x.dim() == 3, "x must be 3D");

    dim = dim < 0 ? dim + 3 : dim;
    TORCH_CHECK(dim >= 0 && dim < 3, "dim must be in [-3, 2]");

    const int batch = static_cast<int>(x.size(0));
    const int d1 = static_cast<int>(x.size(1));
    const int d2 = static_cast<int>(x.size(2));

    c10::cuda::CUDAGuard device_guard(x.device());
    const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());

    if (dim == 1) {
        auto out = torch::empty({batch, d2}, x.options());
        const dim3 block(DIM1_THREADS);
        const dim3 grid((d2 + DIM1_THREADS - 1) / DIM1_THREADS, batch);
        max_dim1_kernel<<<grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            batch,
            d1,
            d2
        );
        return out;
    }

    if (dim == 2) {
        auto out = torch::empty({batch, d1}, x.options());
        const dim3 block(REDUCE_THREADS);
        const dim3 grid(d1, batch);
        max_dim2_kernel<<<grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            batch,
            d1,
            d2
        );
        return out;
    }

    auto out = torch::empty({d1, d2}, x.options());
    const dim3 block(DIM1_THREADS);
    const dim3 grid((d2 + DIM1_THREADS - 1) / DIM1_THREADS, d1);
    max_dim0_kernel<<<grid, block, 0, stream>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        batch,
        d1,
        d2
    );
    return out;
}
"""

_SIG = hashlib.md5(CUDA_SRC.encode("utf-8")).hexdigest()[:16]
_EXT = load_inline(
    name=f"ptxbench_max_reduce_{_SIG}",
    cpp_sources="torch::Tensor max_reduce_cuda(torch::Tensor x, int64_t dim);",
    cuda_sources=CUDA_SRC,
    functions=["max_reduce_cuda"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _EXT.max_reduce_cuda(x, self.dim)
