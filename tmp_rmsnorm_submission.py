import os

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

namespace {

constexpr int kChannels = 64;

__global__ void rmsnorm64_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const int64_t outer,
    const int64_t inner,
    const float eps
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = outer * inner;
    if (idx >= total) {
        return;
    }

    const int64_t n = idx / inner;
    const int64_t s = idx - n * inner;
    const int64_t base = n * static_cast<int64_t>(kChannels) * inner + s;

    float vals[kChannels];
    float sum = 0.0f;

    #pragma unroll
    for (int c = 0; c < kChannels; ++c) {
        const float v = x[base + static_cast<int64_t>(c) * inner];
        vals[c] = v;
        sum = fmaf(v, v, sum);
    }

    const float inv_rms = rsqrtf(sum * (1.0f / 64.0f) + eps);

    #pragma unroll
    for (int c = 0; c < kChannels; ++c) {
        out[base + static_cast<int64_t>(c) * inner] = vals[c] * inv_rms;
    }
}

__global__ void rmsnorm_generic_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const int64_t outer,
    const int64_t channels,
    const int64_t inner,
    const float eps
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = outer * inner;
    if (idx >= total) {
        return;
    }

    const int64_t n = idx / inner;
    const int64_t s = idx - n * inner;
    const int64_t base = n * channels * inner + s;

    float sum = 0.0f;
    for (int64_t c = 0; c < channels; ++c) {
        const float v = x[base + c * inner];
        sum = fmaf(v, v, sum);
    }

    const float inv_rms = rsqrtf(sum / static_cast<float>(channels) + eps);
    for (int64_t c = 0; c < channels; ++c) {
        out[base + c * inner] = x[base + c * inner] * inv_rms;
    }
}

}  // namespace

torch::Tensor rmsnorm_cuda(torch::Tensor x, double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dimensions");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    auto out = torch::empty_like(x);

    const int64_t outer = x.size(0);
    const int64_t channels = x.size(1);
    int64_t inner = 1;
    for (int64_t d = 2; d < x.dim(); ++d) {
        inner *= x.size(d);
    }

    const int threads = 256;
    const int64_t total = outer * inner;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();

    if (channels == kChannels) {
        rmsnorm64_kernel<<<blocks, threads, 0, stream>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            outer,
            inner,
            static_cast<float>(eps)
        );
    } else {
        rmsnorm_generic_kernel<<<blocks, threads, 0, stream>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            outer,
            channels,
            inner,
            static_cast<float>(eps)
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""


_EXT = load_inline(
    name=f"ptxbench_rmsnorm64_{os.getpid()}",
    cpp_sources="torch::Tensor rmsnorm_cuda(torch::Tensor x, double eps);",
    cuda_sources=CUDA_SRC,
    functions=["rmsnorm_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _EXT.rmsnorm_cuda(x, self.eps)
