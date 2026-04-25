import hashlib
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


_CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float softplus_scalar(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

__global__ void softplus_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    long long n
) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    for (long long i = idx; i < n; i += stride) {
        y[i] = softplus_scalar(x[i]);
    }
}

__global__ void softplus4_kernel(
    const float4* __restrict__ x,
    float4* __restrict__ y,
    long long n4
) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    for (long long i = idx; i < n4; i += stride) {
        float4 xv = x[i];
        float4 out;
        out.x = softplus_scalar(xv.x);
        out.y = softplus_scalar(xv.y);
        out.z = softplus_scalar(xv.z);
        out.w = softplus_scalar(xv.w);
        y[i] = out;
    }
}

}  // namespace

torch::Tensor softplus_out_cuda(torch::Tensor x, torch::Tensor y) {
    const auto n = x.numel();
    constexpr int block = 256;
    const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());

    const auto x_ptr = reinterpret_cast<uintptr_t>(x.data_ptr<float>());
    const auto y_ptr = reinterpret_cast<uintptr_t>(y.data_ptr<float>());
    if ((n & 3LL) == 0 && (x_ptr & 15ULL) == 0 && (y_ptr & 15ULL) == 0) {
        const long long n4 = n >> 2;
        const long long grid_ll = (n4 + block - 1) / block;
        const int grid = static_cast<int>(grid_ll < 32768LL ? grid_ll : 32768LL);
        softplus4_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(y.data_ptr<float>()),
            n4
        );
    } else {
        const long long grid_ll = (n + block - 1) / block;
        const int grid = static_cast<int>(grid_ll < 32768LL ? grid_ll : 32768LL);
        softplus_kernel<<<grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            n
        );
    }
    return y;
}

torch::Tensor softplus_cuda(torch::Tensor x) {
    auto y = torch::empty_like(x);
    return softplus_out_cuda(x, y);
}
"""


_CPP_SRC = """
torch::Tensor softplus_cuda(torch::Tensor x);
torch::Tensor softplus_out_cuda(torch::Tensor x, torch::Tensor y);
"""


_EXT = load_inline(
    name=f"ptxbench_softplus_{hashlib.sha1(_CUDA_SRC.encode()).hexdigest()[:16]}",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["softplus_cuda", "softplus_out_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._out = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("ModelNew requires a CUDA tensor")
        if x.dtype != torch.float32:
            raise RuntimeError("ModelNew expects float32 input")
        if not x.is_contiguous():
            x = x.contiguous()
        if (
            self._out is None
            or self._out.shape != x.shape
            or self._out.device != x.device
            or self._out.dtype != x.dtype
        ):
            self._out = torch.empty_like(x)
        return _EXT.softplus_out_cuda(x, self._out)
