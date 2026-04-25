import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

__global__ void has_negative_vec4_kernel(const float4* __restrict__ x, int64_t n_vec, int* __restrict__ flag) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n_vec) {
        float4 v = x[i];
        if (v.x < 0.0f || v.y < 0.0f || v.z < 0.0f || v.w < 0.0f) {
            atomicExch(flag, 1);
        }
    }
}

__global__ void has_negative_scalar_kernel(const float* __restrict__ x, int64_t n, int* __restrict__ flag) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n && x[i] < 0.0f) {
        atomicExch(flag, 1);
    }
}

__device__ __forceinline__ float relu_preserve_nan(float v) {
    return v <= 0.0f ? 0.0f : v;
}

__global__ void relu_vec4_kernel(const float4* __restrict__ x, float4* __restrict__ out, int64_t n_vec) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n_vec) {
        float4 v = x[i];
        v.x = relu_preserve_nan(v.x);
        v.y = relu_preserve_nan(v.y);
        v.z = relu_preserve_nan(v.z);
        v.w = relu_preserve_nan(v.w);
        out[i] = v;
    }
}

__global__ void relu_scalar_kernel(const float* __restrict__ x, float* __restrict__ out, int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        out[i] = relu_preserve_nan(v);
    }
}

}  // namespace

void has_negative_cuda(torch::Tensor x, torch::Tensor flag) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(flag.is_cuda(), "flag must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, "x must be float32");
    TORCH_CHECK(flag.scalar_type() == at::ScalarType::Int, "flag must be int32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(flag.numel() == 1, "flag must have one element");

    const int64_t n = x.numel();
    if (n == 0) {
        return;
    }

    constexpr int threads = 512;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemsetAsync(flag.data_ptr<int>(), 0, sizeof(int), stream));

    auto* x_ptr = x.data_ptr<float>();
    const auto x_addr = reinterpret_cast<std::uintptr_t>(x_ptr);
    if ((x_addr & 15u) == 0u) {
        const int64_t n_vec = n >> 2;
        if ((n_vec << 2) == n) {
            int blocks = static_cast<int>((n_vec + threads - 1) / threads);
            has_negative_vec4_kernel<<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const float4*>(x_ptr),
                n_vec,
                flag.data_ptr<int>()
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return;
        }
    }

    int blocks = static_cast<int>((n + threads - 1) / threads);
    has_negative_scalar_kernel<<<blocks, threads, 0, stream>>>(x_ptr, n, flag.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void relu_out_cuda(torch::Tensor x, torch::Tensor out) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, "x must be float32");
    TORCH_CHECK(out.scalar_type() == at::ScalarType::Float, "out must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(x.numel() == out.numel(), "size mismatch");

    const int64_t n = x.numel();
    if (n == 0) {
        return;
    }

    constexpr int threads = 512;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto* x_ptr = x.data_ptr<float>();
    auto* out_ptr = out.data_ptr<float>();
    const auto x_addr = reinterpret_cast<std::uintptr_t>(x_ptr);
    const auto out_addr = reinterpret_cast<std::uintptr_t>(out_ptr);

    if (((x_addr | out_addr) & 15u) == 0u) {
        const int64_t n_vec = n >> 2;
        if ((n_vec << 2) == n) {
            int blocks = static_cast<int>((n_vec + threads - 1) / threads);
            relu_vec4_kernel<<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const float4*>(x_ptr),
                reinterpret_cast<float4*>(out_ptr),
                n_vec
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return;
        }
    }

    int blocks = static_cast<int>((n + threads - 1) / threads);
    relu_scalar_kernel<<<blocks, threads, 0, stream>>>(x_ptr, out_ptr, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


module = load_inline(
    name="ptxbench_relu_scan_sm89",
    cpp_sources="""
void has_negative_cuda(torch::Tensor x, torch::Tensor flag);
void relu_out_cuda(torch::Tensor x, torch::Tensor out);
""",
    cuda_sources=CUDA_SRC,
    functions=["has_negative_cuda", "relu_out_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--extra-device-vectorization"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._out = None
        self._out_sig = None
        self._flag = None
        self._flag_device = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._flag is None or self._flag_device != x.device:
            self._flag = torch.empty(1, device=x.device, dtype=torch.int32)
            self._flag_device = x.device

        module.has_negative_cuda(x, self._flag)
        if self._flag.item() == 0:
            return x

        out_sig = (tuple(x.shape), x.device, x.dtype)
        if self._out is None or self._out_sig != out_sig:
            self._out = torch.empty_like(x)
            self._out_sig = out_sig

        module.relu_out_cuda(x, self._out)
        return self._out
