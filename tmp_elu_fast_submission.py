import hashlib
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


_CPP_SRC = """
void has_negative_cuda(torch::Tensor x, torch::Tensor flag);
torch::Tensor elu_out_cuda(torch::Tensor x, torch::Tensor out, double alpha);
"""


_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

__global__ void has_negative_vec4_kernel(
    const float4* __restrict__ x,
    int64_t n_vec,
    int* __restrict__ flag
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < n_vec; i += stride) {
        float4 v = x[i];
        if (v.x < 0.0f || v.y < 0.0f || v.z < 0.0f || v.w < 0.0f) {
            atomicExch(flag, 1);
            return;
        }
    }
}

__global__ void has_negative_scalar_kernel(
    const float* __restrict__ x,
    int64_t n,
    int* __restrict__ flag
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        if (x[i] < 0.0f) {
            atomicExch(flag, 1);
            return;
        }
    }
}

__device__ __forceinline__ float elu_scalar(float x, float alpha) {
    return x > 0.0f ? x : alpha * (expf(x) - 1.0f);
}

__global__ void elu_vec4_kernel(
    const float4* __restrict__ x,
    float4* __restrict__ out,
    int64_t n_vec,
    float alpha
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < n_vec; i += stride) {
        float4 v = x[i];
        float4 r;
        r.x = elu_scalar(v.x, alpha);
        r.y = elu_scalar(v.y, alpha);
        r.z = elu_scalar(v.z, alpha);
        r.w = elu_scalar(v.w, alpha);
        out[i] = r;
    }
}

__global__ void elu_scalar_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int64_t n,
    float alpha
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        out[i] = elu_scalar(x[i], alpha);
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
            int64_t blocks64 = (n_vec + threads - 1) / threads;
            int blocks = static_cast<int>(blocks64 < 32768 ? blocks64 : 32768);
            has_negative_vec4_kernel<<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const float4*>(x_ptr),
                n_vec,
                flag.data_ptr<int>()
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return;
        }
    }

    int64_t blocks64 = (n + threads - 1) / threads;
    int blocks = static_cast<int>(blocks64 < 32768 ? blocks64 : 32768);
    has_negative_scalar_kernel<<<blocks, threads, 0, stream>>>(x_ptr, n, flag.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor elu_out_cuda(torch::Tensor x, torch::Tensor out, double alpha) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, "x must be float32");
    TORCH_CHECK(out.scalar_type() == at::ScalarType::Float, "out must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(x.numel() == out.numel(), "size mismatch");

    const int64_t n = x.numel();
    if (n == 0) {
        return out;
    }

    constexpr int threads = 512;
    const float alpha_f = static_cast<float>(alpha);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto* x_ptr = x.data_ptr<float>();
    auto* out_ptr = out.data_ptr<float>();
    const auto x_addr = reinterpret_cast<std::uintptr_t>(x_ptr);
    const auto out_addr = reinterpret_cast<std::uintptr_t>(out_ptr);

    if (((x_addr | out_addr) & 15u) == 0u) {
        const int64_t n_vec = n >> 2;
        if ((n_vec << 2) == n) {
            int64_t blocks64 = (n_vec + threads - 1) / threads;
            int blocks = static_cast<int>(blocks64 < 32768 ? blocks64 : 32768);
            elu_vec4_kernel<<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const float4*>(x_ptr),
                reinterpret_cast<float4*>(out_ptr),
                n_vec,
                alpha_f
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return out;
        }
    }

    int64_t blocks64 = (n + threads - 1) / threads;
    int blocks = static_cast<int>(blocks64 < 32768 ? blocks64 : 32768);
    elu_scalar_kernel<<<blocks, threads, 0, stream>>>(x_ptr, out_ptr, n, alpha_f);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""


_EXT = load_inline(
    name=f"ptxbench_elu_sm89_{hashlib.sha1(_CUDA_SRC.encode()).hexdigest()[:16]}",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["has_negative_cuda", "elu_out_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--extra-device-vectorization"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self._out = None
        self._out_sig = None
        self._flag = None
        self._flag_device = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("ModelNew requires a CUDA tensor")
        if x.dtype != torch.float32:
            raise RuntimeError("ModelNew expects float32 input")
        if not x.is_contiguous():
            x = x.contiguous()
        if x.numel() == 0:
            return x

        if self._flag is None or self._flag_device != x.device:
            self._flag = torch.empty(1, device=x.device, dtype=torch.int32)
            self._flag_device = x.device

        _EXT.has_negative_cuda(x, self._flag)
        if self._flag.item() == 0:
            return x

        out_sig = (tuple(x.shape), x.device, x.dtype)
        if self._out is None or self._out_sig != out_sig:
            self._out = torch.empty_like(x)
            self._out_sig = out_sig
        return _EXT.elu_out_cuda(x, self._out, self.alpha)
