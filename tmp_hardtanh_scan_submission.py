import hashlib
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


_CPP_SRC = """
void has_oob_cuda(torch::Tensor x, torch::Tensor flag);
torch::Tensor hardtanh_out_cuda(torch::Tensor x, torch::Tensor out);
"""


_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

__device__ __forceinline__ bool out_of_bounds(float x) {
    return x < -1.0f || x > 1.0f;
}

__device__ __forceinline__ float hardtanh_scalar(float x) {
    return x < -1.0f ? -1.0f : (x > 1.0f ? 1.0f : x);
}

__global__ void has_oob_vec4_kernel(
    const float4* __restrict__ x,
    int64_t n_vec,
    int* __restrict__ flag
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < n_vec; i += stride) {
        float4 v = x[i];
        if (out_of_bounds(v.x) || out_of_bounds(v.y) || out_of_bounds(v.z) || out_of_bounds(v.w)) {
            atomicExch(flag, 1);
            return;
        }
    }
}

__global__ void has_oob_scalar_kernel(
    const float* __restrict__ x,
    int64_t n,
    int* __restrict__ flag
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        if (out_of_bounds(x[i])) {
            atomicExch(flag, 1);
            return;
        }
    }
}

__global__ void hardtanh_vec4_kernel(
    const float4* __restrict__ x,
    float4* __restrict__ out,
    int64_t n_vec
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < n_vec; i += stride) {
        float4 v = x[i];
        float4 r;
        r.x = hardtanh_scalar(v.x);
        r.y = hardtanh_scalar(v.y);
        r.z = hardtanh_scalar(v.z);
        r.w = hardtanh_scalar(v.w);
        out[i] = r;
    }
}

__global__ void hardtanh_scalar_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int64_t n
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        out[i] = hardtanh_scalar(x[i]);
    }
}

}  // namespace

void has_oob_cuda(torch::Tensor x, torch::Tensor flag) {
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

    if ((x_addr & 15u) == 0u && (n & 3LL) == 0) {
        const int64_t n_vec = n >> 2;
        int64_t blocks64 = (n_vec + threads - 1) / threads;
        int blocks = static_cast<int>(blocks64 < 32768 ? blocks64 : 32768);
        has_oob_vec4_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(x_ptr),
            n_vec,
            flag.data_ptr<int>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }

    int64_t blocks64 = (n + threads - 1) / threads;
    int blocks = static_cast<int>(blocks64 < 32768 ? blocks64 : 32768);
    has_oob_scalar_kernel<<<blocks, threads, 0, stream>>>(x_ptr, n, flag.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor hardtanh_out_cuda(torch::Tensor x, torch::Tensor out) {
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
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto* x_ptr = x.data_ptr<float>();
    auto* out_ptr = out.data_ptr<float>();
    const auto x_addr = reinterpret_cast<std::uintptr_t>(x_ptr);
    const auto out_addr = reinterpret_cast<std::uintptr_t>(out_ptr);

    if (((x_addr | out_addr) & 15u) == 0u && (n & 3LL) == 0) {
        const int64_t n_vec = n >> 2;
        int64_t blocks64 = (n_vec + threads - 1) / threads;
        int blocks = static_cast<int>(blocks64 < 32768 ? blocks64 : 32768);
        hardtanh_vec4_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(x_ptr),
            reinterpret_cast<float4*>(out_ptr),
            n_vec
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }

    int64_t blocks64 = (n + threads - 1) / threads;
    int blocks = static_cast<int>(blocks64 < 32768 ? blocks64 : 32768);
    hardtanh_scalar_kernel<<<blocks, threads, 0, stream>>>(x_ptr, out_ptr, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""


_EXT = load_inline(
    name=f"ptxbench_hardtanh_scan_sm89_{hashlib.sha1(_CUDA_SRC.encode()).hexdigest()[:16]}",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["has_oob_cuda", "hardtanh_out_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3",
        "--extra-device-vectorization",
        "-gencode=arch=compute_89,code=sm_89",
    ],
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

        _EXT.has_oob_cuda(x, self._flag)
        if self._flag.item() == 0:
            return x

        out_sig = (tuple(x.shape), x.device, x.dtype)
        if self._out is None or self._out_sig != out_sig:
            self._out = torch.empty_like(x)
            self._out_sig = out_sig

        return _EXT.hardtanh_out_cuda(x, self._out)
