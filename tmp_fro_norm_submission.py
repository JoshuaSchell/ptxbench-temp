import hashlib
import os

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"


_CPP_SRC = """
torch::Tensor fro_norm_cuda(torch::Tensor x, torch::Tensor partials, torch::Tensor out);
"""


_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")

constexpr int THREADS = 256;

__device__ __forceinline__ float warp_sum(float value) {
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(mask, value, offset);
    }
    return value;
}

template <int THREADS_>
__device__ __forceinline__ float block_sum(float value) {
    __shared__ float shared[THREADS_ / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_sum(value);
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < (THREADS_ / 32) ? shared[lane] : 0.0f;
    if (warp == 0) {
        value = warp_sum(value);
    }
    return value;
}

template <bool VEC4>
__launch_bounds__(THREADS, 2) __global__ void sumsq_kernel(
    const float* __restrict__ x,
    float* __restrict__ partials,
    long long n
) {
    float sum = 0.0f;
    if constexpr (VEC4) {
        const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
        const long long n4 = n >> 2;
        const long long start = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
        const long long step = static_cast<long long>(gridDim.x) * blockDim.x;
        for (long long idx = start; idx < n4; idx += step) {
            float4 v = x4[idx];
            sum = fmaf(v.x, v.x, sum);
            sum = fmaf(v.y, v.y, sum);
            sum = fmaf(v.z, v.z, sum);
            sum = fmaf(v.w, v.w, sum);
        }
    } else {
        const long long start = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
        const long long step = static_cast<long long>(gridDim.x) * blockDim.x;
        for (long long idx = start; idx < n; idx += step) {
            float v = x[idx];
            sum = fmaf(v, v, sum);
        }
    }

    sum = block_sum<THREADS>(sum);
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = sum;
    }
}

__launch_bounds__(THREADS, 1) __global__ void finalize_invnorm_kernel(
    float* __restrict__ partials,
    int nblocks
) {
    float sum = 0.0f;
    for (int idx = threadIdx.x; idx < nblocks; idx += blockDim.x) {
        sum += partials[idx];
    }
    sum = block_sum<THREADS>(sum);
    if (threadIdx.x == 0) {
        partials[0] = rsqrtf(sum);
    }
}

template <bool VEC4>
__launch_bounds__(THREADS, 2) __global__ void scale_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const float* __restrict__ inv_norm_ptr,
    long long n
) {
    const float inv_norm = inv_norm_ptr[0];
    if constexpr (VEC4) {
        const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
        float4* __restrict__ out4 = reinterpret_cast<float4*>(out);
        const long long n4 = n >> 2;
        const long long start = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
        const long long step = static_cast<long long>(gridDim.x) * blockDim.x;
        for (long long idx = start; idx < n4; idx += step) {
            float4 v = x4[idx];
            float4 r;
            r.x = v.x * inv_norm;
            r.y = v.y * inv_norm;
            r.z = v.z * inv_norm;
            r.w = v.w * inv_norm;
            out4[idx] = r;
        }
    } else {
        const long long start = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
        const long long step = static_cast<long long>(gridDim.x) * blockDim.x;
        for (long long idx = start; idx < n; idx += step) {
            out[idx] = x[idx] * inv_norm;
        }
    }
}

torch::Tensor fro_norm_cuda(torch::Tensor x, torch::Tensor partials, torch::Tensor out) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    CHECK_FLOAT(x);
    CHECK_CUDA(partials);
    CHECK_CONTIGUOUS(partials);
    CHECK_FLOAT(partials);
    CHECK_CUDA(out);
    CHECK_CONTIGUOUS(out);
    CHECK_FLOAT(out);
    TORCH_CHECK(partials.dim() == 1, "partials must be a 1D tensor");
    TORCH_CHECK(out.sizes().equals(x.sizes()), "out shape mismatch");
    TORCH_CHECK(x.device() == partials.device() && x.device() == out.device(), "device mismatch");

    const long long n = x.numel();
    if (n == 0) {
        return out;
    }

    c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());

    const auto x_ptr_u = reinterpret_cast<std::uintptr_t>(x.data_ptr<float>());
    const auto out_ptr_u = reinterpret_cast<std::uintptr_t>(out.data_ptr<float>());
    const bool can_vec4 = ((x_ptr_u | out_ptr_u) & 15u) == 0u && (n & 3ll) == 0;

    const auto* props = at::cuda::getDeviceProperties(x.get_device());
    const int max_blocks = static_cast<int>(partials.numel());
    const long long work_items = can_vec4 ? (n >> 2) : n;
    int blocks = static_cast<int>((work_items + THREADS - 1) / THREADS);
    blocks = blocks < 1 ? 1 : blocks;
    blocks = blocks > props->multiProcessorCount * 8 ? props->multiProcessorCount * 8 : blocks;
    blocks = blocks > max_blocks ? max_blocks : blocks;

    if (can_vec4) {
        sumsq_kernel<true><<<blocks, THREADS, 0, stream>>>(
            x.data_ptr<float>(),
            partials.data_ptr<float>(),
            n
        );
    } else {
        sumsq_kernel<false><<<blocks, THREADS, 0, stream>>>(
            x.data_ptr<float>(),
            partials.data_ptr<float>(),
            n
        );
    }

    finalize_invnorm_kernel<<<1, THREADS, 0, stream>>>(
        partials.data_ptr<float>(),
        blocks
    );

    if (can_vec4) {
        scale_kernel<true><<<blocks, THREADS, 0, stream>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            partials.data_ptr<float>(),
            n
        );
    } else {
        scale_kernel<false><<<blocks, THREADS, 0, stream>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            partials.data_ptr<float>(),
            n
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""


_EXT = load_inline(
    name="ptxbench_fro_norm_" + hashlib.sha1((_CPP_SRC + _CUDA_SRC).encode("utf-8")).hexdigest()[:16],
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["fro_norm_cuda"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._partials = None
        self._out = None
        self._device_index = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("x must be CUDA")
        if x.dtype != torch.float32:
            raise RuntimeError("x must be float32")
        if not x.is_contiguous():
            raise RuntimeError("x must be contiguous")

        device_index = x.device.index
        if self._partials is None or self._device_index != device_index:
            self._partials = torch.empty(1024, device=x.device, dtype=torch.float32)
            self._device_index = device_index

        if (
            self._out is None
            or self._out.device != x.device
            or self._out.shape != x.shape
            or self._out.dtype != x.dtype
        ):
            self._out = torch.empty_like(x)

        return _EXT.fro_norm_cuda(x, self._partials, self._out)
