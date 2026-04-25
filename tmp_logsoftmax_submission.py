import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_2D(x) TORCH_CHECK((x).dim() == 2, #x " must be 2D")

__device__ __forceinline__ void update_pair(float value, float& current_max, float& current_sum) {
    if (value <= current_max) {
        current_sum += expf(value - current_max);
    } else {
        current_sum = current_sum * expf(current_max - value) + 1.0f;
        current_max = value;
    }
}

__device__ __forceinline__ void combine_pair(float other_max, float other_sum, float& current_max, float& current_sum) {
    if (other_sum == 0.0f) {
        return;
    }
    if (current_sum == 0.0f) {
        current_max = other_max;
        current_sum = other_sum;
        return;
    }
    if (current_max >= other_max) {
        current_sum += other_sum * expf(other_max - current_max);
    } else {
        current_sum = current_sum * expf(current_max - other_max) + other_sum;
        current_max = other_max;
    }
}

template <int THREADS, bool VEC4>
__launch_bounds__(THREADS, 2) __global__ void row_stats_kernel(
    const float* __restrict__ x,
    float2* __restrict__ stats,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const float* row_ptr = x + static_cast<long long>(row) * cols;
    float local_max = -INFINITY;
    float local_sum = 0.0f;

    if constexpr (VEC4) {
        const float4* row_vec = reinterpret_cast<const float4*>(row_ptr);
        int cols4 = cols >> 2;
        for (int idx = threadIdx.x; idx < cols4; idx += THREADS) {
            float4 values = row_vec[idx];
            update_pair(values.x, local_max, local_sum);
            update_pair(values.y, local_max, local_sum);
            update_pair(values.z, local_max, local_sum);
            update_pair(values.w, local_max, local_sum);
        }
    } else {
        for (int idx = threadIdx.x; idx < cols; idx += THREADS) {
            update_pair(row_ptr[idx], local_max, local_sum);
        }
    }

    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_max = __shfl_down_sync(mask, local_max, offset);
        float other_sum = __shfl_down_sync(mask, local_sum, offset);
        combine_pair(other_max, other_sum, local_max, local_sum);
    }

    __shared__ float warp_max[32];
    __shared__ float warp_sum[32];

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warp_count = THREADS >> 5;

    if (lane == 0) {
        warp_max[warp] = local_max;
        warp_sum[warp] = local_sum;
    }
    __syncthreads();

    if (warp == 0) {
        float block_max = lane < warp_count ? warp_max[lane] : -INFINITY;
        float block_sum = lane < warp_count ? warp_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_max = __shfl_down_sync(mask, block_max, offset);
            float other_sum = __shfl_down_sync(mask, block_sum, offset);
            combine_pair(other_max, other_sum, block_max, block_sum);
        }
        if (lane == 0) {
            stats[row] = make_float2(block_max, block_sum);
        }
    }
}

template <int THREADS, bool VEC4>
__launch_bounds__(THREADS, 2) __global__ void logsoftmax_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const float2* __restrict__ stats,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const float* row_ptr = x + static_cast<long long>(row) * cols;
    float* out_ptr = out + static_cast<long long>(row) * cols;
    float2 row_stats = stats[row];
    float log_denom = row_stats.x + logf(row_stats.y);

    if constexpr (VEC4) {
        const float4* row_vec = reinterpret_cast<const float4*>(row_ptr);
        float4* out_vec = reinterpret_cast<float4*>(out_ptr);
        int cols4 = cols >> 2;
        for (int idx = threadIdx.x; idx < cols4; idx += THREADS) {
            float4 values = row_vec[idx];
            float4 result;
            result.x = values.x - log_denom;
            result.y = values.y - log_denom;
            result.z = values.z - log_denom;
            result.w = values.w - log_denom;
            out_vec[idx] = result;
        }
    } else {
        for (int idx = threadIdx.x; idx < cols; idx += THREADS) {
            out_ptr[idx] = row_ptr[idx] - log_denom;
        }
    }
}

torch::Tensor log_softmax_cuda(torch::Tensor x, torch::Tensor stats) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    CHECK_FLOAT(x);
    CHECK_2D(x);
    CHECK_CUDA(stats);
    CHECK_CONTIGUOUS(stats);
    CHECK_FLOAT(stats);
    TORCH_CHECK(stats.dim() == 2, "stats must be 2D");

    const int rows = static_cast<int>(x.size(0));
    const int cols = static_cast<int>(x.size(1));
    TORCH_CHECK(stats.size(0) >= rows && stats.size(1) >= 2, "stats has insufficient shape");

    auto out = torch::empty_like(x);

    c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());

    constexpr int threads = 256;
    const dim3 grid(rows);

    const auto x_ptr = reinterpret_cast<std::uintptr_t>(x.data_ptr<float>());
    const auto out_ptr = reinterpret_cast<std::uintptr_t>(out.data_ptr<float>());
    const bool can_vec4 = ((x_ptr | out_ptr) & 15u) == 0u && (cols & 3) == 0;

    if (can_vec4) {
        row_stats_kernel<threads, true><<<grid, threads, 0, stream>>>(
            x.data_ptr<float>(),
            reinterpret_cast<float2*>(stats.data_ptr<float>()),
            rows,
            cols
        );
        logsoftmax_kernel<threads, true><<<grid, threads, 0, stream>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            reinterpret_cast<float2*>(stats.data_ptr<float>()),
            rows,
            cols
        );
    } else {
        row_stats_kernel<threads, false><<<grid, threads, 0, stream>>>(
            x.data_ptr<float>(),
            reinterpret_cast<float2*>(stats.data_ptr<float>()),
            rows,
            cols
        );
        logsoftmax_kernel<threads, false><<<grid, threads, 0, stream>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            reinterpret_cast<float2*>(stats.data_ptr<float>()),
            rows,
            cols
        );
    }

    return out;
}
"""


module = load_inline(
    name="ptxbench_logsoftmax_sm89_v2",
    cpp_sources="torch::Tensor log_softmax_cuda(torch::Tensor x, torch::Tensor stats);",
    cuda_sources=CUDA_SRC,
    functions=["log_softmax_cuda"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim
        self._stats = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resolved_dim = self.dim if self.dim >= 0 else x.dim() + self.dim
        if resolved_dim != 1:
            raise RuntimeError("ModelNew only supports log_softmax over dim=1")
        rows = x.size(0)
        if self._stats is None or self._stats.device != x.device or self._stats.dtype != x.dtype or self._stats.size(0) < rows:
            self._stats = torch.empty((rows, 2), device=x.device, dtype=x.dtype)
        return module.log_softmax_cuda(x, self._stats)
