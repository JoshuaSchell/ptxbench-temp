import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


_CPP_SRC = """
torch::Tensor row_scale_cuda(torch::Tensor a, torch::Tensor b);
"""


_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_DIM(x, d) TORCH_CHECK((x).dim() == (d), #x " must have dimension " #d)

constexpr int THREADS = 256;
constexpr int VEC = 4;
constexpr int UNROLL = 2;
constexpr int TILE = THREADS * VEC * UNROLL;

__global__ void row_scale_vec_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int n,
    int m,
    int vec_cols
) {
    const int row = blockIdx.y;
    if (row >= n) {
        return;
    }
    const float scale = __ldg(a + row);
    const int row_offset = row * m;
    const int base_col = blockIdx.x * TILE + threadIdx.x * VEC;

    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
        const int col = base_col + u * THREADS * VEC;
        if (col < vec_cols) {
            const float4 v = *reinterpret_cast<const float4*>(b + row_offset + col);
            float4 r;
            r.x = v.x * scale;
            r.y = v.y * scale;
            r.z = v.z * scale;
            r.w = v.w * scale;
            *reinterpret_cast<float4*>(out + row_offset + col) = r;
        }
    }
}

__global__ void row_scale_tail_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int n,
    int m,
    int start_col
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int width = m - start_col;
    const int total = n * width;
    if (idx >= total) {
        return;
    }
    const int row = idx / width;
    const int col = start_col + (idx - row * width);
    out[row * m + col] = __ldg(a + row) * b[row * m + col];
}

__global__ void row_scale_strided_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int n,
    int m,
    long long a_s0,
    long long b_s0,
    long long b_s1
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < m) {
        out[row * m + col] = a[row * a_s0] * b[row * b_s0 + col * b_s1];
    }
}

torch::Tensor row_scale_cuda(torch::Tensor a, torch::Tensor b) {
    CHECK_CUDA(a);
    CHECK_CUDA(b);
    CHECK_FLOAT(a);
    CHECK_FLOAT(b);
    CHECK_DIM(a, 1);
    CHECK_DIM(b, 2);
    TORCH_CHECK(a.size(0) == b.size(0), "A and B shape mismatch");

    const int n = static_cast<int>(b.size(0));
    const int m = static_cast<int>(b.size(1));
    auto out = torch::empty({n, m}, b.options());
    if (n == 0 || m == 0) {
        return out;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.device().index());

    if (a.is_contiguous() && b.is_contiguous()) {
        const int vec_cols = (m / VEC) * VEC;
        if (vec_cols > 0) {
            dim3 block(THREADS);
            dim3 grid((vec_cols + TILE - 1) / TILE, n);
            row_scale_vec_kernel<<<grid, block, 0, stream>>>(
                a.data_ptr<float>(),
                b.data_ptr<float>(),
                out.data_ptr<float>(),
                n,
                m,
                vec_cols
            );
        }
        if (vec_cols < m) {
            const int width = m - vec_cols;
            const int total = n * width;
            const int block = 256;
            const int grid = (total + block - 1) / block;
            row_scale_tail_kernel<<<grid, block, 0, stream>>>(
                a.data_ptr<float>(),
                b.data_ptr<float>(),
                out.data_ptr<float>(),
                n,
                m,
                vec_cols
            );
        }
    } else {
        dim3 block(32, 8);
        dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);
        row_scale_strided_kernel<<<grid, block, 0, stream>>>(
            a.data_ptr<float>(),
            b.data_ptr<float>(),
            out.data_ptr<float>(),
            n,
            m,
            a.stride(0),
            b.stride(0),
            b.stride(1)
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""


_EXT = load_inline(
    name="ptxbench_rowscale_sm89",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["row_scale_cuda"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return _EXT.row_scale_cuda(A, B)
