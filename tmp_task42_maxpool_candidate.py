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

constexpr int TILE_W = 32;
constexpr int TILE_H = 8;
constexpr int SMEM_STRIDE = TILE_W + 4;
constexpr int LOAD_W = TILE_W + 3;
constexpr int LOAD_H = TILE_H + 3;

__device__ __forceinline__ float load_or_neg_inf(
    const float* __restrict__ x,
    int iy,
    int ix,
    int h,
    int w) {
    if (static_cast<unsigned>(iy) < static_cast<unsigned>(h) &&
        static_cast<unsigned>(ix) < static_cast<unsigned>(w)) {
        return __ldg(x + static_cast<long long>(iy) * w + ix);
    }
    return -FLT_MAX;
}

__global__ __launch_bounds__(256, 2) void maxpool_border_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int h,
    int w,
    int out_h,
    int out_w,
    int plane_stride_in,
    int plane_stride_out,
    int start_bx,
    int start_by) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x + start_bx;
    const int by = blockIdx.y + start_by;
    const int plane = blockIdx.z;

    const int out_x0 = bx * TILE_W;
    const int out_y0 = by * TILE_H;
    const int in_x0 = out_x0 - 1;
    const int in_y0 = out_y0 - 1;

    __shared__ float tile[LOAD_H][SMEM_STRIDE];

    const float* plane_x = x + static_cast<long long>(plane) * plane_stride_in;
    float* plane_out = out + static_cast<long long>(plane) * plane_stride_out;

    const int linear_tid = ty * blockDim.x + tx;
    for (int idx = linear_tid; idx < LOAD_H * SMEM_STRIDE; idx += blockDim.x * blockDim.y) {
        const int sy = idx / SMEM_STRIDE;
        const int sx = idx - sy * SMEM_STRIDE;
        float value = -FLT_MAX;
        if (sx < LOAD_W) {
            value = load_or_neg_inf(plane_x, in_y0 + sy, in_x0 + sx, h, w);
        }
        tile[sy][sx] = value;
    }
    __syncthreads();

    const int ox = out_x0 + tx;
    const int oy = out_y0 + ty;
    if (ox < out_w && oy < out_h) {
        const int base = ty * SMEM_STRIDE + tx;
        const float a00 = ((float*)tile)[base];
        const float a01 = ((float*)tile)[base + 1];
        const float a02 = ((float*)tile)[base + 2];
        const float a03 = ((float*)tile)[base + 3];
        const float a10 = ((float*)tile)[base + SMEM_STRIDE];
        const float a11 = ((float*)tile)[base + SMEM_STRIDE + 1];
        const float a12 = ((float*)tile)[base + SMEM_STRIDE + 2];
        const float a13 = ((float*)tile)[base + SMEM_STRIDE + 3];
        const float a20 = ((float*)tile)[base + 2 * SMEM_STRIDE];
        const float a21 = ((float*)tile)[base + 2 * SMEM_STRIDE + 1];
        const float a22 = ((float*)tile)[base + 2 * SMEM_STRIDE + 2];
        const float a23 = ((float*)tile)[base + 2 * SMEM_STRIDE + 3];
        const float a30 = ((float*)tile)[base + 3 * SMEM_STRIDE];
        const float a31 = ((float*)tile)[base + 3 * SMEM_STRIDE + 1];
        const float a32 = ((float*)tile)[base + 3 * SMEM_STRIDE + 2];
        const float a33 = ((float*)tile)[base + 3 * SMEM_STRIDE + 3];

        const float m0 = fmaxf(fmaxf(a00, a01), fmaxf(a02, a03));
        const float m1 = fmaxf(fmaxf(a10, a11), fmaxf(a12, a13));
        const float m2 = fmaxf(fmaxf(a20, a21), fmaxf(a22, a23));
        const float m3 = fmaxf(fmaxf(a30, a31), fmaxf(a32, a33));
        plane_out[static_cast<long long>(oy) * out_w + ox] = fmaxf(fmaxf(m0, m1), fmaxf(m2, m3));
    }
}

__global__ __launch_bounds__(256, 2) void maxpool_interior_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int w,
    int out_w,
    int plane_stride_in,
    int plane_stride_out) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x + 1;
    const int by = blockIdx.y + 1;
    const int plane = blockIdx.z;

    const int out_x0 = bx * TILE_W;
    const int out_y0 = by * TILE_H;
    const int in_x0 = out_x0 - 1;
    const int in_y0 = out_y0 - 1;

    __shared__ float tile[LOAD_H][SMEM_STRIDE];

    const float* plane_x = x + static_cast<long long>(plane) * plane_stride_in;
    float* plane_out = out + static_cast<long long>(plane) * plane_stride_out;

    const int linear_tid = ty * blockDim.x + tx;
    for (int idx = linear_tid; idx < LOAD_H * SMEM_STRIDE; idx += blockDim.x * blockDim.y) {
        const int sy = idx / SMEM_STRIDE;
        const int sx = idx - sy * SMEM_STRIDE;
        float value = -FLT_MAX;
        if (sx < LOAD_W) {
            value = __ldg(plane_x + static_cast<long long>(in_y0 + sy) * w + (in_x0 + sx));
        }
        tile[sy][sx] = value;
    }
    __syncthreads();

    const int ox = out_x0 + tx;
    const int oy = out_y0 + ty;
    const int base = ty * SMEM_STRIDE + tx;

    const float a00 = ((float*)tile)[base];
    const float a01 = ((float*)tile)[base + 1];
    const float a02 = ((float*)tile)[base + 2];
    const float a03 = ((float*)tile)[base + 3];
    const float a10 = ((float*)tile)[base + SMEM_STRIDE];
    const float a11 = ((float*)tile)[base + SMEM_STRIDE + 1];
    const float a12 = ((float*)tile)[base + SMEM_STRIDE + 2];
    const float a13 = ((float*)tile)[base + SMEM_STRIDE + 3];
    const float a20 = ((float*)tile)[base + 2 * SMEM_STRIDE];
    const float a21 = ((float*)tile)[base + 2 * SMEM_STRIDE + 1];
    const float a22 = ((float*)tile)[base + 2 * SMEM_STRIDE + 2];
    const float a23 = ((float*)tile)[base + 2 * SMEM_STRIDE + 3];
    const float a30 = ((float*)tile)[base + 3 * SMEM_STRIDE];
    const float a31 = ((float*)tile)[base + 3 * SMEM_STRIDE + 1];
    const float a32 = ((float*)tile)[base + 3 * SMEM_STRIDE + 2];
    const float a33 = ((float*)tile)[base + 3 * SMEM_STRIDE + 3];

    const float m0 = fmaxf(fmaxf(a00, a01), fmaxf(a02, a03));
    const float m1 = fmaxf(fmaxf(a10, a11), fmaxf(a12, a13));
    const float m2 = fmaxf(fmaxf(a20, a21), fmaxf(a22, a23));
    const float m3 = fmaxf(fmaxf(a30, a31), fmaxf(a32, a33));
    plane_out[static_cast<long long>(oy) * out_w + ox] = fmaxf(fmaxf(m0, m1), fmaxf(m2, m3));
}

torch::Tensor maxpool2d_4x4_s1_p1_out(torch::Tensor x, torch::Tensor out) {
    CHECK_CUDA(x);
    CHECK_CUDA(out);
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(out);
    CHECK_FLOAT(x);
    CHECK_FLOAT(out);
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    TORCH_CHECK(out.dim() == 4, "out must be 4D");
    TORCH_CHECK(x.size(2) >= 2 && x.size(3) >= 2, "input spatial dimensions must be at least 2");

    const auto n = static_cast<int>(x.size(0));
    const auto c = static_cast<int>(x.size(1));
    const auto h = static_cast<int>(x.size(2));
    const auto w = static_cast<int>(x.size(3));
    const auto out_h = h - 1;
    const auto out_w = w - 1;
    TORCH_CHECK(out.size(0) == n && out.size(1) == c && out.size(2) == out_h && out.size(3) == out_w,
        "out has incorrect shape");

    const int plane_stride_in = h * w;
    const int plane_stride_out = out_h * out_w;
    const int planes = n * c;
    const int grid_x = (out_w + TILE_W - 1) / TILE_W;
    const int grid_y = (out_h + TILE_H - 1) / TILE_H;

    c10::cuda::CUDAGuard device_guard(x.device());
    const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());

    const dim3 block(TILE_W, TILE_H);

    if (grid_x > 2 && grid_y > 2) {
        const dim3 interior_grid(grid_x - 2, grid_y - 2, planes);
        maxpool_interior_kernel<<<interior_grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            w,
            out_w,
            plane_stride_in,
            plane_stride_out
        );
    }

    const dim3 top_grid(grid_x, 1, planes);
    maxpool_border_kernel<<<top_grid, block, 0, stream>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        h,
        w,
        out_h,
        out_w,
        plane_stride_in,
        plane_stride_out,
        0,
        0
    );

    if (grid_y > 1) {
        const dim3 bottom_grid(grid_x, 1, planes);
        maxpool_border_kernel<<<bottom_grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            h,
            w,
            out_h,
            out_w,
            plane_stride_in,
            plane_stride_out,
            0,
            grid_y - 1
        );
    }

    if (grid_y > 2) {
        const dim3 left_grid(1, grid_y - 2, planes);
        maxpool_border_kernel<<<left_grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            h,
            w,
            out_h,
            out_w,
            plane_stride_in,
            plane_stride_out,
            0,
            1
        );
        if (grid_x > 1) {
            const dim3 right_grid(1, grid_y - 2, planes);
            maxpool_border_kernel<<<right_grid, block, 0, stream>>>(
                x.data_ptr<float>(),
                out.data_ptr<float>(),
                h,
                w,
                out_h,
                out_w,
                plane_stride_in,
                plane_stride_out,
                grid_x - 1,
                1
            );
        }
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

"""


module = load_inline(
    name="ptxbench_maxpool2d_4x4_s1_p1_v2",
    cpp_sources="torch::Tensor maxpool2d_4x4_s1_p1_out(torch::Tensor x, torch::Tensor out);",
    cuda_sources=CUDA_SRC,
    functions=["maxpool2d_4x4_s1_p1_out"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-Xptxas", "-dlcm=ca"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int) -> None:
        super().__init__()
        if kernel_size != 4 or stride != 1 or padding != 1 or dilation != 1:
            raise ValueError("ModelNew is specialized for kernel_size=4, stride=1, padding=1, dilation=1")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("ModelNew requires a CUDA tensor")
        if x.dtype != torch.float32:
            raise RuntimeError("ModelNew expects float32 input")
        if not x.is_contiguous():
            raise RuntimeError("ModelNew expects contiguous input")
        out_shape = (x.shape[0], x.shape[1], x.shape[2] - 1, x.shape[3] - 1)
        out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        return module.maxpool2d_4x4_s1_p1_out(x, out)
