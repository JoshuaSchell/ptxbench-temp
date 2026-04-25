import hashlib

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")

__device__ __forceinline__ float max_nan(float a, float b) {
    return isnan(b) || b > a ? b : a;
}

__global__ __launch_bounds__(128, 4) void maxpool3d_generic_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int C,
    int D,
    int H,
    int W,
    int outD,
    int outH,
    int outW
) {
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    if (ow >= outW || oh >= outH) {
        return;
    }

    const int ncod = blockIdx.z;
    const int od = ncod % outD;
    const int nc = ncod / outD;
    const int c = nc % C;
    const int n = nc / C;

    const int in_d0 = od * 2 - 1;
    const int in_h0 = oh * 2 - 1;
    const int in_w0 = ow * 2 - 1;
    const int base_nc = ((n * C + c) * D) * H * W;
    float m = -CUDART_INF_F;

#pragma unroll
    for (int kd = 0; kd < 3; ++kd) {
        const int id = in_d0 + kd * 3;
        if ((unsigned int)id >= (unsigned int)D) {
            continue;
        }
        const int dbase = base_nc + id * H * W;
#pragma unroll
        for (int kh = 0; kh < 3; ++kh) {
            const int ih = in_h0 + kh * 3;
            if ((unsigned int)ih >= (unsigned int)H) {
                continue;
            }
            const int hbase = dbase + ih * W;
#pragma unroll
            for (int kw = 0; kw < 3; ++kw) {
                const int iw = in_w0 + kw * 3;
                if ((unsigned int)iw < (unsigned int)W) {
                    m = max_nan(m, __ldg(x + hbase + iw));
                }
            }
        }
    }

    const int out_idx = ((((n * C + c) * outD + od) * outH + oh) * outW + ow);
    y[out_idx] = m;
}

__global__ __launch_bounds__(128, 4) void maxpool3d_interior_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int C,
    int D,
    int H,
    int W,
    int outD,
    int outH,
    int outW,
    int fullD
) {
    const int ow = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (ow >= outW || oh >= outH) {
        return;
    }

    const int ncod = blockIdx.z;
    const int od = (ncod % fullD) + 1;
    const int nc = ncod / fullD;
    const int c = nc % C;
    const int n = nc / C;

    const int in_d0 = od * 2 - 1;
    const int in_h0 = oh * 2 - 1;
    const int in_w0 = ow * 2 - 1;
    const int base_nc = ((n * C + c) * D) * H * W;
    float m = -CUDART_INF_F;

#pragma unroll
    for (int kd = 0; kd < 3; ++kd) {
        const int dbase = base_nc + (in_d0 + kd * 3) * H * W;
#pragma unroll
        for (int kh = 0; kh < 3; ++kh) {
            const int hbase = dbase + (in_h0 + kh * 3) * W;
#pragma unroll
            for (int kw = 0; kw < 3; ++kw) {
                m = max_nan(m, __ldg(x + hbase + in_w0 + kw * 3));
            }
        }
    }

    const int out_idx = ((((n * C + c) * outD + od) * outH + oh) * outW + ow);
    y[out_idx] = m;
}

__global__ __launch_bounds__(128, 4) void maxpool3d_front_plane_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int C,
    int D,
    int H,
    int W,
    int outD,
    int outH,
    int outW
) {
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    if (ow >= outW || oh >= outH) {
        return;
    }

    const int nc = blockIdx.z;
    const int c = nc % C;
    const int n = nc / C;

    const int in_h0 = oh * 2 - 1;
    const int in_w0 = ow * 2 - 1;
    const int base_nc = ((n * C + c) * D) * H * W;
    float m = -CUDART_INF_F;

#pragma unroll
    for (int kd = 0; kd < 3; ++kd) {
        const int id = -1 + kd * 3;
        if ((unsigned int)id >= (unsigned int)D) {
            continue;
        }
        const int dbase = base_nc + id * H * W;
#pragma unroll
        for (int kh = 0; kh < 3; ++kh) {
            const int ih = in_h0 + kh * 3;
            if ((unsigned int)ih >= (unsigned int)H) {
                continue;
            }
            const int hbase = dbase + ih * W;
#pragma unroll
            for (int kw = 0; kw < 3; ++kw) {
                const int iw = in_w0 + kw * 3;
                if ((unsigned int)iw < (unsigned int)W) {
                    m = max_nan(m, __ldg(x + hbase + iw));
                }
            }
        }
    }

    const int out_idx = ((((n * C + c) * outD) * outH + oh) * outW + ow);
    y[out_idx] = m;
}

__global__ __launch_bounds__(128, 4) void maxpool3d_top_plane_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int C,
    int D,
    int H,
    int W,
    int outD,
    int outH,
    int outW
) {
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int od = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (ow >= outW || od >= outD) {
        return;
    }

    const int nc = blockIdx.z;
    const int c = nc % C;
    const int n = nc / C;

    const int in_d0 = od * 2 - 1;
    const int in_w0 = ow * 2 - 1;
    const int base_nc = ((n * C + c) * D) * H * W;
    float m = -CUDART_INF_F;

#pragma unroll
    for (int kd = 0; kd < 3; ++kd) {
        const int dbase = base_nc + (in_d0 + kd * 3) * H * W;
#pragma unroll
        for (int kh = 0; kh < 3; ++kh) {
            const int ih = -1 + kh * 3;
            if ((unsigned int)ih >= (unsigned int)H) {
                continue;
            }
            const int hbase = dbase + ih * W;
#pragma unroll
            for (int kw = 0; kw < 3; ++kw) {
                const int iw = in_w0 + kw * 3;
                if ((unsigned int)iw < (unsigned int)W) {
                    m = max_nan(m, __ldg(x + hbase + iw));
                }
            }
        }
    }

    const int out_idx = ((((n * C + c) * outD + od) * outH) * outW + ow);
    y[out_idx] = m;
}

__global__ __launch_bounds__(128, 4) void maxpool3d_left_edge_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int C,
    int D,
    int H,
    int W,
    int outD,
    int outH,
    int outW
) {
    const int oh = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int od = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (oh >= outH || od >= outD) {
        return;
    }

    const int nc = blockIdx.z;
    const int c = nc % C;
    const int n = nc / C;

    const int in_d0 = od * 2 - 1;
    const int in_h0 = oh * 2 - 1;
    const int base_nc = ((n * C + c) * D) * H * W;
    float m = -CUDART_INF_F;

#pragma unroll
    for (int kd = 0; kd < 3; ++kd) {
        const int dbase = base_nc + (in_d0 + kd * 3) * H * W;
#pragma unroll
        for (int kh = 0; kh < 3; ++kh) {
            const int hbase = dbase + (in_h0 + kh * 3) * W;
#pragma unroll
            for (int kw = 0; kw < 3; ++kw) {
                const int iw = -1 + kw * 3;
                if ((unsigned int)iw < (unsigned int)W) {
                    m = max_nan(m, __ldg(x + hbase + iw));
                }
            }
        }
    }

    const int out_idx = ((((n * C + c) * outD + od) * outH + oh) * outW);
    y[out_idx] = m;
}

torch::Tensor maxpool3d_k3s2p1d3_cuda(torch::Tensor x) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    CHECK_FLOAT(x);
    TORCH_CHECK(x.dim() == 5, "x must be a 5D tensor");

    const at::cuda::CUDAGuard device_guard(x.device());

    const int N = static_cast<int>(x.size(0));
    const int C = static_cast<int>(x.size(1));
    const int D = static_cast<int>(x.size(2));
    const int H = static_cast<int>(x.size(3));
    const int W = static_cast<int>(x.size(4));

    const int outD = (D + 2 - 7) / 2 + 1;
    const int outH = (H + 2 - 7) / 2 + 1;
    const int outW = (W + 2 - 7) / 2 + 1;
    TORCH_CHECK(outD >= 0 && outH >= 0 && outW >= 0, "invalid output size for maxpool3d");

    auto y = torch::empty({N, C, outD, outH, outW}, x.options());
    if (y.numel() == 0) {
        return y;
    }

    const int fullD = (D >= 7) ? std::min(outD - 1, (D - 6) / 2) : 0;
    const int fullH = (H >= 7) ? std::min(outH - 1, (H - 6) / 2) : 0;
    const int fullW = (W >= 7) ? std::min(outW - 1, (W - 6) / 2) : 0;

    constexpr int BX = 32;
    constexpr int BY = 4;
    const dim3 block(BX, BY);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (fullD == outD - 1 && fullH == outH - 1 && fullW == outW - 1 && outD > 1 && outH > 1 && outW > 1) {
        const dim3 grid_interior(
            (fullW + BX - 1) / BX,
            (fullH + BY - 1) / BY,
            N * C * fullD
        );
        maxpool3d_interior_kernel<<<grid_interior, block, 0, stream>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            C,
            D,
            H,
            W,
            outD,
            outH,
            outW,
            fullD
        );

        const dim3 grid_front(
            (outW + BX - 1) / BX,
            (outH + BY - 1) / BY,
            N * C
        );
        maxpool3d_front_plane_kernel<<<grid_front, block, 0, stream>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            C,
            D,
            H,
            W,
            outD,
            outH,
            outW
        );

        const dim3 grid_top(
            (outW + BX - 1) / BX,
            (outD - 1 + BY - 1) / BY,
            N * C
        );
        maxpool3d_top_plane_kernel<<<grid_top, block, 0, stream>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            C,
            D,
            H,
            W,
            outD,
            outH,
            outW
        );

        const dim3 grid_left(
            (outH - 1 + BX - 1) / BX,
            (outD - 1 + BY - 1) / BY,
            N * C
        );
        maxpool3d_left_edge_kernel<<<grid_left, block, 0, stream>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            C,
            D,
            H,
            W,
            outD,
            outH,
            outW
        );
    } else {
        const dim3 grid_generic(
            (outW + BX - 1) / BX,
            (outH + BY - 1) / BY,
            N * C * outD
        );
        maxpool3d_generic_kernel<<<grid_generic, block, 0, stream>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            C,
            D,
            H,
            W,
            outD,
            outH,
            outW
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
"""

_SIG = hashlib.sha1(CUDA_SRC.encode("utf-8")).hexdigest()[:16]
module = load_inline(
    name=f"ptxbench_maxpool3d_{_SIG}",
    cpp_sources="torch::Tensor maxpool3d_k3s2p1d3_cuda(torch::Tensor x);",
    cuda_sources=CUDA_SRC,
    functions=["maxpool3d_k3s2p1d3_cuda"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        stride: int = None,
        padding: int = 0,
        dilation: int = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        stride = kernel_size if stride is None else stride
        if kernel_size != 3 or stride != 2 or padding != 1 or dilation != 3:
            raise ValueError("ModelNew is specialized for kernel_size=3, stride=2, padding=1, dilation=3")
        if return_indices or ceil_mode:
            raise ValueError("ModelNew does not support return_indices or ceil_mode")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.dtype != torch.float32:
            raise ValueError("x must be float32")
        if x.dim() != 5:
            raise ValueError("x must be a 5D tensor")
        return module.maxpool3d_k3s2p1d3_cuda(x.contiguous())
