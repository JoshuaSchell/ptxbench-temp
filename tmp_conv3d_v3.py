import hashlib
from pathlib import Path

import nvidia.cudnn
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


_CUDNN_ROOT = Path(list(nvidia.cudnn.__path__)[0])
_CUDNN_INCLUDE = _CUDNN_ROOT / "include"
_CUDNN_LIB = _CUDNN_ROOT / "lib" / "libcudnn.so.9"


CPP_SRC = r"""
torch::Tensor conv3d_cuda(torch::Tensor x, torch::Tensor weight);
"""


CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cudnn.h>

#define CHECK_INPUT(x) \
    TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor"); \
    TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32"); \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

#define CUDNN_CHECK(expr) \
    TORCH_CHECK((expr) == CUDNN_STATUS_SUCCESS, "cuDNN failure: ", cudnnGetErrorString(expr))

struct ConvState {
    cudnnHandle_t handle = nullptr;
    cudnnTensorDescriptor_t x_desc = nullptr;
    cudnnTensorDescriptor_t y_desc = nullptr;
    cudnnFilterDescriptor_t w_desc = nullptr;
    cudnnConvolutionDescriptor_t conv_desc = nullptr;
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    at::Tensor output;
    at::Tensor workspace;
    int64_t device_index = -1;
    int64_t batch = -1;
};

static ConvState& state() {
    static ConvState value;
    return value;
}

static void ensure_handle(ConvState& s, int device_index) {
    if (s.handle != nullptr && s.device_index == device_index) {
        return;
    }
    if (s.x_desc == nullptr) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&s.x_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&s.y_desc));
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&s.w_desc));
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&s.conv_desc));
    }
    c10::cuda::CUDAGuard guard(device_index);
    if (s.handle == nullptr) {
        CUDNN_CHECK(cudnnCreate(&s.handle));
    }
    s.device_index = device_index;
}

static void set_tensor_desc(cudnnTensorDescriptor_t desc, const torch::Tensor& tensor) {
    int dims[5];
    int strides[5];
    for (int i = 0; i < 5; ++i) {
        dims[i] = static_cast<int>(tensor.size(i));
        strides[i] = static_cast<int>(tensor.stride(i));
    }
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(desc, CUDNN_DATA_FLOAT, 5, dims, strides));
}

static void configure(ConvState& s, const torch::Tensor& x, const torch::Tensor& weight) {
    const int n = static_cast<int>(x.size(0));
    if (s.output.defined() && s.batch == n) {
        return;
    }

    int w_dims[5] = {64, 3, 3, 5, 7};
    int conv_pad[3] = {0, 0, 0};
    int conv_stride[3] = {1, 1, 1};
    int conv_dilation[3] = {1, 1, 1};

    CUDNN_CHECK(cudnnSetFilterNdDescriptor(s.w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 5, w_dims));
    CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(
        s.conv_desc,
        3,
        conv_pad,
        conv_stride,
        conv_dilation,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT
    ));
    CUDNN_CHECK(cudnnSetConvolutionMathType(s.conv_desc, CUDNN_DEFAULT_MATH));
    set_tensor_desc(s.x_desc, x);

    int y_dims[5];
    CUDNN_CHECK(cudnnGetConvolutionNdForwardOutputDim(s.conv_desc, s.x_desc, s.w_desc, 5, y_dims));
    s.output = torch::empty({y_dims[0], y_dims[1], y_dims[2], y_dims[3], y_dims[4]}, x.options());
    set_tensor_desc(s.y_desc, s.output);

    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t search_bytes = free_mem / 4;
    const size_t max_search = static_cast<size_t>(1536) * 1024 * 1024;
    if (search_bytes > max_search) {
        search_bytes = max_search;
    }
    if (search_bytes < (1u << 20)) {
        search_bytes = (1u << 20);
    }
    auto search_workspace = torch::empty({static_cast<long long>(search_bytes)}, x.options().dtype(torch::kUInt8));

    int returned = 0;
    cudnnConvolutionFwdAlgoPerf_t perf[8];
    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithmEx(
        s.handle,
        s.x_desc,
        x.data_ptr<float>(),
        s.w_desc,
        weight.data_ptr<float>(),
        s.conv_desc,
        s.y_desc,
        s.output.data_ptr<float>(),
        8,
        &returned,
        perf,
        search_workspace.data_ptr(),
        search_bytes
    ));
    TORCH_CHECK(returned > 0, "No cuDNN forward algorithm found");

    bool found = false;
    size_t workspace_bytes = 0;
    for (int i = 0; i < returned; ++i) {
        if (perf[i].status != CUDNN_STATUS_SUCCESS) {
            continue;
        }
        s.algo = perf[i].algo;
        workspace_bytes = perf[i].memory;
        found = true;
        break;
    }
    TORCH_CHECK(found, "No successful cuDNN forward algorithm");

    if (workspace_bytes > 0) {
        s.workspace = torch::empty({static_cast<long long>(workspace_bytes)}, x.options().dtype(torch::kUInt8));
    } else {
        s.workspace = torch::empty({0}, x.options().dtype(torch::kUInt8));
    }
    s.batch = n;
}

torch::Tensor conv3d_cuda(torch::Tensor x, torch::Tensor weight) {
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    TORCH_CHECK(x.dim() == 5, "x must be 5D");
    TORCH_CHECK(weight.dim() == 5, "weight must be 5D");
    TORCH_CHECK(x.size(1) == 3 && x.size(2) == 16 && x.size(3) == 128 && x.size(4) == 128,
        "x must have shape [N, 3, 16, 128, 128]");
    TORCH_CHECK(weight.size(0) == 64 && weight.size(1) == 3 && weight.size(2) == 3 && weight.size(3) == 5 && weight.size(4) == 7,
        "weight must have shape [64, 3, 3, 5, 7]");

    auto& s = state();
    const int device_index = static_cast<int>(x.get_device());
    ensure_handle(s, device_index);
    auto stream = at::cuda::getCurrentCUDAStream(device_index);
    CUDNN_CHECK(cudnnSetStream(s.handle, stream.stream()));
    configure(s, x, weight);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    void* workspace_ptr = s.workspace.numel() == 0 ? nullptr : s.workspace.data_ptr();
    size_t workspace_bytes = static_cast<size_t>(s.workspace.numel());
    CUDNN_CHECK(cudnnConvolutionForward(
        s.handle,
        &alpha,
        s.x_desc,
        x.data_ptr<float>(),
        s.w_desc,
        weight.data_ptr<float>(),
        s.conv_desc,
        s.algo,
        workspace_ptr,
        workspace_bytes,
        &beta,
        s.y_desc,
        s.output.data_ptr<float>()
    ));
    return s.output;
}
"""


MODULE_NAME = f"ptxbench_conv3d_v3_{hashlib.md5(CUDA_SRC.encode('utf-8')).hexdigest()[:16]}"
module = load_inline(
    name=MODULE_NAME,
    cpp_sources=CPP_SRC,
    cuda_sources=CUDA_SRC,
    functions=["conv3d_cuda"],
    extra_cuda_cflags=["-O3", f"-I{_CUDNN_INCLUDE}"],
    extra_ldflags=[str(_CUDNN_LIB), f"-Wl,-rpath,{_CUDNN_LIB.parent}"],
    verbose=False,
)


def _triple_like(value):
    if isinstance(value, int):
        return (value, value, value)
    return tuple(int(v) for v in value)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if (
            in_channels != 3
            or out_channels != 64
            or tuple(int(v) for v in kernel_size) != (3, 5, 7)
            or _triple_like(stride) != (1, 1, 1)
            or _triple_like(padding) != (0, 0, 0)
            or _triple_like(dilation) != (1, 1, 1)
            or groups != 1
            or bias
        ):
            raise ValueError("ModelNew is specialized for the benchmark configuration")
        seed = torch.initial_seed()
        torch.manual_seed(seed)
        ref = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.register_buffer("weight", ref.weight.detach().contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return module.conv3d_cuda(x, self.weight)
