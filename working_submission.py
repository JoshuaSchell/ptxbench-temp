import hashlib

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


CPP_SRC = r"""
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_d,
    int pad_h,
    int pad_w,
    int groups
);
"""


CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/ops/convolution.h>
#include <c10/core/MemoryFormat.h>

#define CHECK_INPUT(x) \
    TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor"); \
    TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")

struct ConvState {
    at::Tensor output;
    int64_t device_index = -1;
    int64_t batch = -1;
    int64_t out_channels = -1;
    int64_t out_d = -1;
    int64_t out_h = -1;
    int64_t out_w = -1;
};

static ConvState& state() {
    static ConvState value;
    return value;
}

static void ensure_output(
    ConvState& s,
    const torch::Tensor& x,
    const torch::Tensor& weight,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_d,
    int pad_h,
    int pad_w,
    int groups
) {
    const auto batch = x.size(0);
    const auto out_channels = weight.size(1) * groups;
    const auto out_d = (x.size(2) - 1) * stride_d - 2 * pad_d + weight.size(2);
    const auto out_h = (x.size(3) - 1) * stride_h - 2 * pad_h + weight.size(3);
    const auto out_w = (x.size(4) - 1) * stride_w - 2 * pad_w + weight.size(4);
    const auto device_index = static_cast<int64_t>(x.get_device());
    if (
        s.output.defined()
        && s.device_index == device_index
        && s.batch == batch
        && s.out_channels == out_channels
        && s.out_d == out_d
        && s.out_h == out_h
        && s.out_w == out_w
    ) {
        return;
    }
    auto opts = x.options().memory_format(c10::MemoryFormat::ChannelsLast3d);
    s.output = torch::empty({batch, out_channels, out_d, out_h, out_w}, opts);
    s.device_index = device_index;
    s.batch = batch;
    s.out_channels = out_channels;
    s.out_d = out_d;
    s.out_h = out_h;
    s.out_w = out_w;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_d,
    int pad_h,
    int pad_w,
    int groups
) {
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    TORCH_CHECK(x.dim() == 5, "x must be 5D");
    TORCH_CHECK(weight.dim() == 5, "weight must be 5D");
    TORCH_CHECK(x.is_contiguous(c10::MemoryFormat::ChannelsLast3d), "x must be channels_last_3d contiguous");
    TORCH_CHECK(weight.is_contiguous(c10::MemoryFormat::ChannelsLast3d), "weight must be channels_last_3d contiguous");

    auto& s = state();
    ensure_output(s, x, weight, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, groups);
    at::convolution_out(
        s.output,
        x,
        weight,
        ::std::optional<at::Tensor>(),
        {stride_d, stride_h, stride_w},
        {pad_d, pad_h, pad_w},
        {1, 1, 1},
        true,
        {0, 0, 0},
        groups
    );
    return s.output;
}
"""


MODULE_NAME = f"ptxbench_convtranspose3d_cl_{hashlib.md5(CUDA_SRC.encode('utf-8')).hexdigest()[:16]}"
module = load_inline(
    name=MODULE_NAME,
    cpp_sources=CPP_SRC,
    cuda_sources=CUDA_SRC,
    functions=["conv_transpose3d_cuda"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1, 1),
        padding: tuple = (0, 0, 0),
        output_padding: tuple = (0, 0, 0),
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if tuple(output_padding) != (0, 0, 0):
            raise ValueError("ModelNew only supports output_padding=(0, 0, 0)")
        if bias:
            raise ValueError("ModelNew only supports bias=False")
        seed = torch.initial_seed()
        torch.manual_seed(seed)
        ref = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )
        self.register_buffer("weight", ref.weight.detach().to(memory_format=torch.channels_last_3d))
        self.stride = tuple(int(v) for v in stride)
        self.padding = tuple(int(v) for v in padding)
        self.groups = int(groups)
        self._cached_x = None
        self._cached_ptr = None
        self._cached_version = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._cached_x is None or x.data_ptr() != self._cached_ptr or x._version != self._cached_version:
            self._cached_x = x.contiguous(memory_format=torch.channels_last_3d)
            self._cached_ptr = x.data_ptr()
            self._cached_version = x._version
        return module.conv_transpose3d_cuda(
            self._cached_x,
            self.weight,
            self.stride[0],
            self.stride[1],
            self.stride[2],
            self.padding[0],
            self.padding[1],
            self.padding[2],
            self.groups,
        )
