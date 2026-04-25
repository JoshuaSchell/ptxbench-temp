import hashlib
import time

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


CPP_SRC = r"""
torch::Tensor conv2d_cuda(torch::Tensor x, torch::Tensor weight);
"""


CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/ops/cudnn_convolution.h>

#define CHECK_INPUT(x) \
    TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor"); \
    TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")

struct ConvState {
    at::Tensor output;
    int64_t device_index = -1;
    int64_t batch = -1;
    int64_t height = -1;
    int64_t width = -1;
};

static ConvState& state() {
    static ConvState value;
    return value;
}

static void ensure_output(ConvState& s, const torch::Tensor& x) {
    const auto batch = x.size(0);
    const auto height = x.size(2);
    const auto width = x.size(3);
    const auto out_h = height - 5 + 1;
    const auto out_w = width - 7 + 1;
    const auto device_index = static_cast<int64_t>(x.get_device());
    if (s.output.defined() && s.device_index == device_index && s.batch == batch && s.height == height && s.width == width) {
        return;
    }
    s.output = torch::empty({batch, 128, out_h, out_w}, x.options());
    s.device_index = device_index;
    s.batch = batch;
    s.height = height;
    s.width = width;
}

torch::Tensor conv2d_cuda(torch::Tensor x, torch::Tensor weight) {
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    auto& s = state();
    ensure_output(s, x);
    at::cudnn_convolution_out(
        s.output,
        x,
        weight,
        {0, 0},
        {1, 1},
        {1, 1},
        1,
        true,
        false,
        true
    );
    return s.output;
}
"""


MODULE_NAME = f"ptxbench_conv2d_nchw_out_{hashlib.md5(CUDA_SRC.encode('utf-8')).hexdigest()[:16]}"
module = load_inline(
    name=MODULE_NAME,
    cpp_sources=CPP_SRC,
    cuda_sources=CUDA_SRC,
    functions=["conv2d_cuda"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        seed = torch.initial_seed()
        torch.manual_seed(seed)
        ref = nn.Conv2d(64, 128, (5, 7), bias=False)
        self.register_buffer("weight", ref.weight.detach().cuda().contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return module.conv2d_cuda(x, self.weight)


def _bench():
    torch.manual_seed(0)
    x = torch.rand(8, 64, 512, 256, device="cuda")
    seed = torch.initial_seed()
    torch.manual_seed(seed)
    ref = nn.Conv2d(64, 128, (5, 7), bias=False).cuda()
    torch.manual_seed(seed)
    model = ModelNew().cuda()
    y0 = ref(x)
    y1 = model(x)
    print("max abs diff", (y0 - y1).abs().max().item())

    for _ in range(10):
        ref(x)
        model(x)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(30):
        ref(x)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    for _ in range(30):
        model(x)
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    print("eager", (t1 - t0) / 30 * 1000)
    print("model", (t3 - t2) / 30 * 1000)


if __name__ == "__main__":
    _bench()
