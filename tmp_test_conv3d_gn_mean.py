import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptxbench.runtime import PTXModuleRunner
from ptxbench.spec import PTXKernelSpec


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.hardswish(x)
        x = self.group_norm(x)
        x = torch.mean(x, dim=[2, 3, 4])
        return x


PTX_SOURCES = {
    "conv": open("tmp_conv3d_gn_mean_kernel.ptx", "r", encoding="utf-8").read(),
    "finalize": open("tmp_conv3d_gn_mean_kernel.ptx", "r", encoding="utf-8").read(),
}

PTX_KERNELS = {
    "conv": PTXKernelSpec(
        entry="conv_hswish_channel_sums_kernel",
        grid=lambda x, w, b, sums, sqs, batch: (int(batch) * 16, 1, 1),
        block=(256, 1, 1),
        arg_types=("tensor", "tensor", "tensor", "tensor", "tensor", "uint32"),
    ),
    "finalize": PTXKernelSpec(
        entry="finalize_groupnorm_mean_kernel",
        grid=lambda sums, sqs, gw, gb, out, batch: (int(batch), 1, 1),
        block=(32, 1, 1),
        arg_types=("tensor", "tensor", "tensor", "tensor", "tensor", "uint32"),
    ),
}


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super().__init__()
        seed = torch.initial_seed()
        torch.manual_seed(seed)
        ref = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias).to(device="cuda", dtype=torch.float32)
        gn = nn.GroupNorm(num_groups, out_channels).to(device="cuda", dtype=torch.float32)
        self.register_buffer("weight", ref.weight.detach().contiguous())
        if ref.bias is None:
            self.register_buffer("conv_bias", torch.zeros((out_channels,), device="cuda", dtype=torch.float32))
        else:
            self.register_buffer("conv_bias", ref.bias.detach().contiguous())
        self.register_buffer("gn_weight", gn.weight.detach().contiguous())
        self.register_buffer("gn_bias", gn.bias.detach().contiguous())
        self.register_buffer("sums", torch.empty((1024, 16), device="cuda", dtype=torch.float32))
        self.register_buffer("sqs", torch.empty((1024, 16), device="cuda", dtype=torch.float32))
        self.runner = PTXModuleRunner(PTX_SOURCES, PTX_KERNELS)

    def forward(self, x):
        out = torch.empty((x.shape[0], 16), device=x.device, dtype=x.dtype)
        self.runner.launch("conv", x, self.weight, self.conv_bias, self.sums, self.sqs, x.shape[0])
        self.runner.launch("finalize", self.sums, self.sqs, self.gn_weight, self.gn_bias, out, x.shape[0])
        return out


def bench(fn, x, iters=20, warm=5):
    for _ in range(warm):
        y = fn(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        y = fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / iters


def main():
    device = "cuda"
    torch.manual_seed(0)
    ref = Model(3, 16, 4).to(device=device, dtype=torch.float32)
    cand = ModelNew(3, 16, 4).to(device=device, dtype=torch.float32)
    x = torch.rand((1024, 3, 16, 32, 32), device=device, dtype=torch.float32)
    with torch.no_grad():
        y_ref = ref(x)
        y_cand = cand(x)
        diff = (y_ref - y_cand).abs()
        print("max", diff.max().item(), "mean", diff.mean().item(), "allclose", torch.allclose(y_ref, y_cand, atol=1e-4, rtol=1e-4))
        print("ref", bench(ref, x), "cand", bench(cand, x))


if __name__ == "__main__":
    main()
