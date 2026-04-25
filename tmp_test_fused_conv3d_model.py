import torch
import torch.nn as nn
from ptxbench.runtime import PTXModuleRunner
from ptxbench.spec import PTXKernelSpec


with open("/home/josh/code/ptxbench-temp/tmp_fused_conv3d_relu_gelu_sigmoid_bias.ptx", "r", encoding="utf-8") as f:
    _PTX = f.read()


PTX_SOURCES = {"fused": _PTX}

PTX_KERNELS = {
    "fused": PTXKernelSpec(
        entry="fused_conv3d_relu_gelu_sigmoid_bias_kernel",
        grid=lambda x, w, cb, ob, out: (8, 16, int(x.shape[0]) * 8 * 4),
        block=(8, 4, 4),
        arg_types=("tensor", "tensor", "tensor", "tensor", "tensor"),
    ),
}


class Ref(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(8, 32, 3)
        self.bias = nn.Parameter(torch.randn(32, 1, 1, 1))

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        x = torch.nn.functional.gelu(x)
        x = torch.sigmoid(x)
        x = x + self.bias
        return x


class Cand(nn.Module):
    def __init__(self):
        super().__init__()
        seed = torch.initial_seed()
        torch.manual_seed(seed)
        ref = Ref()
        self.register_buffer(
            "w",
            ref.conv.weight.detach().view(4, 8, 8, 3, 3, 3).permute(0, 3, 4, 5, 2, 1).contiguous(),
        )
        self.register_buffer("conv_bias", ref.conv.bias.detach().contiguous())
        self.register_buffer("bias", ref.bias.detach().reshape(32).contiguous())
        self.runner = PTXModuleRunner(PTX_SOURCES, PTX_KERNELS)
        self._out = None

    def forward(self, x):
        if self._out is None or self._out.shape[0] != x.shape[0] or self._out.device != x.device or self._out.dtype != x.dtype:
            self._out = torch.empty((x.shape[0], 32, 30, 62, 62), device=x.device, dtype=x.dtype)
        self.runner.launch("fused", x, self.w, self.conv_bias, self.bias, self._out)
        return self._out
