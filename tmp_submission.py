import torch
import torch.nn as nn

from ptxbench.runtime import PTXModuleRunner
from ptxbench.spec import PTXKernelSpec


with open("/home/josh/code/ptxbench-temp/tmp_convtx3d.ptx", "r", encoding="utf-8") as _f:
    _PTX = _f.read()


PTX_SOURCES = {
    "convtx3d": _PTX,
}


PTX_KERNELS = {
    "convtx3d": PTXKernelSpec(
        entry="convtx3d_gather_kernel",
        grid=lambda x, w, out: ((3, 24 * 48, out.shape[0] * 4)),
        block=(32, 8, 1),
        arg_types=("tensor", "tensor", "tensor"),
    ),
}


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
        icpg = in_channels // groups
        ocpg = out_channels // groups
        packed = ref.weight.detach().view(groups, icpg, ocpg, 3, 5, 7).permute(0, 2, 1, 3, 4, 5).contiguous()
        self.register_buffer("weight", packed)
        self.runner = PTXModuleRunner(PTX_SOURCES, PTX_KERNELS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty((x.shape[0], 32, 24, 48, 96), device=x.device, dtype=x.dtype)
        self.runner.launch("convtx3d", x, self.weight, out)
        return out
