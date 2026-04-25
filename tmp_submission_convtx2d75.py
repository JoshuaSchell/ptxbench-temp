import torch
import torch.nn as nn

from ptxbench.runtime import PTXModuleRunner
from ptxbench.spec import PTXKernelSpec

with open("/home/josh/code/ptxbench-temp/tmp_convtx2d75.ptx", "r", encoding="utf-8") as _f:
    _PTX = _f.read()

PTX_SOURCES = {
    "zero4": _PTX,
    "p0": _PTX,
    "p1": _PTX,
    "p2": _PTX,
}

PTX_KERNELS = {
    "zero4": PTXKernelSpec(
        entry="zero4_kernel",
        grid=lambda out, n4: ((int((n4 + 255) // 256), 1, 1)),
        block=(256, 1, 1),
        arg_types=("tensor", "uint32"),
    ),
    "p0": PTXKernelSpec(
        entry="deconv75_p0",
        grid=lambda x, wp, out: ((4096, 4, int(x.shape[0]))),
        block=(128, 1, 1),
        arg_types=("tensor", "tensor", "tensor"),
    ),
    "p1": PTXKernelSpec(
        entry="deconv75_p1",
        grid=lambda x, wp, out: ((4080, 4, int(x.shape[0]))),
        block=(128, 1, 1),
        arg_types=("tensor", "tensor", "tensor"),
    ),
    "p2": PTXKernelSpec(
        entry="deconv75_p2",
        grid=lambda x, wp, out: ((4080, 4, int(x.shape[0]))),
        block=(128, 1, 1),
        arg_types=("tensor", "tensor", "tensor"),
    ),
}


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert in_channels == 32
        assert out_channels == 64
        assert kernel_size == (3, 5)
        assert stride == (2, 3)
        assert padding == (1, 2)
        assert dilation == (2, 1)
        assert groups == 4
        assert not bias

        seed = torch.initial_seed()
        torch.manual_seed(seed)
        ref = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        w = ref.weight.detach().contiguous()
        pack = torch.empty((3, 4, 6, 8, 16), dtype=w.dtype)
        for g in range(4):
            ic0 = g * 8
            for ic in range(8):
                src_ic = ic0 + ic
                for oc in range(16):
                    pack[0, g, 0, ic, oc] = w[src_ic, oc, 0, 2]
                    pack[0, g, 1, ic, oc] = w[src_ic, oc, 1, 2]
                    pack[0, g, 2, ic, oc] = w[src_ic, oc, 2, 2]
                    pack[0, g, 3, ic, oc] = 0.0
                    pack[0, g, 4, ic, oc] = 0.0
                    pack[0, g, 5, ic, oc] = 0.0
                    pack[1, g, 0, ic, oc] = w[src_ic, oc, 0, 0]
                    pack[1, g, 1, ic, oc] = w[src_ic, oc, 0, 3]
                    pack[1, g, 2, ic, oc] = w[src_ic, oc, 1, 0]
                    pack[1, g, 3, ic, oc] = w[src_ic, oc, 1, 3]
                    pack[1, g, 4, ic, oc] = w[src_ic, oc, 2, 0]
                    pack[1, g, 5, ic, oc] = w[src_ic, oc, 2, 3]
                    pack[2, g, 0, ic, oc] = w[src_ic, oc, 0, 1]
                    pack[2, g, 1, ic, oc] = w[src_ic, oc, 0, 4]
                    pack[2, g, 2, ic, oc] = w[src_ic, oc, 1, 1]
                    pack[2, g, 3, ic, oc] = w[src_ic, oc, 1, 4]
                    pack[2, g, 4, ic, oc] = w[src_ic, oc, 2, 1]
                    pack[2, g, 5, ic, oc] = w[src_ic, oc, 2, 4]
        self.register_buffer("weight_pack", pack)
        self.runner = PTXModuleRunner(PTX_SOURCES, PTX_KERNELS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty((x.shape[0], 64, 257, 766), device=x.device, dtype=x.dtype)
        self.runner.launch("zero4", out, out.numel() // 4)
        self.runner.launch("p0", x, self.weight_pack, out)
        self.runner.launch("p1", x, self.weight_pack, out)
        self.runner.launch("p2", x, self.weight_pack, out)
        return out
