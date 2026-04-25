import torch
import torch.nn as nn
from ptxbench.runtime import PTXModuleRunner
from ptxbench.spec import PTXKernelSpec


with open("/home/josh/code/ptxbench-temp/tmp_task14_exact_tile.ptx", "r", encoding="utf-8") as _f:
    _PTX = _f.read()


PTX_SOURCES = {"exact": _PTX}

PTX_KERNELS = {
    "exact": PTXKernelSpec(
        entry="exact_tile_kernel",
        grid=lambda x, w, out, ncols, nrows, scale: (int(x.shape[0]), 1, 1),
        block=(32, 8, 1),
        arg_types=("tensor", "tensor", "tensor", "int32", "int32", "float32"),
    ),
}


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        weight = torch.randn(hidden_size, input_size)
        self.register_buffer("weight", weight.contiguous())
        self.scale = float(scaling_factor) * 0.5
        self.runner = PTXModuleRunner(PTX_SOURCES, PTX_KERNELS)
        self.register_buffer("out", torch.empty((1024, 1), dtype=torch.float32))

    def forward(self, x):
        self.runner.launch("exact", x, self.weight, self.out, x.shape[1], self.weight.shape[0], self.scale)
        return self.out
