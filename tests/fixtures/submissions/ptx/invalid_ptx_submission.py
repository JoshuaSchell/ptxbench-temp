import torch
import torch.nn as nn

from ptxbench.runtime import PTXModuleRunner
from ptxbench.spec import PTXKernelSpec


PTX_SOURCES = {
    "relu": ".version 8.0\n.target sm_89\n.visible .entry broken_kernel(\n"
}

PTX_KERNELS = {
    "relu": PTXKernelSpec(
        entry="broken_kernel",
        grid=(1, 1, 1),
        block=(1, 1, 1),
        arg_types=("tensor", "tensor", "uint32"),
    )
}


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.runner = PTXModuleRunner(PTX_SOURCES, PTX_KERNELS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        self.runner.launch("relu", x, out, x.numel())
        return out
