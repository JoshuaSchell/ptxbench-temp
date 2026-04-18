import torch
import torch.nn as nn

from ptxbench.runtime import PTXModuleRunner
from ptxbench.spec import PTXKernelSpec


PTX_SOURCES = {
    "relu": r"""
.version 8.0
.target sm_89
.address_size 64

.visible .entry relu_kernel(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u32 n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<6>;
    .reg .f32 %f<3>;

    ld.param.u64 %rd1, [input_ptr];
    ld.param.u64 %rd2, [output_ptr];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mad.lo.s32 %r5, %r3, %r4, %r2;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE;

    mul.wide.u32 %rd3, %r5, 4;
    add.s64 %rd4, %rd1, %rd3;
    add.s64 %rd5, %rd2, %rd3;
    ld.global.f32 %f1, [%rd4];
    max.f32 %f2, %f1, 0f00000000;
    st.global.f32 [%rd5], %f2;

DONE:
    ret;
}
"""
}


PTX_KERNELS = {
    "relu": PTXKernelSpec(
        entry="relu_kernel",
        grid=lambda x, out, n: ((int((n + 255) // 256), 1, 1)),
        block=(256, 1, 1),
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
