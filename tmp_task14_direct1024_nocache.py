import torch
import torch.nn as nn
from ptxbench.runtime import PTXModuleRunner
from ptxbench.spec import PTXKernelSpec


PTX_SOURCES = {
    "direct": r"""
.version 8.0
.target sm_89
.address_size 64

.visible .entry direct_kernel(
    .param .u64 x_ptr,
    .param .u64 w_ptr,
    .param .u64 out_ptr,
    .param .u32 ncols,
    .param .u32 nrows,
    .param .f32 scale
)
{
    .reg .pred %p<6>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<20>;
    .reg .f32 %f<16>;

    .shared .align 4 .b8 smem[4096];

    ld.param.u64 %rd1, [x_ptr];
    ld.param.u64 %rd2, [w_ptr];
    ld.param.u64 %rd3, [out_ptr];
    ld.param.u32 %r1, [ncols];
    ld.param.u32 %r2, [nrows];
    ld.param.f32 %f1, [scale];

    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %tid.x;
    mov.u32 %r5, %ntid.x;

    setp.ge.u32 %p1, %r3, 1024;
    @%p1 bra DONE;

    mul.wide.u32 %rd4, %r3, %r1;
    shl.b64 %rd5, %rd4, 2;
    add.s64 %rd6, %rd1, %rd5;

    mov.f32 %f2, 0f00000000;
    mov.u32 %r6, %r4;

OUTER:
    setp.ge.u32 %p2, %r6, %r2;
    @%p2 bra OUTER_DONE;

    mul.wide.u32 %rd7, %r6, %r1;
    shl.b64 %rd8, %rd7, 2;
    add.s64 %rd9, %rd2, %rd8;
    mov.f32 %f3, 0f00000000;
    mov.u32 %r7, 0;

INNER:
    setp.ge.u32 %p3, %r7, %r1;
    @%p3 bra INNER_DONE;
    mul.wide.u32 %rd10, %r7, 4;
    add.s64 %rd11, %rd6, %rd10;
    add.s64 %rd12, %rd9, %rd10;
    ld.global.nc.f32 %f4, [%rd11];
    ld.global.nc.f32 %f5, [%rd12];
    fma.rn.f32 %f3, %f4, %f5, %f3;
    add.u32 %r7, %r7, 1;
    bra INNER;

INNER_DONE:
    add.f32 %f2, %f2, %f3;
    add.u32 %r6, %r6, %r5;
    bra OUTER;

OUTER_DONE:
    mov.u32 %r20, smem;
    shl.b32 %r21, %r4, 2;
    add.s32 %r22, %r20, %r21;
    st.shared.f32 [%r22], %f2;
    bar.sync 0;

    mov.u32 %r23, 512;
REDUCE:
    setp.lt.u32 %p4, %r4, %r23;
    @!%p4 bra NEXT;
    add.u32 %r24, %r4, %r23;
    shl.b32 %r25, %r24, 2;
    add.s32 %r26, %r20, %r25;
    ld.shared.f32 %f6, [%r22];
    ld.shared.f32 %f7, [%r26];
    add.f32 %f8, %f6, %f7;
    st.shared.f32 [%r22], %f8;
NEXT:
    bar.sync 0;
    shr.u32 %r23, %r23, 1;
    setp.ne.u32 %p5, %r23, 0;
    @%p5 bra REDUCE;

    setp.ne.u32 %p5, %r4, 0;
    @%p5 bra DONE;

    ld.shared.f32 %f9, [smem];
    mul.f32 %f10, %f9, %f1;
    mul.wide.u32 %rd13, %r3, 4;
    add.s64 %rd14, %rd3, %rd13;
    st.global.f32 [%rd14], %f10;

DONE:
    ret;
}
"""
}

PTX_KERNELS = {
    "direct": PTXKernelSpec(
        entry="direct_kernel",
        grid=lambda x, w, out, ncols, nrows, scale: (int(x.shape[0]), 1, 1),
        block=(1024, 1, 1),
        arg_types=("tensor", "tensor", "tensor", "uint32", "uint32", "float32"),
    ),
}


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        seed = torch.initial_seed()
        torch.manual_seed(seed)
        weight = torch.randn(hidden_size, input_size)
        self.weight = nn.Parameter(weight)
        self.scale = float(scaling_factor) * 0.5
        self.runner = PTXModuleRunner(PTX_SOURCES, PTX_KERNELS)

    def forward(self, x):
        out = torch.empty((x.shape[0], 1), device=x.device, dtype=x.dtype)
        self.runner.launch("direct", x, self.weight, out, x.shape[1], self.weight.shape[0], self.scale)
        return out
