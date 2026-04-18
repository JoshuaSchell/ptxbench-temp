import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float value = x[idx];
        out[idx] = value > 0.0f ? value : 0.0f;
    }
}

torch::Tensor relu_cuda(torch::Tensor x) {
    auto out = torch::zeros_like(x);
    int n = static_cast<int>(x.numel());
    int block = 256;
    int grid = (n + block - 1) / block;
    relu_kernel<<<grid, block>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
"""


module = load_inline(
    name="ptxbench_relu_cuda",
    cpp_sources="torch::Tensor relu_cuda(torch::Tensor x);",
    cuda_sources=CUDA_SRC,
    functions=["relu_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return module.relu_cuda(x)
