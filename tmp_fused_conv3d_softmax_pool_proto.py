import hashlib
import time

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


CPP_SRC = r"""
torch::Tensor fused_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor out);
"""


CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_INPUT(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor"); TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")

__global__ void fused_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out
) {
    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;
    const int ow2 = block_id % 7;
    const int t0 = block_id / 7;
    const int oh2 = t0 % 7;
    const int t1 = t0 / 7;
    const int od2 = t1 % 3;
    const int n = t1 / 3;

    const int d0 = od2 * 4;
    const int h0 = oh2 * 4;
    const int w0 = ow2 * 4;

    __shared__ float sx[648];
    __shared__ float sw[1296];
    __shared__ float sb[16];
    __shared__ float sprobs[1024];

    for (int i = tid; i < 648; i += 64) {
        const int c = i / 216;
        const int rem = i - c * 216;
        const int d = rem / 36;
        const int rem2 = rem - d * 36;
        const int h = rem2 / 6;
        const int ww = rem2 - h * 6;
        const int x_idx = (((n * 3 + c) * 16 + (d0 + d)) * 32 + (h0 + h)) * 32 + (w0 + ww);
        sx[i] = x[x_idx];
    }
    for (int i = tid; i < 1296; i += 64) {
        sw[i] = w[i];
    }
    for (int i = tid; i < 16; i += 64) {
        sb[i] = b[i];
    }
    __syncthreads();

    if (tid < 64) {
        const int pd = tid >> 4;
        const int rem = tid & 15;
        const int ph = rem >> 2;
        const int pw = rem & 3;
        float acc[16];
        #pragma unroll
        for (int oc = 0; oc < 16; ++oc) {
            acc[oc] = sb[oc];
        }
        #pragma unroll
        for (int ic = 0; ic < 3; ++ic) {
            #pragma unroll
            for (int kd = 0; kd < 3; ++kd) {
                #pragma unroll
                for (int kh = 0; kh < 3; ++kh) {
                    #pragma unroll
                    for (int kw = 0; kw < 3; ++kw) {
                        const int xv = ((ic * 6 + (pd + kd)) * 6 + (ph + kh)) * 6 + (pw + kw);
                        const float xval = sx[xv];
                        const int wbase = (((ic * 3 + kd) * 3 + kh) * 3 + kw);
                        #pragma unroll
                        for (int oc = 0; oc < 16; ++oc) {
                            acc[oc] = fmaf(xval, sw[oc * 81 + wbase], acc[oc]);
                        }
                    }
                }
            }
        }

        float vmax = acc[0];
        #pragma unroll
        for (int oc = 1; oc < 16; ++oc) {
            vmax = vmax > acc[oc] ? vmax : acc[oc];
        }
        float esum = 0.0f;
        float probs[16];
        #pragma unroll
        for (int oc = 0; oc < 16; ++oc) {
            const float p = expf(acc[oc] - vmax);
            probs[oc] = p;
            esum += p;
        }
        const float inv = 1.0f / esum;
        #pragma unroll
        for (int oc = 0; oc < 16; ++oc) {
            sprobs[oc * 64 + tid] = probs[oc] * inv;
        }
    }
    __syncthreads();

    if (tid < 16) {
        float vmax = sprobs[tid * 64];
        #pragma unroll
        for (int i = 1; i < 64; ++i) {
            const float v = sprobs[tid * 64 + i];
            vmax = vmax > v ? vmax : v;
        }
        const int out_idx = ((((n * 16 + tid) * 3 + od2) * 7 + oh2) * 7 + ow2);
        out[out_idx] = vmax;
    }
}

torch::Tensor fused_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor out) {
    CHECK_INPUT(x);
    CHECK_INPUT(w);
    CHECK_INPUT(b);
    CHECK_INPUT(out);
    fused_kernel<<<dim3((unsigned)(x.size(0) * 147)), dim3(64)>>>(x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>());
    return out;
}
"""


module = load_inline(
    name=f"tmp_fused_conv3d_softmax_pool_{hashlib.md5(CUDA_SRC.encode()).hexdigest()[:16]}",
    cpp_sources=CPP_SRC,
    cuda_sources=CUDA_SRC,
    functions=["fused_cuda"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class Ref(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(3, 16, 3)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.conv(x)
        x = torch.softmax(x, dim=1)
        x = self.pool1(x)
        x = self.pool2(x)
        return x


class Cand(nn.Module):
    def __init__(self):
        super().__init__()
        seed = torch.initial_seed()
        torch.manual_seed(seed)
        ref = nn.Conv3d(3, 16, 3)
        self.register_buffer("w", ref.weight.detach().contiguous())
        self.register_buffer("b", ref.bias.detach().contiguous())
        self.out = None
        self.batch = None

    def forward(self, x):
        if self.out is None or self.batch != x.shape[0]:
            self.out = torch.empty((x.shape[0], 16, 3, 7, 7), device=x.device, dtype=x.dtype)
            self.batch = x.shape[0]
        return module.fused_cuda(x, self.w, self.b, self.out)


def bench(fn, x, iters=50, warmup=10):
    for _ in range(warmup):
        y = fn(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        y = fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters, y


def main():
    device = "cuda"
    seed = 1234
    torch.manual_seed(seed)
    ref = Ref().to(device=device, dtype=torch.float32)
    cand = Cand().to(device=device, dtype=torch.float32)
    x = torch.rand(128, 3, 16, 32, 32, device=device)
    y0 = ref(x)
    y1 = cand(x)
    print("shape", y0.shape, y1.shape)
    print("max abs", (y0 - y1).abs().max().item())
    print("allclose", torch.allclose(y0, y1, atol=1e-4, rtol=1e-4))
    t_ref, _ = bench(ref, x)
    t_cand, _ = bench(cand, x)
    print("ref ms", t_ref)
    print("cand ms", t_cand)


if __name__ == "__main__":
    main()
