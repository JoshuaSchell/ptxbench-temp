import hashlib

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


CPP_SRC = r"""
void matmul_out_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C);
"""


CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>

#define CHECK_INPUT(x) TORCH_CHECK((x).is_cuda() && (x).is_contiguous() && (x).scalar_type() == at::kFloat, #x " must be a contiguous CUDA float32 tensor")
#define CHECK_CUBLAS(expr) TORCH_CHECK((expr) == CUBLAS_STATUS_SUCCESS, "cuBLASLt call failed: ", #expr)

namespace {

struct MatmulCache {
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatmulAlgo_t algo{};
    at::Tensor workspace;
    bool ready = false;
    int device = -1;
    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;
    size_t workspace_size = 0;

    MatmulCache() {
        CHECK_CUBLAS(cublasLtCreate(&handle));
    }

    ~MatmulCache() {
        if (op_desc != nullptr) {
            cublasLtMatmulDescDestroy(op_desc);
        }
        if (a_desc != nullptr) {
            cublasLtMatrixLayoutDestroy(a_desc);
        }
        if (b_desc != nullptr) {
            cublasLtMatrixLayoutDestroy(b_desc);
        }
        if (c_desc != nullptr) {
            cublasLtMatrixLayoutDestroy(c_desc);
        }
        if (handle != nullptr) {
            cublasLtDestroy(handle);
        }
    }

    void reset() {
        if (op_desc != nullptr) {
            cublasLtMatmulDescDestroy(op_desc);
            op_desc = nullptr;
        }
        if (a_desc != nullptr) {
            cublasLtMatrixLayoutDestroy(a_desc);
            a_desc = nullptr;
        }
        if (b_desc != nullptr) {
            cublasLtMatrixLayoutDestroy(b_desc);
            b_desc = nullptr;
        }
        if (c_desc != nullptr) {
            cublasLtMatrixLayoutDestroy(c_desc);
            c_desc = nullptr;
        }
        workspace = at::Tensor();
        workspace_size = 0;
        ready = false;
    }
};

MatmulCache& cache() {
    static MatmulCache value;
    return value;
}

void ensure_initialized(const at::Tensor& A, const at::Tensor& B) {
    auto& state = cache();
    const int device = A.get_device();
    const auto m = A.size(0);
    const auto k = A.size(1);
    const auto n = B.size(1);

    if (state.ready && state.device == device && state.m == m && state.n == n && state.k == k) {
        return;
    }

    state.reset();
    state.device = device;
    state.m = m;
    state.n = n;
    state.k = k;

    CHECK_CUBLAS(cublasLtMatmulDescCreate(&state.op_desc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));

    cublasOperation_t trans = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        state.op_desc,
        CUBLASLT_MATMUL_DESC_TRANSA,
        &trans,
        sizeof(trans)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        state.op_desc,
        CUBLASLT_MATMUL_DESC_TRANSB,
        &trans,
        sizeof(trans)));

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&state.a_desc, CUDA_R_32F, m, k, k));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&state.b_desc, CUDA_R_32F, k, n, n));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&state.c_desc, CUDA_R_32F, m, n, n));

    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
        state.a_desc,
        CUBLASLT_MATRIX_LAYOUT_ORDER,
        &order,
        sizeof(order)));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
        state.b_desc,
        CUBLASLT_MATRIX_LAYOUT_ORDER,
        &order,
        sizeof(order)));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
        state.c_desc,
        CUBLASLT_MATRIX_LAYOUT_ORDER,
        &order,
        sizeof(order)));

    cublasLtMatmulPreference_t pref = nullptr;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
    size_t max_workspace = 64ull << 20;
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &max_workspace,
        sizeof(max_workspace)));

    cublasLtMatmulHeuristicResult_t heuristics[32];
    int returned = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
        state.handle,
        state.op_desc,
        state.a_desc,
        state.b_desc,
        state.c_desc,
        state.c_desc,
        pref,
        32,
        heuristics,
        &returned));
    CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(pref));
    TORCH_CHECK(returned > 0, "No cuBLASLt algorithm found");

    state.algo = heuristics[0].algo;
    if (heuristics[0].workspaceSize > 0) {
        state.workspace = at::empty(
            {static_cast<long long>(heuristics[0].workspaceSize)},
            A.options().dtype(at::kByte));
        state.workspace_size = heuristics[0].workspaceSize;
    }
    state.ready = true;
}

}  // namespace

void matmul_out_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && C.dim() == 2, "expected 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "dimension mismatch");
    TORCH_CHECK(C.size(0) == A.size(0) && C.size(1) == B.size(1), "output shape mismatch");

    c10::cuda::CUDAGuard device_guard(A.device());
    ensure_initialized(A, B);
    auto& state = cache();
    auto stream = at::cuda::getCurrentCUDAStream(A.get_device());
    const float alpha = 1.0f;
    const float beta = 0.0f;
    void* workspace_ptr = state.workspace_size > 0 ? state.workspace.data_ptr() : nullptr;

    CHECK_CUBLAS(cublasLtMatmul(
        state.handle,
        state.op_desc,
        &alpha,
        A.data_ptr<float>(),
        state.a_desc,
        B.data_ptr<float>(),
        state.b_desc,
        &beta,
        C.data_ptr<float>(),
        state.c_desc,
        C.data_ptr<float>(),
        state.c_desc,
        &state.algo,
        workspace_ptr,
        state.workspace_size,
        stream.stream()));
}
"""


MODULE_NAME = f"ptxbench_matmul_tf32_{hashlib.md5(CUDA_SRC.encode('utf-8')).hexdigest()[:16]}"

module = load_inline(
    name=MODULE_NAME,
    cpp_sources=CPP_SRC,
    cuda_sources=CUDA_SRC,
    functions=["matmul_out_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    extra_ldflags=["-lcublasLt"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._output_2d = None
        self._output_shape = None
        self._output_dtype = None
        self._output_device = None

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        n, m, k = A.shape
        l = B.shape[1]
        flat_rows = n * m
        out_shape = (flat_rows, l)
        device = A.device
        dtype = A.dtype

        if (
            self._output_2d is None
            or self._output_shape != out_shape
            or self._output_dtype != dtype
            or self._output_device != device
        ):
            self._output_2d = torch.empty(out_shape, device=device, dtype=dtype)
            self._output_shape = out_shape
            self._output_dtype = dtype
            self._output_device = device

        module.matmul_out_cuda(A.view(flat_rows, k), B, self._output_2d)
        return self._output_2d.view(n, m, l)
