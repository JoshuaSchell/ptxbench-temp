import hashlib

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


CPP_SRC = r"""
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""


CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <limits>
#include <memory>
#include <mutex>
#include <vector>

#define CHECK_INPUT(x) TORCH_CHECK((x).is_cuda() && (x).is_contiguous() && (x).scalar_type() == at::kFloat, #x " must be a contiguous CUDA float32 tensor")
#define CHECK_CUBLAS(expr) TORCH_CHECK((expr) == CUBLAS_STATUS_SUCCESS, "cuBLASLt call failed: ", #expr)
#define CHECK_CUDA(expr) TORCH_CHECK((expr) == cudaSuccess, "CUDA call failed: ", #expr)

namespace {

struct LtHandleCache {
    std::vector<cublasLtHandle_t> handles;
    std::mutex mutex;

    cublasLtHandle_t get(int device) {
        std::lock_guard<std::mutex> guard(mutex);
        if (device >= static_cast<int>(handles.size())) {
            handles.resize(device + 1, nullptr);
        }
        if (handles[device] == nullptr) {
            CHECK_CUBLAS(cublasLtCreate(&handles[device]));
        }
        return handles[device];
    }

    ~LtHandleCache() {
        for (auto handle : handles) {
            if (handle != nullptr) {
                cublasLtDestroy(handle);
            }
        }
    }
};

LtHandleCache& handle_cache() {
    static LtHandleCache cache;
    return cache;
}

struct Plan {
    bool ready = false;
    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;
    size_t workspace_size = 0;
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatmulAlgo_t algo{};
    at::Tensor workspace;

    void clear() {
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

    ~Plan() {
        clear();
    }
};

std::vector<std::unique_ptr<Plan>>& plan_cache() {
    static std::vector<std::unique_ptr<Plan>> cache;
    return cache;
}

std::mutex& plan_mutex() {
    static std::mutex mutex;
    return mutex;
}

double benchmark_algo(
    cublasLtHandle_t handle,
    cublasLtMatmulDesc_t op_desc,
    cublasLtMatrixLayout_t a_desc,
    cublasLtMatrixLayout_t b_desc,
    cublasLtMatrixLayout_t c_desc,
    const at::Tensor& A,
    const at::Tensor& B,
    const cublasLtMatmulAlgo_t& algo,
    void* workspace_ptr,
    size_t workspace_size
) {
    auto scratch = at::empty({A.size(0), B.size(1)}, A.options());
    auto stream = at::cuda::getCurrentCUDAStream(A.get_device());
    const float alpha = 1.0f;
    const float beta = 0.0f;

    for (int i = 0; i < 2; ++i) {
        CHECK_CUBLAS(cublasLtMatmul(
            handle,
            op_desc,
            &alpha,
            A.data_ptr<float>(),
            a_desc,
            B.data_ptr<float>(),
            b_desc,
            &beta,
            scratch.data_ptr<float>(),
            c_desc,
            scratch.data_ptr<float>(),
            c_desc,
            &algo,
            workspace_ptr,
            workspace_size,
            stream.stream()));
    }

    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    CHECK_CUDA(cudaEventCreate(&start_event));
    CHECK_CUDA(cudaEventCreate(&stop_event));
    CHECK_CUDA(cudaEventRecord(start_event, stream.stream()));

    constexpr int iters = 5;
    for (int i = 0; i < iters; ++i) {
        CHECK_CUBLAS(cublasLtMatmul(
            handle,
            op_desc,
            &alpha,
            A.data_ptr<float>(),
            a_desc,
            B.data_ptr<float>(),
            b_desc,
            &beta,
            scratch.data_ptr<float>(),
            c_desc,
            scratch.data_ptr<float>(),
            c_desc,
            &algo,
            workspace_ptr,
            workspace_size,
            stream.stream()));
    }

    CHECK_CUDA(cudaEventRecord(stop_event, stream.stream()));
    CHECK_CUDA(cudaEventSynchronize(stop_event));
    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
    CHECK_CUDA(cudaEventDestroy(start_event));
    CHECK_CUDA(cudaEventDestroy(stop_event));
    return static_cast<double>(elapsed_ms) / static_cast<double>(iters);
}

Plan& get_plan(const at::Tensor& A, const at::Tensor& B) {
    const auto m = A.size(0);
    const auto k = A.size(1);
    const auto n = B.size(1);
    const auto device = A.get_device();

    std::lock_guard<std::mutex> guard(plan_mutex());
    auto& cache = plan_cache();
    if (device >= static_cast<int>(cache.size())) {
        cache.resize(device + 1);
    }
    if (!cache[device]) {
        cache[device] = std::make_unique<Plan>();
    }

    Plan& plan = *cache[device];
    if (plan.ready && plan.m == m && plan.n == n && plan.k == k) {
        return plan;
    }

    plan.clear();
    plan.m = m;
    plan.n = n;
    plan.k = k;

    cublasLtHandle_t handle = handle_cache().get(device);
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&plan.op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasOperation_t trans = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(plan.op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(plan.op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans)));

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan.a_desc, CUDA_R_32F, m, k, k));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan.b_desc, CUDA_R_32F, k, n, n));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan.c_desc, CUDA_R_32F, m, n, n));

    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(plan.a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(plan.b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(plan.c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

    cublasLtMatmulPreference_t pref;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
    size_t max_workspace = 64ull << 20;
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &max_workspace,
        sizeof(max_workspace)));

    cublasLtMatmulHeuristicResult_t heuristics[16];
    int returned = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
        handle,
        plan.op_desc,
        plan.a_desc,
        plan.b_desc,
        plan.c_desc,
        plan.c_desc,
        pref,
        16,
        heuristics,
        &returned));
    CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(pref));
    TORCH_CHECK(returned > 0, "No cuBLASLt algorithm found for the requested GEMM");

    double best_ms = std::numeric_limits<double>::infinity();
    int best_idx = -1;
    at::Tensor best_workspace;

    for (int i = 0; i < returned; ++i) {
        if (heuristics[i].state != CUBLAS_STATUS_SUCCESS) {
            continue;
        }
        at::Tensor workspace;
        void* workspace_ptr = nullptr;
        if (heuristics[i].workspaceSize > 0) {
            workspace = at::empty({static_cast<long long>(heuristics[i].workspaceSize)}, A.options().dtype(at::kByte));
            workspace_ptr = workspace.data_ptr();
        }
        const double elapsed_ms = benchmark_algo(
            handle,
            plan.op_desc,
            plan.a_desc,
            plan.b_desc,
            plan.c_desc,
            A,
            B,
            heuristics[i].algo,
            workspace_ptr,
            heuristics[i].workspaceSize);
        if (elapsed_ms < best_ms) {
            best_ms = elapsed_ms;
            best_idx = i;
            best_workspace = workspace;
        }
    }

    TORCH_CHECK(best_idx >= 0, "No successful cuBLASLt algorithm survived autotuning");
    plan.algo = heuristics[best_idx].algo;
    plan.workspace_size = heuristics[best_idx].workspaceSize;
    plan.workspace = best_workspace;
    plan.ready = true;
    return plan;
}

}  // namespace

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "expected 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "dimension mismatch");

    auto C = at::empty({A.size(0), B.size(1)}, A.options());
    auto& plan = get_plan(A, B);
    auto handle = handle_cache().get(A.get_device());
    auto stream = at::cuda::getCurrentCUDAStream(A.get_device());

    const float alpha = 1.0f;
    const float beta = 0.0f;
    void* workspace_ptr = plan.workspace_size > 0 ? plan.workspace.data_ptr() : nullptr;

    CHECK_CUBLAS(cublasLtMatmul(
        handle,
        plan.op_desc,
        &alpha,
        A.data_ptr<float>(),
        plan.a_desc,
        B.data_ptr<float>(),
        plan.b_desc,
        &beta,
        C.data_ptr<float>(),
        plan.c_desc,
        C.data_ptr<float>(),
        plan.c_desc,
        &plan.algo,
        workspace_ptr,
        plan.workspace_size,
        stream.stream()));

    return C;
}
"""


_MODULE_NAME = f"ptxbench_cuda_gemm_{hashlib.md5(CUDA_SRC.encode('utf-8')).hexdigest()[:16]}"

_module = load_inline(
    name=_MODULE_NAME,
    cpp_sources=CPP_SRC,
    cuda_sources=CUDA_SRC,
    functions=["matmul_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    extra_ldflags=["-lcublasLt"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return _module.matmul_cuda(A, B)
