import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>

#define CHECK_INPUT(x) \
    TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor"); \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous"); \
    TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")

static inline void check_cublas_lt(cublasStatus_t status, const char* msg) {
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, msg, " failed with status ", static_cast<int>(status));
}

struct BmmCache {
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulAlgo_t algo{};
    at::Tensor workspace;
    at::Tensor output;
    int64_t device_index = -1;
    int64_t batch = -1;
    int64_t m = -1;
    int64_t n = -1;
    int64_t k = -1;
    bool initialized = false;

    BmmCache() {
        check_cublas_lt(cublasLtCreate(&handle), "cublasLtCreate");
    }

    ~BmmCache() {
        if (pref) {
            cublasLtMatmulPreferenceDestroy(pref);
        }
        if (a_desc) {
            cublasLtMatrixLayoutDestroy(a_desc);
        }
        if (b_desc) {
            cublasLtMatrixLayoutDestroy(b_desc);
        }
        if (c_desc) {
            cublasLtMatrixLayoutDestroy(c_desc);
        }
        if (op_desc) {
            cublasLtMatmulDescDestroy(op_desc);
        }
        if (handle) {
            cublasLtDestroy(handle);
        }
    }
};

static BmmCache& cache() {
    static BmmCache state;
    return state;
}

static void reset_descriptors(BmmCache& state) {
    if (state.pref) {
        cublasLtMatmulPreferenceDestroy(state.pref);
        state.pref = nullptr;
    }
    if (state.a_desc) {
        cublasLtMatrixLayoutDestroy(state.a_desc);
        state.a_desc = nullptr;
    }
    if (state.b_desc) {
        cublasLtMatrixLayoutDestroy(state.b_desc);
        state.b_desc = nullptr;
    }
    if (state.c_desc) {
        cublasLtMatrixLayoutDestroy(state.c_desc);
        state.c_desc = nullptr;
    }
    if (state.op_desc) {
        cublasLtMatmulDescDestroy(state.op_desc);
        state.op_desc = nullptr;
    }
}

static void ensure_initialized(
    int64_t device_index,
    int64_t batch,
    int64_t m,
    int64_t n,
    int64_t k,
    const at::TensorOptions& options
) {
    auto& state = cache();
    if (
        state.initialized &&
        state.device_index == device_index &&
        state.batch == batch &&
        state.m == m &&
        state.n == n &&
        state.k == k
    ) {
        return;
    }

    reset_descriptors(state);

    check_cublas_lt(
        cublasLtMatmulDescCreate(&state.op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
        "cublasLtMatmulDescCreate"
    );

    cublasOperation_t trans = CUBLAS_OP_N;
    check_cublas_lt(
        cublasLtMatmulDescSetAttribute(state.op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)),
        "set TRANSA"
    );
    check_cublas_lt(
        cublasLtMatmulDescSetAttribute(state.op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans)),
        "set TRANSB"
    );

    check_cublas_lt(cublasLtMatrixLayoutCreate(&state.a_desc, CUDA_R_32F, m, k, k), "create A layout");
    check_cublas_lt(cublasLtMatrixLayoutCreate(&state.b_desc, CUDA_R_32F, k, n, n), "create B layout");
    check_cublas_lt(cublasLtMatrixLayoutCreate(&state.c_desc, CUDA_R_32F, m, n, n), "create C layout");

    cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
    check_cublas_lt(
        cublasLtMatrixLayoutSetAttribute(state.a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)),
        "set A order"
    );
    check_cublas_lt(
        cublasLtMatrixLayoutSetAttribute(state.b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)),
        "set B order"
    );
    check_cublas_lt(
        cublasLtMatrixLayoutSetAttribute(state.c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)),
        "set C order"
    );

    int batch_count = static_cast<int>(batch);
    long long stride_a = m * k;
    long long stride_b = k * n;
    long long stride_c = m * n;
    check_cublas_lt(
        cublasLtMatrixLayoutSetAttribute(
            state.a_desc,
            CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
            &batch_count,
            sizeof(batch_count)
        ),
        "set A batch count"
    );
    check_cublas_lt(
        cublasLtMatrixLayoutSetAttribute(
            state.b_desc,
            CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
            &batch_count,
            sizeof(batch_count)
        ),
        "set B batch count"
    );
    check_cublas_lt(
        cublasLtMatrixLayoutSetAttribute(
            state.c_desc,
            CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
            &batch_count,
            sizeof(batch_count)
        ),
        "set C batch count"
    );
    check_cublas_lt(
        cublasLtMatrixLayoutSetAttribute(
            state.a_desc,
            CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &stride_a,
            sizeof(stride_a)
        ),
        "set A stride"
    );
    check_cublas_lt(
        cublasLtMatrixLayoutSetAttribute(
            state.b_desc,
            CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &stride_b,
            sizeof(stride_b)
        ),
        "set B stride"
    );
    check_cublas_lt(
        cublasLtMatrixLayoutSetAttribute(
            state.c_desc,
            CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &stride_c,
            sizeof(stride_c)
        ),
        "set C stride"
    );

    check_cublas_lt(cublasLtMatmulPreferenceCreate(&state.pref), "create matmul preference");

    state.workspace = torch::empty(
        {64 * 1024 * 1024},
        options.device(torch::kCUDA, device_index).dtype(torch::kUInt8)
    );
    size_t workspace_bytes = static_cast<size_t>(state.workspace.numel());
    check_cublas_lt(
        cublasLtMatmulPreferenceSetAttribute(
            state.pref,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_bytes,
            sizeof(workspace_bytes)
        ),
        "set workspace size"
    );

    cublasLtMatmulHeuristicResult_t heuristics[64];
    int heuristic_count = 0;
    check_cublas_lt(
        cublasLtMatmulAlgoGetHeuristic(
            state.handle,
            state.op_desc,
            state.a_desc,
            state.b_desc,
            state.c_desc,
            state.c_desc,
            state.pref,
            64,
            heuristics,
            &heuristic_count
        ),
        "cublasLtMatmulAlgoGetHeuristic"
    );
    TORCH_CHECK(heuristic_count > 0, "No cublasLt algorithm found for this shape");

    state.algo = heuristics[0].algo;
    state.output = torch::empty({batch, m, n}, options.device(torch::kCUDA, device_index));
    state.device_index = device_index;
    state.batch = batch;
    state.m = m;
    state.n = n;
    state.k = k;
    state.initialized = true;
}

torch::Tensor bmm_cuda(torch::Tensor a, torch::Tensor b) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    TORCH_CHECK(a.dim() == 3 && b.dim() == 3, "Expected 3D tensors");
    TORCH_CHECK(a.size(0) == b.size(0), "Batch dimensions must match");
    TORCH_CHECK(a.size(2) == b.size(1), "Inner dimensions must match");

    c10::cuda::CUDAGuard device_guard(a.device());
    const auto batch = a.size(0);
    const auto m = a.size(1);
    const auto k = a.size(2);
    const auto n = b.size(2);

    ensure_initialized(a.get_device(), batch, m, n, k, a.options());
    auto& state = cache();

    const float alpha = 1.0f;
    const float beta = 0.0f;
    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

    check_cublas_lt(
        cublasLtMatmul(
            state.handle,
            state.op_desc,
            &alpha,
            a.data_ptr<float>(),
            state.a_desc,
            b.data_ptr<float>(),
            state.b_desc,
            &beta,
            state.output.data_ptr<float>(),
            state.c_desc,
            state.output.data_ptr<float>(),
            state.c_desc,
            &state.algo,
            state.workspace.data_ptr(),
            static_cast<size_t>(state.workspace.numel()),
            stream.stream()
        ),
        "cublasLtMatmul"
    );

    return state.output;
}
"""

CPP_SRC = "torch::Tensor bmm_cuda(torch::Tensor a, torch::Tensor b);"

module = load_inline(
    name="ptxbench_bmm_cublaslt_cached_v1",
    cpp_sources=CPP_SRC,
    cuda_sources=CUDA_SRC,
    functions=["bmm_cuda"],
    extra_cuda_cflags=["-O3"],
    extra_ldflags=["-lcublasLt"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return module.bmm_cuda(A, B)
