from pathlib import Path

import pytest

from ptxbench.static_checker import validate_submission_static


def _ptx_submission(
    forward_body: str,
    *,
    ptx_sources_expr: str | None = None,
    extra_imports: str = "",
    extra_methods: str = "",
) -> str:
    ptx_sources = ptx_sources_expr or """{
    "relu": ".version 8.0\\n.target sm_89\\n.address_size 64\\n.visible .entry relu_kernel() { ret; }"
}"""
    return f"""
import torch
import torch.nn as nn
from ptxbench.runtime import PTXModuleRunner
from ptxbench.spec import PTXKernelSpec
{extra_imports}

PTX_SOURCES = {ptx_sources}

PTX_KERNELS = {{
    "relu": PTXKernelSpec(
        entry="relu_kernel",
        grid=lambda x, out, n: ((int((n + 255) // 256), 1, 1)),
        block=(256, 1, 1),
        arg_types=("tensor", "tensor", "uint32"),
    )
}}


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.runner = PTXModuleRunner(PTX_SOURCES, PTX_KERNELS)
{extra_methods}

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
{forward_body}
"""


def test_static_checker_accepts_ptx_fixture() -> None:
    source = Path("tests/fixtures/submissions/ptx/relu_submission.py").read_text(encoding="utf-8")
    result = validate_submission_static(source, backend="ptx")
    assert result.valid


def test_static_checker_allows_empty_like_and_tensor_numel() -> None:
    result = validate_submission_static(
        _ptx_submission(
            """
        n = x.numel()
        out = torch.empty_like(x)
        self.runner.launch("relu", x, out, n)
        return out
        """
        ),
        backend="ptx",
    )
    assert result.valid


def test_static_checker_rejects_returning_input_tensor() -> None:
    result = validate_submission_static(
        _ptx_submission(
            """
        self.runner.launch("relu", x, y, x.numel())
        return x
        """
        ),
        backend="ptx",
    )
    assert not result.valid
    assert "ptx_forbidden:return_input_tensor" in result.errors


def test_static_checker_rejects_missing_launch_call() -> None:
    result = validate_submission_static(
        _ptx_submission(
            """
        out = torch.empty_like(x)
        return out
        """
        ),
        backend="ptx",
    )
    assert not result.valid
    assert "ptx_required:launch_call" in result.errors


def test_static_checker_rejects_ptx_sources_open_read() -> None:
    result = validate_submission_static(
        _ptx_submission(
            """
        out = torch.empty_like(x)
        self.runner.launch("relu", x, out, x.numel())
        return out
        """,
            ptx_sources_expr='open("kernel.ptx").read()',
        ),
        backend="ptx",
    )
    assert not result.valid
    assert "ptx_forbidden:open" in result.errors
    assert "ptx_forbidden:literal_ptx_sources" in result.errors


def test_static_checker_rejects_syntax_error() -> None:
    result = validate_submission_static("def broken(:\n    pass\n", backend="ptx")
    assert not result.valid
    assert "ptx_static:syntax_error" in result.errors


def test_static_checker_rejects_helper_method_torch_compute() -> None:
    result = validate_submission_static(
        _ptx_submission(
            """
        out = torch.empty_like(x)
        self.runner.launch("relu", x, out, x.numel())
        return out
        """,
            extra_methods="""
    def helper(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.add(x, y)
""",
        ),
        backend="ptx",
    )
    assert not result.valid
    assert "ptx_forbidden:torch_compute" in result.errors


def test_static_checker_rejects_eval_and_getattr_fallbacks() -> None:
    result = validate_submission_static(
        _ptx_submission(
            """
        out = torch.empty_like(x)
        fn = getattr(torch, "relu")
        value = eval("1")
        self.runner.launch("relu", x, out, x.numel())
        return out
        """
        ),
        backend="ptx",
    )
    assert not result.valid
    assert "ptx_forbidden:getattr" in result.errors
    assert "ptx_forbidden:eval" in result.errors


def test_static_checker_rejects_tensor_binary_ops_in_forward() -> None:
    result = validate_submission_static(
        _ptx_submission(
            """
        out = x + y
        return out
        """
        ),
        backend="ptx",
    )
    assert not result.valid
    assert "ptx_forbidden:tensor_binop" in result.errors


def test_static_checker_rejects_tensor_reduction_methods() -> None:
    result = validate_submission_static(
        _ptx_submission(
            """
        out = torch.empty_like(x)
        self.runner.launch("relu", x, out, x.numel())
        return x.sum()
        """
        ),
        backend="ptx",
    )
    assert not result.valid
    assert "ptx_forbidden:tensor_compute" in result.errors


def test_static_checker_rejects_torch_compute_ops() -> None:
    result = validate_submission_static(
        _ptx_submission(
            """
        out = torch.add(x, y)
        return out
        """
        ),
        backend="ptx",
    )
    assert not result.valid
    assert "ptx_forbidden:torch_compute" in result.errors


def test_static_checker_rejects_torch_ops_namespace() -> None:
    result = validate_submission_static(
        _ptx_submission(
            """
        out = torch.ops.aten.add.Tensor(x, y)
        return out
        """
        ),
        backend="ptx",
    )
    assert not result.valid
    assert "ptx_forbidden:torch_ops" in result.errors


def test_static_checker_rejects_cuda_inline_extensions() -> None:
    result = validate_submission_static(
        _ptx_submission(
            """
        out = torch.empty_like(x)
        self.runner.launch("relu", x, out, x.numel())
        return out
        """,
            extra_imports="from torch.utils.cpp_extension import load_inline",
        ),
        backend="ptx",
    )
    assert not result.valid
    assert "ptx_forbidden:cuda_inline" in result.errors


@pytest.mark.parametrize(
    ("extra_imports", "expected_error"),
    [
        ("import subprocess", "ptx_forbidden:subprocess"),
        ("import ctypes", "ptx_forbidden:ctypes"),
    ],
)
def test_static_checker_rejects_subprocess_and_ctypes(extra_imports: str, expected_error: str) -> None:
    result = validate_submission_static(
        _ptx_submission(
            """
        out = torch.empty_like(x)
        self.runner.launch("relu", x, out, x.numel())
        return out
        """,
            extra_imports=extra_imports,
        ),
        backend="ptx",
    )
    assert not result.valid
    assert expected_error in result.errors


def test_static_checker_rejects_torch_fallback() -> None:
    result = validate_submission_static(
        """
import torch
class ModelNew:
    def forward(self, x):
        try:
            return torch.relu(x)
        except Exception:
            return x
PTX_SOURCES = {}
PTX_KERNELS = {}
        """,
        backend="ptx",
    )
    assert not result.valid
    assert any(error.startswith("strict:try_except") or error == "ptx_forbidden:try_except" for error in result.errors)
