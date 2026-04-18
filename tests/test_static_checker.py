from pathlib import Path

import pytest

from ptxbench.static_checker import validate_submission_static


def _ptx_submission(forward_body: str, *, extra_imports: str = "") -> str:
    return f"""
import torch
import torch.nn as nn
from ptxbench.runtime import PTXModuleRunner
from ptxbench.spec import PTXKernelSpec
{extra_imports}

PTX_SOURCES = {{
    "relu": ".version 8.0\\n.target sm_89\\n.address_size 64\\n.visible .entry relu_kernel() {{ ret; }}"
}}

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
        return x
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
    assert any(error.startswith("strict:try_except") for error in result.errors)
