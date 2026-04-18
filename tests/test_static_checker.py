from pathlib import Path

from ptxbench.static_checker import validate_submission_static


def test_static_checker_accepts_ptx_fixture() -> None:
    source = Path("tests/fixtures/submissions/ptx/relu_submission.py").read_text(encoding="utf-8")
    result = validate_submission_static(source, backend="ptx")
    assert result.valid


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
