from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ptxbench.dataset import construct_dataset
from ptxbench.eval import evaluate_submission


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for PTX integration tests")
def test_ptx_relu_submission_smoke() -> None:
    problem = construct_dataset(level=1, problem_ids=[19]).get_problem(19)
    submission = Path("tests/fixtures/submissions/ptx/relu_submission.py").resolve()
    result = evaluate_submission(
        problem=problem,
        submission_path=submission,
        backend="ptx",
        num_correct_trials=2,
        num_perf_trials=5,
    )
    assert result.compiled
    assert result.correctness
    assert result.runtime_ms > 0
    assert result.ref_runtime_ms > 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for PTX integration tests")
def test_invalid_ptx_submission_reports_assembly_failure() -> None:
    problem = construct_dataset(level=1, problem_ids=[19]).get_problem(19)
    submission = Path("tests/fixtures/submissions/ptx/invalid_ptx_submission.py").resolve()
    result = evaluate_submission(
        problem=problem,
        submission_path=submission,
        backend="ptx",
        num_correct_trials=1,
        num_perf_trials=2,
    )
    assert result.compiled
    assert not result.correctness
    assert result.assembled is False
    assert "assembly_error" in result.metadata
