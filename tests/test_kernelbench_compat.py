import math
from pathlib import Path

import pytest

from ptxbench.analysis import (
    fastp,
    geometric_mean_speed_ratio_correct_and_faster_only,
    geometric_mean_speed_ratio_correct_only,
)
from ptxbench.dataset import (
    LEVEL_REPRESENTATIVE_IDS,
    construct_dataset,
    fetch_ref_arch_from_dataset,
    get_code_hash,
)
from ptxbench.eval import EvalResult
from ptxbench.static_checker import validate_submission_static

torch = pytest.importorskip("torch")


def test_get_code_hash_ignores_comments() -> None:
    code_v1 = """
import torch
# batch size
B = 1
"""
    code_v2 = """
import torch
'''
comment changed
'''
B = 1
"""
    assert get_code_hash(code_v1) == get_code_hash(code_v2)


def test_get_code_hash_changes_for_real_code_changes() -> None:
    assert get_code_hash("B = 1") != get_code_hash("B = 64")


def test_dataset_accessors_match_kernelbench_style() -> None:
    dataset = construct_dataset(level=1)
    problem = dataset.get_problem_by_id(1)

    assert problem.problem_id == 1
    assert problem.name.startswith("1_")
    assert len(problem.code) > 0
    assert isinstance(problem.hash, str)
    assert len(problem.hash) == 32
    assert dataset.get_problem_ids() == sorted(dataset.get_problem_ids())


def test_dataset_subset_range_and_sampling_are_reproducible() -> None:
    dataset = construct_dataset(level=1)

    subset = dataset.subset(problem_ids=[1, 3, 5])
    assert subset.get_problem_ids() == [1, 3, 5]

    ranged = dataset.subset(id_range=(1, 5))
    assert ranged.get_problem_ids() == [1, 2, 3, 4, 5]

    sample_a = dataset.sample(5, seed=42)
    sample_b = dataset.sample(5, seed=42)
    sample_c = dataset.sample(5, seed=7)
    assert sample_a.get_problem_ids() == sample_b.get_problem_ids()
    assert sample_a.get_problem_ids() != sample_c.get_problem_ids()


def test_representative_subset_matches_upstream_ids() -> None:
    dataset = construct_dataset(level=1)
    subset = dataset.get_representative_subset()
    assert subset.get_problem_ids() == LEVEL_REPRESENTATIVE_IDS[1]


def test_fetch_ref_arch_from_dataset_matches_problem_lookup() -> None:
    dataset = construct_dataset(level=1)
    path, name, code = fetch_ref_arch_from_dataset(dataset, problem_id=19)
    problem = dataset.get_problem_by_id(19)

    assert path == problem.path
    assert name == problem.name
    assert code == problem.code


def test_validate_submission_static_is_tuple_like() -> None:
    result = validate_submission_static("x = 1 + 1")
    valid, errors, warnings = result

    assert isinstance(valid, bool)
    assert isinstance(errors, list)
    assert isinstance(warnings, list)


def test_validate_submission_static_precision_warning_and_forbidden_override() -> None:
    code = """
class ModelNew:
    def forward(self, x):
        x = x.half()
        return x
"""
    result = validate_submission_static(code, precision="fp32")
    assert any("precision" in message for message in result.warnings)

    strict_result = validate_submission_static(
        code,
        precision="float32",
        forbidden=["precision_downgrade"],
        warnings=[],
    )
    assert not strict_result.valid
    assert any("precision" in message for message in strict_result.errors)


def test_validate_submission_static_custom_warning_list_reclassifies_bypass() -> None:
    code = """
try:
    result = custom_kernel(x)
except:
    result = torch.matmul(x, w)
"""
    result = validate_submission_static(
        code,
        backend="cuda",
        forbidden=[],
        warnings=["code_bypass"],
    )
    assert any("try_except" in message for message in result.warnings)


def test_score_examples_match_kernelbench_reference() -> None:
    abs_tol = 1e-7

    assert math.isclose(
        geometric_mean_speed_ratio_correct_only(
            [1, 0, 1, 1, 0],
            [0.1, 0.15, 0.2, 0.05, 0.3],
            [0.2, 0.15, 0.3, 0.01, 0.2],
            5,
        ),
        1.185631101,
        abs_tol=abs_tol,
    )
    assert math.isclose(
        geometric_mean_speed_ratio_correct_and_faster_only(
            [1, 0, 1, 1, 0],
            [0.1, 0.15, 0.2, 0.05, 0.3],
            [0.2, 0.15, 0.3, 0.01, 0.2],
            5,
        ),
        5.0,
        abs_tol=abs_tol,
    )
    assert math.isclose(
        fastp(
            [1, 0, 1, 1, 0],
            [0.1, 0.15, 0.2, 0.05, 0.3],
            [0.2, 0.15, 0.3, 0.01, 0.2],
            5,
            1.0,
        ),
        0.2,
        abs_tol=abs_tol,
    )


def test_eval_result_to_dict_has_kernelbench_style_fields() -> None:
    result = EvalResult(
        backend="ptx",
        problem_id=19,
        problem_name="19_ReLU.py",
        source_path=str(Path("runs/sample.py")),
        compiled=True,
        assembled=True,
        loaded=True,
        correctness=True,
        runtime_ms=1.5,
        ref_runtime_ms=2.0,
        speedup_vs_torch=4 / 3,
        metadata={"correctness_trials": "5/5"},
    )
    payload = result.to_dict()

    assert payload["backend"] == "ptx"
    assert payload["problem_id"] == 19
    assert payload["compiled"] is True
    assert payload["assembled"] is True
    assert payload["loaded"] is True
    assert payload["correctness"] is True
    assert payload["runtime_ms"] == 1.5
    assert payload["ref_runtime_ms"] == 2.0
    assert payload["metadata"]["correctness_trials"] == "5/5"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for PTX integration tests")
def test_eval_submission_separates_ptx_assembly_failures() -> None:
    from ptxbench.eval import evaluate_submission

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
    assert result.assembled is False
    assert result.loaded is False
    assert "assembly_error" in result.metadata
