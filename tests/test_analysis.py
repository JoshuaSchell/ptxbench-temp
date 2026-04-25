from pathlib import Path
import importlib.util

import pytest

from ptxbench.analysis import (
    classify_paper_failure_category,
    classify_result_stage,
    compute_agentic_budget_summary,
    compute_backend_summary,
    compute_family_backend_summaries,
    compute_fast_p,
    compute_joint_backend_summary,
)
from ptxbench.dataset import construct_dataset
from ptxbench.eval import build_evaluation_failure_result


def _load_benchmark_analysis_script():
    script_path = Path("scripts/benchmark_eval_analysis.py").resolve()
    spec = importlib.util.spec_from_file_location("ptxbench_benchmark_eval_analysis_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compute_fast_p() -> None:
    score = compute_fast_p(
        correct=[True, True, False],
        ref_ms=[2.0, 2.0, 2.0],
        candidate_ms=[1.0, 3.0, 1.0],
        p=1.0,
    )
    assert score == 1 / 3


def test_joint_backend_summary() -> None:
    summary = compute_joint_backend_summary(
        ptx_correct=[True, True, False],
        ptx_ref_ms=[2.0, 2.0, 2.0],
        ptx_ms=[1.0, 3.0, 1.0],
        cuda_correct=[True, True, True],
        cuda_ref_ms=[2.0, 2.0, 2.0],
        cuda_ms=[1.5, 2.5, 1.0],
    )
    assert summary.total_tasks == 3
    assert summary.jointly_correct_tasks == 2
    assert summary.ptx_head_to_head_win_rate == 0.5


def test_backend_summary_tracks_failure_breakdown() -> None:
    rows = [
        {"compiled": False, "correctness": False, "runtime_ms": -1.0, "ref_runtime_ms": -1.0, "metadata": {}},
        {"compiled": True, "assembled": False, "loaded": False, "correctness": False, "runtime_ms": -1.0, "ref_runtime_ms": -1.0, "metadata": {"assembly_error": "bad"}},
        {"compiled": True, "assembled": True, "loaded": True, "correctness": True, "runtime_ms": 1.0, "ref_runtime_ms": 2.0, "metadata": {}},
    ]
    summary = compute_backend_summary(rows)
    assert summary.total_tasks == 3
    assert summary.correct_tasks == 1
    assert summary.failure_breakdown["compile"] == 1
    assert summary.failure_breakdown["assemble"] == 1
    assert summary.failure_breakdown["success"] == 1
    assert summary.paper_failure_breakdown["import_or_compile_error"] == 1


def test_classify_paper_failure_category_major_cases() -> None:
    cases = [
        ({"correctness": True, "speedup_vs_eager": 1.2, "metadata": {}}, "success_fast_eager"),
        ({"correctness": True, "speedup_vs_compile_default": 0.8, "metadata": {}}, "success_slow_compile"),
        ({"correctness": False, "compiled": False, "metadata": {"missing_submission": True}}, "missing_submission"),
        ({"correctness": False, "compiled": False, "metadata": {"generation_failure": {"error": "x"}}}, "generation_failure"),
        ({"correctness": False, "compiled": False, "metadata": {"static_errors": ["fallback"]}}, "static_guardrail"),
        ({"correctness": False, "compiled": False, "metadata": {"compile_error": "Submission does not define ModelNew"}}, "contract_error"),
        ({"correctness": False, "compiled": False, "metadata": {"compile_error": "No module named x"}}, "import_or_compile_error"),
        ({"correctness": False, "compiled": True, "assembled": False, "metadata": {"assembly_error": "ptxas failed"}}, "ptxas_assembly_error"),
        ({"correctness": False, "compiled": True, "assembled": True, "loaded": False, "metadata": {"load_error": "bad cubin"}}, "cubin_load_error"),
        ({"correctness": False, "compiled": True, "assembled": True, "loaded": True, "metadata": {"runtime_error": "cuLaunchKernel failed"}}, "kernel_launch_error"),
        ({"correctness": False, "compiled": True, "metadata": {"runtime_error": "CUDA error: illegal memory access"}}, "illegal_memory_or_cuda_error"),
        ({"correctness": False, "compiled": True, "metadata": {"oom_error": True}}, "oom"),
        ({"correctness": False, "compiled": True, "metadata": {"timeout_error": "slow"}}, "timeout"),
        ({"correctness": False, "compiled": True, "metadata": {"correctness_mismatch": {"shape_mismatch": True}}}, "correctness_shape"),
        ({"correctness": False, "compiled": True, "metadata": {"correctness_mismatch": {"dtype_mismatch": True}}}, "correctness_dtype"),
        ({"correctness": False, "compiled": True, "metadata": {"correctness_mismatch": {"kind": "tensor_mismatch"}}}, "correctness_numeric"),
        ({"correctness": False, "compiled": True, "metadata": {"evaluator_crash": True}}, "evaluator_crash"),
    ]
    for row, expected in cases:
        assert classify_paper_failure_category(row) == expected


def test_backend_summary_tracks_compile_default_fastp() -> None:
    rows = [
        {
            "compiled": True,
            "assembled": True,
            "loaded": True,
            "correctness": True,
            "runtime_ms": 1.0,
            "ref_runtime_ms": 2.0,
            "ref_runtime_compile_default_ms": 1.5,
            "metadata": {},
        },
        {
            "compiled": True,
            "assembled": True,
            "loaded": True,
            "correctness": True,
            "runtime_ms": 2.0,
            "ref_runtime_ms": 4.0,
            "ref_runtime_compile_default_ms": 1.0,
            "metadata": {},
        },
    ]
    summary = compute_backend_summary(rows)
    assert summary.fast_p_vs_compile_default[1.0] == 0.5
    assert summary.fast_p_vs_compile_default[2.0] == 0.0


def test_classify_result_stage_defaults_to_correctness_failure() -> None:
    row = {
        "compiled": True,
        "assembled": True,
        "loaded": True,
        "correctness": False,
        "metadata": {},
    }
    assert classify_result_stage(row) == "correctness"


def test_build_evaluation_failure_result_classifies_as_oom() -> None:
    problem = construct_dataset(level=1, problem_ids=[19]).get_problem(19)
    result = build_evaluation_failure_result(
        problem,
        backend="ptx",
        source_path=Path("runs/failure_submission.py"),
        metadata={"runtime_error": "CUDA out of memory", "oom_error": True},
    )

    row = result.to_dict()
    assert row["compiled"] is True
    assert row["correctness"] is False
    assert classify_result_stage(row) == "oom"


def test_compute_agentic_budget_summary_uses_first_success_steps() -> None:
    rows = [
        {"correctness": True, "first_compile_step": 1, "first_correct_step": 2},
        {"correctness": False, "first_compile_step": 2, "first_correct_step": None},
        {"correctness": True, "first_compile_step": 1, "first_correct_step": 3},
    ]
    summary = compute_agentic_budget_summary(rows)
    assert summary.total_tasks == 3
    assert summary.correct_at_budget == 2 / 3
    assert summary.median_steps_to_first_compile == 1.0
    assert summary.median_steps_to_first_correct == 2.5


def test_compute_family_backend_summaries_groups_by_primary_family() -> None:
    rows = [
        {
            "task_family_tags": ["elementwise"],
            "compiled": True,
            "assembled": True,
            "loaded": True,
            "correctness": True,
            "runtime_ms": 1.0,
            "ref_runtime_ms": 2.0,
            "metadata": {},
        },
        {
            "task_family_tags": ["norm"],
            "compiled": True,
            "assembled": True,
            "loaded": True,
            "correctness": False,
            "runtime_ms": -1.0,
            "ref_runtime_ms": -1.0,
            "metadata": {"correctness_errors": ["mismatch"]},
        },
    ]
    summaries = compute_family_backend_summaries(rows)
    assert summaries["elementwise"].correct_tasks == 1
    assert summaries["norm"].failure_breakdown["correctness"] == 1


def test_validate_paired_protocol_parity_rejects_seed_mismatch() -> None:
    benchmark_analysis = _load_benchmark_analysis_script()
    shared_problem_ids = [1, 2, 3]
    ptx_protocol = {
        "protocol_version": "paper-v1",
        "level": 1,
        "track": "oneshot",
        "precision": "fp32",
        "arch": "sm_89",
        "one_shot": True,
        "num_correct_trials": 5,
        "num_perf_trials": 100,
        "generation_timeout_seconds": 1800,
        "official_eval_seed": 42,
    }
    cuda_protocol = {
        **ptx_protocol,
        "official_eval_seed": 99,
    }

    with pytest.raises(ValueError, match="Protocol parity mismatch"):
        benchmark_analysis.validate_paired_protocol_parity(
            ptx_run_manifest={"protocol": ptx_protocol, "problems": shared_problem_ids},
            cuda_run_manifest={"protocol": cuda_protocol, "problems": shared_problem_ids},
            ptx_eval_manifest={"protocol": ptx_protocol, "problem_ids": shared_problem_ids},
            cuda_eval_manifest={"protocol": cuda_protocol, "problem_ids": shared_problem_ids},
            ptx_problem_ids=shared_problem_ids,
            cuda_problem_ids=shared_problem_ids,
        )


def test_benchmark_analysis_serializes_compile_default_metrics() -> None:
    benchmark_analysis = _load_benchmark_analysis_script()
    rows = [
        {
            "compiled": True,
            "assembled": True,
            "loaded": True,
            "correctness": True,
            "runtime_ms": 1.0,
            "ref_runtime_ms": 2.0,
            "ref_runtime_compile_default_ms": 1.5,
            "metadata": {},
        }
    ]
    serialized = benchmark_analysis._serialize_backend_summary(compute_backend_summary(rows))
    assert serialized["fast1_vs_compile_default"] == 1.0
    assert serialized["fast2_vs_compile_default"] == 0.0
