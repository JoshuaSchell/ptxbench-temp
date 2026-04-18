from __future__ import annotations

from dataclasses import dataclass
import math
import statistics


FAILURE_STAGES = ("success", "compile", "assemble", "load", "runtime", "correctness", "timeout", "oom", "evaluator_crash")


def geometric_mean_speed_ratio_correct_only(
    is_correct: list[bool],
    baseline_speed: list[float],
    actual_speed: list[float],
    n: int,
) -> float:
    del n
    speedups = [
        baseline / actual
        for correct, baseline, actual in zip(is_correct, baseline_speed, actual_speed, strict=True)
        if correct and baseline > 0 and actual > 0
    ]
    return _geomean(speedups)


def geometric_mean_speed_ratio_correct_and_faster_only(
    is_correct: list[bool],
    baseline_speed: list[float],
    actual_speed: list[float],
    n: int,
) -> float:
    del n
    speedups = [
        baseline / actual
        for correct, baseline, actual in zip(is_correct, baseline_speed, actual_speed, strict=True)
        if correct and baseline > 0 and actual > 0 and (baseline / actual) > 1.0
    ]
    return _geomean(speedups)


def fastp(
    is_correct: list[bool],
    baseline_speed: list[float],
    actual_speed: list[float],
    n: int,
    p: float,
) -> float:
    total = n
    if total == 0:
        return 0.0
    passing = 0
    for correct, baseline, actual in zip(is_correct, baseline_speed, actual_speed, strict=True):
        if not correct or baseline <= 0 or actual <= 0:
            continue
        if (baseline / actual) > p:
            passing += 1
    return passing / total


def compute_fast_p(correct: list[bool], ref_ms: list[float], candidate_ms: list[float], p: float) -> float:
    return fastp(correct, ref_ms, candidate_ms, len(correct), p)


def _geomean(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.exp(sum(math.log(x) for x in values) / len(values))


@dataclass(frozen=True)
class JointComparisonSummary:
    total_tasks: int
    jointly_correct_tasks: int
    fast_p_ptx_vs_torch: dict[float, float]
    fast_p_cuda_vs_torch: dict[float, float]
    ptx_head_to_head_win_rate: float
    ptx_vs_cuda_geomean_speedup: float


@dataclass(frozen=True)
class BackendRunSummary:
    total_tasks: int
    correct_tasks: int
    correctness_rate: float
    fast_p_vs_torch: dict[float, float]
    geomean_speedup_vs_torch_correct_only: float
    geomean_speedup_vs_torch_correct_and_faster_only: float
    failure_breakdown: dict[str, int]


@dataclass(frozen=True)
class AgenticBudgetSummary:
    total_tasks: int
    correct_at_budget: float
    median_steps_to_first_compile: float | None
    median_steps_to_first_correct: float | None


def classify_result_stage(row: dict) -> str:
    metadata = row.get("metadata", {})
    explicit_category = metadata.get("failure_category")
    if isinstance(explicit_category, str) and explicit_category in FAILURE_STAGES:
        return explicit_category
    if row.get("correctness"):
        return "success"
    if metadata.get("timeout_error"):
        return "timeout"
    if metadata.get("oom_error"):
        return "oom"
    if metadata.get("evaluator_crash") or metadata.get("evaluator_error"):
        return "evaluator_crash"
    if not row.get("compiled", False):
        return "compile"
    if row.get("assembled") is False:
        return "assemble"
    if row.get("loaded") is False:
        return "load"
    if "runtime_error" in metadata:
        return "runtime"
    if metadata.get("correctness_errors"):
        return "correctness"
    if any(key in metadata for key in ("assembly_error", "load_error", "runtime_error")):
        if "assembly_error" in metadata:
            return "assemble"
        if "load_error" in metadata:
            return "load"
        return "runtime"
    return "correctness"


def compute_backend_summary(
    rows: list[dict],
    thresholds: tuple[float, ...] = (0.0, 1.0, 2.0),
) -> BackendRunSummary:
    correct = [bool(row["correctness"]) for row in rows]
    ref_ms = [float(row["ref_runtime_ms"]) for row in rows]
    candidate_ms = [float(row["runtime_ms"]) for row in rows]
    failure_breakdown = {stage: 0 for stage in FAILURE_STAGES}
    for row in rows:
        failure_breakdown[classify_result_stage(row)] += 1
    total = len(rows)
    correct_tasks = sum(correct)
    return BackendRunSummary(
        total_tasks=total,
        correct_tasks=correct_tasks,
        correctness_rate=(correct_tasks / total) if total else 0.0,
        fast_p_vs_torch={threshold: compute_fast_p(correct, ref_ms, candidate_ms, threshold) for threshold in thresholds},
        geomean_speedup_vs_torch_correct_only=geometric_mean_speed_ratio_correct_only(correct, ref_ms, candidate_ms, total),
        geomean_speedup_vs_torch_correct_and_faster_only=geometric_mean_speed_ratio_correct_and_faster_only(
            correct,
            ref_ms,
            candidate_ms,
            total,
        ),
        failure_breakdown=failure_breakdown,
    )


def compute_joint_backend_summary(
    ptx_correct: list[bool],
    ptx_ref_ms: list[float],
    ptx_ms: list[float],
    cuda_correct: list[bool],
    cuda_ref_ms: list[float],
    cuda_ms: list[float],
    thresholds: tuple[float, ...] = (0.0, 1.0, 2.0),
) -> JointComparisonSummary:
    if not (
        len(ptx_correct)
        == len(ptx_ref_ms)
        == len(ptx_ms)
        == len(cuda_correct)
        == len(cuda_ref_ms)
        == len(cuda_ms)
    ):
        raise ValueError("All metric arrays must have the same length")

    total = len(ptx_correct)
    fast_p_ptx = {threshold: compute_fast_p(ptx_correct, ptx_ref_ms, ptx_ms, threshold) for threshold in thresholds}
    fast_p_cuda = {threshold: compute_fast_p(cuda_correct, cuda_ref_ms, cuda_ms, threshold) for threshold in thresholds}

    joint_ratios: list[float] = []
    wins = 0
    jointly_correct = 0
    for ptx_ok, ptx_runtime, cuda_ok, cuda_runtime in zip(
        ptx_correct,
        ptx_ms,
        cuda_correct,
        cuda_ms,
        strict=True,
    ):
        if not (ptx_ok and cuda_ok):
            continue
        if ptx_runtime <= 0 or cuda_runtime <= 0:
            continue
        jointly_correct += 1
        ratio = cuda_runtime / ptx_runtime
        joint_ratios.append(ratio)
        if ptx_runtime < cuda_runtime:
            wins += 1

    return JointComparisonSummary(
        total_tasks=total,
        jointly_correct_tasks=jointly_correct,
        fast_p_ptx_vs_torch=fast_p_ptx,
        fast_p_cuda_vs_torch=fast_p_cuda,
        ptx_head_to_head_win_rate=(wins / jointly_correct) if jointly_correct else 0.0,
        ptx_vs_cuda_geomean_speedup=_geomean(joint_ratios),
    )


def compute_agentic_budget_summary(rows: list[dict]) -> AgenticBudgetSummary:
    total = len(rows)
    correct_tasks = sum(bool(row.get("correctness")) for row in rows)
    compile_steps = _median_numeric(
        row.get("first_compile_step")
        for row in rows
        if row.get("first_compile_step") is not None
    )
    correct_steps = _median_numeric(
        row.get("first_correct_step")
        for row in rows
        if row.get("first_correct_step") is not None
    )
    return AgenticBudgetSummary(
        total_tasks=total,
        correct_at_budget=(correct_tasks / total) if total else 0.0,
        median_steps_to_first_compile=compile_steps,
        median_steps_to_first_correct=correct_steps,
    )


def compute_family_backend_summaries(
    rows: list[dict],
    thresholds: tuple[float, ...] = (0.0, 1.0, 2.0),
) -> dict[str, BackendRunSummary]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        family_tags = row.get("task_family_tags") or row.get("metadata", {}).get("task_family_tags") or ["unclassified"]
        family = family_tags[0]
        grouped.setdefault(family, []).append(row)
    return {
        family: compute_backend_summary(family_rows, thresholds=thresholds)
        for family, family_rows in sorted(grouped.items())
    }


def _median_numeric(values) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return float(statistics.median(clean))
