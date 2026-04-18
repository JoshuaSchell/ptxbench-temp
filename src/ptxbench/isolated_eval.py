from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import subprocess
import sys
import tempfile

from .config import DEFAULT_ARCH, DEFAULT_OFFICIAL_EVAL_SEED
from .dataset import Problem
from .eval import (
    DEFAULT_NUM_CORRECT_TRIALS,
    DEFAULT_NUM_PERF_TRIALS,
    DEFAULT_NUM_WARMUP,
    EvalResult,
    evaluate_submission,
)


WORKER_MODULE = "ptxbench.eval_worker"
STREAM_CAPTURE_LIMIT = 4000


def _trim_stream(value: str | bytes | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if len(value) <= STREAM_CAPTURE_LIMIT:
        return value
    return value[-STREAM_CAPTURE_LIMIT:]


def _serialize_problem(problem: Problem) -> dict[str, Any]:
    return {
        "problem_id": problem.problem_id,
        "level": problem.level,
        "name": problem.name,
        "path": str(problem.path),
        "code": problem.code,
    }


def deserialize_problem(payload: dict[str, Any]) -> Problem:
    return Problem(
        problem_id=int(payload["problem_id"]),
        level=int(payload["level"]),
        name=str(payload["name"]),
        path=Path(payload["path"]),
        code=str(payload["code"]),
    )


def _build_failure_payload(
    problem: Problem,
    *,
    backend: str,
    source_path: Path,
    category: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    category = category.lower()
    payload_metadata = dict(metadata or {})
    payload_metadata["failure_category"] = category

    if category == "compile":
        compiled = False
        assembled = False if backend == "ptx" else None
        loaded = False if backend == "ptx" else None
    elif category == "assemble":
        compiled = True
        assembled = False if backend == "ptx" else None
        loaded = False if backend == "ptx" else None
    elif category == "load":
        compiled = True
        assembled = True if backend == "ptx" else None
        loaded = False if backend == "ptx" else None
    elif category in {"runtime", "correctness", "oom"}:
        compiled = True
        assembled = True if backend == "ptx" else None
        loaded = True if backend == "ptx" else None
    else:
        compiled = False
        assembled = False if backend == "ptx" else None
        loaded = False if backend == "ptx" else None

    if category == "oom":
        payload_metadata["oom_error"] = True
        payload_metadata.setdefault("runtime_error", "CUDA out of memory")

    result = EvalResult(
        backend=backend,
        problem_id=problem.problem_id,
        problem_name=problem.name,
        source_path=str(source_path),
        task_family_tags=list(problem.task_family_tags),
        compiled=compiled,
        assembled=assembled,
        loaded=loaded,
        correctness=False,
        metadata=payload_metadata,
    )
    return result.to_dict()


def classify_failure_category(payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata", {})
    explicit_category = metadata.get("failure_category")
    if isinstance(explicit_category, str) and explicit_category:
        return explicit_category
    if payload.get("correctness"):
        return "success"
    if metadata.get("timeout_error"):
        return "timeout"
    if metadata.get("oom_error"):
        return "oom"
    if metadata.get("evaluator_crash") or metadata.get("evaluator_error"):
        return "evaluator_crash"
    if not payload.get("compiled", False) or "compile_error" in metadata:
        return "compile"
    if payload.get("assembled") is False or "assembly_error" in metadata:
        return "assemble"
    if payload.get("loaded") is False or "load_error" in metadata:
        return "load"
    if "runtime_error" in metadata or "init_error" in metadata:
        return "runtime"
    return "correctness"


def annotate_eval_payload(
    payload: dict[str, Any],
    *,
    mode: str,
    timeout_seconds: int | None = None,
    returncode: int | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
) -> dict[str, Any]:
    normalized = dict(payload)
    metadata = dict(normalized.get("metadata", {}))
    metadata["failure_category"] = classify_failure_category(normalized)
    isolated_metadata = dict(metadata.get("isolated_eval", {}))
    isolated_metadata["mode"] = mode
    if timeout_seconds is not None:
        isolated_metadata["timeout_seconds"] = timeout_seconds
    if returncode is not None:
        isolated_metadata["returncode"] = returncode
    if stdout:
        isolated_metadata["stdout_tail"] = _trim_stream(stdout)
    if stderr:
        isolated_metadata["stderr_tail"] = _trim_stream(stderr)
    metadata["isolated_eval"] = isolated_metadata
    normalized["metadata"] = metadata
    return normalized


def _worker_command(request_path: Path, output_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        WORKER_MODULE,
        "--input",
        str(request_path),
        "--output",
        str(output_path),
    ]


def _subprocess_looks_like_oom(process: subprocess.CompletedProcess[str]) -> bool:
    haystack = "\n".join(part for part in (process.stdout, process.stderr) if part)
    lowered = haystack.lower()
    return "out of memory" in lowered or "cudnn_status_alloc_failed" in lowered


def _exception_looks_like_oom(exc: Exception) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cudnn_status_alloc_failed" in message


def evaluate_submission_payload_safely(
    problem: Problem,
    submission_path: Path,
    *,
    backend: str,
    device: Any = "cuda:0",
    precision: str = "fp32",
    arch: str = DEFAULT_ARCH,
    num_correct_trials: int = DEFAULT_NUM_CORRECT_TRIALS,
    num_perf_trials: int = DEFAULT_NUM_PERF_TRIALS,
    num_warmup: int = DEFAULT_NUM_WARMUP,
    run_static_checks: bool = True,
    seed: int = DEFAULT_OFFICIAL_EVAL_SEED,
    timeout_seconds: int = 300,
    in_process: bool = False,
) -> dict[str, Any]:
    if in_process:
        try:
            payload = evaluate_submission(
                problem=problem,
                submission_path=submission_path,
                backend=backend,
                device=device,
                precision=precision,
                arch=arch,
                num_correct_trials=num_correct_trials,
                num_perf_trials=num_perf_trials,
                num_warmup=num_warmup,
                run_static_checks=run_static_checks,
                seed=seed,
            ).to_dict()
        except Exception as exc:
            payload = _build_failure_payload(
                problem,
                backend=backend,
                source_path=submission_path,
                category="oom" if _exception_looks_like_oom(exc) else "evaluator_crash",
                metadata={
                    "runtime_error": str(exc),
                    "evaluator_crash": not _exception_looks_like_oom(exc),
                    "evaluator_error": not _exception_looks_like_oom(exc),
                },
            )
        return annotate_eval_payload(payload, mode="in_process")

    request_payload = {
        "problem": _serialize_problem(problem),
        "submission_path": str(submission_path),
        "backend": backend,
        "device": str(device),
        "precision": precision,
        "arch": arch,
        "num_correct_trials": num_correct_trials,
        "num_perf_trials": num_perf_trials,
        "num_warmup": num_warmup,
        "run_static_checks": run_static_checks,
        "seed": seed,
    }

    with tempfile.TemporaryDirectory(prefix="ptxbench-eval-") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        request_path = tmpdir / "request.json"
        output_path = tmpdir / "output.json"
        request_path.write_text(json.dumps(request_payload, indent=2), encoding="utf-8")

        try:
            process = subprocess.run(
                _worker_command(request_path, output_path),
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            payload = _build_failure_payload(
                problem,
                backend=backend,
                source_path=submission_path,
                category="timeout",
                metadata={
                    "timeout_error": f"evaluation exceeded {timeout_seconds} seconds",
                },
            )
            return annotate_eval_payload(
                payload,
                mode="subprocess",
                timeout_seconds=timeout_seconds,
                stdout=_trim_stream(exc.stdout),
                stderr=_trim_stream(exc.stderr),
            )

        if output_path.exists():
            try:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                payload = _build_failure_payload(
                    problem,
                    backend=backend,
                    source_path=submission_path,
                    category="evaluator_crash",
                    metadata={
                        "evaluator_crash": True,
                        "runtime_error": f"worker produced invalid JSON: {exc}",
                    },
                )
            return annotate_eval_payload(
                payload,
                mode="subprocess",
                timeout_seconds=timeout_seconds,
                returncode=process.returncode,
                stdout=process.stdout,
                stderr=process.stderr,
            )

        category = "oom" if _subprocess_looks_like_oom(process) else "evaluator_crash"
        metadata: dict[str, Any] = {}
        if category == "oom":
            metadata["runtime_error"] = process.stderr.strip() or process.stdout.strip() or "CUDA out of memory"
        else:
            metadata["evaluator_crash"] = True
            metadata["evaluator_error"] = True
            metadata["runtime_error"] = (
                process.stderr.strip()
                or process.stdout.strip()
                or f"worker exited with return code {process.returncode}"
            )
        payload = _build_failure_payload(
            problem,
            backend=backend,
            source_path=submission_path,
            category=category,
            metadata=metadata,
        )
        return annotate_eval_payload(
            payload,
            mode="subprocess",
            timeout_seconds=timeout_seconds,
            returncode=process.returncode,
            stdout=process.stdout,
            stderr=process.stderr,
        )
