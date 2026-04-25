from __future__ import annotations

from pathlib import Path
import fnmatch
import py_compile
import subprocess


ROOT_GENERATED_PATTERNS = (
    "tmp_*",
    "tmp*.py",
    "final_*",
    "final*.py",
    "candidate_*",
    "candidate*.py",
    "answer*.py",
    "scratch_*",
    "scratch*.py",
    "modelnew_*",
    "modelnew*.py",
    "model_new*",
    "generated_*",
    "generated*.py",
    "bench_*",
    "bench*.py",
    "submission_*",
    "submission*.py",
    "working_submission.py",
    "wrapper_submission.py",
    "ptxbench_answer*.py",
)

ROOT_GPU_ARTIFACT_PATTERNS = (
    "*.ptx",
    "*.cubin",
    "*.cu",
    "*.cuda",
    "*.o",
    "*.so",
    "*.out",
)


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


def test_generated_artifacts_are_not_tracked() -> None:
    offenders: list[str] = []
    for raw_path in _tracked_files():
        path = Path(raw_path)
        parts = path.parts
        name = path.name

        if any(part.endswith(".egg-info") for part in parts):
            offenders.append(raw_path)
        elif parts and parts[0] in {"build", "dist"}:
            offenders.append(raw_path)
        elif any(part in {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".ptxbench_cache"} for part in parts):
            offenders.append(raw_path)
        elif len(parts) == 1 and any(fnmatch.fnmatch(name, pattern) for pattern in ROOT_GENERATED_PATTERNS):
            offenders.append(raw_path)
        elif len(parts) == 1 and any(fnmatch.fnmatch(name, pattern) for pattern in ROOT_GPU_ARTIFACT_PATTERNS):
            offenders.append(raw_path)

    assert offenders == []


def test_gitignore_has_newlines_and_ignores_root_generated_files() -> None:
    gitignore_path = Path(".gitignore")
    lines = gitignore_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) >= 20

    result = subprocess.run(
        [
            "git",
            "check-ignore",
            "answer.py",
            "tmp_test.ptx",
            "candidate_submission.py",
            "final_kernel.cu",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.splitlines() == [
        "answer.py",
        "tmp_test.ptx",
        "candidate_submission.py",
        "final_kernel.cu",
    ]


def test_launch_entrypoints_compile() -> None:
    for path in (
        Path("scripts/generate_samples.py"),
        Path("scripts/run_level1_paired.py"),
        Path("scripts/run_experiment.py"),
        Path("src/ptxbench/experiment_specs.py"),
    ):
        py_compile.compile(str(path), doraise=True)
