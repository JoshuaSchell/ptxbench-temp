from pathlib import Path
import importlib.util
import shutil
import subprocess
import sys

from ptxbench.dataset import construct_dataset
from ptxbench.eval import build_missing_submission_result
from ptxbench.generation import clear_generation_failure, default_run_dir, generation_failure_path, write_generation_failure
from ptxbench.workflow import (
    GenerationChunkTask,
    chunk_metadata_dir,
    chunk_problem_ids,
    resolve_problem_ids,
    write_backend_generation_summary,
)


def _load_build_generation_command():
    module_path = Path("scripts/run_level1_paired.py")
    spec = importlib.util.spec_from_file_location("run_level1_paired_for_tests", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_generation_command


def test_generation_failure_artifacts_round_trip() -> None:
    scratch = Path("tests") / ".tmp_generation_flow"
    scratch.mkdir(parents=True, exist_ok=True)
    output_path = scratch / "001_sample.py"
    try:
        failure_path = write_generation_failure(
            output_path,
            prompt="prompt text",
            metadata={"problem_id": 1, "error_type": "TimeoutError"},
        )

        assert failure_path == generation_failure_path(output_path)
        assert failure_path.exists()
        assert '"error_type": "TimeoutError"' in failure_path.read_text(encoding="utf-8")

        clear_generation_failure(output_path)
        assert not failure_path.exists()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_build_missing_submission_result_counts_as_failure() -> None:
    problem = construct_dataset(level=1, problem_ids=[19]).get_problem(19)
    result = build_missing_submission_result(
        problem,
        backend="ptx",
        expected_path=Path("runs/missing_submission.py"),
        metadata={"missing_submission": True},
    )

    payload = result.to_dict()
    assert payload["compiled"] is False
    assert payload["correctness"] is False
    assert payload["runtime_ms"] == -1.0
    assert payload["ref_runtime_ms"] == -1.0
    assert payload["metadata"]["missing_submission"] is True


def test_chunk_problem_ids_splits_into_stable_batches() -> None:
    assert chunk_problem_ids([1, 2, 3, 4, 5, 6, 7], 3) == [[1, 2, 3], [4, 5, 6], [7]]


def test_resolve_problem_ids_uses_level_specific_pilot_ids() -> None:
    assert resolve_problem_ids("pilot", None, 4) == [1, 5, 10, 15, 20]


def test_chunk_metadata_dir_uses_isolated_chunk_folder() -> None:
    run_dir = Path("runs/demo/ptx/level1")
    assert chunk_metadata_dir(run_dir, None) == run_dir
    assert chunk_metadata_dir(run_dir, "chunk_003") == run_dir / "_chunks" / "chunk_003"


def test_generation_chunk_task_exposes_stable_chunk_label() -> None:
    task = GenerationChunkTask(backend="ptx", level=1, chunk_index=7, chunk_total=20, problem_ids=[31, 32, 33])
    assert task.chunk_label == "chunk_007"


def test_write_backend_generation_summary_collects_failed_problem_ids() -> None:
    run_name = "test-parallel-summary"
    run_dir = default_run_dir(run_name, "ptx", 1)
    problem = construct_dataset(level=1, problem_ids=[19]).get_problem(19)
    submission_path = run_dir / f"{problem.problem_id:03d}_{problem.path.stem}.py"
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        write_generation_failure(
            submission_path,
            prompt="prompt text",
            metadata={"problem_id": problem.problem_id, "error_type": "TimeoutError"},
        )

        write_backend_generation_summary(
            run_name=run_name,
            backend="ptx",
            level=1,
            problem_ids=[19],
            chunk_total=4,
        )

        summary_path = run_dir / "generation_summary.json"
        payload = summary_path.read_text(encoding="utf-8")
        assert '"failed": 1' in payload
        assert '"missing": 0' in payload
        assert '"failed_problem_ids": [\n    19\n  ]' in payload
    finally:
        shutil.rmtree(run_dir.parent.parent, ignore_errors=True)


def test_generate_samples_dry_run_accepts_model_metadata_flags() -> None:
    run_name = "test-generate-metadata-flags"
    command = [
        sys.executable,
        "scripts/generate_samples.py",
        "--provider",
        "litellm",
        "--backend",
        "ptx",
        "--level",
        "1",
        "--run-name",
        run_name,
        "--model",
        "dummy-model",
        "--reasoning-effort",
        "medium",
        "--model-verbosity",
        "low",
        "--provider-extra-arg",
        "metadata-only=true",
        "--model-family",
        "dummy-family",
        "--paper-model-label",
        "dummy_label",
        "--claim-scope",
        "pilot",
        "--problem-ids",
        "19",
        "--dry-run",
    ]
    run_dir = default_run_dir(run_name, "ptx", 1)
    try:
        subprocess.run(command, check=True, cwd=Path.cwd())
        manifest = (run_dir / "run_manifest.json").read_text(encoding="utf-8")
        assert '"reasoning_effort": "medium"' in manifest
        assert '"model_verbosity": "low"' in manifest
        assert '"provider_extra_args": [\n      "metadata-only=true"\n    ]' in manifest
        assert '"claim_scope": [\n    "pilot"\n  ]' in manifest
    finally:
        shutil.rmtree(run_dir.parent.parent, ignore_errors=True)


def test_build_generation_command_is_generate_samples_dry_run_compatible() -> None:
    run_name = "test-build-generation-command"
    build_generation_command = _load_build_generation_command()
    command = build_generation_command(
        python_exe=sys.executable,
        provider="litellm",
        model="dummy-model",
        reasoning_effort=None,
        model_verbosity=None,
        provider_extra_args=[],
        model_family=None,
        paper_model_label=None,
        claim_scope=[],
        track="oneshot",
        backend="ptx",
        level=1,
        run_name=run_name,
        problem_ids=[19],
        arch="sm_89",
        timeout_seconds=1,
        official_eval_seed=42,
        max_steps=5,
        max_wall_clock_minutes=10,
        max_tool_calls=4,
        dev_eval_seed=7,
        dev_eval_correct_trials=2,
        dev_eval_perf_trials=5,
        dev_eval_profile_enabled=False,
        dev_eval_profile_trials=1,
        dev_eval_profile_metrics=[],
        codex_bin="codex",
        codex_sandbox="read-only",
        codex_home=None,
        codex_config=[],
        claude_bin="claude",
        claude_extra_args=[],
        chunk_label="chunk_001",
    )
    run_dir = default_run_dir(run_name, "ptx", 1)
    try:
        assert "--reasoning-effort" not in command
        assert "--model-verbosity" not in command
        subprocess.run([*command, "--dry-run"], check=True, cwd=Path.cwd())
        assert (run_dir / "_chunks" / "chunk_001" / "run_manifest.json").exists()
    finally:
        shutil.rmtree(run_dir.parent.parent, ignore_errors=True)
