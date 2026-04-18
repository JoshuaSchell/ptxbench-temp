from pathlib import Path
import shutil

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
