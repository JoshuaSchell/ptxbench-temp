from pathlib import Path
import json
import subprocess
import tempfile

from ptxbench.dataset import Problem
from ptxbench.eval import EvalResult
from ptxbench.isolated_eval import evaluate_submission_payload_safely


def _problem() -> Problem:
    return Problem(
        problem_id=19,
        level=1,
        name="19_ReLU.py",
        path=Path("vendor/KernelBench-upstream/KernelBench/level1/19_ReLU.py"),
        code="class Model:\n    pass\n",
    )


def _workspace_tempdir() -> tempfile.TemporaryDirectory[str]:
    base = Path(".pytest_tmp").resolve()
    base.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=base)


def test_isolated_eval_returns_worker_payload(monkeypatch) -> None:
    with _workspace_tempdir() as tmpdir_str:
        submission_path = Path(tmpdir_str) / "submission.py"
        submission_path.write_text("print('submission')\n", encoding="utf-8")

        def fake_run(command, **kwargs):
            output_path = Path(command[command.index("--output") + 1])
            payload = EvalResult(
                backend="ptx",
                problem_id=19,
                problem_name="19_ReLU.py",
                source_path=str(submission_path),
                compiled=True,
                assembled=True,
                loaded=True,
                correctness=True,
                runtime_ms=1.0,
                ref_runtime_ms=2.0,
                speedup_vs_torch=2.0,
            ).to_dict()
            output_path.write_text(json.dumps(payload), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="worker ok", stderr="")

        monkeypatch.setattr("ptxbench.isolated_eval.subprocess.run", fake_run)
        payload = evaluate_submission_payload_safely(
            _problem(),
            submission_path,
            backend="ptx",
            timeout_seconds=12,
        )

        assert payload["correctness"] is True
        assert payload["metadata"]["failure_category"] == "success"
        assert payload["metadata"]["isolated_eval"]["mode"] == "subprocess"
        assert payload["metadata"]["isolated_eval"]["returncode"] == 0


def test_isolated_eval_timeout_returns_failed_payload(monkeypatch) -> None:
    with _workspace_tempdir() as tmpdir_str:
        submission_path = Path(tmpdir_str) / "submission.py"
        submission_path.write_text("print('submission')\n", encoding="utf-8")

        def fake_run(command, **kwargs):
            raise subprocess.TimeoutExpired(command, kwargs["timeout"], output="partial stdout", stderr="still running")

        monkeypatch.setattr("ptxbench.isolated_eval.subprocess.run", fake_run)
        payload = evaluate_submission_payload_safely(
            _problem(),
            submission_path,
            backend="ptx",
            timeout_seconds=7,
        )

        assert payload["correctness"] is False
        assert payload["metadata"]["failure_category"] == "timeout"
        assert payload["metadata"]["isolated_eval"]["mode"] == "subprocess"
        assert payload["metadata"]["isolated_eval"]["timeout_seconds"] == 7


def test_isolated_eval_classifies_subprocess_oom(monkeypatch) -> None:
    with _workspace_tempdir() as tmpdir_str:
        submission_path = Path(tmpdir_str) / "submission.py"
        submission_path.write_text("print('submission')\n", encoding="utf-8")

        def fake_run(command, **kwargs):
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="CUDA out of memory")

        monkeypatch.setattr("ptxbench.isolated_eval.subprocess.run", fake_run)
        payload = evaluate_submission_payload_safely(
            _problem(),
            submission_path,
            backend="ptx",
            timeout_seconds=7,
        )

        assert payload["correctness"] is False
        assert payload["metadata"]["failure_category"] == "oom"
        assert payload["metadata"]["oom_error"] is True


def test_isolated_eval_classifies_worker_crash(monkeypatch) -> None:
    with _workspace_tempdir() as tmpdir_str:
        submission_path = Path(tmpdir_str) / "submission.py"
        submission_path.write_text("print('submission')\n", encoding="utf-8")

        def fake_run(command, **kwargs):
            return subprocess.CompletedProcess(command, 9, stdout="", stderr="worker crashed")

        monkeypatch.setattr("ptxbench.isolated_eval.subprocess.run", fake_run)
        payload = evaluate_submission_payload_safely(
            _problem(),
            submission_path,
            backend="ptx",
            timeout_seconds=7,
        )

        assert payload["correctness"] is False
        assert payload["metadata"]["failure_category"] == "evaluator_crash"
        assert payload["metadata"]["isolated_eval"]["returncode"] == 9
