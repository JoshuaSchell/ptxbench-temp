from pathlib import Path
import importlib.util
import sys
import tempfile

from ptxbench.dataset import Problem
from ptxbench.eval import EvalResult


def _load_run_and_check_script():
    script_path = Path("scripts/run_and_check.py").resolve()
    spec = importlib.util.spec_from_file_location("ptxbench_run_and_check_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_run_and_check_defaults_compile_baseline_to_false(monkeypatch, capsys) -> None:
    run_and_check = _load_run_and_check_script()
    calls: list[dict] = []

    def fake_evaluate_submission(**kwargs):
        calls.append(kwargs)
        return EvalResult(
            backend="ptx",
            problem_id=19,
            problem_name="19_ReLU.py",
            source_path=str(kwargs["submission_path"]),
            compiled=True,
            assembled=True,
            loaded=True,
            correctness=True,
            runtime_ms=1.0,
            ref_runtime_ms=2.0,
            speedup_vs_torch=2.0,
        )

    monkeypatch.setattr(run_and_check, "load_problem", lambda *args, **kwargs: _problem())
    monkeypatch.setattr(run_and_check, "evaluate_submission", fake_evaluate_submission)

    with _workspace_tempdir() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        submission_path = tmpdir / "submission.py"
        submission_path.write_text("class ModelNew:\n    pass\n", encoding="utf-8")

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_and_check.py",
                "--backend",
                "ptx",
                "--submission",
                str(submission_path),
                "--reference-file",
                str(tmpdir / "reference.py"),
            ],
        )

        run_and_check.main()

    assert calls
    assert calls[0]["measure_compile_default_baseline"] is False
    captured = capsys.readouterr()
    assert "\"correctness\": true" in captured.out
