from pathlib import Path
import importlib.util
import json

from ptxbench.statistics import wilson_interval


def _load_report_script():
    script_path = Path("scripts/make_paper_report.py").resolve()
    spec = importlib.util.spec_from_file_location("ptxbench_make_paper_report_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wilson_interval_bounds_known_case() -> None:
    low, high = wilson_interval(5, 10)
    assert round(low, 3) == 0.237
    assert round(high, 3) == 0.763


def test_paired_outcome_classification() -> None:
    report = _load_report_script()
    assert report.classify_paired_outcome({"correctness": True, "runtime_ms": 1.0}, {"correctness": True, "runtime_ms": 2.0}) == "ptx_win"
    assert report.classify_paired_outcome({"correctness": True, "runtime_ms": 2.0}, {"correctness": True, "runtime_ms": 1.0}) == "cuda_win"
    assert report.classify_paired_outcome({"correctness": True, "runtime_ms": 1.0}, {"correctness": True, "runtime_ms": 1.0}) == "tie"
    assert report.classify_paired_outcome({"correctness": True}, {"correctness": False}) == "ptx_only_correct"
    assert report.classify_paired_outcome({"correctness": False}, {"correctness": True}) == "cuda_only_correct"
    assert report.classify_paired_outcome({"correctness": False}, {"correctness": False}) == "both_wrong"


def _row(*, backend: str, problem_id: int, correct: bool, runtime_ms: float) -> dict:
    return {
        "backend": backend,
        "problem_id": problem_id,
        "problem_name": f"{problem_id}_Problem.py",
        "track": "oneshot",
        "compiled": correct,
        "assembled": correct if backend == "ptx" else None,
        "loaded": correct if backend == "ptx" else None,
        "correctness": correct,
        "runtime_ms": runtime_ms,
        "ref_runtime_ms": 2.0,
        "ref_runtime_eager_ms": 2.0,
        "ref_runtime_compile_default_ms": 1.5,
        "speedup_vs_torch": 2.0 / runtime_ms if correct else 0.0,
        "speedup_vs_eager": 2.0 / runtime_ms if correct else 0.0,
        "speedup_vs_compile_default": 1.5 / runtime_ms if correct else None,
        "failure_category": "success" if correct else "compile",
        "paper_failure_category": "success_fast_compile" if correct else "import_or_compile_error",
        "metadata": {
            "failure_category": "success" if correct else "compile",
            "paper_failure_category": "success_fast_compile" if correct else "import_or_compile_error",
            "ptx_resource_summary": {
                "num_artifacts": 1,
                "num_functions": 1,
                "max_registers": 32,
                "max_spill_stores_bytes": 0,
                "max_spill_loads_bytes": 0,
                "max_shared_memory_bytes": 0,
                "max_local_memory_bytes": 0,
                "max_constant_memory_bytes": 16,
                "max_stack_frame_bytes": 0,
                "any_spills": False,
            },
        },
    }


def test_make_paper_report_writes_synthetic_outputs(tmp_path, monkeypatch) -> None:
    report = _load_report_script()
    monkeypatch.setattr(report, "REPO_ROOT", tmp_path)
    run_name = "synthetic"
    level = 1
    for backend in ("ptx", "cuda"):
        run_dir = tmp_path / "runs" / run_name / backend / f"level{level}"
        result_dir = tmp_path / "results" / "timing" / run_name / backend / f"level{level}"
        run_dir.mkdir(parents=True)
        result_dir.mkdir(parents=True)
        (run_dir / "run_manifest.json").write_text(
            json.dumps({"provider": "codex", "model": "gpt-5.4", "track": "oneshot"}),
            encoding="utf-8",
        )
        rows = [
            _row(backend=backend, problem_id=1, correct=True, runtime_ms=1.0 if backend == "ptx" else 1.5),
            _row(backend=backend, problem_id=2, correct=False, runtime_ms=-1.0),
        ]
        (result_dir / "summary.json").write_text(json.dumps(rows), encoding="utf-8")

    out_dir = tmp_path / "paper-out"
    assert report.main(["--run-name", run_name, "--levels", "1", "--out-dir", str(out_dir)]) == 0
    for name in (
        "main_results.csv",
        "paired_results.csv",
        "failure_breakdown.csv",
        "paper_failure_breakdown.csv",
        "ptx_resource_metrics.csv",
        "paper_tables.md",
        "report_manifest.json",
    ):
        assert (out_dir / name).exists()
    assert "ptx_win" in (out_dir / "paired_results.csv").read_text(encoding="utf-8")
