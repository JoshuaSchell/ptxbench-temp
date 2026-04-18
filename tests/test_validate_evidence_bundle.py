from pathlib import Path
import importlib.util
import json
import tempfile


def _load_validator_script():
    script_path = Path("scripts/validate_evidence_bundle.py").resolve()
    spec = importlib.util.spec_from_file_location("ptxbench_validate_evidence_bundle_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _workspace_tempdir() -> tempfile.TemporaryDirectory[str]:
    base = Path(".pytest_tmp").resolve()
    base.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=base)


def _result_payload(*, backend: str, problem_id: int) -> dict:
    return {
        "backend": backend,
        "problem_id": problem_id,
        "problem_name": "19_ReLU.py",
        "track": "oneshot",
        "submission_hash": "deadbeef",
        "compiled": True,
        "assembled": True if backend == "ptx" else None,
        "loaded": True if backend == "ptx" else None,
        "correctness": True,
        "runtime_ms": 1.0,
        "ref_runtime_ms": 2.0,
        "ref_runtime_eager_ms": 2.0,
        "ref_runtime_compile_default_ms": 1.5,
        "speedup_vs_torch": 2.0,
        "speedup_vs_eager": 2.0,
        "speedup_vs_compile_default": 1.5,
        "failure_category": "success",
        "num_correct_trials": 5,
        "num_perf_trials": 100,
        "seed": 42,
        "arch": "sm_89",
        "precision": "fp32",
        "gpu_name": "Fake GPU",
        "torch_version": "2.8.0",
        "cuda_version": "12.8",
        "ptxas_version": "12.8",
        "repo_commit": "repo-commit",
        "kernelbench_commit": "kernelbench-commit",
        "metadata": {
            "failure_category": "success",
        },
    }


def test_validate_evidence_bundle_accepts_tiny_fake_run() -> None:
    with _workspace_tempdir() as tmpdir_str:
        tmp_path = Path(tmpdir_str)
        validator = _load_validator_script()
        run_name = "tiny-run"
        level = 1
        backends = ["ptx", "cuda"]
        problem_ids = [19]

        (tmp_path / "runs" / run_name).mkdir(parents=True, exist_ok=True)
        (tmp_path / "results" / "analysis").mkdir(parents=True, exist_ok=True)

        (tmp_path / "runs" / run_name / "paper_run_manifest.json").write_text(
            json.dumps({"run_name": run_name, "track": "oneshot", "problem_ids": problem_ids}),
            encoding="utf-8",
        )
        (tmp_path / "results" / "analysis" / f"{run_name}_level{level}.json").write_text(
            json.dumps(
                {
                    "run_name": run_name,
                    "level": level,
                    "track": "oneshot",
                    "backend_summaries": {"ptx": {}, "cuda": {}},
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "results" / "analysis" / f"{run_name}_level{level}.md").write_text("# analysis\n", encoding="utf-8")

        for backend in backends:
            run_dir = tmp_path / "runs" / run_name / backend / f"level{level}"
            results_dir = tmp_path / "results" / "timing" / run_name / backend / f"level{level}"
            run_dir.mkdir(parents=True, exist_ok=True)
            results_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "run_name": run_name,
                        "backend": backend,
                        "track": "oneshot",
                        "level": level,
                        "problems": problem_ids,
                    }
                ),
                encoding="utf-8",
            )
            (results_dir / "eval_manifest.json").write_text(
                json.dumps(
                    {
                        "run_name": run_name,
                        "backend": backend,
                        "track": "oneshot",
                        "level": level,
                        "problem_ids": problem_ids,
                    }
                ),
                encoding="utf-8",
            )
            payload = _result_payload(backend=backend, problem_id=19)
            (results_dir / "019.json").write_text(json.dumps(payload), encoding="utf-8")
            (results_dir / "summary.json").write_text(json.dumps([payload]), encoding="utf-8")

        valid, issues, stats = validator.validate_evidence_bundle(
            repo_root=tmp_path,
            run_name=run_name,
            level=level,
            track="oneshot",
            backends=backends,
        )

        assert valid is True
        assert issues == []
        assert stats["backends"] == 2
        assert stats["problems"] == 2
