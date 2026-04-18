from pathlib import Path
import importlib.util
import json
import shutil
import sys

from ptxbench.eval import EvalResult
from ptxbench.run_metadata import default_paper_protocol, protocol_signature, sha256_text


def _load_eval_script():
    script_path = Path("scripts/eval_from_generations.py").resolve()
    spec = importlib.util.spec_from_file_location("ptxbench_eval_from_generations_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _eval_protocol(*, track: str) -> dict:
    protocol = default_paper_protocol(level=1, track=track).to_dict()
    protocol["level"] = 1
    protocol["track"] = track
    protocol["precision"] = "fp32"
    protocol["arch"] = "sm_89"
    protocol["num_correct_trials"] = 5
    protocol["num_perf_trials"] = 100
    return protocol


def test_eval_from_generations_reuses_existing_result_when_hash_and_protocol_match(monkeypatch) -> None:
    eval_script = _load_eval_script()
    run_name = "test-eval-resume"
    run_dir = Path("runs") / run_name / "ptx" / "level1"
    output_dir = Path("results") / "timing" / run_name / "ptx" / "level1"
    submission_src = Path("tests/fixtures/submissions/ptx/relu_submission.py")
    submission_path = run_dir / "019_19_ReLU.py"
    protocol = _eval_protocol(track="agentic")
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        submission_path.write_text(submission_src.read_text(encoding="utf-8"), encoding="utf-8")
        (run_dir / "run_manifest.json").write_text(
            json.dumps({"track": "agentic", "protocol": protocol}),
            encoding="utf-8",
        )
        submission_hash = sha256_text(submission_path.read_text(encoding="utf-8"))
        submission_path.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "prompt": "prompt",
                    "raw_response": "response",
                    "metadata": {
                        "track": "agentic",
                        "episode_id": "episode-19",
                        "step_count": 3,
                        "budget_used": {"tool_calls_used": 2},
                        "first_compile_step": 1,
                        "first_correct_step": 2,
                        "final_submission_hash": submission_hash,
                        "task_family_tags": ["elementwise"],
                    },
                }
            ),
            encoding="utf-8",
        )
        (output_dir / "019.json").write_text(
            json.dumps(
                {
                    "backend": "ptx",
                    "problem_id": 19,
                    "problem_name": "19_ReLU.py",
                    "source_path": str(submission_path),
                    "compiled": True,
                    "assembled": True,
                    "loaded": True,
                    "correctness": True,
                    "runtime_ms": 1.0,
                    "ref_runtime_ms": 2.0,
                    "speedup_vs_torch": 2.0,
                    "metadata": {
                        "eval_protocol_signature": protocol_signature(protocol),
                        "evaluated_submission_hash": submission_hash,
                    },
                }
            ),
            encoding="utf-8",
        )

        def fail_if_re_evaluated(**kwargs):
            raise AssertionError("resume path should not re-evaluate when protocol and hash match")

        monkeypatch.setattr(eval_script, "evaluate_submission", fail_if_re_evaluated)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "eval_from_generations.py",
                "--run-name",
                run_name,
                "--backend",
                "ptx",
                "--level",
                "1",
                "--problem-ids",
                "19",
            ],
        )
        eval_script.main()

        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        assert len(summary) == 1
        assert summary[0]["track"] == "agentic"
        assert summary[0]["episode_id"] == "episode-19"
        assert summary[0]["first_correct_step"] == 2
        assert summary[0]["metadata"]["generation"]["budget_used"]["tool_calls_used"] == 2
    finally:
        shutil.rmtree(Path("runs") / run_name, ignore_errors=True)
        shutil.rmtree(Path("results") / "timing" / run_name, ignore_errors=True)


def test_eval_from_generations_re_evaluates_when_submission_hash_changes(monkeypatch) -> None:
    eval_script = _load_eval_script()
    run_name = "test-eval-resume-stale"
    run_dir = Path("runs") / run_name / "ptx" / "level1"
    output_dir = Path("results") / "timing" / run_name / "ptx" / "level1"
    submission_src = Path("tests/fixtures/submissions/ptx/relu_submission.py")
    submission_path = run_dir / "019_19_ReLU.py"
    protocol = _eval_protocol(track="oneshot")
    calls: list[Path] = []
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        submission_path.write_text(submission_src.read_text(encoding="utf-8"), encoding="utf-8")
        (run_dir / "run_manifest.json").write_text(
            json.dumps({"track": "oneshot", "protocol": protocol}),
            encoding="utf-8",
        )
        (output_dir / "019.json").write_text(
            json.dumps(
                {
                    "backend": "ptx",
                    "problem_id": 19,
                    "problem_name": "19_ReLU.py",
                    "source_path": str(submission_path),
                    "compiled": True,
                    "assembled": True,
                    "loaded": True,
                    "correctness": True,
                    "runtime_ms": 1.0,
                    "ref_runtime_ms": 2.0,
                    "speedup_vs_torch": 2.0,
                    "metadata": {
                        "eval_protocol_signature": protocol_signature(protocol),
                        "evaluated_submission_hash": "stale-hash",
                    },
                }
            ),
            encoding="utf-8",
        )

        def fake_evaluate_submission(**kwargs):
            calls.append(kwargs["submission_path"])
            return EvalResult(
                backend="ptx",
                problem_id=19,
                problem_name="19_ReLU.py",
                source_path=str(kwargs["submission_path"]),
                compiled=True,
                assembled=True,
                loaded=True,
                correctness=True,
                runtime_ms=3.0,
                ref_runtime_ms=6.0,
                speedup_vs_torch=2.0,
            )

        monkeypatch.setattr(eval_script, "evaluate_submission", fake_evaluate_submission)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "eval_from_generations.py",
                "--run-name",
                run_name,
                "--backend",
                "ptx",
                "--level",
                "1",
                "--problem-ids",
                "19",
            ],
        )
        eval_script.main()

        assert calls == [submission_path.resolve()]
        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary[0]["runtime_ms"] == 3.0
        assert summary[0]["metadata"]["evaluated_submission_hash"] == sha256_text(
            submission_path.read_text(encoding="utf-8")
        )
    finally:
        shutil.rmtree(Path("runs") / run_name, ignore_errors=True)
        shutil.rmtree(Path("results") / "timing" / run_name, ignore_errors=True)
