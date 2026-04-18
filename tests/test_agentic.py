from pathlib import Path
import json
import shutil
from unittest.mock import patch

import pytest

from ptxbench.agentic import AgenticEpisodeBudget, format_agentic_observation, run_agentic_episode
from ptxbench.dataset import construct_dataset
from ptxbench.eval import evaluate_submission
from ptxbench.providers import ProviderResponse

torch = pytest.importorskip("torch")


def test_agentic_episode_records_budget_and_step_metadata() -> None:
    run_name = "test-agentic-metadata"
    problem = construct_dataset(level=1, problem_ids=[19]).get_problem(19)
    responses = iter(
        [
            "class ModelNew:\n    pass\n",
            "class ModelNew:\n    value = 2\n",
            "class ModelNew:\n    value = 3\n",
        ]
    )

    def fake_provider(**kwargs) -> ProviderResponse:
        return ProviderResponse(content=next(responses), metadata={"provider": "fake", "model": "fake-model"})

    observations = [
        {"static_check": {"valid": False, "errors": ["strict:pass_statement"], "warnings": []}, "compiled": False, "correctness": False},
        {
            "static_check": {"valid": True, "errors": [], "warnings": []},
            "compiled": True,
            "assembled": True,
            "loaded": True,
            "correctness": True,
            "runtime_ms": 1.0,
            "ref_runtime_ms": 2.0,
            "speedup_vs_torch": 2.0,
            "metadata": {},
        },
    ]

    episode_dir = Path("runs") / run_name
    try:
        with patch("ptxbench.agentic.run_agentic_validation_pass", side_effect=observations):
            artifacts = run_agentic_episode(
                problem=problem,
                backend="ptx",
                provider="codex",
                model="fake-model",
                run_name=run_name,
                level=1,
                temperature=0.0,
                max_tokens=100,
                budget=AgenticEpisodeBudget(max_steps=3, max_tool_calls=2, max_wall_clock_minutes=20),
                provider_fn=fake_provider,
            )
        assert artifacts.metadata["track"] == "agentic"
        assert artifacts.metadata["step_count"] == 3
        assert artifacts.metadata["first_compile_step"] == 2
        assert artifacts.metadata["first_correct_step"] == 2
        assert artifacts.metadata["budget_used"]["tool_calls_used"] == 2
        assert Path(artifacts.metadata["episode_manifest_path"]).exists()
        payload = json.loads(artifacts.output_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        assert payload["metadata"]["track"] == "agentic"
        assert payload["metadata"]["final_submission_hash"]
    finally:
        shutil.rmtree(episode_dir, ignore_errors=True)


def test_format_agentic_observation_includes_profile_feedback() -> None:
    rendered = format_agentic_observation(
        {
            "static_check": {"valid": True, "errors": [], "warnings": []},
            "compiled": True,
            "assembled": True,
            "loaded": True,
            "correctness": True,
            "runtime_ms": 1.0,
            "ref_runtime_ms": 2.0,
            "speedup_vs_torch": 2.0,
            "profile": {
                "tool": "ncu",
                "status": "collected",
                "metrics": {
                    "gpu__time_duration.sum": 1234.0,
                    "sm__cycles_active.avg": 56.0,
                },
            },
        }
    )

    assert "dev_eval: correct" in rendered
    assert "profile[ncu]:" in rendered
    assert "gpu__time_duration.sum=" in rendered


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for PTX integration tests")
def test_agentic_ptx_episode_repairs_after_failed_attempt() -> None:
    run_name = "test-agentic-gpu"
    problem = construct_dataset(level=1, problem_ids=[19]).get_problem(19)
    invalid_source = Path("tests/fixtures/submissions/ptx/invalid_ptx_submission.py").read_text(encoding="utf-8")
    valid_source = Path("tests/fixtures/submissions/ptx/relu_submission.py").read_text(encoding="utf-8")
    responses = iter([invalid_source, valid_source, valid_source])

    def fake_provider(**kwargs) -> ProviderResponse:
        return ProviderResponse(content=next(responses), metadata={"provider": "fake", "model": "fake-model"})

    episode_dir = Path("runs") / run_name
    try:
        artifacts = run_agentic_episode(
            problem=problem,
            backend="ptx",
            provider="codex",
            model="fake-model",
            run_name=run_name,
            level=1,
            temperature=0.0,
            max_tokens=100,
            budget=AgenticEpisodeBudget(
                max_steps=3,
                max_tool_calls=2,
                max_wall_clock_minutes=20,
                dev_eval_correct_trials=1,
                dev_eval_perf_trials=2,
            ),
            provider_fn=fake_provider,
        )
        assert artifacts.metadata["first_compile_step"] == 1
        assert artifacts.metadata["first_correct_step"] == 2
        result = evaluate_submission(
            problem=problem,
            submission_path=artifacts.output_path,
            backend="ptx",
            num_correct_trials=1,
            num_perf_trials=2,
        )
        assert result.correctness
    finally:
        shutil.rmtree(episode_dir, ignore_errors=True)
