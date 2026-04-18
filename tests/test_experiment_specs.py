from pathlib import Path
import shutil

from ptxbench.experiment_specs import (
    EXPERIMENT_SPECS_DIR,
    available_experiment_specs,
    build_wsl_experiment_command,
    load_experiment_spec,
    render_experiment_summary,
    resolve_experiment_spec_path,
)


def test_available_experiment_specs_lists_checked_in_specs() -> None:
    specs = available_experiment_specs()
    names = {path.name for path in specs}
    assert "level1_matched_oneshot_gpt54.toml" in names
    assert "level1_matched_agentic_gpt54.toml" in names
    assert "level1_pilot_oneshot_gpt54.toml" in names
    assert "level1_pilot_agentic_gpt54.toml" in names


def test_resolve_experiment_spec_path_supports_basename_lookup() -> None:
    resolved = resolve_experiment_spec_path("level1_matched_agentic_gpt54")
    assert resolved == (EXPERIMENT_SPECS_DIR / "level1_matched_agentic_gpt54.toml").resolve()


def test_load_experiment_spec_reads_agentic_budget() -> None:
    spec = load_experiment_spec(EXPERIMENT_SPECS_DIR / "level1_matched_agentic_gpt54.toml")
    assert spec.track == "agentic"
    assert spec.level == 1
    assert spec.locked is True
    assert spec.canonical is True
    assert spec.machine_label == "WSL2 Ubuntu on local RTX 4090"
    assert spec.official_eval_seed == 42
    assert spec.max_steps == 5
    assert spec.max_tool_calls == 4
    assert spec.dev_eval_seed == 7
    assert spec.dev_eval_correct_trials == 2
    assert spec.run_name == "level1-matched-agentic-gpt54"
    assert spec.required_outputs


def test_load_pilot_spec_reads_fixed_problem_subset() -> None:
    spec = load_experiment_spec(EXPERIMENT_SPECS_DIR / "level1_pilot_agentic_gpt54.toml")
    assert spec.phase == "pilot"
    assert spec.locked is True
    assert spec.canonical is False
    assert spec.problem_ids == [1, 3, 19, 23, 40]
    assert spec.official_eval_seed == 42
    assert spec.dev_eval_seed == 7


def test_build_wsl_experiment_command_includes_agentic_flags() -> None:
    spec = load_experiment_spec(EXPERIMENT_SPECS_DIR / "level1_matched_agentic_gpt54.toml")
    command = build_wsl_experiment_command(spec, python_exe="python")
    assert command[:2] == ["python", "scripts/run_level1_paired_wsl.py"]
    assert "--track" in command
    assert "agentic" in command
    assert "--official-eval-seed" in command
    assert "--max-steps" in command
    assert "--dev-eval-seed" in command
    assert "--dev-eval-correct-trials" in command


def test_build_wsl_experiment_command_omits_agentic_flags_for_oneshot() -> None:
    spec = load_experiment_spec(EXPERIMENT_SPECS_DIR / "level1_matched_oneshot_gpt54.toml")
    command = build_wsl_experiment_command(spec, python_exe="python")
    assert "--track" in command
    assert "oneshot" in command
    assert "--official-eval-seed" in command
    assert "--max-steps" not in command
    assert "--dev-eval-seed" not in command
    assert "--dev-eval-correct-trials" not in command


def test_build_wsl_experiment_command_includes_pilot_problem_ids() -> None:
    spec = load_experiment_spec(EXPERIMENT_SPECS_DIR / "level1_pilot_oneshot_gpt54.toml")
    command = build_wsl_experiment_command(spec, python_exe="python")
    assert "--problem-ids" in command
    assert "1,3,19,23,40" in command


def test_render_experiment_summary_mentions_notes() -> None:
    spec = load_experiment_spec(EXPERIMENT_SPECS_DIR / "level1_matched_oneshot_gpt54.toml")
    summary = render_experiment_summary(spec)
    assert "level1-matched-oneshot-gpt54" in summary
    assert "locked/canonical: True / True" in summary
    assert "comparison_goal:" in summary
    assert "required_outputs:" in summary
    assert "notes:" in summary
    assert "official_eval_seed: 42" in summary
    assert "KernelBench-style one-shot claims" in summary


def test_load_experiment_spec_reads_optional_agentic_profiler() -> None:
    scratch = Path("tests") / ".tmp_experiment_specs"
    scratch.mkdir(parents=True, exist_ok=True)
    spec_path = scratch / "agentic_profile.toml"
    try:
        spec_path.write_text(
            """
[experiment]
name = "profiled-agentic"
run_name = "profiled-agentic-run"
model = "gpt-5.4"
track = "agentic"
level = 1

[agentic]
profile_enabled = true
profile_trials = 2
profile_metrics = ["gpu__time_duration.sum", "sm__cycles_active.avg"]
""".strip(),
            encoding="utf-8",
        )

        spec = load_experiment_spec(spec_path)

        assert spec.dev_eval_profile_enabled is True
        assert spec.dev_eval_profile_trials == 2
        assert spec.dev_eval_profile_metrics == ["gpu__time_duration.sum", "sm__cycles_active.avg"]

        command = build_wsl_experiment_command(spec, python_exe="python")
        assert "--dev-eval-profile" in command
        assert "--dev-eval-profile-trials" in command
        assert "2" in command
        assert "--dev-eval-profile-metric" in command
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
