from __future__ import annotations

from pathlib import Path
import sys
import tomllib

from ptxbench.experiment_specs import (
    EXPERIMENT_SPECS_DIR,
    available_experiment_specs,
    build_experiment_command,
    load_experiment_spec,
)


REQUIRED_EXPERIMENT_KEYS = ("run_name", "provider", "model", "track", "level", "phase")


def main() -> None:
    issues: list[str] = []
    specs = available_experiment_specs()
    if not specs:
        issues.append(f"no experiment specs found under {EXPERIMENT_SPECS_DIR}")

    run_names: dict[str, Path] = {}
    for spec_path in specs:
        _check_spec(spec_path, run_names, issues)

    if issues:
        print("FAIL experiment spec check")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)

    print(f"PASS experiment spec check ({len(specs)} specs)")


def _check_spec(spec_path: Path, run_names: dict[str, Path], issues: list[str]) -> None:
    try:
        payload = tomllib.loads(spec_path.read_text(encoding="utf-8"))
        experiment = payload.get("experiment", {})
        for key in REQUIRED_EXPERIMENT_KEYS:
            if key not in experiment:
                issues.append(f"{spec_path}: missing [experiment].{key}")

        spec = load_experiment_spec(spec_path)
    except Exception as exc:
        issues.append(f"{spec_path}: failed to parse/load: {type(exc).__name__}: {exc}")
        return

    previous = run_names.get(spec.run_name)
    if previous is not None:
        issues.append(f"{spec_path}: duplicate run_name {spec.run_name!r}; first seen in {previous}")
    run_names[spec.run_name] = spec_path

    if not spec.locked:
        issues.append(f"{spec_path}: checked-in paper spec must set [lock].locked = true")
    if spec.canonical:
        if spec.phase != "full":
            issues.append(f"{spec_path}: canonical specs must use phase = \"full\"")
        if not (spec.claim_scope or spec.comparison_goal or spec.kernelbench_parity_scope):
            issues.append(f"{spec_path}: canonical specs must declare claim scope or comparison goal")
    if spec.canonical and spec.problem_ids is not None and "spread" in spec.run_name:
        issues.append(f"{spec_path}: spread/subset specs must not be canonical")

    if _requires_reasoning_effort(spec.model, spec.provider) and not spec.reasoning_effort:
        issues.append(f"{spec_path}: GPT-5.5 and Claude specs must record reasoning_effort")

    _check_required_outputs(spec_path, spec.run_name, spec.level, spec.required_outputs, issues)
    _check_dry_run_command(spec_path, spec, issues)


def _requires_reasoning_effort(model: str, provider: str) -> bool:
    return model.startswith("gpt-5.5") or provider == "claude-code" or model.startswith("claude-")


def _check_required_outputs(spec_path: Path, run_name: str, level: int, required_outputs: list[str], issues: list[str]) -> None:
    required = {
        f"runs/{run_name}/paper_run_manifest.json",
        f"results/timing/{run_name}/ptx/level{level}/summary.json",
        f"results/timing/{run_name}/cuda/level{level}/summary.json",
        f"results/analysis/{run_name}_level{level}.json",
        f"results/analysis/{run_name}_level{level}.md",
    }
    observed = set(required_outputs)
    for path in sorted(required - observed):
        issues.append(f"{spec_path}: missing required output {path}")
    for path in required_outputs:
        if "{run_name}" in path or "{level}" in path:
            issues.append(f"{spec_path}: unresolved placeholder in required output {path}")
        if path.startswith("runs/") and not path.startswith(f"runs/{run_name}/"):
            issues.append(f"{spec_path}: run output does not match run_name: {path}")
        if path.startswith("results/timing/") and f"results/timing/{run_name}/" not in path:
            issues.append(f"{spec_path}: timing output does not match run_name: {path}")
        if path.startswith("results/analysis/") and f"{run_name}_level{level}" not in path:
            issues.append(f"{spec_path}: analysis output does not match run_name/level: {path}")


def _check_dry_run_command(spec_path: Path, spec, issues: list[str]) -> None:
    try:
        command = build_experiment_command(spec, python_exe=sys.executable)
    except Exception as exc:
        issues.append(f"{spec_path}: dry-run command build failed: {type(exc).__name__}: {exc}")
        return
    expected = ["scripts/run_level_paired.py", "--provider", spec.provider, "--model", spec.model]
    for token in expected:
        if token not in command:
            issues.append(f"{spec_path}: dry-run command missing {token!r}")


if __name__ == "__main__":
    main()
