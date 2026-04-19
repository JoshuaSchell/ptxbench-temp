from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shlex
import tomllib

from .config import (
    DEFAULT_AGENTIC_MAX_STEPS,
    DEFAULT_AGENTIC_MAX_TOOL_CALLS,
    DEFAULT_AGENTIC_MAX_WALL_CLOCK_MINUTES,
    DEFAULT_ARCH,
    DEFAULT_DEV_EVAL_CORRECT_TRIALS,
    DEFAULT_DEV_EVAL_PROFILE_ENABLED,
    DEFAULT_DEV_EVAL_PROFILE_METRICS,
    DEFAULT_DEV_EVAL_PROFILE_TRIALS,
    DEFAULT_DEV_EVAL_PERF_TRIALS,
    DEFAULT_DEV_EVAL_SEED,
    DEFAULT_DEV_EVAL_TIMEOUT_SECONDS,
    DEFAULT_GENERATION_TIMEOUT_SECONDS,
    DEFAULT_LEVELS,
    DEFAULT_NUM_CORRECT_TRIALS,
    DEFAULT_NUM_PERF_TRIALS,
    DEFAULT_OFFICIAL_EVAL_SEED,
    DEFAULT_PRECISION,
)
from .run_metadata import normalize_problem_ids


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_SPECS_DIR = REPO_ROOT / "experiments"


@dataclass(frozen=True)
class ExperimentSpec:
    spec_path: Path
    name: str
    description: str
    run_name: str
    spec_version: str
    phase: str
    provider: str
    model: str
    track: str
    level: int
    arch: str = DEFAULT_ARCH
    precision: str = DEFAULT_PRECISION
    num_correct_trials: int = DEFAULT_NUM_CORRECT_TRIALS
    num_perf_trials: int = DEFAULT_NUM_PERF_TRIALS
    official_eval_seed: int = DEFAULT_OFFICIAL_EVAL_SEED
    timeout_seconds: int = DEFAULT_GENERATION_TIMEOUT_SECONDS
    problem_ids: list[int] | None = None
    chunk_size: int | None = None
    max_concurrent_chunks: int | None = None
    parallel_backends: bool = False
    device: str = "cuda:0"
    codex_bin: str = "codex"
    codex_home: str | None = None
    codex_config: list[str] = field(default_factory=list)
    max_steps: int = DEFAULT_AGENTIC_MAX_STEPS
    max_wall_clock_minutes: int = DEFAULT_AGENTIC_MAX_WALL_CLOCK_MINUTES
    max_tool_calls: int = DEFAULT_AGENTIC_MAX_TOOL_CALLS
    dev_eval_seed: int = DEFAULT_DEV_EVAL_SEED
    dev_eval_correct_trials: int = DEFAULT_DEV_EVAL_CORRECT_TRIALS
    dev_eval_perf_trials: int = DEFAULT_DEV_EVAL_PERF_TRIALS
    dev_eval_timeout_seconds: int = DEFAULT_DEV_EVAL_TIMEOUT_SECONDS
    dev_eval_profile_enabled: bool = DEFAULT_DEV_EVAL_PROFILE_ENABLED
    dev_eval_profile_trials: int = DEFAULT_DEV_EVAL_PROFILE_TRIALS
    dev_eval_profile_metrics: list[str] = field(default_factory=lambda: list(DEFAULT_DEV_EVAL_PROFILE_METRICS))
    locked: bool = False
    canonical: bool = False
    machine_label: str = ""
    comparison_goal: str = ""
    kernelbench_parity_scope: str | None = None
    required_outputs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def problem_ids_arg(self) -> str | None:
        if not self.problem_ids:
            return None
        return ",".join(str(problem_id) for problem_id in self.problem_ids)


def available_experiment_specs(spec_root: Path | None = None) -> list[Path]:
    root = spec_root or EXPERIMENT_SPECS_DIR
    if not root.exists():
        return []
    return sorted(root.glob("*.toml"))


def resolve_experiment_spec_path(spec_ref: str, spec_root: Path | None = None) -> Path:
    root = spec_root or EXPERIMENT_SPECS_DIR
    candidate = Path(spec_ref)
    if candidate.exists():
        return candidate.resolve()
    root_candidate = root / spec_ref
    if root_candidate.exists():
        return root_candidate.resolve()
    suffix_candidate = root / f"{spec_ref}.toml"
    if suffix_candidate.exists():
        return suffix_candidate.resolve()
    raise FileNotFoundError(f"Experiment spec not found: {spec_ref}")


def load_experiment_spec(spec_path: Path) -> ExperimentSpec:
    payload = tomllib.loads(spec_path.read_text(encoding="utf-8"))
    experiment = payload.get("experiment", {})
    agentic = payload.get("agentic", {})
    lock = payload.get("lock", {})
    claims = payload.get("claims", {})
    evidence = payload.get("evidence", {})
    notes = payload.get("notes", {})

    track = str(experiment.get("track", "oneshot"))
    level = int(experiment.get("level", 1))
    if track not in {"oneshot", "agentic"}:
        raise ValueError(f"Unsupported track in {spec_path}: {track}")
    if level not in DEFAULT_LEVELS:
        raise ValueError(f"Unsupported level in {spec_path}: {level}")

    raw_problem_ids = experiment.get("problem_ids")
    problem_ids = None
    if raw_problem_ids is not None:
        problem_ids = normalize_problem_ids([int(value) for value in raw_problem_ids])

    return ExperimentSpec(
        spec_path=spec_path.resolve(),
        name=str(experiment["name"]),
        description=str(experiment.get("description", "")),
        run_name=str(experiment["run_name"]),
        spec_version=str(experiment.get("spec_version", "v1")),
        phase=str(experiment.get("phase", "pilot")),
        provider=str(experiment.get("provider", "codex")),
        model=str(experiment["model"]),
        track=track,
        level=level,
        arch=str(experiment.get("arch", DEFAULT_ARCH)),
        precision=str(experiment.get("precision", DEFAULT_PRECISION)),
        num_correct_trials=int(experiment.get("num_correct_trials", DEFAULT_NUM_CORRECT_TRIALS)),
        num_perf_trials=int(experiment.get("num_perf_trials", DEFAULT_NUM_PERF_TRIALS)),
        official_eval_seed=int(experiment.get("official_eval_seed", DEFAULT_OFFICIAL_EVAL_SEED)),
        timeout_seconds=int(experiment.get("timeout_seconds", DEFAULT_GENERATION_TIMEOUT_SECONDS)),
        problem_ids=problem_ids,
        chunk_size=_optional_int(experiment.get("chunk_size")),
        max_concurrent_chunks=_optional_int(experiment.get("max_concurrent_chunks")),
        parallel_backends=bool(experiment.get("parallel_backends", False)),
        device=str(experiment.get("device", "cuda:0")),
        codex_bin=str(experiment.get("codex_bin", "codex")),
        codex_home=_optional_str(experiment.get("codex_home")),
        codex_config=[str(value) for value in experiment.get("codex_config", [])],
        max_steps=int(agentic.get("max_steps", DEFAULT_AGENTIC_MAX_STEPS)),
        max_wall_clock_minutes=int(agentic.get("max_wall_clock_minutes", DEFAULT_AGENTIC_MAX_WALL_CLOCK_MINUTES)),
        max_tool_calls=int(agentic.get("max_tool_calls", DEFAULT_AGENTIC_MAX_TOOL_CALLS)),
        dev_eval_seed=int(agentic.get("dev_eval_seed", DEFAULT_DEV_EVAL_SEED)),
        dev_eval_correct_trials=int(agentic.get("dev_eval_correct_trials", DEFAULT_DEV_EVAL_CORRECT_TRIALS)),
        dev_eval_perf_trials=int(agentic.get("dev_eval_perf_trials", DEFAULT_DEV_EVAL_PERF_TRIALS)),
        dev_eval_timeout_seconds=int(agentic.get("dev_eval_timeout_seconds", DEFAULT_DEV_EVAL_TIMEOUT_SECONDS)),
        dev_eval_profile_enabled=bool(agentic.get("profile_enabled", DEFAULT_DEV_EVAL_PROFILE_ENABLED)),
        dev_eval_profile_trials=int(agentic.get("profile_trials", DEFAULT_DEV_EVAL_PROFILE_TRIALS)),
        dev_eval_profile_metrics=[str(value) for value in agentic.get("profile_metrics", DEFAULT_DEV_EVAL_PROFILE_METRICS)],
        locked=bool(lock.get("locked", False)),
        canonical=bool(lock.get("canonical", False)),
        machine_label=str(lock.get("machine_label", "")),
        comparison_goal=str(claims.get("comparison_goal", "")),
        kernelbench_parity_scope=_optional_str(claims.get("kernelbench_parity_scope")),
        required_outputs=[str(value) for value in evidence.get("required_outputs", [])],
        notes=[str(value) for value in notes.get("items", [])],
    )


def build_experiment_command(spec: ExperimentSpec, *, python_exe: str) -> list[str]:
    command = [
        python_exe,
        "scripts/run_level_paired.py",
        "--run-name",
        spec.run_name,
        "--phase",
        spec.phase,
        "--provider",
        spec.provider,
        "--track",
        spec.track,
        "--model",
        spec.model,
        "--level",
        str(spec.level),
        "--arch",
        spec.arch,
        "--precision",
        spec.precision,
        "--num-correct-trials",
        str(spec.num_correct_trials),
        "--num-perf-trials",
        str(spec.num_perf_trials),
        "--official-eval-seed",
        str(spec.official_eval_seed),
        "--timeout-seconds",
        str(spec.timeout_seconds),
        "--device",
        spec.device,
        "--codex-bin",
        spec.codex_bin,
    ]
    if spec.problem_ids_arg:
        command.extend(["--problem-ids", spec.problem_ids_arg])
    if spec.chunk_size is not None:
        command.extend(["--chunk-size", str(spec.chunk_size)])
    if spec.max_concurrent_chunks is not None:
        command.extend(["--max-concurrent-chunks", str(spec.max_concurrent_chunks)])
    if spec.codex_home:
        command.extend(["--codex-home", spec.codex_home])
    for config_override in spec.codex_config:
        command.extend(["--codex-config", config_override])
    if spec.parallel_backends:
        command.append("--parallel-backends")
    if spec.track == "agentic":
        command.extend(
            [
                "--max-steps",
                str(spec.max_steps),
                "--max-wall-clock-minutes",
                str(spec.max_wall_clock_minutes),
                "--max-tool-calls",
                str(spec.max_tool_calls),
                "--dev-eval-seed",
                str(spec.dev_eval_seed),
                "--dev-eval-correct-trials",
                str(spec.dev_eval_correct_trials),
                "--dev-eval-perf-trials",
                str(spec.dev_eval_perf_trials),
            ]
        )
        if spec.dev_eval_profile_enabled:
            command.extend(
                [
                    "--dev-eval-profile",
                    "--dev-eval-profile-trials",
                    str(spec.dev_eval_profile_trials),
                ]
            )
            for metric in spec.dev_eval_profile_metrics:
                command.extend(["--dev-eval-profile-metric", metric])
    return command


def render_experiment_summary(spec: ExperimentSpec) -> str:
    lines = [
        f"name: {spec.name}",
        f"spec: {spec.spec_path}",
        f"description: {spec.description}",
        f"run_name: {spec.run_name}",
        f"spec_version: {spec.spec_version}",
        f"locked/canonical: {spec.locked} / {spec.canonical}",
        f"track: {spec.track}",
        f"phase: {spec.phase}",
        f"provider/model: {spec.provider} / {spec.model}",
        f"level: {spec.level}",
        f"arch/precision: {spec.arch} / {spec.precision}",
        f"official eval trials: correctness={spec.num_correct_trials}, perf={spec.num_perf_trials}",
        f"official_eval_seed: {spec.official_eval_seed}",
        f"generation timeout_seconds: {spec.timeout_seconds}",
        f"problem_ids: {spec.problem_ids_arg or 'default for phase'}",
    ]
    if spec.machine_label:
        lines.append(f"machine: {spec.machine_label}")
    if spec.comparison_goal:
        lines.append(f"comparison_goal: {spec.comparison_goal}")
    if spec.kernelbench_parity_scope:
        lines.append(f"kernelbench_parity_scope: {spec.kernelbench_parity_scope}")
    if spec.track == "agentic":
        lines.append(
            "agentic budget: "
            f"max_steps={spec.max_steps}, max_wall_clock_minutes={spec.max_wall_clock_minutes}, "
            f"max_tool_calls={spec.max_tool_calls}, dev_eval_seed={spec.dev_eval_seed}, "
            f"dev_eval_correct_trials={spec.dev_eval_correct_trials}, "
            f"dev_eval_perf_trials={spec.dev_eval_perf_trials}, "
            f"dev_eval_timeout_seconds={spec.dev_eval_timeout_seconds}"
        )
        if spec.dev_eval_profile_enabled:
            lines.append(
                "agentic profiler: "
                f"enabled=True, profile_trials={spec.dev_eval_profile_trials}, "
                f"profile_metrics={','.join(spec.dev_eval_profile_metrics)}"
            )
    if spec.required_outputs:
        lines.append("required_outputs:")
        lines.extend(f"- {path}" for path in spec.required_outputs)
    if spec.notes:
        lines.append("notes:")
        lines.extend(f"- {note}" for note in spec.notes)
    return "\n".join(lines)


def shell_render_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
