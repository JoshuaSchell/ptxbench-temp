from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import json
import platform
import subprocess
import sys
import threading

from ptxbench.config import (
    DEFAULT_ARCH,
    DEFAULT_AGENTIC_MAX_STEPS,
    DEFAULT_AGENTIC_MAX_TOOL_CALLS,
    DEFAULT_AGENTIC_MAX_WALL_CLOCK_MINUTES,
    DEFAULT_DEV_EVAL_CORRECT_TRIALS,
    DEFAULT_DEV_EVAL_PERF_TRIALS,
    DEFAULT_DEV_EVAL_SEED,
    DEFAULT_FULL_GENERATION_TIMEOUT_SECONDS,
    DEFAULT_GENERATION_TIMEOUT_SECONDS,
    DEFAULT_GENERATION_CHUNK_SIZE,
    DEFAULT_LEVELS,
    DEFAULT_NUM_CORRECT_TRIALS,
    DEFAULT_NUM_PERF_TRIALS,
    DEFAULT_OFFICIAL_EVAL_SEED,
    DEFAULT_PRECISION,
    DEFAULT_TRACK,
    LEVEL_PILOT_PROBLEM_IDS,
    REPO_ROOT,
)
from ptxbench.dataset import construct_dataset
from ptxbench.generation import default_run_dir, generation_failure_path, prompt_template_hash
from ptxbench.run_metadata import default_paper_protocol, detect_runtime_environment, normalize_problem_ids
from ptxbench.workflow import (
    GenerationChunkTask,
    chunk_problem_ids,
    inspect_chunk_generation,
    resolve_problem_ids,
    update_chunk_status,
    write_backend_generation_summary,
)


def run_command(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True, cwd=str(REPO_ROOT))


def write_paper_run_manifest(
    *,
    run_name: str,
    phase: str,
    provider: str,
    model: str | None,
    reasoning_effort: str | None,
    model_verbosity: str | None,
    provider_extra_args: list[str],
    model_family: str | None,
    paper_model_label: str | None,
    claim_scope: list[str],
    codex_config: list[str],
    claude_extra_args: list[str],
    level: int,
    track: str,
    problem_ids: list[int],
    arch: str,
    precision: str,
    timeout_seconds: int,
    num_correct_trials: int,
    num_perf_trials: int,
    official_eval_seed: int,
    max_steps: int,
    max_wall_clock_minutes: int,
    max_tool_calls: int,
    dev_eval_seed: int,
    dev_eval_correct_trials: int,
    dev_eval_perf_trials: int,
    dev_eval_profile_enabled: bool,
    dev_eval_profile_trials: int,
    dev_eval_profile_metrics: list[str],
    required_outputs: list[str] | None = None,
) -> None:
    protocol = default_paper_protocol(level=level, track=track).to_dict()
    protocol["level"] = level
    protocol["track"] = track
    protocol["arch"] = arch
    protocol["precision"] = precision
    protocol["num_correct_trials"] = num_correct_trials
    protocol["num_perf_trials"] = num_perf_trials
    protocol["official_eval_seed"] = official_eval_seed
    protocol["generation_timeout_seconds"] = timeout_seconds
    if track == "agentic":
        protocol["max_steps"] = max_steps
        protocol["max_wall_clock_minutes"] = max_wall_clock_minutes
        protocol["max_tool_calls"] = max_tool_calls
        protocol["dev_eval_seed"] = dev_eval_seed
        protocol["dev_eval_correct_trials"] = dev_eval_correct_trials
        protocol["dev_eval_perf_trials"] = dev_eval_perf_trials
        protocol["dev_eval_profile_enabled"] = dev_eval_profile_enabled
        protocol["dev_eval_profile_trials"] = dev_eval_profile_trials if dev_eval_profile_enabled else None
        protocol["dev_eval_profile_metrics"] = dev_eval_profile_metrics if dev_eval_profile_enabled else []
    manifest = {
        "run_name": run_name,
        "phase": phase,
        "provider": provider,
        "model": model,
        "model_metadata": {
            "model_family": model_family,
            "paper_model_label": paper_model_label,
            "reasoning_effort": reasoning_effort,
            "model_verbosity": model_verbosity,
            "provider_extra_args": list(provider_extra_args),
            "codex_config": list(codex_config),
            "claude_extra_args": list(claude_extra_args),
        },
        "claim_scope": list(claim_scope),
        "track": track,
        "problem_ids": problem_ids,
        "protocol": protocol,
        "environment": detect_runtime_environment(),
    }
    if required_outputs:
        manifest["required_outputs"] = list(required_outputs)
    manifest_path = REPO_ROOT / "runs" / run_name / "paper_run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_backend_run_manifest(
    *,
    run_name: str,
    backend: str,
    level: int,
    track: str,
    provider: str,
    model: str,
    reasoning_effort: str | None,
    model_verbosity: str | None,
    provider_extra_args: list[str],
    model_family: str | None,
    paper_model_label: str | None,
    claim_scope: list[str],
    codex_config: list[str],
    claude_extra_args: list[str],
    problem_ids: list[int],
    arch: str,
    precision: str,
    num_correct_trials: int,
    num_perf_trials: int,
    timeout_seconds: int,
    official_eval_seed: int,
    max_steps: int,
    max_wall_clock_minutes: int,
    max_tool_calls: int,
    dev_eval_seed: int,
    dev_eval_correct_trials: int,
    dev_eval_perf_trials: int,
    dev_eval_profile_enabled: bool,
    dev_eval_profile_trials: int,
    dev_eval_profile_metrics: list[str],
) -> None:
    problems = list(construct_dataset(level=level, problem_ids=problem_ids))
    run_dir = default_run_dir(run_name, backend, level)
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol = default_paper_protocol(level=level, track=track).to_dict()
    protocol["arch"] = arch
    protocol["precision"] = precision
    protocol["num_correct_trials"] = num_correct_trials
    protocol["num_perf_trials"] = num_perf_trials
    protocol["official_eval_seed"] = official_eval_seed
    protocol["generation_timeout_seconds"] = timeout_seconds
    if track == "agentic":
        protocol["max_steps"] = max_steps
        protocol["max_wall_clock_minutes"] = max_wall_clock_minutes
        protocol["max_tool_calls"] = max_tool_calls
        protocol["dev_eval_seed"] = dev_eval_seed
        protocol["dev_eval_correct_trials"] = dev_eval_correct_trials
        protocol["dev_eval_perf_trials"] = dev_eval_perf_trials
        protocol["dev_eval_profile_enabled"] = dev_eval_profile_enabled
        protocol["dev_eval_profile_trials"] = dev_eval_profile_trials if dev_eval_profile_enabled else None
        protocol["dev_eval_profile_metrics"] = dev_eval_profile_metrics if dev_eval_profile_enabled else []
    payload = {
        "protocol": protocol,
        "provider": provider,
        "track": track,
        "backend": backend,
        "level": level,
        "run_name": run_name,
        "model": model,
        "model_metadata": {
            "model_family": model_family,
            "paper_model_label": paper_model_label,
            "reasoning_effort": reasoning_effort,
            "model_verbosity": model_verbosity,
            "provider_extra_args": list(provider_extra_args),
            "codex_config": list(codex_config),
            "claude_extra_args": list(claude_extra_args),
        },
        "claim_scope": list(claim_scope),
        "temperature": 0.0,
        "max_tokens": 12000,
        "timeout_seconds": timeout_seconds,
        "arch": arch,
        "problems": [problem.problem_id for problem in problems],
        "problem_names": {str(problem.problem_id): problem.name for problem in problems},
        "prompt_template_hash": prompt_template_hash(backend, arch=arch, track=track),
        "environment": detect_runtime_environment(),
        "chunked_generation": True,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_generation_command(
    *,
    python_exe: str,
    provider: str,
    model: str,
    reasoning_effort: str | None,
    model_verbosity: str | None,
    provider_extra_args: list[str],
    model_family: str | None,
    paper_model_label: str | None,
    claim_scope: list[str],
    track: str,
    backend: str,
    level: int,
    run_name: str,
    problem_ids: list[int],
    arch: str,
    timeout_seconds: int,
    official_eval_seed: int,
    max_steps: int,
    max_wall_clock_minutes: int,
    max_tool_calls: int,
    dev_eval_seed: int,
    dev_eval_correct_trials: int,
    dev_eval_perf_trials: int,
    dev_eval_profile_enabled: bool,
    dev_eval_profile_trials: int,
    dev_eval_profile_metrics: list[str],
    codex_bin: str,
    codex_sandbox: str,
    codex_home: str | None,
    codex_config: list[str],
    claude_bin: str,
    claude_extra_args: list[str],
    chunk_label: str,
) -> list[str]:
    command = [
        python_exe,
        "scripts/generate_samples.py",
        "--provider",
        provider,
        "--model",
        model,
        "--reasoning-effort",
        reasoning_effort or "",
        "--model-verbosity",
        model_verbosity or "",
        "--track",
        track,
        "--backend",
        backend,
        "--level",
        str(level),
        "--run-name",
        run_name,
        "--problem-ids",
        ",".join(str(problem_id) for problem_id in problem_ids),
        "--arch",
        arch,
        "--official-eval-seed",
        str(official_eval_seed),
        "--timeout-seconds",
        str(timeout_seconds),
        "--skip-existing",
        "--continue-on-error",
        "--chunk-label",
        chunk_label,
    ]
    if model_family:
        command.extend(["--model-family", model_family])
    if paper_model_label:
        command.extend(["--paper-model-label", paper_model_label])
    for scope in claim_scope:
        command.extend(["--claim-scope", scope])
    for extra_arg in provider_extra_args:
        command.extend(["--provider-extra-arg", extra_arg])
    if track == "agentic":
        command.extend(
            [
                "--max-steps",
                str(max_steps),
                "--max-wall-clock-minutes",
                str(max_wall_clock_minutes),
                "--max-tool-calls",
                str(max_tool_calls),
                "--dev-eval-seed",
                str(dev_eval_seed),
                "--dev-eval-correct-trials",
                str(dev_eval_correct_trials),
                "--dev-eval-perf-trials",
                str(dev_eval_perf_trials),
            ]
        )
        if dev_eval_profile_enabled:
            command.extend(
                [
                    "--dev-eval-profile",
                    "--dev-eval-profile-trials",
                    str(dev_eval_profile_trials),
                ]
            )
            for metric in dev_eval_profile_metrics:
                command.extend(["--dev-eval-profile-metric", metric])
    if provider == "codex":
        command.extend(["--codex-bin", codex_bin])
        command.extend(["--codex-sandbox", codex_sandbox])
        if codex_home:
            command.extend(["--codex-home", codex_home])
        for config_override in codex_config:
            command.extend(["--codex-config", config_override])
    if provider == "claude-code":
        command.extend(["--claude-bin", claude_bin])
        for extra_arg in claude_extra_args:
            command.extend(["--claude-extra-arg", extra_arg])
    return command


def run_generation_chunk(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True, cwd=str(REPO_ROOT))


def execute_generation_tasks(
    *,
    tasks: list[GenerationChunkTask],
    run_name: str,
    provider: str,
    model: str,
    reasoning_effort: str | None,
    model_verbosity: str | None,
    provider_extra_args: list[str],
    model_family: str | None,
    paper_model_label: str | None,
    claim_scope: list[str],
    track: str,
    arch: str,
    timeout_seconds: int,
    precision: str,
    num_correct_trials: int,
    num_perf_trials: int,
    official_eval_seed: int,
    max_steps: int,
    max_wall_clock_minutes: int,
    max_tool_calls: int,
    dev_eval_seed: int,
    dev_eval_correct_trials: int,
    dev_eval_perf_trials: int,
    dev_eval_profile_enabled: bool,
    dev_eval_profile_trials: int,
    dev_eval_profile_metrics: list[str],
    python_exe: str,
    codex_bin: str,
    codex_sandbox: str,
    codex_home: str | None,
    codex_config: list[str],
    claude_bin: str,
    claude_extra_args: list[str],
    max_concurrent_chunks: int,
) -> None:
    status_lock = threading.Lock()
    exceptions: list[BaseException] = []
    tasks_by_backend: dict[str, list[GenerationChunkTask]] = {}
    for task in tasks:
        tasks_by_backend.setdefault(task.backend, []).append(task)
    problem_ids_by_backend = {
        backend: [problem_id for task in backend_tasks for problem_id in task.problem_ids]
        for backend, backend_tasks in tasks_by_backend.items()
    }

    for backend, backend_tasks in tasks_by_backend.items():
        write_backend_run_manifest(
            run_name=run_name,
            backend=backend,
            level=backend_tasks[0].level,
            track=track,
            provider=provider,
            model=model,
            reasoning_effort=reasoning_effort,
            model_verbosity=model_verbosity,
            provider_extra_args=provider_extra_args,
            model_family=model_family,
            paper_model_label=paper_model_label,
            claim_scope=claim_scope,
            codex_config=codex_config,
            claude_extra_args=claude_extra_args,
            problem_ids=problem_ids_by_backend[backend],
            arch=arch,
            precision=precision,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            timeout_seconds=timeout_seconds,
            official_eval_seed=official_eval_seed,
            max_steps=max_steps,
            max_wall_clock_minutes=max_wall_clock_minutes,
            max_tool_calls=max_tool_calls,
            dev_eval_seed=dev_eval_seed,
            dev_eval_correct_trials=dev_eval_correct_trials,
            dev_eval_perf_trials=dev_eval_perf_trials,
            dev_eval_profile_enabled=dev_eval_profile_enabled,
            dev_eval_profile_trials=dev_eval_profile_trials,
            dev_eval_profile_metrics=dev_eval_profile_metrics,
        )
        write_backend_generation_summary(
            run_name=run_name,
            backend=backend,
            level=backend_tasks[0].level,
            problem_ids=problem_ids_by_backend[backend],
            chunk_total=len(backend_tasks),
            track=track,
        )
        for task in backend_tasks:
            with status_lock:
                update_chunk_status(
                    run_name=run_name,
                    backend=backend,
                    level=task.level,
                    chunk_index=task.chunk_index,
                    chunk_total=task.chunk_total,
                    problem_ids=task.problem_ids,
                    status="queued",
                    track=track,
                )

    def worker(task: GenerationChunkTask) -> None:
        with status_lock:
            update_chunk_status(
                run_name=run_name,
                backend=task.backend,
                level=task.level,
                chunk_index=task.chunk_index,
                chunk_total=task.chunk_total,
                problem_ids=task.problem_ids,
                status="running",
                track=track,
            )

        command = build_generation_command(
            python_exe=python_exe,
            provider=provider,
            model=model,
            reasoning_effort=reasoning_effort,
            model_verbosity=model_verbosity,
            provider_extra_args=provider_extra_args,
            model_family=model_family,
            paper_model_label=paper_model_label,
            claim_scope=claim_scope,
            track=track,
            backend=task.backend,
            level=task.level,
            run_name=run_name,
            problem_ids=task.problem_ids,
            arch=arch,
            timeout_seconds=timeout_seconds,
            official_eval_seed=official_eval_seed,
            max_steps=max_steps,
            max_wall_clock_minutes=max_wall_clock_minutes,
            max_tool_calls=max_tool_calls,
            dev_eval_seed=dev_eval_seed,
            dev_eval_correct_trials=dev_eval_correct_trials,
            dev_eval_perf_trials=dev_eval_perf_trials,
            dev_eval_profile_enabled=dev_eval_profile_enabled,
            dev_eval_profile_trials=dev_eval_profile_trials,
            dev_eval_profile_metrics=dev_eval_profile_metrics,
            codex_bin=codex_bin,
            codex_sandbox=codex_sandbox,
            codex_home=codex_home,
            codex_config=codex_config,
            claude_bin=claude_bin,
            claude_extra_args=claude_extra_args,
            chunk_label=task.chunk_label,
        )
        try:
            run_generation_chunk(command)
        except subprocess.CalledProcessError as exc:
            counts = inspect_chunk_generation(
                run_name=run_name,
                backend=task.backend,
                level=task.level,
                problem_ids=task.problem_ids,
            )
            with status_lock:
                update_chunk_status(
                    run_name=run_name,
                    backend=task.backend,
                    level=task.level,
                    chunk_index=task.chunk_index,
                    chunk_total=task.chunk_total,
                    problem_ids=task.problem_ids,
                    status="failed",
                    counts=counts,
                    error=str(exc),
                    track=track,
                )
                write_backend_generation_summary(
                    run_name=run_name,
                    backend=task.backend,
                    level=task.level,
                    problem_ids=problem_ids_by_backend[task.backend],
                    chunk_total=len(tasks_by_backend[task.backend]),
                    track=track,
                )
            raise

        counts = inspect_chunk_generation(
            run_name=run_name,
            backend=task.backend,
            level=task.level,
            problem_ids=task.problem_ids,
        )
        with status_lock:
            update_chunk_status(
                run_name=run_name,
                backend=task.backend,
                level=task.level,
                chunk_index=task.chunk_index,
                chunk_total=task.chunk_total,
                problem_ids=task.problem_ids,
                status="completed_with_failures" if counts["failed"] else "completed",
                counts=counts,
                track=track,
            )
            write_backend_generation_summary(
                run_name=run_name,
                backend=task.backend,
                level=task.level,
                problem_ids=problem_ids_by_backend[task.backend],
                chunk_total=len(tasks_by_backend[task.backend]),
                track=track,
            )

    with ThreadPoolExecutor(max_workers=max_concurrent_chunks) as executor:
        futures = [executor.submit(worker, task) for task in tasks]
        for future in as_completed(futures):
            try:
                future.result()
            except BaseException as exc:
                exceptions.append(exc)

    if exceptions:
        raise RuntimeError(f"{len(exceptions)} generation chunk(s) failed") from exceptions[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the paper-grade PTX-vs-CUDA workflow for a KernelBench level.")
    parser.add_argument("--phase", choices=["smoke", "pilot", "full"], default="pilot")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--provider", default="codex", choices=["litellm", "codex", "claude-code"])
    parser.add_argument("--track", default=DEFAULT_TRACK, choices=["oneshot", "agentic"])
    parser.add_argument("--model")
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument("--model-verbosity", default="")
    parser.add_argument("--provider-extra-arg", action="append", default=[])
    parser.add_argument("--model-family", default="")
    parser.add_argument("--paper-model-label", default="")
    parser.add_argument("--claim-scope", action="append", default=[])
    parser.add_argument("--level", type=int, choices=DEFAULT_LEVELS, default=1)
    parser.add_argument("--problem-ids")
    parser.add_argument("--arch", default=DEFAULT_ARCH)
    parser.add_argument("--precision", default=DEFAULT_PRECISION, choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--num-correct-trials", type=int, default=DEFAULT_NUM_CORRECT_TRIALS)
    parser.add_argument("--num-perf-trials", type=int, default=DEFAULT_NUM_PERF_TRIALS)
    parser.add_argument("--official-eval-seed", type=int, default=DEFAULT_OFFICIAL_EVAL_SEED)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_GENERATION_TIMEOUT_SECONDS)
    parser.add_argument("--chunk-size", type=int, help="Problems per generation chunk. Defaults to chunked full runs and unchunked smoke/pilot runs.")
    parser.add_argument(
        "--max-concurrent-chunks",
        type=int,
        default=1,
        help="Maximum number of chunk-generation subprocesses to run at once.",
    )
    parser.add_argument(
        "--parallel-backends",
        action="store_true",
        help="Generate PTX and CUDA chunks concurrently instead of finishing PTX before CUDA.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument(
        "--codex-sandbox",
        default="read-only",
        choices=["read-only", "workspace-write", "danger-full-access"],
    )
    parser.add_argument("--codex-home")
    parser.add_argument("--codex-config", action="append", default=[])
    parser.add_argument("--claude-bin", default="claude")
    parser.add_argument("--claude-extra-arg", action="append", default=[])
    parser.add_argument("--max-steps", type=int, default=DEFAULT_AGENTIC_MAX_STEPS)
    parser.add_argument("--max-wall-clock-minutes", type=int, default=DEFAULT_AGENTIC_MAX_WALL_CLOCK_MINUTES)
    parser.add_argument("--max-tool-calls", type=int, default=DEFAULT_AGENTIC_MAX_TOOL_CALLS)
    parser.add_argument("--dev-eval-seed", type=int, default=DEFAULT_DEV_EVAL_SEED)
    parser.add_argument("--dev-eval-correct-trials", type=int, default=DEFAULT_DEV_EVAL_CORRECT_TRIALS)
    parser.add_argument("--dev-eval-perf-trials", type=int, default=DEFAULT_DEV_EVAL_PERF_TRIALS)
    parser.add_argument("--dev-eval-profile", action="store_true")
    parser.add_argument("--dev-eval-profile-trials", type=int, default=1)
    parser.add_argument("--dev-eval-profile-metric", action="append", default=[])
    parser.add_argument("--required-output", action="append", default=[])
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()

    if platform.system() != "Linux":
        raise RuntimeError("scripts/run_level1_paired.py is intended to run on Linux.")
    if not args.skip_generation and not args.model:
        raise ValueError("--model is required unless --skip-generation is used")

    problem_ids = resolve_problem_ids(args.phase, args.problem_ids, args.level)
    problem_ids_arg = ",".join(str(problem_id) for problem_id in problem_ids)
    timeout_seconds = args.timeout_seconds
    if args.phase == "full" and timeout_seconds == DEFAULT_GENERATION_TIMEOUT_SECONDS:
        timeout_seconds = DEFAULT_FULL_GENERATION_TIMEOUT_SECONDS
    chunk_size = args.chunk_size or (DEFAULT_GENERATION_CHUNK_SIZE if args.phase == "full" else len(problem_ids))
    max_concurrent_chunks = max(1, args.max_concurrent_chunks)

    write_paper_run_manifest(
        run_name=args.run_name,
        phase=args.phase,
        provider=args.provider,
        model=args.model,
        reasoning_effort=args.reasoning_effort or None,
        model_verbosity=args.model_verbosity or None,
        provider_extra_args=args.provider_extra_arg,
        model_family=args.model_family or None,
        paper_model_label=args.paper_model_label or None,
        claim_scope=args.claim_scope,
        codex_config=args.codex_config,
        claude_extra_args=args.claude_extra_arg,
        level=args.level,
        track=args.track,
        problem_ids=problem_ids,
        arch=args.arch,
        precision=args.precision,
        timeout_seconds=timeout_seconds,
        num_correct_trials=args.num_correct_trials,
        num_perf_trials=args.num_perf_trials,
        official_eval_seed=args.official_eval_seed,
        max_steps=args.max_steps,
        max_wall_clock_minutes=args.max_wall_clock_minutes,
        max_tool_calls=args.max_tool_calls,
        dev_eval_seed=args.dev_eval_seed,
        dev_eval_correct_trials=args.dev_eval_correct_trials,
        dev_eval_perf_trials=args.dev_eval_perf_trials,
        dev_eval_profile_enabled=args.dev_eval_profile,
        dev_eval_profile_trials=args.dev_eval_profile_trials,
        dev_eval_profile_metrics=args.dev_eval_profile_metric,
        required_outputs=args.required_output,
    )

    python_exe = sys.executable

    if not args.skip_smoke:
        for backend, submission in (
            ("ptx", "tests/fixtures/submissions/ptx/relu_submission.py"),
            ("cuda", "tests/fixtures/submissions/cuda/relu_submission.py"),
        ):
            run_command(
                [
                    python_exe,
                    "scripts/run_and_check.py",
                    "--backend",
                    backend,
                    "--level",
                    "1",
                    "--problem-id",
                    "19",
                    "--submission",
                    submission,
                    "--num-correct-trials",
                    "2",
                    "--num-perf-trials",
                    "5",
                ]
            )

    if not args.skip_generation:
        backend_order = ("ptx", "cuda")
        if args.parallel_backends:
            chunks = chunk_problem_ids(problem_ids, chunk_size)
            tasks: list[GenerationChunkTask] = []
            for chunk_index, chunk_ids in enumerate(chunks, start=1):
                for backend in backend_order:
                    tasks.append(
                        GenerationChunkTask(
                            backend=backend,
                            level=args.level,
                            chunk_index=chunk_index,
                            chunk_total=len(chunks),
                            problem_ids=chunk_ids,
                        )
                    )
            execute_generation_tasks(
                tasks=tasks,
                run_name=args.run_name,
                provider=args.provider,
                model=args.model,
                reasoning_effort=args.reasoning_effort or None,
                model_verbosity=args.model_verbosity or None,
                provider_extra_args=args.provider_extra_arg,
                model_family=args.model_family or None,
                paper_model_label=args.paper_model_label or None,
                claim_scope=args.claim_scope,
                track=args.track,
                arch=args.arch,
                timeout_seconds=timeout_seconds,
                precision=args.precision,
                num_correct_trials=args.num_correct_trials,
                num_perf_trials=args.num_perf_trials,
                official_eval_seed=args.official_eval_seed,
                max_steps=args.max_steps,
                max_wall_clock_minutes=args.max_wall_clock_minutes,
                max_tool_calls=args.max_tool_calls,
                dev_eval_seed=args.dev_eval_seed,
                dev_eval_correct_trials=args.dev_eval_correct_trials,
                dev_eval_perf_trials=args.dev_eval_perf_trials,
                dev_eval_profile_enabled=args.dev_eval_profile,
                dev_eval_profile_trials=args.dev_eval_profile_trials,
                dev_eval_profile_metrics=args.dev_eval_profile_metric,
                python_exe=python_exe,
                codex_bin=args.codex_bin,
                codex_sandbox=args.codex_sandbox,
                codex_home=args.codex_home,
                codex_config=args.codex_config,
                claude_bin=args.claude_bin,
                claude_extra_args=args.claude_extra_arg,
                max_concurrent_chunks=max_concurrent_chunks,
            )
        else:
            for backend in backend_order:
                chunks = chunk_problem_ids(problem_ids, chunk_size)
                tasks = [
                    GenerationChunkTask(
                        backend=backend,
                        level=args.level,
                        chunk_index=chunk_index,
                        chunk_total=len(chunks),
                        problem_ids=chunk_ids,
                    )
                    for chunk_index, chunk_ids in enumerate(chunks, start=1)
                ]
                execute_generation_tasks(
                    tasks=tasks,
                    run_name=args.run_name,
                    provider=args.provider,
                    model=args.model,
                    reasoning_effort=args.reasoning_effort or None,
                    model_verbosity=args.model_verbosity or None,
                    provider_extra_args=args.provider_extra_arg,
                    model_family=args.model_family or None,
                    paper_model_label=args.paper_model_label or None,
                    claim_scope=args.claim_scope,
                    track=args.track,
                    arch=args.arch,
                    timeout_seconds=timeout_seconds,
                    precision=args.precision,
                    num_correct_trials=args.num_correct_trials,
                    num_perf_trials=args.num_perf_trials,
                    official_eval_seed=args.official_eval_seed,
                    max_steps=args.max_steps,
                    max_wall_clock_minutes=args.max_wall_clock_minutes,
                    max_tool_calls=args.max_tool_calls,
                    dev_eval_seed=args.dev_eval_seed,
                    dev_eval_correct_trials=args.dev_eval_correct_trials,
                    dev_eval_perf_trials=args.dev_eval_perf_trials,
                    dev_eval_profile_enabled=args.dev_eval_profile,
                    dev_eval_profile_trials=args.dev_eval_profile_trials,
                    dev_eval_profile_metrics=args.dev_eval_profile_metric,
                    python_exe=python_exe,
                    codex_bin=args.codex_bin,
                    codex_sandbox=args.codex_sandbox,
                    codex_home=args.codex_home,
                    codex_config=args.codex_config,
                    claude_bin=args.claude_bin,
                    claude_extra_args=args.claude_extra_arg,
                    max_concurrent_chunks=max_concurrent_chunks,
                )

    if not args.skip_eval:
        for backend in ("ptx", "cuda"):
            run_command(
                [
                    python_exe,
                    "scripts/eval_from_generations.py",
                    "--run-name",
                    args.run_name,
                    "--backend",
                    backend,
                    "--level",
                    str(args.level),
                    "--problem-ids",
                    problem_ids_arg,
                    "--device",
                    args.device,
                    "--precision",
                    args.precision,
                     "--arch",
                     args.arch,
                     "--num-correct-trials",
                     str(args.num_correct_trials),
                     "--num-perf-trials",
                     str(args.num_perf_trials),
                     "--official-eval-seed",
                     str(args.official_eval_seed),
                 ]
             )

    if not args.skip_analysis:
        run_command(
            [
                python_exe,
                "scripts/benchmark_eval_analysis.py",
                "--run-name",
                args.run_name,
                "--level",
                str(args.level),
            ]
        )


if __name__ == "__main__":
    main()
