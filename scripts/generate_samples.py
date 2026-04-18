from __future__ import annotations

import argparse
from pathlib import Path
import json
import traceback

from ptxbench.agentic import AgenticEpisodeBudget, run_agentic_episode
from ptxbench.config import DEFAULT_GENERATION_TIMEOUT_SECONDS, DEFAULT_LEVELS, REPO_ROOT
from ptxbench.dataset import construct_dataset
from ptxbench.generation import (
    build_generation_prompt,
    clear_generation_failure,
    default_run_dir,
    extract_python_source,
    prompt_template_hash,
    write_generation_artifacts,
    write_generation_failure,
)
from ptxbench.profiler import normalize_profile_metrics
from ptxbench.providers import generate_with_codex_cli, generate_with_litellm
from ptxbench.run_metadata import default_paper_protocol, detect_runtime_environment, normalize_problem_ids, sha256_text
from ptxbench.workflow import chunk_metadata_dir, parse_problem_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PTXBench submissions with a one-shot prompt.")
    parser.add_argument("--provider", default="litellm", choices=["litellm", "codex"])
    parser.add_argument("--backend", required=True, choices=["ptx", "cuda"])
    parser.add_argument("--level", required=True, type=int, choices=DEFAULT_LEVELS)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--track", default="oneshot", choices=["oneshot", "agentic"])
    parser.add_argument("--model", required=True, help="Model name for the selected provider, for example gpt-5.4")
    parser.add_argument("--problem-ids")
    parser.add_argument("--max-problems", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--arch", default="sm_89")
    parser.add_argument("--official-eval-seed", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_GENERATION_TIMEOUT_SECONDS)
    parser.add_argument("--dry-run", action="store_true", help="Write prompts only and skip the model call.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip problems with an existing generated .py file.")
    parser.add_argument("--continue-on-error", action="store_true", help="Record per-problem failures and continue.")
    parser.add_argument("--chunk-label", help="Optional chunk label for parallel-safe manifest/summary writes.")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--codex-sandbox", default="read-only", choices=["read-only", "workspace-write", "danger-full-access"])
    parser.add_argument("--codex-home", help="Optional CODEX_HOME override when using --provider codex")
    parser.add_argument("--codex-config", action="append", default=[], help="Extra `codex exec -c key=value` overrides")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-wall-clock-minutes", type=int)
    parser.add_argument("--max-tool-calls", type=int)
    parser.add_argument("--dev-eval-seed", type=int)
    parser.add_argument("--dev-eval-correct-trials", type=int)
    parser.add_argument("--dev-eval-perf-trials", type=int)
    parser.add_argument("--dev-eval-profile", action="store_true")
    parser.add_argument("--dev-eval-profile-trials", type=int)
    parser.add_argument(
        "--dev-eval-profile-metric",
        action="append",
        default=[],
        help="Nsight metric name(s) for agentic dev-eval profiling. Accepts repeated flags or comma-separated values.",
    )
    args = parser.parse_args()

    protocol = default_paper_protocol(level=args.level, track=args.track).to_dict()
    protocol["level"] = args.level
    protocol["track"] = args.track
    protocol["arch"] = args.arch
    if args.official_eval_seed is not None:
        protocol["official_eval_seed"] = args.official_eval_seed
    protocol["generation_timeout_seconds"] = args.timeout_seconds
    if args.max_steps is not None:
        protocol["max_steps"] = args.max_steps
    if args.max_wall_clock_minutes is not None:
        protocol["max_wall_clock_minutes"] = args.max_wall_clock_minutes
    if args.max_tool_calls is not None:
        protocol["max_tool_calls"] = args.max_tool_calls
    if args.dev_eval_seed is not None:
        protocol["dev_eval_seed"] = args.dev_eval_seed
    if args.dev_eval_correct_trials is not None:
        protocol["dev_eval_correct_trials"] = args.dev_eval_correct_trials
    if args.dev_eval_perf_trials is not None:
        protocol["dev_eval_perf_trials"] = args.dev_eval_perf_trials
    if args.dev_eval_profile:
        protocol["dev_eval_profile_enabled"] = True
        if args.dev_eval_profile_trials is not None:
            protocol["dev_eval_profile_trials"] = args.dev_eval_profile_trials
        protocol["dev_eval_profile_metrics"] = list(normalize_profile_metrics(args.dev_eval_profile_metric))
    problem_ids = normalize_problem_ids(parse_problem_ids(args.problem_ids))
    dataset = construct_dataset(level=args.level, problem_ids=problem_ids)
    problems = list(dataset)
    if args.max_problems is not None:
        problems = problems[: args.max_problems]

    run_dir = default_run_dir(args.run_name, args.backend, args.level)
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = chunk_metadata_dir(run_dir, args.chunk_label)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    run_manifest = {
        "protocol": protocol,
        "provider": args.provider,
        "track": args.track,
        "backend": args.backend,
        "level": args.level,
        "run_name": args.run_name,
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout_seconds": args.timeout_seconds,
        "arch": args.arch,
        "problems": [problem.problem_id for problem in problems],
        "problem_names": {str(problem.problem_id): problem.name for problem in problems},
        "prompt_template_hash": prompt_template_hash(args.backend, arch=args.arch, track=args.track),
        "environment": detect_runtime_environment(),
    }
    if args.chunk_label:
        run_manifest["chunk_label"] = args.chunk_label
    (metadata_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    generation_summary = {
        "total": len(problems),
        "track": args.track,
        "generated": 0,
        "skipped": 0,
        "failed": 0,
        "failed_problem_ids": [],
    }

    agentic_budget = AgenticEpisodeBudget(
        max_steps=args.max_steps or int(protocol["max_steps"]),
        max_wall_clock_minutes=args.max_wall_clock_minutes or int(protocol["max_wall_clock_minutes"] or 0),
        max_tool_calls=args.max_tool_calls if args.max_tool_calls is not None else int(protocol["max_tool_calls"]),
        dev_eval_correct_trials=(
            args.dev_eval_correct_trials
            if args.dev_eval_correct_trials is not None
            else int(protocol["dev_eval_correct_trials"] or 0)
        ),
        dev_eval_perf_trials=(
            args.dev_eval_perf_trials
            if args.dev_eval_perf_trials is not None
            else int(protocol["dev_eval_perf_trials"] or 0)
        ),
        dev_eval_seed=args.dev_eval_seed if args.dev_eval_seed is not None else int(protocol["dev_eval_seed"] or 7),
        dev_eval_profile_enabled=bool(protocol.get("dev_eval_profile_enabled", False) or args.dev_eval_profile),
        dev_eval_profile_trials=(
            args.dev_eval_profile_trials
            if args.dev_eval_profile_trials is not None
            else int(protocol.get("dev_eval_profile_trials", 1) or 1)
        ),
        dev_eval_profile_metrics=normalize_profile_metrics(
            args.dev_eval_profile_metric or protocol.get("dev_eval_profile_metrics")
        ),
    )

    for problem in problems:
        prompt = build_generation_prompt(problem, backend=args.backend, arch=args.arch, track=args.track)
        output_path = run_dir / f"{problem.problem_id:03d}_{problem.path.stem}.py"
        if args.skip_existing and output_path.exists():
            clear_generation_failure(output_path)
            generation_summary["skipped"] += 1
            print(f"[skip] {args.backend} problem {problem.problem_id}: existing submission")
            continue

        try:
            if args.dry_run:
                write_generation_artifacts(
                    output_path=output_path,
                    prompt=prompt,
                    response_text="",
                    extracted_source="",
                    metadata={"dry_run": True, "problem_id": problem.problem_id, "track": args.track},
                )
                clear_generation_failure(output_path)
                generation_summary["generated"] += 1
                print(f"[dry-run] {args.backend} problem {problem.problem_id}")
                continue

            if args.track == "agentic":
                artifacts = run_agentic_episode(
                    problem=problem,
                    backend=args.backend,
                    provider=args.provider,
                    model=args.model,
                    run_name=args.run_name,
                    level=args.level,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    arch=args.arch,
                    timeout_seconds=args.timeout_seconds,
                    codex_bin=args.codex_bin,
                    codex_home=Path(args.codex_home).resolve() if args.codex_home else None,
                    codex_sandbox=args.codex_sandbox,
                    codex_config=args.codex_config,
                    budget=agentic_budget,
                )
                extracted = artifacts.extracted_source
                provider_metadata = artifacts.metadata
                raw_response = artifacts.response_text
            else:
                if args.provider == "litellm":
                    provider_response = generate_with_litellm(
                        prompt=prompt,
                        model=args.model,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        timeout_seconds=args.timeout_seconds,
                    )
                else:
                    provider_response = generate_with_codex_cli(
                        prompt=prompt,
                        model=args.model,
                        working_dir=REPO_ROOT,
                        codex_bin=args.codex_bin,
                        sandbox=args.codex_sandbox,
                        codex_home=Path(args.codex_home).resolve() if args.codex_home else None,
                        config_overrides=args.codex_config,
                        timeout_seconds=args.timeout_seconds,
                    )

                extracted = extract_python_source(provider_response.content)
                raw_response = provider_response.content
                provider_metadata = {
                    "problem_id": problem.problem_id,
                    "problem_name": problem.name,
                    "model": args.model,
                    "provider": args.provider,
                    "backend": args.backend,
                    "track": args.track,
                    "arch": args.arch,
                    "timeout_seconds": args.timeout_seconds,
                    "task_family_tags": list(problem.task_family_tags),
                    "prompt_hash": sha256_text(prompt),
                    **provider_response.metadata,
                }
                write_generation_artifacts(
                    output_path=output_path,
                    prompt=prompt,
                    response_text=raw_response,
                    extracted_source=extracted,
                    metadata=provider_metadata,
                )
                clear_generation_failure(output_path)
            generation_summary["generated"] += 1
            print(f"[generated] {args.backend} problem {problem.problem_id}")
        except Exception as exc:
            failure_metadata = {
                "problem_id": problem.problem_id,
                "problem_name": problem.name,
                "model": args.model,
                "provider": args.provider,
                "backend": args.backend,
                "track": args.track,
                "arch": args.arch,
                "timeout_seconds": args.timeout_seconds,
                "task_family_tags": list(problem.task_family_tags),
                "prompt_hash": sha256_text(prompt),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "error_traceback": traceback.format_exc(),
            }
            write_generation_failure(output_path, prompt=prompt, metadata=failure_metadata)
            generation_summary["failed"] += 1
            generation_summary["failed_problem_ids"].append(problem.problem_id)
            print(
                f"[failed] {args.backend} problem {problem.problem_id}: "
                f"{failure_metadata['error_type']}: {failure_metadata['error_message']}"
            )
            if not args.continue_on_error:
                (metadata_dir / "generation_summary.json").write_text(
                    json.dumps(generation_summary, indent=2),
                    encoding="utf-8",
                )
                raise

    (metadata_dir / "generation_summary.json").write_text(json.dumps(generation_summary, indent=2), encoding="utf-8")
    if generation_summary["failed"]:
        print(
            f"[summary] {args.backend}: generated={generation_summary['generated']} "
            f"skipped={generation_summary['skipped']} failed={generation_summary['failed']}"
        )
    else:
        print(
            f"[summary] {args.backend}: generated={generation_summary['generated']} "
            f"skipped={generation_summary['skipped']} failed=0"
        )


if __name__ == "__main__":
    main()
