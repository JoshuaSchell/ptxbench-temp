from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any, Callable
import json

from .config import (
    DEFAULT_AGENTIC_MAX_STEPS,
    DEFAULT_AGENTIC_MAX_TOOL_CALLS,
    DEFAULT_AGENTIC_MAX_WALL_CLOCK_MINUTES,
    DEFAULT_ARCH,
    DEFAULT_DEV_EVAL_CORRECT_TRIALS,
    DEFAULT_DEV_EVAL_PROFILE_ENABLED,
    DEFAULT_DEV_EVAL_PROFILE_METRICS,
    DEFAULT_DEV_EVAL_PROFILE_TOOL,
    DEFAULT_DEV_EVAL_PROFILE_TRIALS,
    DEFAULT_DEV_EVAL_PERF_TRIALS,
    DEFAULT_DEV_EVAL_SEED,
    DEFAULT_DEV_EVAL_WARMUP,
)
from .dataset import Problem
from .eval import (
    _validate_submission_contract,
    evaluate_submission,
    load_submission_module,
    unload_submission_module,
)
from .generation import (
    build_agentic_step_prompt,
    clear_generation_failure,
    default_episode_dir,
    default_run_dir,
    extract_python_source,
    write_generation_artifacts,
)
from .profiler import ProfileRequest, format_profile_summary, normalize_profile_metrics
from .providers import ProviderResponse, generate_with_codex_cli, generate_with_litellm
from .run_metadata import sha256_text
from .runtime import PTXAssemblyError, PTXCompileArtifact, compile_ptx_source
from .static_checker import validate_submission_static


@dataclass(frozen=True)
class AgenticEpisodeBudget:
    max_steps: int = DEFAULT_AGENTIC_MAX_STEPS
    max_wall_clock_minutes: int = DEFAULT_AGENTIC_MAX_WALL_CLOCK_MINUTES
    max_tool_calls: int = DEFAULT_AGENTIC_MAX_TOOL_CALLS
    dev_eval_correct_trials: int = DEFAULT_DEV_EVAL_CORRECT_TRIALS
    dev_eval_perf_trials: int = DEFAULT_DEV_EVAL_PERF_TRIALS
    dev_eval_seed: int = DEFAULT_DEV_EVAL_SEED
    dev_eval_warmup: int = DEFAULT_DEV_EVAL_WARMUP
    dev_eval_profile_enabled: bool = DEFAULT_DEV_EVAL_PROFILE_ENABLED
    dev_eval_profile_tool: str = DEFAULT_DEV_EVAL_PROFILE_TOOL
    dev_eval_profile_trials: int = DEFAULT_DEV_EVAL_PROFILE_TRIALS
    dev_eval_profile_metrics: tuple[str, ...] = DEFAULT_DEV_EVAL_PROFILE_METRICS

    @property
    def wall_clock_budget_seconds(self) -> int:
        return int(self.max_wall_clock_minutes * 60)

    @property
    def profile_request(self) -> ProfileRequest | None:
        if not self.dev_eval_profile_enabled:
            return None
        return ProfileRequest(
            enabled=True,
            tool=self.dev_eval_profile_tool,
            num_trials=self.dev_eval_profile_trials,
            metrics=normalize_profile_metrics(self.dev_eval_profile_metrics),
        )


@dataclass
class AgenticEpisodeStep:
    step_index: int
    prompt_path: str
    response_path: str
    submission_path: str
    prompt_hash: str
    submission_hash: str
    observation: dict[str, Any] | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "prompt_path": self.prompt_path,
            "response_path": self.response_path,
            "submission_path": self.submission_path,
            "prompt_hash": self.prompt_hash,
            "submission_hash": self.submission_hash,
            "observation": self.observation,
            "provider_metadata": self.provider_metadata,
        }


@dataclass(frozen=True)
class AgenticEpisodeArtifacts:
    output_path: Path
    prompt: str
    response_text: str
    extracted_source: str
    metadata: dict[str, Any]


def run_agentic_episode(
    *,
    problem: Problem,
    backend: str,
    provider: str,
    model: str,
    run_name: str,
    level: int,
    temperature: float,
    max_tokens: int,
    arch: str = DEFAULT_ARCH,
    timeout_seconds: int = 900,
    codex_bin: str = "codex",
    codex_home: Path | None = None,
    codex_sandbox: str = "read-only",
    codex_config: list[str] | None = None,
    budget: AgenticEpisodeBudget | None = None,
    provider_fn: Callable[..., ProviderResponse] | None = None,
) -> AgenticEpisodeArtifacts:
    budget = budget or AgenticEpisodeBudget()
    codex_config = codex_config or []
    run_dir = default_run_dir(run_name, backend, level)
    output_path = run_dir / f"{problem.problem_id:03d}_{problem.path.stem}.py"
    episode_dir = default_episode_dir(run_name, backend, level, problem)
    steps_dir = episode_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)

    start = monotonic()
    episode_id = f"{run_name}-{backend}-level{level}-{problem.problem_id:03d}"
    max_feedback_rounds = min(max(0, budget.max_tool_calls), max(0, budget.max_steps - 1))
    tool_calls_used = 0
    step_records: list[AgenticEpisodeStep] = []
    last_source: str | None = None
    last_prompt = ""
    last_response = ""
    last_provider_metadata: dict[str, Any] = {}
    last_observation_summary: str | None = None
    first_compile_step: int | None = None
    first_correct_step: int | None = None
    terminated_reason = "max_steps_reached"
    provider_error: dict[str, Any] | None = None

    for step_index in range(1, budget.max_steps + 1):
        remaining_seconds = budget.wall_clock_budget_seconds - int(monotonic() - start)
        if step_index > 1 and remaining_seconds <= 0 and last_source is not None:
            terminated_reason = "wall_clock_budget_exhausted"
            break

        prompt = build_agentic_step_prompt(
            problem,
            backend=backend,
            arch=arch,
            step_index=step_index,
            max_steps=budget.max_steps,
            max_tool_calls=max_feedback_rounds,
            previous_source=last_source,
            previous_observation=last_observation_summary,
        )
        prompt_hash = sha256_text(prompt)
        prompt_path = steps_dir / f"step_{step_index:03d}.prompt.txt"
        response_path = steps_dir / f"step_{step_index:03d}.response.txt"
        submission_path = steps_dir / f"step_{step_index:03d}.submission.py"
        prompt_path.write_text(prompt, encoding="utf-8")

        call_timeout = timeout_seconds
        if budget.max_wall_clock_minutes:
            call_timeout = max(1, min(timeout_seconds, remaining_seconds if remaining_seconds > 0 else 1))

        try:
            provider_response = _generate_agentic_step(
                prompt=prompt,
                provider=provider,
                provider_fn=provider_fn,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=call_timeout,
                codex_bin=codex_bin,
                codex_home=codex_home,
                codex_sandbox=codex_sandbox,
                codex_config=codex_config,
            )
        except Exception as exc:
            if last_source is None:
                raise
            terminated_reason = "provider_error_after_partial_episode"
            provider_error = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "failed_step": step_index,
            }
            break

        extracted_source = extract_python_source(provider_response.content)
        if not extracted_source:
            if last_source is None:
                raise ValueError("Agentic provider returned an empty response")
            terminated_reason = "empty_response_after_partial_episode"
            provider_error = {
                "error_type": "EmptyResponse",
                "error_message": "Agentic provider returned an empty response",
                "failed_step": step_index,
            }
            break

        response_path.write_text(provider_response.content, encoding="utf-8")
        submission_path.write_text(extracted_source, encoding="utf-8")
        submission_hash = sha256_text(extracted_source)

        last_prompt = prompt
        last_response = provider_response.content
        last_provider_metadata = dict(provider_response.metadata)
        last_source = extracted_source

        observation_payload = None
        should_observe = step_index < budget.max_steps and tool_calls_used < max_feedback_rounds
        if should_observe:
            observation_payload = run_agentic_validation_pass(
                problem=problem,
                backend=backend,
                submission_path=submission_path,
                arch=arch,
                budget=budget,
            )
            tool_calls_used += 1
            last_observation_summary = format_agentic_observation(observation_payload)
            (steps_dir / f"step_{step_index:03d}.observation.json").write_text(
                json.dumps(observation_payload, indent=2),
                encoding="utf-8",
            )
            if observation_payload.get("compiled") and first_compile_step is None:
                first_compile_step = step_index
            if observation_payload.get("correctness") and first_correct_step is None:
                first_correct_step = step_index
        elif step_index < budget.max_steps:
            terminated_reason = "tool_budget_exhausted"
        else:
            terminated_reason = "completed_budgeted_episode"

        step_records.append(
            AgenticEpisodeStep(
                step_index=step_index,
                prompt_path=str(prompt_path),
                response_path=str(response_path),
                submission_path=str(submission_path),
                prompt_hash=prompt_hash,
                submission_hash=submission_hash,
                observation=observation_payload,
                provider_metadata=dict(provider_response.metadata),
            )
        )

        if not should_observe:
            break

    if last_source is None:
        raise ValueError("Agentic episode produced no submission")

    elapsed_seconds = round(monotonic() - start, 3)
    metadata = {
        "problem_id": problem.problem_id,
        "problem_name": problem.name,
        "model": model,
        "provider": provider,
        "backend": backend,
        "track": "agentic",
        "arch": arch,
        "temperature": temperature,
        "timeout_seconds": timeout_seconds,
        "episode_id": episode_id,
        "step_count": len(step_records),
        "budget_used": {
            "max_steps": budget.max_steps,
            "steps_used": len(step_records),
            "max_tool_calls": max_feedback_rounds,
            "tool_calls_used": tool_calls_used,
            "max_wall_clock_minutes": budget.max_wall_clock_minutes,
            "wall_clock_seconds": elapsed_seconds,
            "wall_clock_budget_seconds": budget.wall_clock_budget_seconds,
            "dev_eval_correct_trials": budget.dev_eval_correct_trials,
            "dev_eval_perf_trials": budget.dev_eval_perf_trials,
            "dev_eval_seed": budget.dev_eval_seed,
            "dev_eval_profile_enabled": budget.dev_eval_profile_enabled,
            "dev_eval_profile_tool": budget.dev_eval_profile_tool if budget.dev_eval_profile_enabled else None,
            "dev_eval_profile_trials": budget.dev_eval_profile_trials if budget.dev_eval_profile_enabled else None,
            "dev_eval_profile_metrics": (
                list(budget.dev_eval_profile_metrics) if budget.dev_eval_profile_enabled else []
            ),
        },
        "task_family_tags": list(problem.task_family_tags),
        "first_compile_step": first_compile_step,
        "first_correct_step": first_correct_step,
        "final_submission_hash": sha256_text(last_source),
        "episode_manifest_path": str(episode_dir / "episode_manifest.json"),
        "terminated_reason": terminated_reason,
        "provider_error": provider_error,
        "prompt_hash": sha256_text(last_prompt),
        **last_provider_metadata,
    }
    episode_manifest = {
        "episode_id": episode_id,
        "track": "agentic",
        "backend": backend,
        "problem_id": problem.problem_id,
        "problem_name": problem.name,
        "task_family_tags": list(problem.task_family_tags),
        "budget": asdict(budget),
        "budget_used": metadata["budget_used"],
        "first_compile_step": first_compile_step,
        "first_correct_step": first_correct_step,
        "final_submission_hash": metadata["final_submission_hash"],
        "terminated_reason": terminated_reason,
        "provider_error": provider_error,
        "steps": [step.to_dict() for step in step_records],
    }
    episode_dir.mkdir(parents=True, exist_ok=True)
    (episode_dir / "episode_manifest.json").write_text(json.dumps(episode_manifest, indent=2), encoding="utf-8")

    write_generation_artifacts(
        output_path=output_path,
        prompt=last_prompt,
        response_text=last_response,
        extracted_source=last_source,
        metadata=metadata,
    )
    clear_generation_failure(output_path)
    return AgenticEpisodeArtifacts(
        output_path=output_path,
        prompt=last_prompt,
        response_text=last_response,
        extracted_source=last_source,
        metadata=metadata,
    )


def run_agentic_validation_pass(
    *,
    problem: Problem,
    backend: str,
    submission_path: Path,
    arch: str,
    budget: AgenticEpisodeBudget,
) -> dict[str, Any]:
    source_text = submission_path.read_text(encoding="utf-8")
    static_result = validate_submission_static(source_text, backend=backend)
    observation: dict[str, Any] = {
        "static_check": {
            "valid": static_result.valid,
            "errors": list(static_result.errors),
            "warnings": list(static_result.warnings),
        },
        "compiled": False,
        "assembled": None,
        "loaded": None,
        "correctness": False,
        "runtime_ms": -1.0,
        "ref_runtime_ms": -1.0,
        "speedup_vs_torch": 0.0,
        "profile": None,
    }
    if not static_result.valid:
        observation["failure_stage"] = "static"
        return observation

    if backend == "ptx":
        assembly_result = run_ptx_assembly_check(submission_path=submission_path, arch=arch)
        observation["assembly_check"] = assembly_result
        observation["compiled"] = bool(assembly_result["compiled"])
        observation["assembled"] = bool(assembly_result["assembled"])
        if not assembly_result["assembled"]:
            observation["failure_stage"] = "assemble"
            return observation

    dev_result = evaluate_submission(
        problem=problem,
        submission_path=submission_path,
        backend=backend,
        arch=arch,
        num_correct_trials=budget.dev_eval_correct_trials,
        num_perf_trials=budget.dev_eval_perf_trials,
        num_warmup=min(budget.dev_eval_warmup, budget.dev_eval_perf_trials),
        seed=budget.dev_eval_seed,
        profile_request=budget.profile_request,
    )
    observation["compiled"] = dev_result.compiled
    observation["assembled"] = dev_result.assembled
    observation["loaded"] = dev_result.loaded
    observation["correctness"] = dev_result.correctness
    observation["runtime_ms"] = dev_result.runtime_ms
    observation["ref_runtime_ms"] = dev_result.ref_runtime_ms
    observation["speedup_vs_torch"] = dev_result.speedup_vs_torch
    observation["metadata"] = dev_result.metadata
    observation["profile"] = dev_result.metadata.get("profile")
    observation["failure_stage"] = _infer_agentic_failure_stage(observation)
    return observation


def run_ptx_assembly_check(*, submission_path: Path, arch: str) -> dict[str, Any]:
    build_dir = submission_path.parent / ".agentic_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    module = None
    source_reports: list[dict[str, Any]] = []
    try:
        module = load_submission_module(submission_path, build_dir=build_dir, backend="ptx")
        _validate_submission_contract(module, "ptx")
        for source_name, source_text in module.PTX_SOURCES.items():
            artifact = compile_ptx_source(source_name=source_name, source_text=source_text, arch=arch)
            entry = None
            if hasattr(module, "PTX_KERNELS") and source_name in module.PTX_KERNELS:
                entry = getattr(module.PTX_KERNELS[source_name], "entry", None)
            source_reports.append(_serialize_ptx_assembly_artifact(artifact, entry=entry))
        return {
            "compiled": True,
            "assembled": True,
            "target_arch": arch,
            "sources": source_reports,
            "error": None,
            "ptxas_error": None,
        }
    except PTXAssemblyError as exc:
        return {
            "compiled": True,
            "assembled": False,
            "target_arch": arch,
            "sources": source_reports,
            "error": str(exc),
            "ptxas_error": str(exc),
        }
    except Exception as exc:
        return {
            "compiled": False,
            "assembled": False,
            "target_arch": arch,
            "sources": source_reports,
            "error": str(exc),
            "ptxas_error": None,
        }
    finally:
        if module is not None:
            unload_submission_module(module)


def _serialize_ptx_assembly_artifact(artifact: PTXCompileArtifact, *, entry: str | None) -> dict[str, Any]:
    selected_report = artifact.assembly_report.for_entry(entry)
    return {
        "source_name": artifact.source_name,
        "entry": entry,
        "target_arch": artifact.arch,
        "registers": selected_report.registers if selected_report is not None else artifact.assembly_report.registers,
        "spill_stores_bytes": (
            selected_report.spill_stores_bytes if selected_report is not None else artifact.assembly_report.spill_stores_bytes
        ),
        "spill_loads_bytes": (
            selected_report.spill_loads_bytes if selected_report is not None else artifact.assembly_report.spill_loads_bytes
        ),
        "shared_memory_bytes": (
            selected_report.shared_memory_bytes if selected_report is not None else artifact.assembly_report.shared_memory_bytes
        ),
        "local_memory_bytes": (
            selected_report.local_memory_bytes if selected_report is not None else artifact.assembly_report.local_memory_bytes
        ),
        "constant_memory_bytes": (
            selected_report.constant_memory_bytes
            if selected_report is not None
            else artifact.assembly_report.constant_memory_bytes
        ),
        "stack_frame_bytes": (
            selected_report.stack_frame_bytes if selected_report is not None else artifact.assembly_report.stack_frame_bytes
        ),
        "log_path": str(artifact.log_path),
        "report": artifact.assembly_report.to_dict(),
    }


def _format_ptx_bytes(value: Any) -> str:
    return "?" if value is None else f"{int(value)}B"


def _format_ptx_assembly_source(source: dict[str, Any]) -> str:
    label = str(source.get("entry") or source.get("source_name") or "kernel")
    registers = source.get("registers")
    return (
        f"{label} regs={'?' if registers is None else int(registers)}"
        f" spills={_format_ptx_bytes(source.get('spill_stores_bytes'))}/{_format_ptx_bytes(source.get('spill_loads_bytes'))}"
        f" smem={_format_ptx_bytes(source.get('shared_memory_bytes'))}"
        f" lmem={_format_ptx_bytes(source.get('local_memory_bytes'))}"
        f" cmem={_format_ptx_bytes(source.get('constant_memory_bytes'))}"
    )


def _truncate_agentic_text(value: Any, *, limit: int = 240) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def format_agentic_observation(observation: dict[str, Any]) -> str:
    lines: list[str] = []
    profile_summary = format_profile_summary(observation.get("profile"))
    static_check = observation.get("static_check", {})
    if static_check:
        if static_check.get("valid"):
            lines.append("static_check: pass")
        else:
            lines.append(
                "static_check: fail - "
                + ", ".join(static_check.get("errors", []) or ["unknown static validation failure"])
            )
            return "\n".join(lines)

    assembly_check = observation.get("assembly_check")
    if assembly_check is not None:
        if assembly_check.get("assembled"):
            arch = assembly_check.get("target_arch") or "unknown"
            source_summaries = [
                _format_ptx_assembly_source(source) for source in (assembly_check.get("sources") or []) if source
            ]
            assembly_line = f"assembly_check: pass arch={arch}"
            if source_summaries:
                rendered_summaries = "; ".join(source_summaries[:2])
                if len(source_summaries) > 2:
                    rendered_summaries += f"; +{len(source_summaries) - 2} more"
                assembly_line += " " + rendered_summaries
            lines.append(assembly_line)
        else:
            arch = assembly_check.get("target_arch") or "unknown"
            lines.append(f"assembly_check: fail arch={arch}")
            error_text = assembly_check.get("ptxas_error") or assembly_check.get("error") or "unknown assembly failure"
            lines.append("ptxas: " + _truncate_agentic_text(error_text))
            return "\n".join(lines)

    if observation.get("compiled") is False and observation.get("metadata", {}).get("compile_error"):
        lines.append(f"dev_eval: compile fail - {observation['metadata']['compile_error']}")
        return "\n".join(lines)

    if observation.get("correctness"):
        lines.append(
            "dev_eval: correct"
            f" runtime_ms={observation.get('runtime_ms', -1.0):.4f}"
            f" ref_runtime_ms={observation.get('ref_runtime_ms', -1.0):.4f}"
            f" speedup_vs_torch={observation.get('speedup_vs_torch', 0.0):.4f}"
        )
        if profile_summary:
            lines.append(profile_summary)
        return "\n".join(lines)

    metadata = observation.get("metadata", {})
    if metadata.get("correctness_errors"):
        lines.append("dev_eval: incorrect - " + "; ".join(metadata["correctness_errors"]))
    elif metadata.get("runtime_error"):
        lines.append("dev_eval: runtime fail - " + str(metadata["runtime_error"]))
    elif metadata.get("load_error"):
        lines.append("dev_eval: load fail - " + str(metadata["load_error"]))
    elif metadata.get("assembly_error"):
        lines.append("dev_eval: assembly fail - " + str(metadata["assembly_error"]))
    elif metadata.get("compile_error"):
        lines.append("dev_eval: compile fail - " + str(metadata["compile_error"]))
    else:
        lines.append("dev_eval: failed without a classified error")
    if profile_summary:
        lines.append(profile_summary)
    return "\n".join(lines)


def _generate_agentic_step(
    *,
    prompt: str,
    provider: str,
    provider_fn: Callable[..., ProviderResponse] | None,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
    codex_bin: str,
    codex_home: Path | None,
    codex_sandbox: str,
    codex_config: list[str],
) -> ProviderResponse:
    if provider_fn is not None:
        return provider_fn(
            prompt=prompt,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )
    if provider == "litellm":
        return generate_with_litellm(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )
    return generate_with_codex_cli(
        prompt=prompt,
        model=model,
        working_dir=Path(__file__).resolve().parents[2],
        codex_bin=codex_bin,
        sandbox=codex_sandbox,
        codex_home=codex_home,
        config_overrides=codex_config,
        timeout_seconds=timeout_seconds,
    )


def _infer_agentic_failure_stage(observation: dict[str, Any]) -> str:
    if observation.get("correctness"):
        return "success"
    metadata = observation.get("metadata", {})
    if not observation.get("compiled", False):
        return "compile"
    if observation.get("assembled") is False:
        return "assemble"
    if observation.get("loaded") is False:
        return "load"
    if metadata.get("runtime_error"):
        return "runtime"
    return "correctness"
