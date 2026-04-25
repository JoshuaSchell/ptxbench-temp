from __future__ import annotations

import argparse
from pathlib import Path
import json

from ptxbench.config import DEFAULT_LEVELS, REPO_ROOT
from ptxbench.analysis import classify_paper_failure_category
from ptxbench.dataset import construct_dataset
from ptxbench.eval import build_missing_submission_result
from ptxbench.generation import generation_failure_path
from ptxbench.isolated_eval import classify_failure_category, evaluate_submission_payload_safely
from ptxbench.run_metadata import (
    default_paper_protocol,
    detect_runtime_environment,
    normalize_problem_ids,
    protocol_signature,
    sha256_text,
)


def parse_problem_ids(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def load_json_if_exists(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_generation_metadata(submission_path: Path) -> dict | None:
    metadata_path = submission_path.with_suffix(".meta.json")
    payload = load_json_if_exists(metadata_path)
    return payload if isinstance(payload, dict) else None


def load_submission_hash(submission_path: Path) -> str | None:
    if not submission_path.exists():
        return None
    return sha256_text(submission_path.read_text(encoding="utf-8"))


def apply_runtime_aliases(payload: dict) -> dict:
    normalized = dict(payload)
    ref_runtime_eager_ms = float(normalized.get("ref_runtime_eager_ms", normalized.get("ref_runtime_ms", -1.0)))
    speedup_vs_eager = float(normalized.get("speedup_vs_eager", normalized.get("speedup_vs_torch", 0.0)))
    normalized["ref_runtime_eager_ms"] = ref_runtime_eager_ms
    normalized["ref_runtime_ms"] = ref_runtime_eager_ms
    normalized["speedup_vs_eager"] = speedup_vs_eager
    normalized["speedup_vs_torch"] = speedup_vs_eager
    normalized.setdefault("ref_runtime_compile_default_ms", None)
    normalized.setdefault("speedup_vs_compile_default", None)
    return normalized


def enrich_result_payload(payload: dict, *, problem, submission_path: Path) -> dict:
    payload = apply_runtime_aliases(payload)
    generation_payload = load_generation_metadata(submission_path)
    generation_metadata = (generation_payload or {}).get("metadata", {})
    track = generation_metadata.get("track", payload.get("track", "oneshot"))
    task_family_tags = list(payload.get("task_family_tags") or generation_metadata.get("task_family_tags") or problem.task_family_tags)
    payload["track"] = track
    payload["task_family_tags"] = task_family_tags
    payload["episode_id"] = generation_metadata.get("episode_id", payload.get("episode_id"))
    payload["step_count"] = generation_metadata.get("step_count", payload.get("step_count"))
    payload["budget_used"] = generation_metadata.get("budget_used", payload.get("budget_used", {}))
    payload["first_compile_step"] = generation_metadata.get("first_compile_step", payload.get("first_compile_step"))
    payload["first_correct_step"] = generation_metadata.get("first_correct_step", payload.get("first_correct_step"))
    payload["final_submission_hash"] = generation_metadata.get(
        "final_submission_hash",
        payload.get("final_submission_hash"),
    )
    if payload.get("compiled") and payload.get("first_compile_step") is None:
        payload["first_compile_step"] = payload.get("step_count") or 1
    if payload.get("correctness") and payload.get("first_correct_step") is None:
        payload["first_correct_step"] = payload.get("step_count") or 1
    metadata = dict(payload.get("metadata", {}))
    metadata["task_family_tags"] = task_family_tags
    if generation_metadata:
        metadata["generation"] = generation_metadata
    payload["metadata"] = metadata
    return payload


def _environment_value(environment: dict | None, *keys: str) -> str | None:
    payload = environment or {}
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return str(value)
    return None


def stamp_eval_metadata(
    payload: dict,
    *,
    protocol: dict,
    submission_path: Path,
    environment: dict | None,
    track: str,
) -> dict:
    payload = apply_runtime_aliases(payload)
    metadata = dict(payload.get("metadata", {}))
    metadata["eval_protocol_signature"] = protocol_signature(protocol)
    metadata["evaluated_submission_hash"] = load_submission_hash(submission_path)
    metadata["failure_category"] = classify_failure_category(payload)
    metadata["paper_failure_category"] = classify_paper_failure_category({**payload, "metadata": metadata})
    payload["metadata"] = metadata
    payload["track"] = track
    payload["submission_hash"] = metadata["evaluated_submission_hash"]
    payload["failure_category"] = metadata["failure_category"]
    payload["paper_failure_category"] = metadata["paper_failure_category"]
    payload["num_correct_trials"] = int(protocol["num_correct_trials"])
    payload["num_perf_trials"] = int(protocol["num_perf_trials"])
    payload["seed"] = int(protocol["official_eval_seed"])
    payload["arch"] = str(protocol["arch"])
    payload["precision"] = str(protocol["precision"])
    payload["gpu_name"] = _environment_value(environment, "gpu_name", "torch_device_name")
    payload["torch_version"] = _environment_value(environment, "torch_version")
    payload["cuda_version"] = _environment_value(environment, "cuda_version", "torch_cuda_version", "cuda_toolkit_release")
    payload["ptxas_version"] = _environment_value(environment, "ptxas_version", "ptxas_release")
    payload["repo_commit"] = _environment_value(environment, "repo_commit")
    payload["kernelbench_commit"] = _environment_value(environment, "kernelbench_commit", "vendor_snapshot_commit")
    return payload


def result_matches_current_eval(
    payload: dict,
    *,
    protocol: dict,
    submission_path: Path,
) -> bool:
    metadata = payload.get("metadata", {})
    stored_protocol = metadata.get("eval_protocol_signature")
    current_protocol = protocol_signature(protocol)
    if stored_protocol != current_protocol:
        return False
    if metadata.get("evaluated_submission_hash") != load_submission_hash(submission_path):
        return False
    return True


def resolve_eval_protocol(
    *,
    level: int,
    track: str,
    run_manifest: dict | None,
    precision: str,
    arch: str,
    num_correct_trials: int,
    num_perf_trials: int,
    official_eval_seed: int | None,
    torch_compile_baseline: bool,
) -> dict:
    protocol = default_paper_protocol(level=level, track=track).to_dict()
    manifest_protocol = (run_manifest or {}).get("protocol", {})
    if isinstance(manifest_protocol, dict):
        protocol.update(manifest_protocol)
    protocol["level"] = level
    protocol["track"] = track
    protocol["precision"] = precision
    protocol["arch"] = arch
    protocol["num_correct_trials"] = num_correct_trials
    protocol["num_perf_trials"] = num_perf_trials
    if official_eval_seed is not None:
        protocol["official_eval_seed"] = official_eval_seed
    protocol["torch_compile_baseline"] = bool(torch_compile_baseline)
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch evaluate generated PTXBench submissions.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--backend", required=True, choices=["ptx", "cuda"])
    parser.add_argument("--level", required=True, type=int, choices=DEFAULT_LEVELS)
    parser.add_argument("--problem-ids")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--arch", default="sm_89")
    parser.add_argument("--num-correct-trials", type=int, default=5)
    parser.add_argument("--num-perf-trials", type=int, default=100)
    parser.add_argument("--official-eval-seed", type=int)
    parser.add_argument("--per-problem-timeout-seconds", type=int, default=300)
    parser.add_argument(
        "--torch-compile-baseline",
        action="store_true",
        help="Also time the reference model with default torch.compile and record speedup_vs_compile_default.",
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Evaluate submissions in the current process instead of an isolated subprocess.",
    )
    parser.add_argument("--overwrite-existing", action="store_true", help="Re-evaluate even when per-problem result JSON already exists.")
    args = parser.parse_args()

    run_dir = REPO_ROOT / "runs" / args.run_name / args.backend / f"level{args.level}"
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    run_manifest = load_json_if_exists(run_dir / "run_manifest.json")
    track = (run_manifest or {}).get("track", "oneshot")

    protocol = resolve_eval_protocol(
        level=args.level,
        track=track,
        run_manifest=run_manifest,
        precision=args.precision,
        arch=args.arch,
        num_correct_trials=args.num_correct_trials,
        num_perf_trials=args.num_perf_trials,
        official_eval_seed=args.official_eval_seed,
        torch_compile_baseline=args.torch_compile_baseline,
    )
    requested_ids = normalize_problem_ids(parse_problem_ids(args.problem_ids))
    dataset = construct_dataset(level=args.level, problem_ids=requested_ids)

    output_dir = REPO_ROOT / "results" / "timing" / args.run_name / args.backend / f"level{args.level}"
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_manifest = {
        "run_name": args.run_name,
        "backend": args.backend,
        "track": track,
        "level": args.level,
        "problem_ids": [problem.problem_id for problem in dataset],
        "protocol": protocol,
        "environment": detect_runtime_environment(),
    }
    (output_dir / "eval_manifest.json").write_text(json.dumps(eval_manifest, indent=2), encoding="utf-8")

    results = []
    for problem in dataset:
        submission_path = run_dir / f"{problem.problem_id:03d}_{problem.path.stem}.py"
        failure_path = generation_failure_path(submission_path)
        result_path = output_dir / f"{problem.problem_id:03d}.json"
        if result_path.exists() and not args.overwrite_existing:
            payload = load_json_if_exists(result_path)
            if isinstance(payload, dict):
                payload = enrich_result_payload(payload, problem=problem, submission_path=submission_path)
                if result_matches_current_eval(payload, protocol=protocol, submission_path=submission_path):
                    payload = stamp_eval_metadata(
                        payload,
                        protocol=protocol,
                        submission_path=submission_path,
                        environment=eval_manifest["environment"],
                        track=track,
                    )
                    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    results.append(payload)
                    continue
        if submission_path.exists():
            payload = evaluate_submission_payload_safely(
                problem=problem,
                submission_path=submission_path,
                backend=args.backend,
                device=args.device,
                precision=args.precision,
                arch=args.arch,
                num_correct_trials=args.num_correct_trials,
                num_perf_trials=args.num_perf_trials,
                seed=int(protocol["official_eval_seed"]),
                timeout_seconds=args.per_problem_timeout_seconds,
                in_process=args.in_process,
                measure_compile_default_baseline=args.torch_compile_baseline,
            )
        else:
            metadata = {
                "missing_submission": True,
            }
            if failure_path.exists():
                failure_payload = json.loads(failure_path.read_text(encoding="utf-8"))
                metadata["generation_failure"] = failure_payload.get("metadata", {})
                metadata["failure_artifact_path"] = str(failure_path)
            result = build_missing_submission_result(
                problem,
                backend=args.backend,
                expected_path=submission_path,
                metadata=metadata,
                )
            payload = result.to_dict()
        payload = enrich_result_payload(payload, problem=problem, submission_path=submission_path)
        payload = stamp_eval_metadata(
            payload,
            protocol=protocol,
            submission_path=submission_path,
            environment=eval_manifest["environment"],
            track=track,
        )
        results.append(payload)
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} evaluation records to {summary_path}")


if __name__ == "__main__":
    main()
