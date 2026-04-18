from __future__ import annotations

import argparse
from pathlib import Path
import json

from ptxbench.config import DEFAULT_LEVELS
from ptxbench.dataset import Problem, construct_dataset
from ptxbench.eval import evaluate_submission
from ptxbench.profiler import ProfileRequest, normalize_profile_metrics


def load_problem(level: int | None, problem_id: int | None, reference_file: str | None) -> Problem:
    if reference_file:
        path = Path(reference_file).resolve()
        return Problem(problem_id=0, level=0, name=path.name, path=path, code=path.read_text(encoding="utf-8"))
    if level is None or problem_id is None:
        raise ValueError("Either --reference-file or both --level and --problem-id are required")
    dataset = construct_dataset(level=level, problem_ids=[problem_id])
    return dataset.get_problem(problem_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a single PTXBench submission against a KernelBench-style problem.")
    parser.add_argument("--backend", required=True, choices=["ptx", "cuda"])
    parser.add_argument("--submission", required=True)
    parser.add_argument("--level", type=int, choices=DEFAULT_LEVELS)
    parser.add_argument("--problem-id", type=int)
    parser.add_argument("--reference-file")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--arch", default="sm_89")
    parser.add_argument("--num-correct-trials", type=int, default=5)
    parser.add_argument("--num-perf-trials", type=int, default=100)
    parser.add_argument("--profile", action="store_true", help="Collect optional Nsight profiler metrics after correctness passes.")
    parser.add_argument("--profile-trials", type=int, default=1)
    parser.add_argument(
        "--profile-metric",
        action="append",
        default=[],
        help="Nsight metric name(s). Accepts repeated flags or comma-separated values.",
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    problem = load_problem(args.level, args.problem_id, args.reference_file)
    result = evaluate_submission(
        problem=problem,
        submission_path=Path(args.submission).resolve(),
        backend=args.backend,
        device=args.device,
        precision=args.precision,
        arch=args.arch,
        num_correct_trials=args.num_correct_trials,
        num_perf_trials=args.num_perf_trials,
        profile_request=ProfileRequest(
            enabled=args.profile,
            num_trials=args.profile_trials,
            metrics=normalize_profile_metrics(args.profile_metric),
        ),
    )

    payload = result.to_dict()
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
