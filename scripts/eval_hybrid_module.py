from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch

from ptxbench.dataset import construct_dataset
from ptxbench.eval import _compare_outputs, get_tolerance, set_seed
from ptxbench.timing import summarize_timings, time_callable_cuda_events


def _load_module(path: Path, name_prefix: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"{name_prefix}_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare_inputs(inputs: list[Any], device: str) -> list[Any]:
    prepared: list[Any] = []
    for item in inputs:
        if isinstance(item, torch.Tensor):
            prepared.append(item.to(device=device, dtype=torch.float32))
        else:
            prepared.append(item)
    return prepared


def _cleanup() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a PTX+torch hybrid module against a vendor reference problem.")
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--problem-id", type=int, required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-correct-trials", type=int, default=1)
    parser.add_argument("--num-perf-trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output")
    args = parser.parse_args()

    dataset = construct_dataset(level=args.level, problem_ids=[args.problem_id])
    problem = dataset.get_problem(args.problem_id)
    reference_module = _load_module(problem.path, "reference")
    candidate_module = _load_module(Path(args.submission).resolve(), "candidate")

    tolerance = get_tolerance("fp32")
    payload: dict[str, Any] = {
        "track": "hybrid",
        "backend": "ptx-hybrid",
        "level": args.level,
        "problem_id": args.problem_id,
        "problem_name": problem.name,
        "source_path": str(Path(args.submission).resolve()),
        "correctness": False,
        "runtime_ms": -1.0,
        "ref_runtime_ms": -1.0,
        "speedup_vs_torch": 0.0,
        "metadata": {},
    }

    passed = 0
    for trial in range(args.num_correct_trials):
        trial_seed = args.seed + trial
        set_seed(trial_seed)
        reference_model = reference_module.Model(*reference_module.get_init_inputs()).to(args.device).eval()
        set_seed(trial_seed)
        candidate_model = candidate_module.ModelNew(*reference_module.get_init_inputs()).to(args.device).eval()
        inputs = _prepare_inputs(list(reference_module.get_inputs()), args.device)
        with torch.no_grad():
            reference_out = reference_model(*inputs)
            candidate_out = candidate_model(*inputs)
        ok, message, details = _compare_outputs(reference_out, candidate_out, atol=tolerance, rtol=tolerance)
        del reference_model, candidate_model, inputs, reference_out, candidate_out
        _cleanup()
        if not ok:
            payload["metadata"]["correctness_error"] = message
            if details is not None:
                payload["metadata"]["correctness_mismatch"] = details
            break
        passed += 1

    payload["correctness"] = passed == args.num_correct_trials
    payload["metadata"]["correctness_trials"] = f"{passed}/{args.num_correct_trials}"

    if payload["correctness"]:
        set_seed(args.seed)
        reference_model = reference_module.Model(*reference_module.get_init_inputs()).to(args.device).eval()
        set_seed(args.seed)
        candidate_model = candidate_module.ModelNew(*reference_module.get_init_inputs()).to(args.device).eval()
        perf_inputs = _prepare_inputs(list(reference_module.get_inputs()), args.device)
        with torch.no_grad():
            reference_samples = time_callable_cuda_events(
                lambda: reference_model(*perf_inputs),
                num_warmup=1,
                num_trials=args.num_perf_trials,
                device=args.device,
            )
            candidate_samples = time_callable_cuda_events(
                lambda: candidate_model(*perf_inputs),
                num_warmup=1,
                num_trials=args.num_perf_trials,
                device=args.device,
            )
        payload["ref_runtime_ms"] = summarize_timings(reference_samples).mean_ms
        payload["runtime_ms"] = summarize_timings(candidate_samples).mean_ms
        if payload["runtime_ms"] > 0:
            payload["speedup_vs_torch"] = payload["ref_runtime_ms"] / payload["runtime_ms"]
        del reference_model, candidate_model, perf_inputs
        _cleanup()

    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
