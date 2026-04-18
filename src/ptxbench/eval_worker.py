from __future__ import annotations

import argparse
from pathlib import Path
import json
import traceback

from .eval import evaluate_submission
from .isolated_eval import annotate_eval_payload, classify_failure_category, deserialize_problem, _build_failure_payload


def _exception_looks_like_oom(exc: Exception) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cudnn_status_alloc_failed" in message


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate one PTXBench submission inside an isolated worker process.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    request = json.loads(input_path.read_text(encoding="utf-8"))
    problem = deserialize_problem(request["problem"])
    submission_path = Path(request["submission_path"])

    try:
        payload = evaluate_submission(
            problem=problem,
            submission_path=submission_path,
            backend=request["backend"],
            device=request["device"],
            precision=request["precision"],
            arch=request["arch"],
            num_correct_trials=int(request["num_correct_trials"]),
            num_perf_trials=int(request["num_perf_trials"]),
            num_warmup=int(request["num_warmup"]),
            run_static_checks=bool(request["run_static_checks"]),
            seed=int(request["seed"]),
        ).to_dict()
    except Exception as exc:
        category = "oom" if _exception_looks_like_oom(exc) else "evaluator_crash"
        metadata = {
            "runtime_error": str(exc),
            "runtime_traceback": traceback.format_exc(),
        }
        if category == "evaluator_crash":
            metadata["evaluator_crash"] = True
            metadata["evaluator_error"] = True
        payload = _build_failure_payload(
            problem,
            backend=request["backend"],
            source_path=submission_path,
            category=category,
            metadata=metadata,
        )

    metadata = dict(payload.get("metadata", {}))
    metadata["failure_category"] = classify_failure_category(payload)
    payload["metadata"] = metadata
    payload = annotate_eval_payload(payload, mode="worker")
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
