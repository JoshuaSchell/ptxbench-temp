from __future__ import annotations

import argparse
from pathlib import Path
import json

from ptxbench.analysis import (
    compute_agentic_budget_summary,
    compute_backend_summary,
    compute_family_backend_summaries,
    compute_joint_backend_summary,
)
from ptxbench.config import DEFAULT_FAST_THRESHOLDS, DEFAULT_LEVELS, REPO_ROOT
from ptxbench.run_metadata import protocol_differences, protocol_signature


def load_backend_summary(run_name: str, backend: str, level: int) -> list[dict]:
    path = REPO_ROOT / "results" / "timing" / run_name / backend / f"level{level}" / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Evaluation summary not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def backend_artifact_paths(run_name: str, backend: str, level: int) -> tuple[Path, Path]:
    run_manifest = REPO_ROOT / "runs" / run_name / backend / f"level{level}" / "run_manifest.json"
    eval_manifest = REPO_ROOT / "results" / "timing" / run_name / backend / f"level{level}" / "eval_manifest.json"
    return run_manifest, eval_manifest


def resolve_protocol(
    *,
    run_name: str,
    level: int,
    ptx_run_manifest: dict | None,
    cuda_run_manifest: dict | None,
    ptx_eval_manifest: dict | None,
    cuda_eval_manifest: dict | None,
) -> dict:
    for manifest in (ptx_run_manifest, cuda_run_manifest, ptx_eval_manifest, cuda_eval_manifest):
        protocol = (manifest or {}).get("protocol")
        if protocol:
            return protocol
    paper_manifest_path = REPO_ROOT / "runs" / run_name / "paper_run_manifest.json"
    paper_manifest = load_json_if_exists(paper_manifest_path)
    return (paper_manifest or {}).get("protocol", {})


def _manifest_protocol(manifest: dict | None) -> dict:
    protocol = (manifest or {}).get("protocol", {})
    return protocol if isinstance(protocol, dict) else {}


def _manifest_problem_ids(manifest: dict | None) -> list[int] | None:
    if not isinstance(manifest, dict):
        return None
    raw_problem_ids = manifest.get("problem_ids", manifest.get("problems"))
    if not isinstance(raw_problem_ids, list):
        return None
    return sorted(int(problem_id) for problem_id in raw_problem_ids)


def _format_protocol_differences(differences: dict[str, tuple[object, object]]) -> str:
    return ", ".join(
        f"{key}: {expected!r} != {observed!r}"
        for key, (expected, observed) in sorted(differences.items())
    )


def validate_paired_protocol_parity(
    *,
    ptx_run_manifest: dict | None,
    cuda_run_manifest: dict | None,
    ptx_eval_manifest: dict | None,
    cuda_eval_manifest: dict | None,
    ptx_problem_ids: list[int],
    cuda_problem_ids: list[int],
) -> dict:
    named_protocols = {
        "PTX run manifest": _manifest_protocol(ptx_run_manifest),
        "CUDA run manifest": _manifest_protocol(cuda_run_manifest),
        "PTX eval manifest": _manifest_protocol(ptx_eval_manifest),
        "CUDA eval manifest": _manifest_protocol(cuda_eval_manifest),
    }
    missing = [name for name, protocol in named_protocols.items() if not protocol]
    if missing:
        raise ValueError(
            "Paired PTX-vs-CUDA analysis requires complete run/eval manifests with protocol metadata. "
            f"Missing protocol data for: {', '.join(missing)}"
        )

    comparisons = (
        ("PTX run manifest", "CUDA run manifest"),
        ("PTX eval manifest", "CUDA eval manifest"),
        ("PTX run manifest", "PTX eval manifest"),
        ("CUDA run manifest", "CUDA eval manifest"),
    )
    for left_name, right_name in comparisons:
        differences = protocol_differences(named_protocols[left_name], named_protocols[right_name])
        if differences:
            raise ValueError(
                f"Protocol parity mismatch between {left_name} and {right_name}: "
                f"{_format_protocol_differences(differences)}"
            )

    if sorted(ptx_problem_ids) != sorted(cuda_problem_ids):
        raise ValueError(
            "Paired PTX-vs-CUDA analysis requires identical evaluated problem IDs for both backends. "
            f"ptx={sorted(ptx_problem_ids)} cuda={sorted(cuda_problem_ids)}"
        )

    ptx_manifest_ids = _manifest_problem_ids(ptx_eval_manifest) or _manifest_problem_ids(ptx_run_manifest)
    cuda_manifest_ids = _manifest_problem_ids(cuda_eval_manifest) or _manifest_problem_ids(cuda_run_manifest)
    if ptx_manifest_ids is not None and sorted(ptx_problem_ids) != ptx_manifest_ids:
        raise ValueError(
            "PTX evaluation summary problem IDs do not match the PTX manifest problem IDs. "
            f"summary={sorted(ptx_problem_ids)} manifest={ptx_manifest_ids}"
        )
    if cuda_manifest_ids is not None and sorted(cuda_problem_ids) != cuda_manifest_ids:
        raise ValueError(
            "CUDA evaluation summary problem IDs do not match the CUDA manifest problem IDs. "
            f"summary={sorted(cuda_problem_ids)} manifest={cuda_manifest_ids}"
        )

    return protocol_signature(named_protocols["PTX eval manifest"])


def render_analysis_markdown(payload: dict) -> str:
    ptx_summary = payload["backend_summaries"]["ptx"]
    cuda_summary = payload["backend_summaries"]["cuda"]
    paired = payload["paired_summary"]
    protocol = payload.get("protocol", {})
    track = payload.get("track", "oneshot")
    lines = [
        f"# PTXBench Analysis: {payload['run_name']} (Level {payload['level']})",
        "",
        "## Headline",
        "",
        f"- Track: {track}",
        f"- PTX correctness rate: {ptx_summary['correctness_rate']:.3f} ({ptx_summary['correct_tasks']}/{ptx_summary['total_tasks']})",
        f"- CUDA correctness rate: {cuda_summary['correctness_rate']:.3f} ({cuda_summary['correct_tasks']}/{cuda_summary['total_tasks']})",
        f"- Jointly correct tasks: {paired['jointly_correct_tasks']}/{paired['total_tasks']}",
        f"- PTX head-to-head win rate vs CUDA: {paired['ptx_head_to_head_win_rate']:.3f}",
        f"- PTX/CUDA geomean speedup on jointly-correct tasks: {paired['ptx_vs_cuda_geomean_speedup']:.3f}",
        "",
        "## Benchmark Metrics",
        "",
        "| Backend | correct@budget | fast_p@1.0 | fast_p@2.0 | Geomean speedup (correct only) |",
        "| --- | --- | --- | --- | --- |",
        (
            f"| PTX | {ptx_summary['correctness_rate']:.3f} | "
            f"{_threshold_value(ptx_summary['fast_p_vs_torch'], 1.0):.3f} | {_threshold_value(ptx_summary['fast_p_vs_torch'], 2.0):.3f} | "
            f"{ptx_summary['geomean_speedup_vs_torch_correct_only']:.3f} |"
        ),
        (
            f"| CUDA | {cuda_summary['correctness_rate']:.3f} | "
            f"{_threshold_value(cuda_summary['fast_p_vs_torch'], 1.0):.3f} | {_threshold_value(cuda_summary['fast_p_vs_torch'], 2.0):.3f} | "
            f"{cuda_summary['geomean_speedup_vs_torch_correct_only']:.3f} |"
        ),
        "",
        "## PTX vs CUDA Head-to-Head",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Jointly correct tasks | {paired['jointly_correct_tasks']}/{paired['total_tasks']} |",
        f"| PTX win rate vs CUDA | {paired['ptx_head_to_head_win_rate']:.3f} |",
        f"| PTX/CUDA geomean speedup | {paired['ptx_vs_cuda_geomean_speedup']:.3f} |",
        "",
        "## Methodology Parity vs KernelBench",
        "",
        "| Dimension | KernelBench-style baseline | PTXBench paper run |",
        "| --- | --- | --- |",
        f"| Task family | KernelBench Level {payload['level']} tasks | Same vendored KernelBench Level {payload['level']} tasks |",
        f"| Generation mode | One-shot | {'Agentic iterative' if track == 'agentic' else 'One-shot'} |",
        f"| Precision | fp32 | {protocol.get('precision', 'fp32')} |",
        f"| Architecture target | NVIDIA GPU | {protocol.get('arch', 'sm_89')} |",
        f"| Correctness trials | 5 | {protocol.get('num_correct_trials', 5)} |",
        f"| Performance trials | 100 | {protocol.get('num_perf_trials', 100)} |",
        "| Primary metric | fast_p vs PyTorch eager | Same, plus paired PTX-vs-CUDA metrics |",
        "| Code target | CUDA kernels | PTX and matched CUDA side-by-side |",
        "",
    ]
    agentic_summary = payload.get("agentic_budget_summary")
    if agentic_summary:
        lines.extend(
            [
                "## Agentic Budget",
                "",
                "| Metric | PTX | CUDA |",
                "| --- | --- | --- |",
                (
                    f"| correct@budget | {payload['backend_summaries']['ptx']['correctness_rate']:.3f} | "
                    f"{payload['backend_summaries']['cuda']['correctness_rate']:.3f} |"
                ),
                (
                    f"| median steps to first compile | "
                    f"{_fmt_optional(payload['backend_summaries']['ptx'].get('agentic_budget_summary', {}).get('median_steps_to_first_compile'))} | "
                    f"{_fmt_optional(payload['backend_summaries']['cuda'].get('agentic_budget_summary', {}).get('median_steps_to_first_compile'))} |"
                ),
                (
                    f"| median steps to first correct | "
                    f"{_fmt_optional(payload['backend_summaries']['ptx'].get('agentic_budget_summary', {}).get('median_steps_to_first_correct'))} | "
                    f"{_fmt_optional(payload['backend_summaries']['cuda'].get('agentic_budget_summary', {}).get('median_steps_to_first_correct'))} |"
                ),
                "",
            ]
        )

    lines.extend(
        [
        "## KernelBench Context",
        "",
        "| Comparison table | Status | Use in paper |",
        "| --- | --- | --- |",
        "| Published KernelBench numbers | Context only | Cite for framing, not direct superiority claims unless rerun with matched settings |",
        "| Local matched rerun against KernelBench protocol | Pending | Use for direct apples-to-apples claims only after matched reruns exist |",
        "",
        "## Failure Breakdown",
        "",
        "| Backend | Success | Compile | Assemble | Load | Runtime | Correctness |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        (
            f"| PTX | {ptx_summary['failure_breakdown']['success']} | {ptx_summary['failure_breakdown']['compile']} | "
            f"{ptx_summary['failure_breakdown']['assemble']} | {ptx_summary['failure_breakdown']['load']} | "
            f"{ptx_summary['failure_breakdown']['runtime']} | {ptx_summary['failure_breakdown']['correctness']} |"
        ),
        (
            f"| CUDA | {cuda_summary['failure_breakdown']['success']} | {cuda_summary['failure_breakdown']['compile']} | "
            f"{cuda_summary['failure_breakdown']['assemble']} | {cuda_summary['failure_breakdown']['load']} | "
            f"{cuda_summary['failure_breakdown']['runtime']} | {cuda_summary['failure_breakdown']['correctness']} |"
        ),
        "",
        "## Task Families",
        "",
        "| Backend | Family | Correctness | fast_p@1.0 | Tasks |",
        "| --- | --- | --- | --- | --- |",
    ]
    )
    for backend in ("ptx", "cuda"):
        for family, summary in payload.get("family_summaries", {}).get(backend, {}).items():
            lines.append(
                f"| {backend.upper()} | {family} | {summary['correctness_rate']:.3f} | "
                f"{_threshold_value(summary['fast_p_vs_torch'], 1.0):.3f} | {summary['total_tasks']} tasks |"
            )
    return "\n".join(lines)


def _fmt_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _threshold_value(values: dict, threshold: float) -> float:
    if threshold in values:
        return float(values[threshold])
    return float(values[str(threshold)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute PTXBench fast_p and paired PTX-vs-CUDA metrics.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--level", required=True, type=int, choices=DEFAULT_LEVELS)
    parser.add_argument("--compare-backends", default="ptx,cuda")
    parser.add_argument("--output")
    args = parser.parse_args()

    backends = [backend.strip() for backend in args.compare_backends.split(",") if backend.strip()]
    if sorted(backends) != ["cuda", "ptx"]:
        raise ValueError("--compare-backends must include exactly 'ptx,cuda'")

    ptx_rows = load_backend_summary(args.run_name, "ptx", args.level)
    cuda_rows = load_backend_summary(args.run_name, "cuda", args.level)
    ptx_run_manifest_path, ptx_eval_manifest_path = backend_artifact_paths(args.run_name, "ptx", args.level)
    cuda_run_manifest_path, cuda_eval_manifest_path = backend_artifact_paths(args.run_name, "cuda", args.level)
    ptx_run_manifest = load_json_if_exists(ptx_run_manifest_path)
    cuda_run_manifest = load_json_if_exists(cuda_run_manifest_path)
    ptx_eval_manifest = load_json_if_exists(ptx_eval_manifest_path)
    cuda_eval_manifest = load_json_if_exists(cuda_eval_manifest_path)
    ptx_by_problem = {row["problem_id"]: row for row in ptx_rows}
    cuda_by_problem = {row["problem_id"]: row for row in cuda_rows}
    parity_protocol = validate_paired_protocol_parity(
        ptx_run_manifest=ptx_run_manifest,
        cuda_run_manifest=cuda_run_manifest,
        ptx_eval_manifest=ptx_eval_manifest,
        cuda_eval_manifest=cuda_eval_manifest,
        ptx_problem_ids=list(ptx_by_problem),
        cuda_problem_ids=list(cuda_by_problem),
    )
    shared_problem_ids = sorted(ptx_by_problem)
    ptx_backend_summary = compute_backend_summary(ptx_rows, thresholds=DEFAULT_FAST_THRESHOLDS)
    cuda_backend_summary = compute_backend_summary(cuda_rows, thresholds=DEFAULT_FAST_THRESHOLDS)
    ptx_family_summaries = compute_family_backend_summaries(ptx_rows, thresholds=DEFAULT_FAST_THRESHOLDS)
    cuda_family_summaries = compute_family_backend_summaries(cuda_rows, thresholds=DEFAULT_FAST_THRESHOLDS)
    track = (ptx_run_manifest or {}).get("track") or (cuda_run_manifest or {}).get("track") or "oneshot"

    summary = compute_joint_backend_summary(
        ptx_correct=[bool(ptx_by_problem[problem_id]["correctness"]) for problem_id in shared_problem_ids],
        ptx_ref_ms=[float(ptx_by_problem[problem_id]["ref_runtime_ms"]) for problem_id in shared_problem_ids],
        ptx_ms=[float(ptx_by_problem[problem_id]["runtime_ms"]) for problem_id in shared_problem_ids],
        cuda_correct=[bool(cuda_by_problem[problem_id]["correctness"]) for problem_id in shared_problem_ids],
        cuda_ref_ms=[float(cuda_by_problem[problem_id]["ref_runtime_ms"]) for problem_id in shared_problem_ids],
        cuda_ms=[float(cuda_by_problem[problem_id]["runtime_ms"]) for problem_id in shared_problem_ids],
        thresholds=DEFAULT_FAST_THRESHOLDS,
    )

    payload = {
        "run_name": args.run_name,
        "level": args.level,
        "track": track,
        "thresholds": list(DEFAULT_FAST_THRESHOLDS),
        "protocol": resolve_protocol(
            run_name=args.run_name,
            level=args.level,
            ptx_run_manifest=ptx_run_manifest,
            cuda_run_manifest=cuda_run_manifest,
            ptx_eval_manifest=ptx_eval_manifest,
            cuda_eval_manifest=cuda_eval_manifest,
        ),
        "protocol_signature": parity_protocol,
        "shared_problem_ids": shared_problem_ids,
        "backend_summaries": {
            "ptx": {
                "total_tasks": ptx_backend_summary.total_tasks,
                "correct_tasks": ptx_backend_summary.correct_tasks,
                "correctness_rate": ptx_backend_summary.correctness_rate,
                "fast_p_vs_torch": ptx_backend_summary.fast_p_vs_torch,
                "fast1_vs_compile_default": _threshold_value(ptx_backend_summary.fast_p_vs_compile_default, 1.0),
                "fast2_vs_compile_default": _threshold_value(ptx_backend_summary.fast_p_vs_compile_default, 2.0),
                "geomean_speedup_vs_torch_correct_only": ptx_backend_summary.geomean_speedup_vs_torch_correct_only,
                "geomean_speedup_vs_torch_correct_and_faster_only": ptx_backend_summary.geomean_speedup_vs_torch_correct_and_faster_only,
                "failure_breakdown": ptx_backend_summary.failure_breakdown,
                "paper_failure_breakdown": ptx_backend_summary.paper_failure_breakdown,
                "agentic_budget_summary": (
                    _serialize_agentic_summary(compute_agentic_budget_summary(ptx_rows)) if track == "agentic" else None
                ),
                "run_manifest": ptx_run_manifest,
                "eval_manifest": ptx_eval_manifest,
            },
            "cuda": {
                "total_tasks": cuda_backend_summary.total_tasks,
                "correct_tasks": cuda_backend_summary.correct_tasks,
                "correctness_rate": cuda_backend_summary.correctness_rate,
                "fast_p_vs_torch": cuda_backend_summary.fast_p_vs_torch,
                "fast1_vs_compile_default": _threshold_value(cuda_backend_summary.fast_p_vs_compile_default, 1.0),
                "fast2_vs_compile_default": _threshold_value(cuda_backend_summary.fast_p_vs_compile_default, 2.0),
                "geomean_speedup_vs_torch_correct_only": cuda_backend_summary.geomean_speedup_vs_torch_correct_only,
                "geomean_speedup_vs_torch_correct_and_faster_only": cuda_backend_summary.geomean_speedup_vs_torch_correct_and_faster_only,
                "failure_breakdown": cuda_backend_summary.failure_breakdown,
                "paper_failure_breakdown": cuda_backend_summary.paper_failure_breakdown,
                "agentic_budget_summary": (
                    _serialize_agentic_summary(compute_agentic_budget_summary(cuda_rows)) if track == "agentic" else None
                ),
                "run_manifest": cuda_run_manifest,
                "eval_manifest": cuda_eval_manifest,
            },
        },
        "family_summaries": {
            "ptx": {
                family: _serialize_backend_summary(summary)
                for family, summary in ptx_family_summaries.items()
            },
            "cuda": {
                family: _serialize_backend_summary(summary)
                for family, summary in cuda_family_summaries.items()
            },
        },
        "paired_summary": {
            "total_tasks": summary.total_tasks,
            "jointly_correct_tasks": summary.jointly_correct_tasks,
            "jointly_correct_rate": (summary.jointly_correct_tasks / summary.total_tasks) if summary.total_tasks else 0.0,
            "fast_p_ptx_vs_torch": summary.fast_p_ptx_vs_torch,
            "fast_p_cuda_vs_torch": summary.fast_p_cuda_vs_torch,
            "ptx_head_to_head_win_rate": summary.ptx_head_to_head_win_rate,
            "ptx_vs_cuda_geomean_speedup": summary.ptx_vs_cuda_geomean_speedup,
        },
    }
    if track == "agentic":
        payload["agentic_budget_summary"] = {
            "ptx": payload["backend_summaries"]["ptx"]["agentic_budget_summary"],
            "cuda": payload["backend_summaries"]["cuda"]["agentic_budget_summary"],
        }
    payload["summary"] = payload["paired_summary"]
    rendered = json.dumps(payload, indent=2)
    print(rendered)

    output_path = (
        Path(args.output).resolve()
        if args.output
        else REPO_ROOT / "results" / "analysis" / f"{args.run_name}_level{args.level}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    output_path.with_suffix(".md").write_text(render_analysis_markdown(payload), encoding="utf-8")

def _serialize_backend_summary(summary) -> dict:
    return {
        "total_tasks": summary.total_tasks,
        "correct_tasks": summary.correct_tasks,
        "correctness_rate": summary.correctness_rate,
        "fast_p_vs_torch": summary.fast_p_vs_torch,
        "fast1_vs_compile_default": _threshold_value(summary.fast_p_vs_compile_default, 1.0),
        "fast2_vs_compile_default": _threshold_value(summary.fast_p_vs_compile_default, 2.0),
        "geomean_speedup_vs_torch_correct_only": summary.geomean_speedup_vs_torch_correct_only,
        "geomean_speedup_vs_torch_correct_and_faster_only": summary.geomean_speedup_vs_torch_correct_and_faster_only,
        "failure_breakdown": summary.failure_breakdown,
        "paper_failure_breakdown": summary.paper_failure_breakdown,
    }


def _serialize_agentic_summary(summary) -> dict:
    return {
        "total_tasks": summary.total_tasks,
        "correct_at_budget": summary.correct_at_budget,
        "median_steps_to_first_compile": summary.median_steps_to_first_compile,
        "median_steps_to_first_correct": summary.median_steps_to_first_correct,
    }


if __name__ == "__main__":
    main()
