from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any
import json
import math
import statistics

from ptxbench.analysis import (
    FAILURE_STAGES,
    PAPER_FAILURE_CATEGORIES,
    classify_paper_failure_category,
    classify_result_stage,
)
from ptxbench.config import DEFAULT_FAST_THRESHOLDS, REPO_ROOT
from ptxbench.statistics import wilson_interval


def classify_paired_outcome(ptx_row: dict[str, Any], cuda_row: dict[str, Any]) -> str:
    ptx_correct = bool(ptx_row.get("correctness"))
    cuda_correct = bool(cuda_row.get("correctness"))
    if ptx_correct and cuda_correct:
        ptx_runtime = float(ptx_row.get("runtime_ms", -1.0))
        cuda_runtime = float(cuda_row.get("runtime_ms", -1.0))
        if ptx_runtime > 0 and cuda_runtime > 0:
            if math.isclose(ptx_runtime, cuda_runtime, rel_tol=1e-12, abs_tol=1e-12):
                return "tie"
            return "ptx_win" if ptx_runtime < cuda_runtime else "cuda_win"
        return "tie"
    if ptx_correct:
        return "ptx_only_correct"
    if cuda_correct:
        return "cuda_only_correct"
    return "both_wrong"


def load_json_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_csv_values(raw: str, cast) -> list:
    return [cast(part.strip()) for part in raw.split(",") if part.strip()]


def _summary_path(run_name: str, backend: str, level: int) -> Path:
    return REPO_ROOT / "results" / "timing" / run_name / backend / f"level{level}" / "summary.json"


def _run_manifest_path(run_name: str, backend: str, level: int) -> Path:
    return REPO_ROOT / "runs" / run_name / backend / f"level{level}" / "run_manifest.json"


def _analysis_path(run_name: str, level: int) -> Path:
    return REPO_ROOT / "results" / "analysis" / f"{run_name}_level{level}.json"


def _median(values: list[float]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value)) and float(value) > 0]
    return float(statistics.median(clean)) if clean else None


def _geomean(values: list[float]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value)) and float(value) > 0]
    if not clean:
        return None
    return math.exp(sum(math.log(value) for value in clean) / len(clean))


def _fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def _safe_speedup(row: dict[str, Any], field: str) -> float | None:
    value = row.get(field)
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) and numeric > 0 else None


def build_main_row(
    *,
    run_name: str,
    level: int,
    backend: str,
    rows: list[dict[str, Any]],
    thresholds: list[float],
    run_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    n = len(rows)
    correct = sum(bool(row.get("correctness")) for row in rows)
    ci_low, ci_high = wilson_interval(correct, n)
    correct_rows = [row for row in rows if row.get("correctness")]
    output: dict[str, Any] = {
        "run_name": run_name,
        "level": level,
        "backend": backend,
        "track": (run_manifest or {}).get("track") or (rows[0].get("track") if rows else ""),
        "model": (run_manifest or {}).get("model", ""),
        "provider": (run_manifest or {}).get("provider", ""),
        "n": n,
        "correct": correct,
        "correctness_rate": correct / n if n else 0.0,
        "correctness_ci_low": ci_low,
        "correctness_ci_high": ci_high,
        "geomean_speedup_correct_only": _geomean([_safe_speedup(row, "speedup_vs_eager") or 0 for row in correct_rows]),
        "median_speedup_correct_only": _median([_safe_speedup(row, "speedup_vs_eager") or 0 for row in correct_rows]),
        "median_runtime_ms_correct_only": _median([float(row.get("runtime_ms", -1.0)) for row in correct_rows]),
        "median_ref_runtime_eager_ms": _median([float(row.get("ref_runtime_eager_ms", row.get("ref_runtime_ms", -1.0))) for row in rows]),
        "median_ref_runtime_compile_default_ms": _median([
            float(row["ref_runtime_compile_default_ms"])
            for row in rows
            if row.get("ref_runtime_compile_default_ms") is not None
        ]),
    }
    for threshold in thresholds:
        fast_eager = [
            row
            for row in rows
            if row.get("correctness") and (_safe_speedup(row, "speedup_vs_eager") or 0.0) > threshold
        ]
        low, high = wilson_interval(len(fast_eager), n)
        key = _threshold_key(threshold)
        output[f"fast_p_vs_eager_{key}"] = len(fast_eager) / n if n else 0.0
        output[f"fast_p_vs_eager_{key}_ci_low"] = low
        output[f"fast_p_vs_eager_{key}_ci_high"] = high
        if any(row.get("speedup_vs_compile_default") is not None for row in rows):
            fast_compile = [
                row
                for row in rows
                if row.get("correctness") and (_safe_speedup(row, "speedup_vs_compile_default") or 0.0) > threshold
            ]
            low, high = wilson_interval(len(fast_compile), n)
            output[f"fast_p_vs_compile_default_{key}"] = len(fast_compile) / n if n else 0.0
            output[f"fast_p_vs_compile_default_{key}_ci_low"] = low
            output[f"fast_p_vs_compile_default_{key}_ci_high"] = high
    return output


def _threshold_key(threshold: float) -> str:
    return str(threshold).replace(".", "_")


def build_paired_rows(
    *,
    run_name: str,
    level: int,
    ptx_rows: list[dict[str, Any]],
    cuda_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ptx_by_id = {int(row["problem_id"]): row for row in ptx_rows}
    cuda_by_id = {int(row["problem_id"]): row for row in cuda_rows}
    if sorted(ptx_by_id) != sorted(cuda_by_id):
        raise ValueError(f"PTX and CUDA summaries evaluated different problem IDs for level {level}")
    paired = []
    for problem_id in sorted(ptx_by_id):
        ptx = ptx_by_id[problem_id]
        cuda = cuda_by_id[problem_id]
        paired.append(
            {
                "run_name": run_name,
                "level": level,
                "problem_id": problem_id,
                "problem_name": ptx.get("problem_name") or cuda.get("problem_name", ""),
                "ptx_correct": bool(ptx.get("correctness")),
                "cuda_correct": bool(cuda.get("correctness")),
                "ptx_runtime_ms": ptx.get("runtime_ms"),
                "cuda_runtime_ms": cuda.get("runtime_ms"),
                "ref_runtime_eager_ms": ptx.get("ref_runtime_eager_ms", ptx.get("ref_runtime_ms")),
                "ref_runtime_compile_default_ms": ptx.get("ref_runtime_compile_default_ms"),
                "ptx_speedup_vs_eager": ptx.get("speedup_vs_eager", ptx.get("speedup_vs_torch")),
                "cuda_speedup_vs_eager": cuda.get("speedup_vs_eager", cuda.get("speedup_vs_torch")),
                "ptx_speedup_vs_compile_default": ptx.get("speedup_vs_compile_default"),
                "cuda_speedup_vs_compile_default": cuda.get("speedup_vs_compile_default"),
                "paired_outcome": classify_paired_outcome(ptx, cuda),
                "ptx_failure_category": ptx.get("failure_category") or ptx.get("metadata", {}).get("failure_category"),
                "cuda_failure_category": cuda.get("failure_category") or cuda.get("metadata", {}).get("failure_category"),
                "ptx_paper_failure_category": ptx.get("paper_failure_category") or classify_paper_failure_category(ptx),
                "cuda_paper_failure_category": cuda.get("paper_failure_category") or classify_paper_failure_category(cuda),
            }
        )
    return paired


def build_ptx_resource_rows(*, run_name: str, level: int, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    resource_rows = []
    for row in rows:
        summary = row.get("metadata", {}).get("ptx_resource_summary", {})
        resource_rows.append(
            {
                "run_name": run_name,
                "level": level,
                "problem_id": row.get("problem_id"),
                "problem_name": row.get("problem_name"),
                "correctness": bool(row.get("correctness")),
                "runtime_ms": row.get("runtime_ms"),
                "speedup_vs_eager": row.get("speedup_vs_eager", row.get("speedup_vs_torch")),
                "max_registers": summary.get("max_registers"),
                "max_spill_stores_bytes": summary.get("max_spill_stores_bytes"),
                "max_spill_loads_bytes": summary.get("max_spill_loads_bytes"),
                "max_shared_memory_bytes": summary.get("max_shared_memory_bytes"),
                "max_local_memory_bytes": summary.get("max_local_memory_bytes"),
                "max_constant_memory_bytes": summary.get("max_constant_memory_bytes"),
                "max_stack_frame_bytes": summary.get("max_stack_frame_bytes"),
                "any_spills": summary.get("any_spills"),
                "num_artifacts": summary.get("num_artifacts"),
                "num_functions": summary.get("num_functions"),
            }
        )
    return resource_rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_breakdown_rows(run_name: str, level: int, backend: str, rows: list[dict[str, Any]], *, paper: bool) -> list[dict[str, Any]]:
    categories = PAPER_FAILURE_CATEGORIES if paper else FAILURE_STAGES
    counts = {category: 0 for category in categories}
    for row in rows:
        category = classify_paper_failure_category(row) if paper else classify_result_stage(row)
        counts[category] = counts.get(category, 0) + 1
    field = "paper_failure_category" if paper else "failure_category"
    return [
        {"run_name": run_name, "level": level, "backend": backend, field: category, "count": count}
        for category, count in counts.items()
        if count
    ]


def _render_markdown(main_rows: list[dict[str, Any]], paired_rows: list[dict[str, Any]], failure_rows: list[dict[str, Any]], ptx_resource_rows: list[dict[str, Any]]) -> str:
    lines = ["# PTXBench Paper Tables", ""]
    lines.extend(["## Main Results", "", "| Level | Backend | Model | Correctness | fast_p@1 | Geomean speedup |", "| --- | --- | --- | --- | --- | --- |"])
    for row in main_rows:
        correctness = f"{float(row['correctness_rate']):.3f} ({float(row['correctness_ci_low']):.3f}, {float(row['correctness_ci_high']):.3f})"
        lines.append(
            f"| {row['level']} | {row['backend']} | {row.get('model', '')} | {correctness} | "
            f"{float(row.get('fast_p_vs_eager_1_0', 0.0)):.3f} | {_fmt_float(row.get('geomean_speedup_correct_only'))} |"
        )
    lines.extend(["", "## Paired Outcomes", "", "| Outcome | Count |", "| --- | --- |"])
    outcome_counts: dict[str, int] = {}
    for row in paired_rows:
        outcome_counts[str(row["paired_outcome"])] = outcome_counts.get(str(row["paired_outcome"]), 0) + 1
    for outcome, count in sorted(outcome_counts.items()):
        lines.append(f"| {outcome} | {count} |")
    lines.extend(["", "## Failure Breakdown", "", "| Backend | Category | Count |", "| --- | --- | --- |"])
    for row in failure_rows:
        lines.append(f"| {row['backend']} | {row.get('paper_failure_category', row.get('failure_category'))} | {row['count']} |")
    correct_resources = [row for row in ptx_resource_rows if row.get("correctness")]
    lines.extend(["", "## PTX Resource Summary For Correct Kernels", "", "| Metric | Median | Max |", "| --- | --- | --- |"])
    for field in ("max_registers", "max_spill_stores_bytes", "max_spill_loads_bytes", "max_shared_memory_bytes", "max_local_memory_bytes", "max_constant_memory_bytes", "max_stack_frame_bytes"):
        values = [float(row[field]) for row in correct_resources if row.get(field) is not None]
        lines.append(f"| {field} | {_fmt_float(_median(values))} | {_fmt_float(max(values) if values else None)} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build deterministic PTXBench paper CSV/Markdown artifacts from result JSON.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--levels", default="1")
    parser.add_argument("--thresholds", default=",".join(str(value) for value in DEFAULT_FAST_THRESHOLDS))
    parser.add_argument("--require-backends", default="ptx,cuda")
    parser.add_argument("--out-dir")
    args = parser.parse_args(argv)

    levels = _parse_csv_values(args.levels, int)
    thresholds = _parse_csv_values(args.thresholds, float)
    backends = _parse_csv_values(args.require_backends, str)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else REPO_ROOT / "results" / "paper" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    main_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    paper_failure_rows: list[dict[str, Any]] = []
    ptx_resource_rows: list[dict[str, Any]] = []
    inputs: list[str] = []

    for level in levels:
        backend_rows: dict[str, list[dict[str, Any]]] = {}
        for backend in backends:
            summary_path = _summary_path(args.run_name, backend, level)
            rows = load_json_if_exists(summary_path)
            if not isinstance(rows, list):
                raise FileNotFoundError(f"Timing summary missing or malformed: {summary_path}")
            manifest = load_json_if_exists(_run_manifest_path(args.run_name, backend, level))
            backend_rows[backend] = [dict(row) for row in rows]
            inputs.append(str(summary_path))
            main_rows.append(
                build_main_row(
                    run_name=args.run_name,
                    level=level,
                    backend=backend,
                    rows=backend_rows[backend],
                    thresholds=thresholds,
                    run_manifest=manifest if isinstance(manifest, dict) else None,
                )
            )
            failure_rows.extend(_build_breakdown_rows(args.run_name, level, backend, backend_rows[backend], paper=False))
            paper_failure_rows.extend(_build_breakdown_rows(args.run_name, level, backend, backend_rows[backend], paper=True))
            if backend == "ptx":
                ptx_resource_rows.extend(build_ptx_resource_rows(run_name=args.run_name, level=level, rows=backend_rows[backend]))
        if "ptx" in backend_rows and "cuda" in backend_rows:
            paired_rows.extend(build_paired_rows(run_name=args.run_name, level=level, ptx_rows=backend_rows["ptx"], cuda_rows=backend_rows["cuda"]))
        analysis_path = _analysis_path(args.run_name, level)
        if analysis_path.exists():
            inputs.append(str(analysis_path))

    _write_csv(out_dir / "main_results.csv", main_rows)
    _write_csv(out_dir / "paired_results.csv", paired_rows)
    _write_csv(out_dir / "failure_breakdown.csv", failure_rows)
    _write_csv(out_dir / "paper_failure_breakdown.csv", paper_failure_rows)
    _write_csv(out_dir / "ptx_resource_metrics.csv", ptx_resource_rows)
    (out_dir / "paper_tables.md").write_text(
        _render_markdown(main_rows, paired_rows, paper_failure_rows, ptx_resource_rows),
        encoding="utf-8",
    )
    manifest = {
        "run_name": args.run_name,
        "levels": levels,
        "thresholds": thresholds,
        "required_backends": backends,
        "inputs": sorted(inputs),
        "outputs": sorted(path.name for path in out_dir.iterdir() if path.is_file()),
    }
    (out_dir / "report_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote paper report artifacts to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
