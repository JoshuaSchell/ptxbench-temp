from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from ptxbench.config import REPO_ROOT


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected dict report payload, got {type(payload).__name__}")
    return payload


def build_coverage_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for backend in ("ptx", "cuda"):
        for level_key, level_payload in sorted(report.get("interactive", {}).get(backend, {}).items()):
            summary = level_payload["summary"]
            rows.append(
                {
                    "backend": backend,
                    "level": level_key,
                    "label": f"{backend.upper()} {level_key.replace('level', 'L')}",
                    "total_tasks": summary["total_tasks"],
                    "correct_tasks": summary["correct_tasks"],
                    "correctness_rate": summary["correctness_rate"],
                    "fast_p_0": summary["fast_p_vs_torch"]["0.0"],
                    "fast_p_1": summary["fast_p_vs_torch"]["1.0"],
                    "fast_p_2": summary["fast_p_vs_torch"]["2.0"],
                    "geomean_correct_only": summary["geomean_speedup_vs_torch_correct_only"],
                    "geomean_correct_and_faster_only": summary["geomean_speedup_vs_torch_correct_and_faster_only"],
                }
            )
    hybrid_rows = report.get("hybrid_level3", [])
    if hybrid_rows:
        correct = [
            bool(row.get("correctness"))
            and float(row.get("runtime_ms", -1.0) or -1.0) > 0
            and float(row.get("ref_runtime_ms", -1.0) or -1.0) > 0
            for row in hybrid_rows
        ]
        speedups = [
            float(row.get("speedup_vs_torch", 0.0) or 0.0)
            for row in hybrid_rows
            if bool(row.get("correctness"))
            and float(row.get("runtime_ms", -1.0) or -1.0) > 0
            and float(row.get("ref_runtime_ms", -1.0) or -1.0) > 0
        ]
        geomean = math.exp(sum(math.log(value) for value in speedups) / len(speedups)) if speedups else 0.0
        total = len(hybrid_rows)
        correct_tasks = sum(correct)
        rows.append(
            {
                "backend": "ptx-hybrid",
                "level": "level3",
                "label": "PTX-Hybrid L3",
                "total_tasks": total,
                "correct_tasks": correct_tasks,
                "correctness_rate": (correct_tasks / total) if total else 0.0,
                "fast_p_0": (correct_tasks / total) if total else 0.0,
                "fast_p_1": (sum(1 for value in speedups if value > 1.0) / total) if total else 0.0,
                "fast_p_2": (sum(1 for value in speedups if value > 2.0) / total) if total else 0.0,
                "geomean_correct_only": geomean,
                "geomean_correct_and_faster_only": geomean,
            }
        )
    return rows


def build_failure_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for backend in ("ptx", "cuda"):
        for level_key, level_payload in sorted(report.get("interactive", {}).get(backend, {}).items()):
            summary = level_payload["summary"]
            for stage, count in summary["failure_breakdown"].items():
                rows.append(
                    {
                        "backend": backend,
                        "level": level_key,
                        "failure_stage": stage,
                        "count": count,
                    }
                )
    return rows


def build_family_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for backend in ("ptx", "cuda"):
        for level_key, level_payload in sorted(report.get("interactive", {}).get(backend, {}).items()):
            for family, summary in sorted(level_payload.get("family_summaries", {}).items()):
                rows.append(
                    {
                        "backend": backend,
                        "level": level_key,
                        "family": family,
                        "total_tasks": summary["total_tasks"],
                        "correct_tasks": summary["correct_tasks"],
                        "correctness_rate": summary["correctness_rate"],
                        "fast_p_1": summary["fast_p_vs_torch"]["1.0"],
                        "fast_p_2": summary["fast_p_vs_torch"]["2.0"],
                        "geomean_correct_only": summary["geomean_speedup_vs_torch_correct_only"],
                    }
                )
    return rows


def build_top_win_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for backend in ("ptx", "cuda"):
        for level_key, level_payload in sorted(report.get("interactive", {}).get(backend, {}).items()):
            for rank, row in enumerate(level_payload.get("top_wins", []), start=1):
                rows.append(
                    {
                        "backend": backend,
                        "level": level_key,
                        "rank": rank,
                        "problem_id": row["problem_id"],
                        "problem_name": row["problem_name"],
                        "speedup_vs_torch": row["speedup_vs_torch"],
                        "runtime_ms": row["runtime_ms"],
                        "ref_runtime_ms": row["ref_runtime_ms"],
                    }
                )
    return rows


def build_overlap_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base_run_name, base_payload in sorted(report.get("comparisons", {}).items()):
        for backend in ("ptx", "cuda"):
            for level_key, overlap in sorted(base_payload.get(backend, {}).items()):
                interactive_summary = overlap["interactive_summary"]
                base_summary = overlap["base_summary"]
                rows.append(
                    {
                        "base_run": base_run_name,
                        "backend": backend,
                        "level": level_key,
                        "shared_tasks": len(overlap["shared_problem_ids"]),
                        "interactive_correct": interactive_summary["correct_tasks"],
                        "base_correct": base_summary["correct_tasks"],
                        "interactive_only_correct": len(overlap["interactive_only_correct_problem_ids"]),
                        "base_only_correct": len(overlap["base_only_correct_problem_ids"]),
                        "interactive_win_rate": overlap["interactive_head_to_head_win_rate"],
                        "interactive_vs_base_geomean_speedup": overlap["interactive_vs_base_geomean_speedup"],
                    }
                )
    return rows


def build_hybrid_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in report.get("hybrid_level3", []):
        rows.append(
            {
                "problem_id": row["problem_id"],
                "problem_name": row["problem_name"],
                "correctness": bool(row["correctness"]),
                "runtime_ms": row["runtime_ms"],
                "ref_runtime_ms": row["ref_runtime_ms"],
                "speedup_vs_torch": row["speedup_vs_torch"],
            }
        )
    return rows


def _load_result_rows(run_name: str, backend: str, level: int) -> dict[int, dict[str, Any]]:
    root = REPO_ROOT / "results" / "timing" / run_name / backend / f"level{level}"
    if not root.exists():
        return {}
    rows: dict[int, dict[str, Any]] = {}
    for path in sorted(root.glob("*.json")):
        if path.name in {"summary.json", "eval_manifest.json"}:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "problem_id" in payload:
            rows[int(payload["problem_id"])] = payload
    return rows


def build_control_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    run_name = str(report.get("interactive_run", "codex-interactive"))
    rows: list[dict[str, Any]] = []
    for level in (1, 2, 3):
        ptx_rows = _load_result_rows(run_name, "ptx", level)
        cuda_rows = _load_result_rows(run_name, "cuda", level)
        if not ptx_rows or not cuda_rows:
            continue
        shared_ids = sorted(set(ptx_rows) & set(cuda_rows))
        if not shared_ids:
            continue

        ptx_correct = sum(bool(ptx_rows[pid].get("correctness")) for pid in shared_ids)
        cuda_correct = sum(bool(cuda_rows[pid].get("correctness")) for pid in shared_ids)

        jointly_correct_ids = [
            pid
            for pid in shared_ids
            if bool(ptx_rows[pid].get("correctness"))
            and bool(cuda_rows[pid].get("correctness"))
            and float(ptx_rows[pid].get("runtime_ms", -1.0) or -1.0) > 0
            and float(cuda_rows[pid].get("runtime_ms", -1.0) or -1.0) > 0
        ]
        ptx_win_count = sum(
            1
            for pid in jointly_correct_ids
            if float(ptx_rows[pid]["runtime_ms"]) < float(cuda_rows[pid]["runtime_ms"])
        )
        ptx_over_cuda = [
            float(ptx_rows[pid]["runtime_ms"]) / float(cuda_rows[pid]["runtime_ms"])
            for pid in jointly_correct_ids
        ]
        geomean_ratio = (
            math.exp(sum(math.log(value) for value in ptx_over_cuda) / len(ptx_over_cuda))
            if ptx_over_cuda
            else 0.0
        )

        rows.append(
            {
                "level": f"level{level}",
                "label": f"L{level}",
                "shared_tasks": len(shared_ids),
                "ptx_correct": ptx_correct,
                "cuda_correct": cuda_correct,
                "ptx_correct_rate": (ptx_correct / len(shared_ids)) if shared_ids else 0.0,
                "cuda_correct_rate": (cuda_correct / len(shared_ids)) if shared_ids else 0.0,
                "jointly_correct": len(jointly_correct_ids),
                "ptx_win_rate": (ptx_win_count / len(jointly_correct_ids)) if jointly_correct_ids else 0.0,
                "ptx_over_cuda_geomean": geomean_ratio,
            }
        )
    return rows


def write_readme(output_dir: Path) -> None:
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Plot Tables",
                "",
                "CSV exports derived from `results/analysis/codex-interactive_report.json`.",
                "",
                "- `coverage_summary.csv`: main summary figure data",
                "- `failure_breakdown.csv`: stacked failure bars by backend/level",
                "- `family_summary.csv`: family-level bars and heatmaps",
                "- `top_wins.csv`: strongest interactive wins",
                "- `overlap_comparison.csv`: direct interactive-vs-base overlap rows",
                "- `hybrid_level3.csv`: exploratory PTX-hybrid results",
                "- `ptx_vs_cuda_control.csv`: PTX vs CUDA on shared interactive tasks",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export figure-ready CSV tables from the interactive analysis report.")
    parser.add_argument(
        "--report-json",
        default=str(REPO_ROOT / "results" / "analysis" / "codex-interactive_report.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results" / "analysis" / "codex-interactive_figures"),
    )
    args = parser.parse_args()

    report_path = Path(args.report_json)
    output_dir = Path(args.output_dir)
    report = _load_report(report_path)

    coverage_rows = build_coverage_rows(report)
    failure_rows = build_failure_rows(report)
    family_rows = build_family_rows(report)
    top_win_rows = build_top_win_rows(report)
    overlap_rows = build_overlap_rows(report)
    hybrid_rows = build_hybrid_rows(report)
    control_rows = build_control_rows(report)

    _write_csv(
        output_dir / "coverage_summary.csv",
        [
            "backend",
            "level",
            "label",
            "total_tasks",
            "correct_tasks",
            "correctness_rate",
            "fast_p_0",
            "fast_p_1",
            "fast_p_2",
            "geomean_correct_only",
            "geomean_correct_and_faster_only",
        ],
        coverage_rows,
    )
    _write_csv(
        output_dir / "failure_breakdown.csv",
        ["backend", "level", "failure_stage", "count"],
        failure_rows,
    )
    _write_csv(
        output_dir / "family_summary.csv",
        [
            "backend",
            "level",
            "family",
            "total_tasks",
            "correct_tasks",
            "correctness_rate",
            "fast_p_1",
            "fast_p_2",
            "geomean_correct_only",
        ],
        family_rows,
    )
    _write_csv(
        output_dir / "top_wins.csv",
        ["backend", "level", "rank", "problem_id", "problem_name", "speedup_vs_torch", "runtime_ms", "ref_runtime_ms"],
        top_win_rows,
    )
    _write_csv(
        output_dir / "overlap_comparison.csv",
        [
            "base_run",
            "backend",
            "level",
            "shared_tasks",
            "interactive_correct",
            "base_correct",
            "interactive_only_correct",
            "base_only_correct",
            "interactive_win_rate",
            "interactive_vs_base_geomean_speedup",
        ],
        overlap_rows,
    )
    _write_csv(
        output_dir / "hybrid_level3.csv",
        ["problem_id", "problem_name", "correctness", "runtime_ms", "ref_runtime_ms", "speedup_vs_torch"],
        hybrid_rows,
    )
    _write_csv(
        output_dir / "ptx_vs_cuda_control.csv",
        [
            "level",
            "label",
            "shared_tasks",
            "ptx_correct",
            "cuda_correct",
            "ptx_correct_rate",
            "cuda_correct_rate",
            "jointly_correct",
            "ptx_win_rate",
            "ptx_over_cuda_geomean",
        ],
        control_rows,
    )
    write_readme(output_dir)


if __name__ == "__main__":
    main()
