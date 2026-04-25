from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from ptxbench.analysis import compute_backend_summary, compute_family_backend_summaries
from ptxbench.config import REPO_ROOT


def load_result_rows(run_name: str, backend: str, level: int) -> list[dict[str, Any]]:
    root = REPO_ROOT / "results" / "timing" / run_name / backend / f"level{level}"
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        if path.name in {"summary.json", "eval_manifest.json"}:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "problem_id" in payload:
            rows.append(payload)
    return rows


def _summary_to_dict(summary: Any) -> dict[str, Any]:
    return {
        "total_tasks": summary.total_tasks,
        "correct_tasks": summary.correct_tasks,
        "correctness_rate": summary.correctness_rate,
        "fast_p_vs_torch": {str(k): v for k, v in summary.fast_p_vs_torch.items()},
        "geomean_speedup_vs_torch_correct_only": summary.geomean_speedup_vs_torch_correct_only,
        "geomean_speedup_vs_torch_correct_and_faster_only": summary.geomean_speedup_vs_torch_correct_and_faster_only,
        "failure_breakdown": dict(summary.failure_breakdown),
        "fast_p_vs_compile_default": {str(k): v for k, v in summary.fast_p_vs_compile_default.items()},
    }


def _family_summaries_to_dict(family_summaries: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {family: _summary_to_dict(summary) for family, summary in family_summaries.items()}


def _geomean(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.exp(sum(math.log(x) for x in values) / len(values))


def build_overlap(interactive_rows: list[dict[str, Any]], base_rows: list[dict[str, Any]]) -> dict[str, Any]:
    interactive_map = {int(row["problem_id"]): row for row in interactive_rows}
    base_map = {int(row["problem_id"]): row for row in base_rows}
    shared_ids = sorted(set(interactive_map) & set(base_map))
    interactive_shared = [interactive_map[pid] for pid in shared_ids]
    base_shared = [base_map[pid] for pid in shared_ids]

    jointly_correct = 0
    interactive_wins = 0
    ratios: list[float] = []
    interactive_only_correct: list[int] = []
    base_only_correct: list[int] = []
    for interactive_row, base_row in zip(interactive_shared, base_shared, strict=True):
        interactive_ok = bool(interactive_row.get("correctness"))
        base_ok = bool(base_row.get("correctness"))
        pid = int(interactive_row["problem_id"])
        if interactive_ok and not base_ok:
            interactive_only_correct.append(pid)
        if base_ok and not interactive_ok:
            base_only_correct.append(pid)
        interactive_ms = float(interactive_row.get("runtime_ms", -1.0) or -1.0)
        base_ms = float(base_row.get("runtime_ms", -1.0) or -1.0)
        if interactive_ok and base_ok and interactive_ms > 0 and base_ms > 0:
            jointly_correct += 1
            if interactive_ms < base_ms:
                interactive_wins += 1
            ratios.append(base_ms / interactive_ms)

    return {
        "shared_problem_ids": shared_ids,
        "shared_problem_names": [row["problem_name"] for row in interactive_shared],
        "interactive_summary": _summary_to_dict(compute_backend_summary(interactive_shared)),
        "base_summary": _summary_to_dict(compute_backend_summary(base_shared)),
        "jointly_correct_tasks": jointly_correct,
        "interactive_head_to_head_win_rate": (interactive_wins / jointly_correct) if jointly_correct else 0.0,
        "interactive_vs_base_geomean_speedup": _geomean(ratios),
        "interactive_only_correct_problem_ids": interactive_only_correct,
        "base_only_correct_problem_ids": base_only_correct,
    }


def collect_top_wins(rows: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    winners = [
        {
            "problem_id": int(row["problem_id"]),
            "problem_name": row["problem_name"],
            "speedup_vs_torch": float(row.get("speedup_vs_torch", 0.0) or 0.0),
            "runtime_ms": float(row.get("runtime_ms", -1.0) or -1.0),
            "ref_runtime_ms": float(row.get("ref_runtime_ms", -1.0) or -1.0),
        }
        for row in rows
        if bool(row.get("correctness"))
        and float(row.get("runtime_ms", -1.0) or -1.0) >= 0.05
        and float(row.get("ref_runtime_ms", -1.0) or -1.0) >= 0.05
        and float(row.get("speedup_vs_torch", 0.0) or 0.0) > 1.0
    ]
    winners.sort(key=lambda row: row["speedup_vs_torch"], reverse=True)
    return winners[:limit]


def collect_hybrid_eval_records() -> list[dict[str, Any]]:
    root = REPO_ROOT / "runs" / "codex-interactive" / "ptx-hybrid" / "level3"
    if not root.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("*_eval.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "problem_id" in payload:
            records.append(payload)
    return records


def build_payload(interactive_run: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "interactive_run": interactive_run,
        "interactive": {},
        "comparisons": {},
        "hybrid_level3": collect_hybrid_eval_records(),
    }

    for backend in ("ptx", "cuda"):
        backend_payload: dict[str, Any] = {}
        for level in (1, 2, 3):
            rows = load_result_rows(interactive_run, backend, level)
            if not rows:
                continue
            backend_payload[f"level{level}"] = {
                "record_count": len(rows),
                "problem_ids": sorted(int(row["problem_id"]) for row in rows),
                "summary": _summary_to_dict(compute_backend_summary(rows)),
                "family_summaries": _family_summaries_to_dict(compute_family_backend_summaries(rows)),
                "top_wins": collect_top_wins(rows),
            }
        if backend_payload:
            payload["interactive"][backend] = backend_payload

    for base_root in sorted((REPO_ROOT / "results" / "timing").iterdir()):
        if not base_root.is_dir():
            continue
        base_run_name = base_root.name
        if base_run_name == interactive_run:
            continue
        run_payload: dict[str, Any] = {}
        for backend in ("ptx", "cuda"):
            backend_payload: dict[str, Any] = {}
            for level in (1, 2, 3):
                interactive_rows = load_result_rows(interactive_run, backend, level)
                base_rows = load_result_rows(base_run_name, backend, level)
                if not interactive_rows or not base_rows:
                    continue
                overlap = build_overlap(interactive_rows, base_rows)
                if overlap["shared_problem_ids"]:
                    backend_payload[f"level{level}"] = overlap
            if backend_payload:
                run_payload[backend] = backend_payload
        if run_payload:
            payload["comparisons"][base_run_name] = run_payload
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = [
        f"# Interactive Analysis: {payload['interactive_run']}",
        "",
        "## Headline",
        "",
    ]

    for backend in ("ptx", "cuda"):
        for level_key, level_payload in sorted(payload.get("interactive", {}).get(backend, {}).items()):
            summary = level_payload["summary"]
            lines.append(
                f"- {backend.upper()} {level_key}: "
                f"{summary['correct_tasks']}/{summary['total_tasks']} correct, "
                f"fast@1.0={summary['fast_p_vs_torch']['1.0']:.3f}, "
                f"geomean(correct)={summary['geomean_speedup_vs_torch_correct_only']:.3f}"
            )

    lines.extend(
        [
            "",
            "## Interactive Coverage",
            "",
            "| Backend | Level | Tasks | Correct | Correctness | fast@1.0 | fast@2.0 | Geomean (correct) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for backend in ("ptx", "cuda"):
        for level_key, level_payload in sorted(payload.get("interactive", {}).get(backend, {}).items()):
            summary = level_payload["summary"]
            lines.append(
                f"| {backend.upper()} | {level_key} | {summary['total_tasks']} | {summary['correct_tasks']} | "
                f"{summary['correctness_rate']:.3f} | {summary['fast_p_vs_torch']['1.0']:.3f} | "
                f"{summary['fast_p_vs_torch']['2.0']:.3f} | {summary['geomean_speedup_vs_torch_correct_only']:.3f} |"
            )

    lines.extend(
        [
            "",
            "## Strongest Interactive Wins",
            "",
            "| Backend | Level | Problem | Speedup vs torch | Runtime ms | Ref ms |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for backend in ("ptx", "cuda"):
        for level_key, level_payload in sorted(payload.get("interactive", {}).get(backend, {}).items()):
            for row in level_payload["top_wins"][:3]:
                lines.append(
                    f"| {backend.upper()} | {level_key} | {row['problem_id']}: {row['problem_name']} | "
                    f"{row['speedup_vs_torch']:.3f} | {row['runtime_ms']:.3f} | {row['ref_runtime_ms']:.3f} |"
                )

    lines.extend(
        [
            "",
            "## Interactive vs Existing Benchmark Runs",
            "",
            "| Base run | Backend | Level | Shared tasks | Interactive correct | Base correct | Interactive-only correct | Base-only correct | Interactive win rate | Geomean interactive/base |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for base_run_name, base_payload in sorted(payload.get("comparisons", {}).items()):
        for backend in ("ptx", "cuda"):
            for level_key, overlap in sorted(base_payload.get(backend, {}).items()):
                interactive_summary = overlap["interactive_summary"]
                base_summary = overlap["base_summary"]
                lines.append(
                    f"| {base_run_name} | {backend.upper()} | {level_key} | {len(overlap['shared_problem_ids'])} | "
                    f"{interactive_summary['correct_tasks']} | {base_summary['correct_tasks']} | "
                    f"{len(overlap['interactive_only_correct_problem_ids'])} | {len(overlap['base_only_correct_problem_ids'])} | "
                    f"{overlap['interactive_head_to_head_win_rate']:.3f} | {overlap['interactive_vs_base_geomean_speedup']:.3f} |"
                )

    hybrid_rows = payload.get("hybrid_level3", [])
    if hybrid_rows:
        lines.extend(
            [
                "",
                "## Hybrid PTX Level 3",
                "",
                "| Problem | Correct | Runtime ms | Ref ms | Speedup vs torch |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for row in hybrid_rows:
            lines.append(
                f"| {row['problem_id']}: {row['problem_name']} | {bool(row['correctness'])} | "
                f"{float(row['runtime_ms']):.3f} | {float(row['ref_runtime_ms']):.3f} | "
                f"{float(row['speedup_vs_torch']):.3f} |"
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The interactive report is rebuilt from per-problem JSONs, not the rolling `summary.json` snapshots, because the lightweight interactive work overwrote those summaries block-by-block.",
            "- Direct interactive-vs-base comparison currently exists only where benchmark runs were actually executed; in this repo that mostly means the Level 1 pilot overlap.",
            "- The hybrid PTX numbers are exploratory and come from `scripts/eval_hybrid_module.py`, not the strict PTX benchmark path.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze codex-interactive results against available base benchmark runs.")
    parser.add_argument("--interactive-run", default="codex-interactive")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    payload = build_payload(args.interactive_run)
    rendered_json = json.dumps(payload, indent=2)
    rendered_md = render_markdown(payload)

    print(rendered_md)

    output_json = Path(args.output_json) if args.output_json else REPO_ROOT / "results" / "analysis" / f"{args.interactive_run}_report.json"
    output_md = Path(args.output_md) if args.output_md else REPO_ROOT / "results" / "analysis" / f"{args.interactive_run}_report.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(rendered_json, encoding="utf-8")
    output_md.write_text(rendered_md, encoding="utf-8")


if __name__ == "__main__":
    main()
