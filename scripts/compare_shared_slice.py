from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from ptxbench.config import REPO_ROOT

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_interactive_vs_base import build_overlap, collect_top_wins, load_result_rows


def _filter_rows(rows: list[dict[str, Any]], allowed_ids: set[int]) -> list[dict[str, Any]]:
    return [row for row in rows if int(row["problem_id"]) in allowed_ids]


def _render_backend_section(backend: str, overlap: dict[str, Any]) -> list[str]:
    interactive_summary = overlap["interactive_summary"]
    base_summary = overlap["base_summary"]
    shared = overlap["shared_problem_ids"]
    lines = [
        f"### {backend.upper()}",
        "",
        f"- Shared tasks: `{len(shared)}`",
        f"- Interactive correct: `{interactive_summary['correct_tasks']}/{interactive_summary['total_tasks']}`",
        f"- Base correct: `{base_summary['correct_tasks']}/{base_summary['total_tasks']}`",
        f"- Interactive-only correct: `{len(overlap['interactive_only_correct_problem_ids'])}`",
        f"- Base-only correct: `{len(overlap['base_only_correct_problem_ids'])}`",
        f"- Interactive win rate on jointly-correct tasks: `{overlap['interactive_head_to_head_win_rate']:.3f}`",
        f"- Geomean interactive/base runtime ratio: `{overlap['interactive_vs_base_geomean_speedup']:.3f}`",
        "",
        "| Metric | Interactive | Base |",
        "| --- | --- | --- |",
        f"| Correctness rate | {interactive_summary['correctness_rate']:.3f} | {base_summary['correctness_rate']:.3f} |",
        f"| fast@1.0 | {interactive_summary['fast_p_vs_torch']['1.0']:.3f} | {base_summary['fast_p_vs_torch']['1.0']:.3f} |",
        f"| fast@2.0 | {interactive_summary['fast_p_vs_torch']['2.0']:.3f} | {base_summary['fast_p_vs_torch']['2.0']:.3f} |",
        f"| Geomean speedup vs torch (correct only) | {interactive_summary['geomean_speedup_vs_torch_correct_only']:.3f} | {base_summary['geomean_speedup_vs_torch_correct_only']:.3f} |",
        "",
    ]
    return lines


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Shared Slice Comparison: {payload['interactive_run']} vs {payload['base_run']}",
        "",
        f"- Level: `{payload['level']}`",
        f"- Requested problem IDs: `{','.join(str(pid) for pid in payload['requested_problem_ids'])}`",
        f"- Effective shared IDs: `{','.join(str(pid) for pid in payload['effective_shared_problem_ids'])}`",
        "",
    ]

    for backend in ("ptx", "cuda"):
        if backend in payload["comparisons"]:
            lines.extend(_render_backend_section(backend, payload["comparisons"][backend]))

    if payload.get("interactive_top_wins"):
        lines.extend(
            [
                "## Interactive Top Wins On This Slice",
                "",
                "| Backend | Problem | Speedup vs torch | Runtime ms | Ref ms |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for backend, rows in payload["interactive_top_wins"].items():
            for row in rows:
                lines.append(
                    f"| {backend.upper()} | {row['problem_id']}: {row['problem_name']} | "
                    f"{row['speedup_vs_torch']:.3f} | {row['runtime_ms']:.3f} | {row['ref_runtime_ms']:.3f} |"
                )
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- This report only compares the explicitly requested slice, not the full run coverage.",
            "- It requires evaluated result JSONs in `results/timing/<run>/<backend>/level<level>/` for both runs.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two evaluated runs on a shared problem slice.")
    parser.add_argument("--interactive-run", required=True)
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--problem-ids", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    args = parser.parse_args()

    requested_ids = {int(part.strip()) for part in args.problem_ids.split(",") if part.strip()}
    payload: dict[str, Any] = {
        "interactive_run": args.interactive_run,
        "base_run": args.base_run,
        "level": args.level,
        "requested_problem_ids": sorted(requested_ids),
        "comparisons": {},
        "interactive_top_wins": {},
    }

    effective_shared: set[int] = set()
    for backend in ("ptx", "cuda"):
        interactive_rows = _filter_rows(load_result_rows(args.interactive_run, backend, args.level), requested_ids)
        base_rows = _filter_rows(load_result_rows(args.base_run, backend, args.level), requested_ids)
        if not interactive_rows or not base_rows:
            continue
        overlap = build_overlap(interactive_rows, base_rows)
        if not overlap["shared_problem_ids"]:
            continue
        payload["comparisons"][backend] = overlap
        payload["interactive_top_wins"][backend] = collect_top_wins(
            _filter_rows(interactive_rows, set(overlap["shared_problem_ids"])),
            limit=5,
        )
        effective_shared.update(overlap["shared_problem_ids"])

    payload["effective_shared_problem_ids"] = sorted(effective_shared)

    rendered_json = json.dumps(payload, indent=2)
    rendered_md = render_markdown(payload)
    print(rendered_md)

    stem = f"{args.interactive_run}_vs_{args.base_run}_level{args.level}_sharedslice"
    output_json = Path(args.output_json) if args.output_json else REPO_ROOT / "results" / "analysis" / f"{stem}.json"
    output_md = Path(args.output_md) if args.output_md else REPO_ROOT / "results" / "analysis" / f"{stem}.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(rendered_json, encoding="utf-8")
    output_md.write_text(rendered_md, encoding="utf-8")


if __name__ == "__main__":
    main()
