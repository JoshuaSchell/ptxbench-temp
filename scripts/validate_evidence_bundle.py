from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import json

from ptxbench.config import DEFAULT_LEVELS, REPO_ROOT
from ptxbench.run_metadata import protocol_signature


DEFAULT_BACKENDS = ("ptx", "cuda")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _load_json(path: Path, *, issues: list[str], label: str) -> Any | None:
    if not path.exists():
        issues.append(f"missing {label}: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        issues.append(f"invalid JSON in {label}: {path} ({exc})")
        return None


def _expect(condition: bool, issues: list[str], message: str) -> None:
    if not condition:
        issues.append(message)


def _require_field(payload: dict[str, Any], field_name: str, *, issues: list[str], path: Path) -> Any:
    if field_name not in payload:
        issues.append(f"missing field {field_name} in {path}")
        return None
    return payload[field_name]


def _validate_string(
    payload: dict[str, Any],
    field_name: str,
    *,
    issues: list[str],
    path: Path,
    allow_none: bool = False,
) -> None:
    value = _require_field(payload, field_name, issues=issues, path=path)
    if value is None and allow_none:
        return
    if not isinstance(value, str) or not value.strip():
        issues.append(f"malformed field {field_name} in {path}: expected non-empty string")


def _validate_int(payload: dict[str, Any], field_name: str, *, issues: list[str], path: Path) -> None:
    value = _require_field(payload, field_name, issues=issues, path=path)
    if not isinstance(value, int) or isinstance(value, bool):
        issues.append(f"malformed field {field_name} in {path}: expected integer")


def _validate_number(payload: dict[str, Any], field_name: str, *, issues: list[str], path: Path) -> None:
    value = _require_field(payload, field_name, issues=issues, path=path)
    if not _is_number(value):
        issues.append(f"malformed field {field_name} in {path}: expected number")


def _validate_bool(payload: dict[str, Any], field_name: str, *, issues: list[str], path: Path) -> None:
    value = _require_field(payload, field_name, issues=issues, path=path)
    if not isinstance(value, bool):
        issues.append(f"malformed field {field_name} in {path}: expected bool")


def _manifest_problem_ids(manifest: dict[str, Any] | None) -> list[int]:
    if not isinstance(manifest, dict):
        return []
    raw = manifest.get("problem_ids", manifest.get("problems", []))
    if not isinstance(raw, list):
        return []
    return [int(problem_id) for problem_id in raw]


def _validate_result_payload(
    payload: dict[str, Any],
    *,
    backend: str,
    track: str,
    expected_problem_id: int,
    path: Path,
    issues: list[str],
    allow_missing_compile_baseline: bool,
    allow_missing_ptx_resources: bool,
) -> None:
    _validate_int(payload, "problem_id", issues=issues, path=path)
    if payload.get("problem_id") != expected_problem_id:
        issues.append(f"problem_id mismatch in {path}: expected {expected_problem_id}, got {payload.get('problem_id')}")
    _validate_string(payload, "backend", issues=issues, path=path)
    if payload.get("backend") != backend:
        issues.append(f"backend mismatch in {path}: expected {backend}, got {payload.get('backend')}")
    _validate_string(payload, "track", issues=issues, path=path)
    if payload.get("track") != track:
        issues.append(f"track mismatch in {path}: expected {track}, got {payload.get('track')}")

    submission_hash = _require_field(payload, "submission_hash", issues=issues, path=path)
    missing_submission = bool(payload.get("metadata", {}).get("missing_submission"))
    if submission_hash is None:
        if not missing_submission:
            issues.append(f"malformed field submission_hash in {path}: unexpected null")
    elif not isinstance(submission_hash, str) or not submission_hash.strip():
        issues.append(f"malformed field submission_hash in {path}: expected hash string")

    _validate_bool(payload, "compiled", issues=issues, path=path)
    assembled = _require_field(payload, "assembled", issues=issues, path=path)
    if assembled is not None and not isinstance(assembled, bool):
        issues.append(f"malformed field assembled in {path}: expected bool or null")
    loaded = _require_field(payload, "loaded", issues=issues, path=path)
    if loaded is not None and not isinstance(loaded, bool):
        issues.append(f"malformed field loaded in {path}: expected bool or null")
    _validate_bool(payload, "correctness", issues=issues, path=path)
    _validate_string(payload, "failure_category", issues=issues, path=path)
    _validate_string(payload, "paper_failure_category", issues=issues, path=path)
    _validate_number(payload, "runtime_ms", issues=issues, path=path)
    _validate_number(payload, "ref_runtime_ms", issues=issues, path=path)
    _validate_number(payload, "speedup_vs_torch", issues=issues, path=path)
    _validate_int(payload, "num_correct_trials", issues=issues, path=path)
    _validate_int(payload, "num_perf_trials", issues=issues, path=path)
    _validate_int(payload, "seed", issues=issues, path=path)
    _validate_string(payload, "arch", issues=issues, path=path)
    _validate_string(payload, "precision", issues=issues, path=path)
    _validate_string(payload, "gpu_name", issues=issues, path=path)
    _validate_string(payload, "torch_version", issues=issues, path=path)
    _validate_string(payload, "cuda_version", issues=issues, path=path)
    _validate_string(payload, "repo_commit", issues=issues, path=path)
    _validate_string(payload, "kernelbench_commit", issues=issues, path=path)

    if not allow_missing_compile_baseline and payload.get("ref_runtime_compile_default_ms") is None:
        issues.append(
            f"missing torch.compile baseline in {path}: pass --allow-missing-compile-baseline only for non-paper or legacy runs"
        )
    if backend == "ptx":
        resource_summary = payload.get("metadata", {}).get("ptx_resource_summary")
        if not allow_missing_ptx_resources:
            _validate_string(payload, "ptxas_version", issues=issues, path=path)
            if not isinstance(resource_summary, dict):
                issues.append(
                    f"missing PTX resource summary in {path}: rerun evaluation with current PTXBench or pass --allow-missing-ptx-resources"
                )

    if "speedup_vs_eager" in payload and _is_number(payload.get("speedup_vs_eager")) and _is_number(payload.get("speedup_vs_torch")):
        if float(payload["speedup_vs_eager"]) != float(payload["speedup_vs_torch"]):
            issues.append(f"alias mismatch in {path}: speedup_vs_torch must equal speedup_vs_eager")
    if "ref_runtime_eager_ms" in payload and _is_number(payload.get("ref_runtime_eager_ms")) and _is_number(payload.get("ref_runtime_ms")):
        if float(payload["ref_runtime_eager_ms"]) != float(payload["ref_runtime_ms"]):
            issues.append(f"alias mismatch in {path}: ref_runtime_ms must equal ref_runtime_eager_ms")


def validate_evidence_bundle(
    *,
    repo_root: Path,
    run_name: str,
    level: int,
    track: str,
    backends: list[str],
    allow_missing_compile_baseline: bool = False,
    allow_missing_ptx_resources: bool = False,
    require_paper_report: bool = False,
) -> tuple[bool, list[str], dict[str, int]]:
    issues: list[str] = []
    stats = {"backends": len(backends), "problems": 0}

    runs_root = repo_root / "runs" / run_name
    results_root = repo_root / "results" / "timing" / run_name
    analysis_json_path = repo_root / "results" / "analysis" / f"{run_name}_level{level}.json"
    analysis_md_path = analysis_json_path.with_suffix(".md")
    paper_report_dir = repo_root / "results" / "paper" / run_name

    paper_manifest = _load_json(runs_root / "paper_run_manifest.json", issues=issues, label="paper run manifest")
    if isinstance(paper_manifest, dict):
        _expect(paper_manifest.get("run_name") == run_name, issues, f"paper manifest run_name mismatch for {run_name}")
        _expect(paper_manifest.get("track") == track, issues, f"paper manifest track mismatch for {run_name}")
        required_outputs = paper_manifest.get("required_outputs", [])
        if isinstance(required_outputs, list):
            for raw_path in required_outputs:
                required_path = repo_root / str(raw_path)
                if not required_path.exists():
                    issues.append(f"missing required output declared by paper manifest: {required_path}")
        elif required_outputs:
            issues.append(f"paper manifest required_outputs must be a list: {runs_root / 'paper_run_manifest.json'}")

    analysis_payload = _load_json(analysis_json_path, issues=issues, label="analysis JSON")
    if not analysis_md_path.exists():
        issues.append(f"missing analysis Markdown: {analysis_md_path}")
    if isinstance(analysis_payload, dict):
        _expect(analysis_payload.get("run_name") == run_name, issues, f"analysis JSON run_name mismatch in {analysis_json_path}")
        _expect(int(analysis_payload.get("level", -1)) == level, issues, f"analysis JSON level mismatch in {analysis_json_path}")
        _expect(analysis_payload.get("track") == track, issues, f"analysis JSON track mismatch in {analysis_json_path}")
        backend_summaries = analysis_payload.get("backend_summaries", {})
        if isinstance(backend_summaries, dict):
            for backend in backends:
                _expect(backend in backend_summaries, issues, f"analysis JSON missing backend summary for {backend}")
        else:
            issues.append(f"analysis JSON missing backend_summaries object: {analysis_json_path}")

    if require_paper_report:
        for name in (
            "main_results.csv",
            "paired_results.csv",
            "failure_breakdown.csv",
            "paper_tables.md",
            "report_manifest.json",
        ):
            path = paper_report_dir / name
            if not path.exists():
                issues.append(f"missing paper report artifact: {path}")

    summaries_by_backend: dict[str, list[dict[str, Any]]] = {}
    protocols_by_backend: dict[str, dict[str, Any]] = {}
    for backend in backends:
        run_dir = runs_root / backend / f"level{level}"
        results_dir = results_root / backend / f"level{level}"
        run_manifest = _load_json(run_dir / "run_manifest.json", issues=issues, label=f"{backend} run manifest")
        eval_manifest = _load_json(results_dir / "eval_manifest.json", issues=issues, label=f"{backend} eval manifest")
        summary_payload = _load_json(results_dir / "summary.json", issues=issues, label=f"{backend} timing summary")

        expected_problem_ids = _manifest_problem_ids(eval_manifest) or _manifest_problem_ids(run_manifest)
        if isinstance(run_manifest, dict):
            _expect(run_manifest.get("track") == track, issues, f"{backend} run manifest track mismatch")
            _expect(int(run_manifest.get("level", -1)) == level, issues, f"{backend} run manifest level mismatch")
            _expect(run_manifest.get("backend") == backend, issues, f"{backend} run manifest backend mismatch")
        if isinstance(eval_manifest, dict):
            _expect(eval_manifest.get("track") == track, issues, f"{backend} eval manifest track mismatch")
            _expect(int(eval_manifest.get("level", -1)) == level, issues, f"{backend} eval manifest level mismatch")
            _expect(eval_manifest.get("backend") == backend, issues, f"{backend} eval manifest backend mismatch")
            protocol = eval_manifest.get("protocol")
            if isinstance(protocol, dict):
                protocols_by_backend[backend] = protocol

        summary_rows: list[dict[str, Any]] = []
        if isinstance(summary_payload, list):
            if not all(isinstance(row, dict) for row in summary_payload):
                issues.append(f"{backend} timing summary must be a list of objects: {results_dir / 'summary.json'}")
            else:
                summary_rows = [dict(row) for row in summary_payload]
        elif summary_payload is not None:
            issues.append(f"{backend} timing summary must be a JSON list: {results_dir / 'summary.json'}")

        summary_problem_ids = sorted(int(row.get("problem_id", -1)) for row in summary_rows)
        if expected_problem_ids:
            _expect(
                summary_problem_ids == sorted(expected_problem_ids),
                issues,
                f"{backend} summary problem IDs do not match manifest problem IDs",
            )
        summaries_by_backend[backend] = summary_rows

        for problem_id in expected_problem_ids:
            stats["problems"] += 1
            result_path = results_dir / f"{problem_id:03d}.json"
            result_payload = _load_json(result_path, issues=issues, label=f"{backend} result {problem_id:03d}")
            if isinstance(result_payload, dict):
                _validate_result_payload(
                    result_payload,
                    backend=backend,
                    track=track,
                    expected_problem_id=problem_id,
                    path=result_path,
                    issues=issues,
                    allow_missing_compile_baseline=allow_missing_compile_baseline,
                    allow_missing_ptx_resources=allow_missing_ptx_resources,
                )

    if {"ptx", "cuda"}.issubset(set(backends)):
        ptx_summary_path = results_root / "ptx" / f"level{level}" / "summary.json"
        cuda_summary_path = results_root / "cuda" / f"level{level}" / "summary.json"
        if "ptx" not in summaries_by_backend:
            issues.append(f"missing PTX summary for paired claim: {ptx_summary_path}")
        if "cuda" not in summaries_by_backend:
            issues.append(f"missing CUDA summary for paired claim: {cuda_summary_path}")
        ptx_ids = sorted(int(row.get("problem_id", -1)) for row in summaries_by_backend.get("ptx", []))
        cuda_ids = sorted(int(row.get("problem_id", -1)) for row in summaries_by_backend.get("cuda", []))
        if ptx_ids != cuda_ids:
            issues.append(f"PTX and CUDA evaluated different problem IDs: ptx={ptx_ids} cuda={cuda_ids}")
        if "ptx" in protocols_by_backend and "cuda" in protocols_by_backend:
            if protocol_signature(protocols_by_backend["ptx"]) != protocol_signature(protocols_by_backend["cuda"]):
                issues.append("PTX and CUDA protocol signatures do not match")

    return not issues, issues, stats


def _parse_backends(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_BACKENDS)
    values = [backend.strip() for backend in raw.split(",") if backend.strip()]
    if not values:
        return list(DEFAULT_BACKENDS)
    invalid = sorted(set(values) - set(DEFAULT_BACKENDS))
    if invalid:
        raise ValueError(f"Unsupported backend(s): {', '.join(invalid)}")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the PTXBench paper-run evidence bundle for a run.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--level", required=True, type=int, choices=DEFAULT_LEVELS)
    parser.add_argument("--track", required=True, choices=["oneshot", "agentic"])
    parser.add_argument("--backends", help="Comma-separated backend list to validate. Defaults to ptx,cuda.")
    parser.add_argument("--allow-missing-compile-baseline", action="store_true")
    parser.add_argument("--allow-missing-ptx-resources", action="store_true")
    parser.add_argument("--require-paper-report", action="store_true")
    args = parser.parse_args(argv)

    valid, issues, stats = validate_evidence_bundle(
        repo_root=REPO_ROOT,
        run_name=args.run_name,
        level=args.level,
        track=args.track,
        backends=_parse_backends(args.backends),
        allow_missing_compile_baseline=args.allow_missing_compile_baseline,
        allow_missing_ptx_resources=args.allow_missing_ptx_resources,
        require_paper_report=args.require_paper_report,
    )
    if valid:
        print(
            f"PASS evidence bundle run={args.run_name} level={args.level} "
            f"track={args.track} backends={stats['backends']} problems={stats['problems']}"
        )
        return 0

    print(
        f"FAIL evidence bundle run={args.run_name} level={args.level} "
        f"track={args.track} issues={len(issues)}"
    )
    for issue in issues[:20]:
        print(f"- {issue}")
    if len(issues) > 20:
        print(f"- ... {len(issues) - 20} more")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
