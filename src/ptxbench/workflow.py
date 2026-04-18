from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json

from .config import LEVEL_PILOT_PROBLEM_IDS
from .dataset import construct_dataset
from .generation import default_run_dir, generation_failure_path
from .run_metadata import normalize_problem_ids


def parse_problem_ids(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def resolve_problem_ids(phase: str, raw_problem_ids: str | None, level: int) -> list[int]:
    requested = normalize_problem_ids(parse_problem_ids(raw_problem_ids))
    if requested is not None:
        return requested
    if phase == "pilot":
        return list(LEVEL_PILOT_PROBLEM_IDS[level])
    if phase == "full":
        return construct_dataset(level=level).get_problem_ids()
    return [19]


def chunk_problem_ids(problem_ids: list[int], chunk_size: int) -> list[list[int]]:
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be greater than 0")
    return [problem_ids[index : index + chunk_size] for index in range(0, len(problem_ids), chunk_size)]


def chunk_metadata_dir(run_dir: Path, chunk_label: str | None) -> Path:
    if not chunk_label:
        return run_dir
    return run_dir / "_chunks" / chunk_label


@dataclass(frozen=True)
class GenerationChunkTask:
    backend: str
    level: int
    chunk_index: int
    chunk_total: int
    problem_ids: list[int]

    @property
    def chunk_label(self) -> str:
        return f"chunk_{self.chunk_index:03d}"


def inspect_chunk_generation(*, run_name: str, backend: str, level: int, problem_ids: list[int]) -> dict[str, int]:
    run_dir = default_run_dir(run_name, backend, level)
    dataset = construct_dataset(level=level, problem_ids=problem_ids)
    generated = 0
    failed = 0
    missing = 0
    for problem in dataset:
        submission_path = run_dir / f"{problem.problem_id:03d}_{problem.path.stem}.py"
        if submission_path.exists():
            generated += 1
        elif generation_failure_path(submission_path).exists():
            failed += 1
        else:
            missing += 1
    return {
        "generated": generated,
        "failed": failed,
        "missing": missing,
        "total": len(problem_ids),
    }


def update_chunk_status(
    *,
    run_name: str,
    backend: str,
    level: int,
    chunk_index: int,
    chunk_total: int,
    problem_ids: list[int],
    status: str,
    counts: dict[str, int] | None = None,
    error: str | None = None,
    track: str = "oneshot",
) -> None:
    path = _chunk_status_path(run_name)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {"run_name": run_name, "chunks": []}

    entry = {
        "backend": backend,
        "track": track,
        "level": level,
        "chunk_index": chunk_index,
        "chunk_total": chunk_total,
        "problem_ids": problem_ids,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if counts is not None:
        entry["counts"] = counts
    if error is not None:
        entry["error"] = error

    chunks = payload.get("chunks", [])
    replacement_index = None
    for index, existing in enumerate(chunks):
        if (
            existing.get("backend") == backend
            and existing.get("track", "oneshot") == track
            and existing.get("chunk_index") == chunk_index
        ):
            replacement_index = index
            break
    if replacement_index is None:
        chunks.append(entry)
    else:
        chunks[replacement_index] = entry

    chunks.sort(key=lambda item: (item.get("track", "oneshot"), item["backend"], item["chunk_index"]))
    payload["chunks"] = chunks
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_backend_generation_summary(
    *,
    run_name: str,
    backend: str,
    level: int,
    problem_ids: list[int],
    chunk_total: int,
    track: str = "oneshot",
) -> None:
    counts = inspect_chunk_generation(run_name=run_name, backend=backend, level=level, problem_ids=problem_ids)
    run_dir = default_run_dir(run_name, backend, level)
    failed_problem_ids: list[int] = []
    for problem in construct_dataset(level=level, problem_ids=problem_ids):
        submission_path = run_dir / f"{problem.problem_id:03d}_{problem.path.stem}.py"
        if generation_failure_path(submission_path).exists():
            failed_problem_ids.append(problem.problem_id)
    payload = {
        "track": track,
        "total": counts["total"],
        "generated": counts["generated"],
        "skipped": 0,
        "failed": counts["failed"],
        "missing": counts["missing"],
        "failed_problem_ids": failed_problem_ids,
        "chunk_total": chunk_total,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "generation_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _chunk_status_path(run_name: str) -> Path:
    return Path(__file__).resolve().parents[2] / "runs" / run_name / "chunk_status.json"
