from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import hashlib
import os
import platform
import re
import socket
import subprocess

from .config import (
    DEFAULT_ARCH,
    DEFAULT_AGENTIC_MAX_STEPS,
    DEFAULT_AGENTIC_MAX_TOOL_CALLS,
    DEFAULT_AGENTIC_MAX_WALL_CLOCK_MINUTES,
    DEFAULT_DEV_EVAL_CORRECT_TRIALS,
    DEFAULT_DEV_EVAL_PROFILE_ENABLED,
    DEFAULT_DEV_EVAL_PROFILE_METRICS,
    DEFAULT_DEV_EVAL_PROFILE_TOOL,
    DEFAULT_DEV_EVAL_PROFILE_TRIALS,
    DEFAULT_DEV_EVAL_PERF_TRIALS,
    DEFAULT_DEV_EVAL_SEED,
    DEFAULT_DEV_EVAL_TIMEOUT_SECONDS,
    DEFAULT_GENERATION_TIMEOUT_SECONDS,
    DEFAULT_LEVELS,
    DEFAULT_NUM_CORRECT_TRIALS,
    DEFAULT_NUM_PERF_TRIALS,
    DEFAULT_OFFICIAL_EVAL_SEED,
    DEFAULT_PRECISION,
    DEFAULT_TRACK,
    DEFAULT_TRACKS,
    LEVEL_PILOT_PROBLEM_IDS,
    REPO_ROOT,
    ensure_vendor_snapshot,
)


PAPER_PROTOCOL_VERSION = "paper-v1"
PROTOCOL_SIGNATURE_KEYS = (
    "protocol_version",
    "level",
    "track",
    "precision",
    "arch",
    "one_shot",
    "num_correct_trials",
    "num_perf_trials",
    "torch_compile_baseline",
    "generation_timeout_seconds",
    "official_eval_seed",
    "max_steps",
    "max_wall_clock_minutes",
    "max_tool_calls",
    "dev_eval_seed",
    "dev_eval_correct_trials",
    "dev_eval_perf_trials",
    "dev_eval_timeout_seconds",
    "dev_eval_profile_enabled",
    "dev_eval_profile_tool",
    "dev_eval_profile_trials",
    "dev_eval_profile_metrics",
)


@dataclass(frozen=True)
class PaperRunProtocol:
    protocol_version: str = PAPER_PROTOCOL_VERSION
    level: int = 1
    track: str = DEFAULT_TRACK
    precision: str = DEFAULT_PRECISION
    arch: str = DEFAULT_ARCH
    one_shot: bool = True
    num_correct_trials: int = DEFAULT_NUM_CORRECT_TRIALS
    num_perf_trials: int = DEFAULT_NUM_PERF_TRIALS
    torch_compile_baseline: bool = False
    generation_timeout_seconds: int = DEFAULT_GENERATION_TIMEOUT_SECONDS
    official_eval_seed: int = DEFAULT_OFFICIAL_EVAL_SEED
    max_steps: int = 1
    max_wall_clock_minutes: int | None = None
    max_tool_calls: int = 0
    dev_eval_seed: int | None = None
    dev_eval_correct_trials: int | None = None
    dev_eval_perf_trials: int | None = None
    dev_eval_timeout_seconds: int | None = None
    dev_eval_profile_enabled: bool = DEFAULT_DEV_EVAL_PROFILE_ENABLED
    dev_eval_profile_tool: str | None = None
    dev_eval_profile_trials: int | None = None
    dev_eval_profile_metrics: tuple[str, ...] = ()
    pilot_problem_ids: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_paper_protocol(level: int = 1, *, track: str = DEFAULT_TRACK) -> PaperRunProtocol:
    if level not in DEFAULT_LEVELS:
        raise ValueError(f"Unsupported level for paper protocol: {level}")
    if track not in DEFAULT_TRACKS:
        raise ValueError(f"Unsupported protocol track: {track}")
    is_agentic = track == "agentic"
    return PaperRunProtocol(
        level=level,
        track=track,
        one_shot=not is_agentic,
        max_steps=DEFAULT_AGENTIC_MAX_STEPS if is_agentic else 1,
        max_wall_clock_minutes=DEFAULT_AGENTIC_MAX_WALL_CLOCK_MINUTES if is_agentic else None,
        max_tool_calls=DEFAULT_AGENTIC_MAX_TOOL_CALLS if is_agentic else 0,
        dev_eval_seed=DEFAULT_DEV_EVAL_SEED if is_agentic else None,
        dev_eval_correct_trials=DEFAULT_DEV_EVAL_CORRECT_TRIALS if is_agentic else None,
        dev_eval_perf_trials=DEFAULT_DEV_EVAL_PERF_TRIALS if is_agentic else None,
        dev_eval_timeout_seconds=DEFAULT_DEV_EVAL_TIMEOUT_SECONDS if is_agentic else None,
        dev_eval_profile_enabled=DEFAULT_DEV_EVAL_PROFILE_ENABLED if is_agentic else False,
        dev_eval_profile_tool=DEFAULT_DEV_EVAL_PROFILE_TOOL if is_agentic else None,
        dev_eval_profile_trials=DEFAULT_DEV_EVAL_PROFILE_TRIALS if is_agentic else None,
        dev_eval_profile_metrics=DEFAULT_DEV_EVAL_PROFILE_METRICS if is_agentic else (),
        pilot_problem_ids=LEVEL_PILOT_PROBLEM_IDS.get(level, ()),
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_problem_ids(problem_ids: list[int] | tuple[int, ...] | None) -> list[int] | None:
    if problem_ids is None:
        return None
    return sorted(dict.fromkeys(int(problem_id) for problem_id in problem_ids))


def protocol_signature(protocol: dict[str, Any] | None) -> dict[str, Any]:
    payload = protocol or {}
    signature: dict[str, Any] = {}
    for key in PROTOCOL_SIGNATURE_KEYS:
        if key in payload:
            value = payload[key]
            if isinstance(value, tuple):
                value = list(value)
            signature[key] = value
    return signature


def protocol_differences(
    expected: dict[str, Any] | None,
    observed: dict[str, Any] | None,
) -> dict[str, tuple[Any, Any]]:
    expected_signature = protocol_signature(expected)
    observed_signature = protocol_signature(observed)
    differences: dict[str, tuple[Any, Any]] = {}
    for key in sorted(set(expected_signature) | set(observed_signature)):
        expected_value = expected_signature.get(key)
        observed_value = observed_signature.get(key)
        if expected_value != observed_value:
            differences[key] = (expected_value, observed_value)
    return differences


def _run_command(command: list[str]) -> str | None:
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if process.returncode != 0:
        return None
    output = process.stdout.strip() or process.stderr.strip()
    return output or None


def _git_commit(repo_root: Path) -> str | None:
    process = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo_root}",
            "-C",
            str(repo_root),
            "rev-parse",
            "HEAD",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        return None
    return process.stdout.strip() or None


def _parse_cuda_release(raw_output: str | None) -> str | None:
    if not raw_output:
        return None
    match = re.search(r"release\s+([0-9]+\.[0-9]+)", raw_output)
    if match:
        return match.group(1)
    return raw_output.splitlines()[0].strip()


def _torch_environment() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on runtime env
        return {
            "torch_available": False,
            "torch_error": str(exc),
        }

    return {
        "torch_available": True,
        "torch_version": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "torch_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "torch_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _gpu_environment() -> dict[str, Any]:
    gpu_query = _run_command(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    gpu_name = None
    driver_version = None
    if gpu_query:
        first_line = gpu_query.splitlines()[0]
        parts = [part.strip() for part in first_line.split(",")]
        if parts:
            gpu_name = parts[0]
        if len(parts) > 1:
            driver_version = parts[1]

    nvcc_output = _run_command(["nvcc", "--version"])
    ptxas_output = _run_command(["ptxas", "--version"])
    return {
        "gpu_name": gpu_name,
        "nvidia_driver_version": driver_version,
        "cuda_toolkit_release": _parse_cuda_release(nvcc_output),
        "ptxas_release": _parse_cuda_release(ptxas_output),
        "nvcc_path": _run_command(["which", "nvcc"]) if os.name != "nt" else _run_command(["where", "nvcc"]),
        "ptxas_path": _run_command(["which", "ptxas"]) if os.name != "nt" else _run_command(["where", "ptxas"]),
    }


def detect_runtime_environment() -> dict[str, Any]:
    snapshot = ensure_vendor_snapshot()
    repo_commit = _git_commit(REPO_ROOT)
    environment = {
        "repo_root": str(REPO_ROOT),
        "repo_commit": repo_commit,
        "hostname": socket.gethostname(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "platform_machine": platform.machine(),
        "python_version": platform.python_version(),
        "is_wsl": "microsoft" in platform.release().lower() or "WSL_DISTRO_NAME" in os.environ,
        "wsl_distro": os.environ.get("WSL_DISTRO_NAME"),
        "cwd": str(Path.cwd()),
        "vendor_snapshot_commit": snapshot.commit,
        "kernelbench_commit": snapshot.commit,
    }
    environment.update(_gpu_environment())
    environment.update(_torch_environment())
    environment["cuda_version"] = environment.get("torch_cuda_version") or environment.get("cuda_toolkit_release")
    environment["ptxas_version"] = environment.get("ptxas_release")
    return environment
