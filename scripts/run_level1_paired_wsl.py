from __future__ import annotations

import argparse
import io
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import tarfile

from ptxbench.config import DEFAULT_LEVELS
try:
    from .sync_to_wsl import DEFAULT_WSL_TARGET, sync_repo_to_wsl
except ImportError:
    from sync_to_wsl import DEFAULT_WSL_TARGET, sync_repo_to_wsl


WINDOWS_REPO_ROOT = Path(__file__).resolve().parents[1]


def windows_path_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    if not drive:
        raise ValueError(f"Expected a Windows drive path, got: {resolved}")
    tail_parts = [part for part in resolved.parts[1:] if part not in {"/", "\\"}]
    return str(PurePosixPath("/mnt", drive, *tail_parts))


def default_wsl_codex_home() -> str | None:
    windows_codex_home = Path.home() / ".codex"
    if not windows_codex_home.exists():
        return None
    return windows_path_to_wsl(windows_codex_home)


def pull_wsl_artifacts(*, distro: str, synced_target: str, run_name: str, level: int) -> None:
    artifact_paths = [
        f"runs/{run_name}",
        f"results/timing/{run_name}",
        f"results/analysis/{run_name}_level{level}.json",
        f"results/analysis/{run_name}_level{level}.md",
    ]
    tar_command = (
        f"cd {shlex.quote(synced_target)} && "
        + "tar --ignore-failed-read -czf - "
        + " ".join(shlex.quote(path) for path in artifact_paths if path)
    )
    process = subprocess.run(
        ["wsl", "-d", distro, "bash", "-lc", tar_command],
        capture_output=True,
        check=True,
    )
    with tarfile.open(fileobj=io.BytesIO(process.stdout), mode="r:gz") as archive:
        archive.extractall(WINDOWS_REPO_ROOT, filter="data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync PTXBench into WSL2 and run the Linux paired workflow for any KernelBench level.")
    parser.add_argument("--distro", default="Ubuntu")
    parser.add_argument("--target", default=DEFAULT_WSL_TARGET)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--phase", choices=["smoke", "pilot", "full"], default="pilot")
    parser.add_argument("--provider", default="codex", choices=["litellm", "codex"])
    parser.add_argument("--track", default="oneshot", choices=["oneshot", "agentic"])
    parser.add_argument("--model")
    parser.add_argument("--level", type=int, choices=DEFAULT_LEVELS, default=1)
    parser.add_argument("--problem-ids")
    parser.add_argument("--arch", default="sm_89")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--num-correct-trials", type=int, default=5)
    parser.add_argument("--num-perf-trials", type=int, default=100)
    parser.add_argument("--official-eval-seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--max-concurrent-chunks", type=int)
    parser.add_argument("--parallel-backends", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--codex-home")
    parser.add_argument("--codex-config", action="append", default=[])
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-wall-clock-minutes", type=int, default=20)
    parser.add_argument("--max-tool-calls", type=int, default=4)
    parser.add_argument("--dev-eval-seed", type=int, default=7)
    parser.add_argument("--dev-eval-correct-trials", type=int, default=2)
    parser.add_argument("--dev-eval-perf-trials", type=int, default=5)
    parser.add_argument("--dev-eval-profile", action="store_true")
    parser.add_argument("--dev-eval-profile-trials", type=int, default=1)
    parser.add_argument("--dev-eval-profile-metric", action="append", default=[])
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-pull-results", action="store_true")
    parser.add_argument("--include-run", action="append", default=[])
    args = parser.parse_args()

    include_runs = set(args.include_run)
    if args.skip_generation:
        include_runs.add(args.run_name)
    synced_target = sync_repo_to_wsl(args.target, args.distro, include_runs)
    codex_home = args.codex_home or (default_wsl_codex_home() if args.provider == "codex" else None)

    linux_command = [
        "source",
        "scripts/setup_wsl_benchmark_env.sh",
        "&&",
        ".venv/bin/python",
        "scripts/run_level_paired.py",
        "--phase",
        args.phase,
        "--run-name",
        args.run_name,
        "--provider",
        args.provider,
        "--track",
        args.track,
        "--level",
        str(args.level),
        "--arch",
        args.arch,
        "--precision",
        args.precision,
        "--num-correct-trials",
        str(args.num_correct_trials),
        "--num-perf-trials",
        str(args.num_perf_trials),
        "--official-eval-seed",
        str(args.official_eval_seed),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--device",
        args.device,
        "--codex-bin",
        args.codex_bin,
        "--max-steps",
        str(args.max_steps),
        "--max-wall-clock-minutes",
        str(args.max_wall_clock_minutes),
        "--max-tool-calls",
        str(args.max_tool_calls),
        "--dev-eval-seed",
        str(args.dev_eval_seed),
        "--dev-eval-correct-trials",
        str(args.dev_eval_correct_trials),
        "--dev-eval-perf-trials",
        str(args.dev_eval_perf_trials),
    ]
    if args.model:
        linux_command.extend(["--model", args.model])
    if args.problem_ids:
        linux_command.extend(["--problem-ids", args.problem_ids])
    if args.chunk_size is not None:
        linux_command.extend(["--chunk-size", str(args.chunk_size)])
    if args.max_concurrent_chunks is not None:
        linux_command.extend(["--max-concurrent-chunks", str(args.max_concurrent_chunks)])
    if codex_home:
        linux_command.extend(["--codex-home", codex_home])
    for config_override in args.codex_config:
        linux_command.extend(["--codex-config", config_override])
    if args.dev_eval_profile:
        linux_command.extend(["--dev-eval-profile", "--dev-eval-profile-trials", str(args.dev_eval_profile_trials)])
        for metric in args.dev_eval_profile_metric:
            linux_command.extend(["--dev-eval-profile-metric", metric])
    for flag, enabled in (
        ("--skip-generation", args.skip_generation),
        ("--skip-eval", args.skip_eval),
        ("--skip-analysis", args.skip_analysis),
        ("--skip-smoke", args.skip_smoke),
        ("--parallel-backends", args.parallel_backends),
    ):
        if enabled:
            linux_command.append(flag)

    bash_command = f"cd {shlex.quote(synced_target)} && " + " ".join(shlex.quote(part) if part != "&&" else "&&" for part in linux_command)
    subprocess.run(
        ["wsl", "-d", args.distro, "bash", "-lc", bash_command],
        check=True,
    )
    if not args.skip_pull_results:
        pull_wsl_artifacts(distro=args.distro, synced_target=synced_target, run_name=args.run_name, level=args.level)


if __name__ == "__main__":
    main()
