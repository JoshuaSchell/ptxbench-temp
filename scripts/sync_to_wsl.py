from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import io
import tarfile
import subprocess

from ptxbench.config import REPO_ROOT


DEFAULT_WSL_TARGET = "~/ptxbench/PTXBench"
EXCLUDED_TOP_LEVEL = {
    ".ptxbench_cache",
    ".pytest_cache",
    ".venv",
    "results",
    "runs",
    "__pycache__",
}
EXCLUDED_NAMES = {"__pycache__"}
EXCLUDED_SUFFIXES = {".pyc"}


def should_include(path: Path, include_run_names: set[str]) -> bool:
    relative = path.relative_to(REPO_ROOT)
    parts = relative.parts
    if not parts:
        return True
    if parts[0] in EXCLUDED_TOP_LEVEL:
        if parts[0] == "runs" and len(parts) > 1 and parts[1] in include_run_names:
            return True
        return False
    if any(part in EXCLUDED_NAMES for part in parts):
        return False
    if path.suffix in EXCLUDED_SUFFIXES:
        return False
    return True


def build_repo_archive(include_run_names: set[str]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(REPO_ROOT.rglob("*")):
            if not should_include(path, include_run_names):
                continue
            relative = path.relative_to(REPO_ROOT)
            archive.add(path, arcname=str(relative))
    buffer.seek(0)
    return buffer.read()


def resolve_wsl_home(distro: str) -> str:
    process = subprocess.run(
        ["wsl", "-d", distro, "bash", "-lc", "printf %s \"$HOME\""],
        capture_output=True,
        text=True,
        check=True,
    )
    return process.stdout.strip()


def normalize_wsl_target(target: str, distro: str) -> str:
    if target.startswith("~/"):
        home = resolve_wsl_home(distro)
        return str(PurePosixPath(home) / target[2:])
    return target


def sync_repo_to_wsl(target: str, distro: str, include_run_names: set[str]) -> str:
    archive_payload = build_repo_archive(include_run_names)
    normalized_target = normalize_wsl_target(target, distro)
    subprocess.run(
        ["wsl", "-d", distro, "bash", "-lc", f"rm -rf '{normalized_target}' && mkdir -p '{normalized_target}'"],
        check=True,
    )
    subprocess.run(
        ["wsl", "-d", distro, "bash", "-lc", f"tar -xzf - -C '{normalized_target}'"],
        input=archive_payload,
        check=True,
    )
    return normalized_target


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync the PTXBench repo into the Linux filesystem inside WSL2.")
    parser.add_argument("--distro", default="Ubuntu")
    parser.add_argument("--target", default=DEFAULT_WSL_TARGET)
    parser.add_argument("--include-run", action="append", default=[])
    args = parser.parse_args()

    synced_target = sync_repo_to_wsl(
        target=args.target,
        distro=args.distro,
        include_run_names=set(args.include_run),
    )
    print(f"Synced PTXBench to WSL distro {args.distro}: {synced_target}")


if __name__ == "__main__":
    main()
