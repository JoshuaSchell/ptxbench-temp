from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from ptxbench.config import (
    EXPECTED_KERNELBENCH_COMMIT,
    KERNELBENCH_TASK_ROOT,
    KERNELBENCH_UPSTREAM_URL,
    VENDOR_ROOT,
)


def _git_command(repo_root: Path, *args: str) -> list[str]:
    return [
        "git",
        "-c",
        f"safe.directory={repo_root}",
        "-C",
        str(repo_root),
        *args,
    ]


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _read_head_commit(repo_root: Path) -> str:
    process = subprocess.run(
        _git_command(repo_root, "rev-parse", "HEAD"),
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"Could not resolve HEAD for {repo_root}: {process.stderr.strip() or process.stdout.strip()}")
    return process.stdout.strip()


def verify_snapshot(*, vendor_root: Path, expected_commit: str) -> Path:
    task_root = vendor_root / "KernelBench"
    if not task_root.exists():
        raise FileNotFoundError(f"KernelBench task root missing at {task_root}")
    if not (vendor_root / ".git").exists():
        raise RuntimeError(f"Vendored snapshot at {vendor_root} is not a git checkout")
    detected_commit = _read_head_commit(vendor_root)
    if detected_commit != expected_commit:
        raise RuntimeError(
            f"Vendored KernelBench snapshot mismatch: expected {expected_commit}, found {detected_commit}"
        )
    return task_root


def bootstrap_snapshot(*, repo_url: str, vendor_root: Path, expected_commit: str) -> Path:
    vendor_root = vendor_root.resolve()
    if vendor_root.exists() and not (vendor_root / ".git").exists():
        try:
            next(vendor_root.iterdir())
        except StopIteration:
            pass
        else:
            raise RuntimeError(
                f"Destination {vendor_root} exists but is not a git checkout. Remove it or choose a different --vendor-root."
            )

    vendor_root.parent.mkdir(parents=True, exist_ok=True)
    if not vendor_root.exists():
        _run(["git", "clone", "--no-tags", repo_url, str(vendor_root)])
    elif not (vendor_root / ".git").exists():
        _run(["git", "clone", "--no-tags", repo_url, str(vendor_root)])
    else:
        _run(_git_command(vendor_root, "remote", "set-url", "origin", repo_url))
        _run(_git_command(vendor_root, "fetch", "--force", "--prune", "origin"))

    _run(_git_command(vendor_root, "fetch", "--force", "origin", expected_commit))
    _run(_git_command(vendor_root, "checkout", "--force", "--detach", expected_commit))
    _run(_git_command(vendor_root, "clean", "-ffd"))
    return verify_snapshot(vendor_root=vendor_root, expected_commit=expected_commit)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clone or update the vendored KernelBench snapshot to the pinned commit.")
    parser.add_argument("--repo-url", default=KERNELBENCH_UPSTREAM_URL)
    parser.add_argument("--vendor-root", default=str(VENDOR_ROOT))
    parser.add_argument("--expected-commit", default=EXPECTED_KERNELBENCH_COMMIT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)

    vendor_root = Path(args.vendor_root).resolve()
    if args.verify_only:
        task_root = verify_snapshot(vendor_root=vendor_root, expected_commit=args.expected_commit)
    else:
        task_root = bootstrap_snapshot(
            repo_url=args.repo_url,
            vendor_root=vendor_root,
            expected_commit=args.expected_commit,
        )

    print(f"KernelBench snapshot ready at {task_root}")
    print(f"Commit: {args.expected_commit}")
    if vendor_root == VENDOR_ROOT.resolve() and task_root == KERNELBENCH_TASK_ROOT.resolve():
        print("PTXBench dataset loading will now use the pinned vendored snapshot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
