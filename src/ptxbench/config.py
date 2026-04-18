from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = REPO_ROOT / "vendor" / "KernelBench-upstream"
KERNELBENCH_TASK_ROOT = VENDOR_ROOT / "KernelBench"
KERNELBENCH_UPSTREAM_URL = "https://github.com/ScalingIntelligence/KernelBench.git"
EXPECTED_KERNELBENCH_COMMIT = "423217d9fda91e0c2d67e4a43bf62f96f6d104f1"

DEFAULT_ARCH = "sm_89"
DEFAULT_LEVELS = (1, 2, 3, 4)
DEFAULT_TRACKS = ("oneshot", "agentic")
DEFAULT_TRACK = "oneshot"
DEFAULT_PRECISION = "fp32"
DEFAULT_NUM_CORRECT_TRIALS = 5
DEFAULT_NUM_PERF_TRIALS = 100
DEFAULT_NUM_WARMUP = 5
DEFAULT_FAST_THRESHOLDS = (0.0, 1.0, 2.0)
DEFAULT_OFFICIAL_EVAL_SEED = 42
DEFAULT_DEV_EVAL_SEED = 7
DEFAULT_AGENTIC_MAX_STEPS = 5
DEFAULT_AGENTIC_MAX_WALL_CLOCK_MINUTES = 20
DEFAULT_AGENTIC_MAX_TOOL_CALLS = 4
DEFAULT_DEV_EVAL_CORRECT_TRIALS = 2
DEFAULT_DEV_EVAL_PERF_TRIALS = 5
DEFAULT_DEV_EVAL_WARMUP = 2
DEFAULT_DEV_EVAL_TIMEOUT_SECONDS = 90
DEFAULT_DEV_EVAL_PROFILE_ENABLED = False
DEFAULT_DEV_EVAL_PROFILE_TOOL = "ncu"
DEFAULT_DEV_EVAL_PROFILE_TRIALS = 1
DEFAULT_PTX_PROFILE_METRICS = (
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
)
DEFAULT_DEV_EVAL_PROFILE_METRICS = (
    *DEFAULT_PTX_PROFILE_METRICS,
)
DEFAULT_GENERATION_TIMEOUT_SECONDS = 900
DEFAULT_FULL_GENERATION_TIMEOUT_SECONDS = 1800
DEFAULT_GENERATION_CHUNK_SIZE = 10
LEVEL_PILOT_PROBLEM_IDS: dict[int, tuple[int, ...]] = {
    1: (1, 3, 19, 23, 40),
    2: (1, 2, 18, 23, 43),
    3: (1, 9, 20, 28, 43),
    4: (1, 5, 10, 15, 20),
}


@dataclass(frozen=True)
class VendorSnapshot:
    commit: str
    root: Path
    task_root: Path


def _vendor_bootstrap_command() -> str:
    return "uv run python scripts/bootstrap_kernelbench.py"


def _git_repo_command(repo_root: Path, *args: str) -> list[str]:
    return [
        "git",
        "-c",
        f"safe.directory={repo_root}",
        "-C",
        str(repo_root),
        *args,
    ]


def detect_vendor_commit() -> str | None:
    git_dir = VENDOR_ROOT / ".git"
    if not git_dir.exists():
        return None

    head_path = git_dir / "HEAD"
    if head_path.exists():
        head_value = head_path.read_text(encoding="utf-8").strip()
        if head_value.startswith("ref:"):
            ref_name = head_value.split(":", 1)[1].strip()
            ref_path = git_dir / Path(ref_name)
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()
            packed_refs = git_dir / "packed-refs"
            if packed_refs.exists():
                for line in packed_refs.read_text(encoding="utf-8").splitlines():
                    if not line or line.startswith("#") or line.startswith("^"):
                        continue
                    commit, packed_ref = line.split(" ", 1)
                    if packed_ref.strip() == ref_name:
                        return commit.strip()
        elif head_value:
            return head_value

    process = subprocess.run(
        _git_repo_command(VENDOR_ROOT, "rev-parse", "HEAD"),
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        return None
    return process.stdout.strip()


def get_vendor_snapshot() -> VendorSnapshot:
    return VendorSnapshot(
        commit=detect_vendor_commit() or "unknown",
        root=VENDOR_ROOT,
        task_root=KERNELBENCH_TASK_ROOT,
    )


def ensure_vendor_snapshot() -> VendorSnapshot:
    if not KERNELBENCH_TASK_ROOT.exists():
        raise FileNotFoundError(
            f"KernelBench snapshot not found at {KERNELBENCH_TASK_ROOT}. "
            f"Run `{_vendor_bootstrap_command()}` to clone the pinned snapshot at commit {EXPECTED_KERNELBENCH_COMMIT}."
        )
    snapshot = get_vendor_snapshot()
    if snapshot.commit == "unknown":
        raise RuntimeError(
            f"KernelBench snapshot at {VENDOR_ROOT} could not be verified. "
            f"PTXBench requires the vendored task set to be a git checkout pinned to commit {EXPECTED_KERNELBENCH_COMMIT}. "
            f"Run `{_vendor_bootstrap_command()}` to recreate the pinned snapshot."
        )
    if snapshot.commit != EXPECTED_KERNELBENCH_COMMIT:
        raise RuntimeError(
            f"KernelBench snapshot commit mismatch at {VENDOR_ROOT}: expected {EXPECTED_KERNELBENCH_COMMIT}, found {snapshot.commit}. "
            f"PTXBench does not permit evaluating a different task set. Run `{_vendor_bootstrap_command()}` to reset the vendored snapshot."
        )
    return snapshot


def default_cache_root() -> Path:
    return REPO_ROOT / ".ptxbench_cache"
