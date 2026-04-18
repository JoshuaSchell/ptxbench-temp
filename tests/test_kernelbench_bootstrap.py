from pathlib import Path
import importlib.util
import subprocess
import tempfile

import pytest

import ptxbench.config as config


def _git(cwd: Path, *args: str, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        text=True,
        capture_output=capture_output,
    )


def _git_stdout(cwd: Path, *args: str) -> str:
    return _git(cwd, *args, capture_output=True).stdout.strip()


def _load_bootstrap_script():
    script_path = Path("scripts/bootstrap_kernelbench.py").resolve()
    spec = importlib.util.spec_from_file_location("ptxbench_bootstrap_kernelbench_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _create_local_kernelbench_repo(tmp_path: Path) -> tuple[Path, str, str]:
    repo_root = tmp_path / "kernelbench-source"
    repo_root.mkdir()
    _git(repo_root, "init")
    _git(repo_root, "config", "user.email", "ptxbench@example.com")
    _git(repo_root, "config", "user.name", "PTXBench Tests")

    task_path = repo_root / "KernelBench" / "level1" / "001_Dummy.py"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_text("class Model:\n    pass\n", encoding="utf-8")
    _git(repo_root, "add", ".")
    _git(repo_root, "commit", "-m", "initial snapshot")
    first_commit = _git_stdout(repo_root, "rev-parse", "HEAD")

    task_path.write_text("class Model:\n    value = 2\n", encoding="utf-8")
    _git(repo_root, "add", ".")
    _git(repo_root, "commit", "-m", "updated snapshot")
    second_commit = _git_stdout(repo_root, "rev-parse", "HEAD")
    return repo_root, first_commit, second_commit


def _workspace_tempdir() -> tempfile.TemporaryDirectory[str]:
    base = Path(".pytest_tmp").resolve()
    base.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=base)


def test_ensure_vendor_snapshot_missing_has_actionable_message(monkeypatch) -> None:
    with _workspace_tempdir() as tmpdir_str:
        missing_vendor_root = Path(tmpdir_str) / "vendor" / "KernelBench-upstream"
        monkeypatch.setattr(config, "VENDOR_ROOT", missing_vendor_root)
        monkeypatch.setattr(config, "KERNELBENCH_TASK_ROOT", missing_vendor_root / "KernelBench")

        with pytest.raises(FileNotFoundError, match="bootstrap_kernelbench.py"):
            config.ensure_vendor_snapshot()


def test_bootstrap_kernelbench_clones_and_updates_without_network() -> None:
    with _workspace_tempdir() as tmpdir_str:
        tmp_path = Path(tmpdir_str)
        bootstrap_script = _load_bootstrap_script()
        repo_root, first_commit, second_commit = _create_local_kernelbench_repo(tmp_path)
        vendor_root = tmp_path / "vendor" / "KernelBench-upstream"

        assert bootstrap_script.main(
            [
                "--repo-url",
                str(repo_root),
                "--vendor-root",
                str(vendor_root),
                "--expected-commit",
                first_commit,
            ]
        ) == 0
        assert (vendor_root / "KernelBench" / "level1" / "001_Dummy.py").exists()
        assert _git_stdout(vendor_root, "rev-parse", "HEAD") == first_commit

        assert bootstrap_script.main(
            [
                "--repo-url",
                str(repo_root),
                "--vendor-root",
                str(vendor_root),
                "--expected-commit",
                second_commit,
            ]
        ) == 0
        assert _git_stdout(vendor_root, "rev-parse", "HEAD") == second_commit

        assert bootstrap_script.main(
            [
                "--vendor-root",
                str(vendor_root),
                "--expected-commit",
                second_commit,
                "--verify-only",
            ]
        ) == 0
