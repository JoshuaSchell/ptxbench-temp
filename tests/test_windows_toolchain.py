from __future__ import annotations

from pathlib import Path
import os
import shutil

import pytest

from ptxbench.config import default_cache_root
from ptxbench.windows_toolchain import _capture_batch_environment


@pytest.mark.skipif(os.name != "nt", reason="Windows-only toolchain bootstrap test")
def test_capture_batch_environment_from_simple_script() -> None:
    tmp_root = default_cache_root() / "test_tmp" / "windows_toolchain"
    tmp_root.mkdir(parents=True, exist_ok=True)
    script_path = tmp_root / "fake_env.cmd"
    script_path.write_text(
        "@echo off\n"
        "set FOO=bar\n"
        "set PATH=C:\\toolchain;%PATH%\n",
        encoding="utf-8",
    )

    try:
        env = _capture_batch_environment(script_path)
        assert env["FOO"] == "bar"
        assert env["PATH"].startswith(r"C:\toolchain;")
    finally:
        script_path.unlink(missing_ok=True)
        shutil.rmtree(tmp_root, ignore_errors=True)
