from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import os
import shutil
import subprocess
import tempfile


def is_windows() -> bool:
    return os.name == "nt"


def _iter_vsdevcmd_candidates() -> list[Path]:
    env_override = os.environ.get("PTXBENCH_VSDEVCMD")
    if env_override:
        return [Path(env_override)]

    candidates: list[Path] = []
    roots = [os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")]
    editions = ("Community", "Professional", "Enterprise", "BuildTools")
    years = ("2022", "2019")
    for root in roots:
        if not root:
            continue
        for year in years:
            for edition in editions:
                candidates.append(Path(root) / "Microsoft Visual Studio" / year / edition / "Common7" / "Tools" / "VsDevCmd.bat")
    return candidates


@lru_cache(maxsize=1)
def find_vsdevcmd() -> Path | None:
    if not is_windows():
        return None
    for candidate in _iter_vsdevcmd_candidates():
        if candidate.exists():
            return candidate
    return None


@lru_cache(maxsize=1)
def find_msvc_tool_root() -> Path | None:
    if not is_windows():
        return None

    env_override = os.environ.get("PTXBENCH_MSVC_ROOT")
    if env_override:
        candidate = Path(env_override)
        if candidate.exists():
            return candidate

    roots = [os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")]
    editions = ("Community", "Professional", "Enterprise", "BuildTools")
    years = ("2022", "2019")
    found_roots: list[Path] = []
    for root in roots:
        if not root:
            continue
        for year in years:
            for edition in editions:
                msvc_parent = Path(root) / "Microsoft Visual Studio" / year / edition / "VC" / "Tools" / "MSVC"
                if not msvc_parent.exists():
                    continue
                found_roots.extend(sorted((path for path in msvc_parent.iterdir() if path.is_dir()), reverse=True))
    return found_roots[0] if found_roots else None


def _capture_batch_environment(batch_path: Path, args: tuple[str, ...] = ()) -> dict[str, str]:
    if not batch_path.exists():
        raise FileNotFoundError(f"Batch script not found: {batch_path}")

    command = " ".join([f'call "{batch_path}"', *args, ">nul", "&&", "set"])
    with tempfile.NamedTemporaryFile("w", suffix=".cmd", delete=False, encoding="utf-8") as handle:
        handle.write("@echo off\n")
        handle.write(command)
        handle.write("\n")
        script_path = Path(handle.name)

    try:
        process = subprocess.run(
            [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", str(script_path)],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        script_path.unlink(missing_ok=True)

    if process.returncode != 0:
        stderr = process.stderr.strip()
        stdout = process.stdout.strip()
        details = stderr or stdout or f"exit code {process.returncode}"
        raise RuntimeError(f"Failed to capture toolchain environment from {batch_path}: {details}")

    env: dict[str, str] = {}
    for line in process.stdout.splitlines():
        if "=" not in line or line.startswith("="):
            continue
        key, value = line.split("=", 1)
        normalized_key = key.upper()
        if normalized_key not in env or key == normalized_key:
            env[normalized_key] = value
    return env


def _missing_toolchain_message(msvc_root: Path | None) -> str:
    if msvc_root is None:
        return (
            "Could not locate an MSVC tool root. Install the Visual Studio C++ workload or set PTXBENCH_MSVC_ROOT."
        )

    missing: list[str] = []
    cl_path = msvc_root / "bin" / "Hostx64" / "x64" / "cl.exe"
    include_dir = msvc_root / "include"
    lib_dir = msvc_root / "lib" / "x64"
    if not cl_path.exists():
        missing.append(f"compiler binary missing at {cl_path}")
    if not include_dir.exists():
        missing.append(f"C++ headers missing at {include_dir}")
    if not lib_dir.exists():
        missing.append(f"x64 libraries missing at {lib_dir}")
    detail = "; ".join(missing) if missing else f"incomplete MSVC tool root at {msvc_root}"
    return (
        f"{detail}. Install the Visual Studio 'Desktop development with C++' workload, "
        "including the MSVC v143 build tools and Windows SDK."
    )


@lru_cache(maxsize=1)
def get_cuda_build_environment() -> dict[str, str]:
    if not is_windows():
        return {}

    current_path = os.environ.get("PATH", "")
    if shutil.which("cl", path=current_path) and os.environ.get("INCLUDE") and os.environ.get("LIB"):
        return {}

    vsdevcmd = find_vsdevcmd()
    if vsdevcmd is None:
        raise FileNotFoundError(
            "Could not locate VsDevCmd.bat. Install Visual Studio Build Tools or set PTXBENCH_VSDEVCMD."
        )

    env = _capture_batch_environment(vsdevcmd, ("-arch=x64", "-host_arch=x64"))
    msvc_root = find_msvc_tool_root()
    if msvc_root is not None:
        cl_dir = msvc_root / "bin" / "Hostx64" / "x64"
        include_dir = msvc_root / "include"
        lib_dir = msvc_root / "lib" / "x64"
        if cl_dir.exists():
            current_path = env.get("PATH", "")
            env["PATH"] = os.pathsep.join([str(cl_dir), current_path]) if current_path else str(cl_dir)
        if include_dir.exists():
            existing_include = env.get("INCLUDE", "")
            env["INCLUDE"] = os.pathsep.join([str(include_dir), existing_include]) if existing_include else str(include_dir)
        if lib_dir.exists():
            existing_lib = env.get("LIB", "")
            env["LIB"] = os.pathsep.join([str(lib_dir), existing_lib]) if existing_lib else str(lib_dir)
    toolchain_path = env.get("PATH", "")
    if shutil.which("cl", path=toolchain_path) is None:
        raise RuntimeError(_missing_toolchain_message(msvc_root))
    if msvc_root is None or not (msvc_root / "include").exists() or not (msvc_root / "lib" / "x64").exists():
        raise RuntimeError(_missing_toolchain_message(msvc_root))
    return env
