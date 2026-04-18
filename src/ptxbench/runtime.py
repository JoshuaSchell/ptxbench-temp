from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import ctypes
import hashlib
import json
import shutil
import subprocess

from .config import DEFAULT_ARCH, default_cache_root
from .spec import PTXKernelSpec


class PTXRuntimeError(RuntimeError):
    pass


class PTXAssemblyError(PTXRuntimeError):
    pass


class PTXLoadError(PTXRuntimeError):
    pass


class PTXLaunchError(PTXRuntimeError):
    pass


@dataclass(frozen=True)
class PTXCompileArtifact:
    source_name: str
    source_hash: str
    arch: str
    ptx_path: Path
    cubin_path: Path
    log_path: Path
    manifest_path: Path
    toolchain: str


_DRIVER_READY = False
_TOOLCHAIN_VERSION: str | None = None
_MODULE_CACHE: dict[tuple[int, str, str], tuple[Any, Any]] = {}


def _decode_error_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _cuda_check(result: Any) -> Any:
    from cuda.bindings import driver as cuda_driver

    if isinstance(result, tuple):
        status = result[0]
        payload = result[1:]
    else:
        status = result
        payload = ()

    if status != cuda_driver.CUresult.CUDA_SUCCESS:
        error_name = _decode_error_text(_cuda_check(cuda_driver.cuGetErrorName(status)))
        error_text = _decode_error_text(_cuda_check(cuda_driver.cuGetErrorString(status)))
        raise PTXLoadError(f"{error_name}: {error_text}")

    if not payload:
        return None
    if len(payload) == 1:
        return payload[0]
    return payload


def initialize_driver() -> None:
    global _DRIVER_READY
    if _DRIVER_READY:
        return
    from cuda.bindings import driver as cuda_driver

    _cuda_check(cuda_driver.cuInit(0))
    _DRIVER_READY = True


def get_ptxas_path() -> str:
    path = shutil.which("ptxas")
    if path is None:
        raise FileNotFoundError("ptxas not found on PATH")
    return path


def get_ptxas_version() -> str:
    global _TOOLCHAIN_VERSION
    if _TOOLCHAIN_VERSION is None:
        process = subprocess.run(
            [get_ptxas_path(), "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode != 0:
            raise PTXAssemblyError(process.stderr.strip() or "Failed to query ptxas version")
        _TOOLCHAIN_VERSION = process.stdout.strip()
    return _TOOLCHAIN_VERSION


def compile_ptx_source(
    source_name: str,
    source_text: str,
    arch: str = DEFAULT_ARCH,
    cache_root: Path | None = None,
    opt_level: int = 3,
) -> PTXCompileArtifact:
    toolchain = get_ptxas_version()
    source_hash = hashlib.sha256(f"{arch}\0{toolchain}\0{source_text}".encode("utf-8")).hexdigest()[:16]
    cache_dir = (cache_root or default_cache_root()) / arch / f"{source_name}-{source_hash}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    ptx_path = cache_dir / f"{source_name}.ptx"
    cubin_path = cache_dir / f"{source_name}.cubin"
    log_path = cache_dir / "ptxas.log"
    manifest_path = cache_dir / "artifact.json"

    ptx_path.write_text(source_text, encoding="utf-8")

    if not cubin_path.exists():
        process = subprocess.run(
            [
                get_ptxas_path(),
                f"--gpu-name={arch}",
                f"-O{opt_level}",
                str(ptx_path),
                "-o",
                str(cubin_path),
                "--warn-on-spills",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        combined_output = "\n".join(part for part in [process.stdout, process.stderr] if part.strip())
        log_path.write_text(combined_output, encoding="utf-8")
        if process.returncode != 0:
            raise PTXAssemblyError(
                f"ptxas failed for {source_name} on {arch}:\n{combined_output or 'no compiler output'}"
            )

    manifest_path.write_text(
        json.dumps(
            {
                "source_name": source_name,
                "source_hash": source_hash,
                "arch": arch,
                "toolchain": toolchain,
                "ptx_path": str(ptx_path),
                "cubin_path": str(cubin_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return PTXCompileArtifact(
        source_name=source_name,
        source_hash=source_hash,
        arch=arch,
        ptx_path=ptx_path,
        cubin_path=cubin_path,
        log_path=log_path,
        manifest_path=manifest_path,
        toolchain=toolchain,
    )


def _stream_handle(stream: Any | None = None) -> int:
    import torch

    if stream is None:
        return int(torch.cuda.current_stream().cuda_stream)
    if hasattr(stream, "cuda_stream"):
        return int(stream.cuda_stream)
    return int(stream)


def _marshal_argument(arg_type: str, value: Any) -> tuple[Any, Any]:
    import torch

    if arg_type == "tensor":
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for arg_type='tensor', got {type(value)}")
        return int(value.data_ptr()), ctypes.c_void_p
    if arg_type == "pointer":
        if isinstance(value, ctypes.c_void_p):
            return value.value, ctypes.c_void_p
        return int(value), ctypes.c_void_p
    if arg_type == "bool":
        return bool(value), ctypes.c_bool
    if arg_type == "float32":
        return float(value), ctypes.c_float
    if arg_type == "float64":
        return float(value), ctypes.c_double
    if arg_type == "int32":
        return int(value), ctypes.c_int32
    if arg_type == "int64":
        return int(value), ctypes.c_int64
    if arg_type == "uint32":
        return int(value), ctypes.c_uint32
    if arg_type == "uint64":
        return int(value), ctypes.c_uint64
    raise TypeError(f"Unsupported PTX argument type: {arg_type}")


class PTXModuleRunner:
    def __init__(
        self,
        ptx_sources: dict[str, str],
        kernel_specs: dict[str, PTXKernelSpec],
        arch: str = DEFAULT_ARCH,
        cache_root: Path | None = None,
    ) -> None:
        if not ptx_sources:
            raise ValueError("PTXModuleRunner requires at least one PTX source")
        self.ptx_sources = dict(ptx_sources)
        self.kernel_specs = dict(kernel_specs)
        self.arch = arch
        self.cache_root = cache_root or default_cache_root()
        self._artifacts: dict[str, PTXCompileArtifact] = {}

    def _ensure_loaded(self, kernel_name: str) -> tuple[PTXCompileArtifact, Any]:
        import torch
        from cuda.bindings import driver as cuda_driver

        if kernel_name not in self.ptx_sources:
            raise KeyError(f"Unknown PTX source '{kernel_name}'")
        if kernel_name not in self.kernel_specs:
            raise KeyError(f"Unknown PTX kernel spec '{kernel_name}'")

        artifact = self._artifacts.get(kernel_name)
        if artifact is None:
            artifact = compile_ptx_source(
                source_name=kernel_name,
                source_text=self.ptx_sources[kernel_name],
                arch=self.arch,
                cache_root=self.cache_root,
            )
            self._artifacts[kernel_name] = artifact

        torch.cuda.init()
        torch.cuda.current_stream()
        initialize_driver()

        device_idx = torch.cuda.current_device()
        spec = self.kernel_specs[kernel_name]
        cache_key = (device_idx, str(artifact.cubin_path), spec.entry)
        if cache_key not in _MODULE_CACHE:
            try:
                module = _cuda_check(cuda_driver.cuModuleLoadData(artifact.cubin_path.read_bytes()))
                function = _cuda_check(cuda_driver.cuModuleGetFunction(module, spec.entry.encode("utf-8")))
            except PTXLoadError:
                raise
            except Exception as exc:
                raise PTXLoadError(f"Failed to load cubin for {kernel_name}: {exc}") from exc
            _MODULE_CACHE[cache_key] = (module, function)
        return artifact, _MODULE_CACHE[cache_key][1]

    def launch(
        self,
        kernel_name: str,
        *runtime_args: Any,
        grid: tuple[int, int, int] | None = None,
        block: tuple[int, int, int] | None = None,
        stream: Any | None = None,
    ) -> PTXCompileArtifact:
        from cuda.bindings import driver as cuda_driver

        artifact, function = self._ensure_loaded(kernel_name)
        spec = self.kernel_specs[kernel_name]
        if len(runtime_args) != len(spec.arg_types):
            raise ValueError(
                f"{kernel_name} expected {len(spec.arg_types)} runtime args, got {len(runtime_args)}"
            )

        launch_grid = spec.resolve_grid(*runtime_args) if grid is None else grid
        launch_block = spec.resolve_block(*runtime_args) if block is None else block
        shared_mem = spec.resolve_shared_mem(*runtime_args)

        kernel_values = []
        kernel_types = []
        for arg_type, value in zip(spec.arg_types, runtime_args, strict=True):
            marshalled_value, marshalled_type = _marshal_argument(arg_type, value)
            kernel_values.append(marshalled_value)
            kernel_types.append(marshalled_type)

        try:
            _cuda_check(
                cuda_driver.cuLaunchKernel(
                    function,
                    int(launch_grid[0]),
                    int(launch_grid[1]),
                    int(launch_grid[2]),
                    int(launch_block[0]),
                    int(launch_block[1]),
                    int(launch_block[2]),
                    int(shared_mem),
                    _stream_handle(stream),
                    kernelParams=(tuple(kernel_values), tuple(kernel_types)),
                    extra=0,
                )
            )
        except PTXLoadError as exc:
            raise PTXLaunchError(str(exc)) from exc
        except Exception as exc:
            raise PTXLaunchError(f"Failed to launch {kernel_name}: {exc}") from exc

        return artifact


def launch_ptx(
    ptx_sources: dict[str, str],
    kernel_specs: dict[str, PTXKernelSpec],
    kernel_name: str,
    *runtime_args: Any,
    arch: str = DEFAULT_ARCH,
    cache_root: Path | None = None,
) -> PTXCompileArtifact:
    runner = PTXModuleRunner(ptx_sources=ptx_sources, kernel_specs=kernel_specs, arch=arch, cache_root=cache_root)
    return runner.launch(kernel_name, *runtime_args)
