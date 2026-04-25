from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import ctypes
import hashlib
import json
import re
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


_PTXAS_CACHE_FORMAT = "ptxas-v2"
_PTXAS_COMPILE_ENTRY_RE = re.compile(r"Compiling entry function '([^']+)' for '([^']+)'")
_PTXAS_FUNCTION_RE = re.compile(r"Function properties for ['\"]?([^'\"\s]+)['\"]?")
_PTXAS_STACK_RE = re.compile(r"(\d+) bytes stack frame")
_PTXAS_SPILL_STORES_RE = re.compile(r"(\d+) bytes spill stores")
_PTXAS_SPILL_LOADS_RE = re.compile(r"(\d+) bytes spill loads")
_PTXAS_REGISTERS_RE = re.compile(r"Used (\d+) registers")
_PTXAS_MEMORY_RE = re.compile(r"(\d+) bytes (smem|lmem|cmem\[\d+\])")


@dataclass(frozen=True)
class PTXKernelAssemblyReport:
    name: str
    arch: str | None = None
    registers: int | None = None
    spill_stores_bytes: int | None = None
    spill_loads_bytes: int | None = None
    shared_memory_bytes: int | None = None
    constant_memory_bytes: int | None = None
    local_memory_bytes: int | None = None
    stack_frame_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arch": self.arch,
            "registers": self.registers,
            "spill_stores_bytes": self.spill_stores_bytes,
            "spill_loads_bytes": self.spill_loads_bytes,
            "shared_memory_bytes": self.shared_memory_bytes,
            "constant_memory_bytes": self.constant_memory_bytes,
            "local_memory_bytes": self.local_memory_bytes,
            "stack_frame_bytes": self.stack_frame_bytes,
        }


@dataclass(frozen=True)
class PTXAssemblyReport:
    source_name: str
    arch: str
    registers: int | None = None
    spill_stores_bytes: int | None = None
    spill_loads_bytes: int | None = None
    shared_memory_bytes: int | None = None
    constant_memory_bytes: int | None = None
    local_memory_bytes: int | None = None
    stack_frame_bytes: int | None = None
    functions: tuple[PTXKernelAssemblyReport, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "arch": self.arch,
            "registers": self.registers,
            "spill_stores_bytes": self.spill_stores_bytes,
            "spill_loads_bytes": self.spill_loads_bytes,
            "shared_memory_bytes": self.shared_memory_bytes,
            "constant_memory_bytes": self.constant_memory_bytes,
            "local_memory_bytes": self.local_memory_bytes,
            "stack_frame_bytes": self.stack_frame_bytes,
            "functions": [function.to_dict() for function in self.functions],
        }

    def for_entry(self, entry: str | None) -> PTXKernelAssemblyReport | None:
        if entry:
            for function in self.functions:
                if function.name == entry:
                    return function
        if len(self.functions) == 1:
            return self.functions[0]
        if not self.functions:
            return None
        return max(
            self.functions,
            key=lambda function: (
                function.registers is not None,
                -1 if function.registers is None else function.registers,
            ),
        )


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
    assembly_report: PTXAssemblyReport


_DRIVER_READY = False
_TOOLCHAIN_VERSION: str | None = None
_MODULE_CACHE: dict[tuple[int, str, str], tuple[Any, Any]] = {}
_PTX_ARTIFACT_LOG: list[dict[str, Any]] = []


def _max_optional_int(values: list[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def clear_ptx_artifact_log() -> None:
    _PTX_ARTIFACT_LOG.clear()


def get_ptx_artifact_log() -> list[dict[str, Any]]:
    deduped: dict[tuple[str, str | None], dict[str, Any]] = {}
    for record in _PTX_ARTIFACT_LOG:
        key = (str(record.get("source_hash")), record.get("entry"))
        deduped[key] = dict(record)
    return list(deduped.values())


def summarize_ptx_artifact_resources(records: list[dict[str, Any]]) -> dict[str, Any]:
    def _max_field(field_name: str) -> int | None:
        values: list[int] = []
        for record in records:
            report = record.get("assembly_report")
            if not isinstance(report, dict):
                continue
            value = report.get(field_name)
            if isinstance(value, int) and not isinstance(value, bool):
                values.append(value)
        return max(values) if values else None

    num_functions = 0
    for record in records:
        report = record.get("assembly_report")
        if isinstance(report, dict) and isinstance(report.get("functions"), list):
            num_functions += len(report["functions"])
    max_spill_stores = _max_field("spill_stores_bytes")
    max_spill_loads = _max_field("spill_loads_bytes")
    return {
        "num_artifacts": len(records),
        "num_functions": num_functions,
        "max_registers": _max_field("registers"),
        "max_spill_stores_bytes": max_spill_stores,
        "max_spill_loads_bytes": max_spill_loads,
        "max_shared_memory_bytes": _max_field("shared_memory_bytes"),
        "max_local_memory_bytes": _max_field("local_memory_bytes"),
        "max_constant_memory_bytes": _max_field("constant_memory_bytes"),
        "max_stack_frame_bytes": _max_field("stack_frame_bytes"),
        "any_spills": bool((max_spill_stores or 0) > 0 or (max_spill_loads or 0) > 0),
    }


def _append_ptx_artifact_record(
    *,
    artifact: PTXCompileArtifact,
    kernel_name: str,
    entry: str,
) -> None:
    _PTX_ARTIFACT_LOG.append(
        {
            "source_name": artifact.source_name,
            "source_hash": artifact.source_hash,
            "arch": artifact.arch,
            "cubin_path": str(artifact.cubin_path),
            "ptx_path": str(artifact.ptx_path),
            "log_path": str(artifact.log_path),
            "toolchain": artifact.toolchain,
            "kernel_name": kernel_name,
            "entry": entry,
            "assembly_report": artifact.assembly_report.to_dict(),
        }
    )


def _new_kernel_assembly_state(name: str, arch: str | None) -> dict[str, Any]:
    return {
        "name": name,
        "arch": arch,
        "registers": None,
        "spill_stores_bytes": None,
        "spill_loads_bytes": None,
        "shared_memory_bytes": None,
        "constant_memory_bytes": None,
        "local_memory_bytes": None,
        "stack_frame_bytes": None,
    }


def _materialize_kernel_assembly_report(state: dict[str, Any]) -> PTXKernelAssemblyReport:
    return PTXKernelAssemblyReport(
        name=str(state["name"]),
        arch=state.get("arch"),
        registers=state.get("registers"),
        spill_stores_bytes=state.get("spill_stores_bytes"),
        spill_loads_bytes=state.get("spill_loads_bytes"),
        shared_memory_bytes=state.get("shared_memory_bytes"),
        constant_memory_bytes=state.get("constant_memory_bytes"),
        local_memory_bytes=state.get("local_memory_bytes"),
        stack_frame_bytes=state.get("stack_frame_bytes"),
    )


def parse_ptxas_output(output: str, *, source_name: str, arch: str) -> PTXAssemblyReport:
    function_states: dict[str, dict[str, Any]] = {}
    current_function: str | None = None

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        compile_match = _PTXAS_COMPILE_ENTRY_RE.search(line)
        if compile_match:
            current_function = compile_match.group(1)
            function_states.setdefault(
                current_function,
                _new_kernel_assembly_state(current_function, compile_match.group(2)),
            )
            function_states[current_function]["arch"] = compile_match.group(2)
            continue

        function_match = _PTXAS_FUNCTION_RE.search(line)
        if function_match:
            current_function = function_match.group(1)
            function_states.setdefault(current_function, _new_kernel_assembly_state(current_function, arch))
            continue

        if current_function is None:
            continue

        state = function_states.setdefault(current_function, _new_kernel_assembly_state(current_function, arch))

        stack_match = _PTXAS_STACK_RE.search(line)
        if stack_match:
            state["stack_frame_bytes"] = int(stack_match.group(1))

        spill_store_match = _PTXAS_SPILL_STORES_RE.search(line)
        if spill_store_match:
            state["spill_stores_bytes"] = int(spill_store_match.group(1))

        spill_load_match = _PTXAS_SPILL_LOADS_RE.search(line)
        if spill_load_match:
            state["spill_loads_bytes"] = int(spill_load_match.group(1))

        register_match = _PTXAS_REGISTERS_RE.search(line)
        if register_match:
            state["registers"] = int(register_match.group(1))

        if "bytes" not in line:
            continue

        constant_memory = 0
        saw_constant_memory = False
        for value_text, kind in _PTXAS_MEMORY_RE.findall(line):
            value = int(value_text)
            if kind == "smem":
                state["shared_memory_bytes"] = value
            elif kind == "lmem":
                state["local_memory_bytes"] = value
            elif kind.startswith("cmem["):
                constant_memory += value
                saw_constant_memory = True
        if saw_constant_memory:
            state["constant_memory_bytes"] = constant_memory
        elif register_match and state["constant_memory_bytes"] is None:
            state["constant_memory_bytes"] = 0
        if register_match:
            if state["shared_memory_bytes"] is None:
                state["shared_memory_bytes"] = 0
            if state["local_memory_bytes"] is None:
                state["local_memory_bytes"] = 0

    function_reports = tuple(_materialize_kernel_assembly_report(state) for state in function_states.values())
    return PTXAssemblyReport(
        source_name=source_name,
        arch=arch,
        registers=_max_optional_int([function.registers for function in function_reports]),
        spill_stores_bytes=_max_optional_int([function.spill_stores_bytes for function in function_reports]),
        spill_loads_bytes=_max_optional_int([function.spill_loads_bytes for function in function_reports]),
        shared_memory_bytes=_max_optional_int([function.shared_memory_bytes for function in function_reports]),
        constant_memory_bytes=_max_optional_int([function.constant_memory_bytes for function in function_reports]),
        local_memory_bytes=_max_optional_int([function.local_memory_bytes for function in function_reports]),
        stack_frame_bytes=_max_optional_int([function.stack_frame_bytes for function in function_reports]),
        functions=function_reports,
    )


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
    source_hash = hashlib.sha256(
        f"{arch}\0{toolchain}\0O{opt_level}\0{_PTXAS_CACHE_FORMAT}\0{source_text}".encode("utf-8")
    ).hexdigest()[:16]
    cache_dir = (cache_root or default_cache_root()) / arch / f"{source_name}-{source_hash}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    ptx_path = cache_dir / f"{source_name}.ptx"
    cubin_path = cache_dir / f"{source_name}.cubin"
    log_path = cache_dir / "ptxas.log"
    manifest_path = cache_dir / "artifact.json"

    ptx_path.write_text(source_text, encoding="utf-8")

    combined_output = ""
    if not cubin_path.exists() or not log_path.exists():
        process = subprocess.run(
            [
                get_ptxas_path(),
                f"--gpu-name={arch}",
                f"-O{opt_level}",
                str(ptx_path),
                "-o",
                str(cubin_path),
                "-v",
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
    elif log_path.exists():
        combined_output = log_path.read_text(encoding="utf-8")

    assembly_report = parse_ptxas_output(combined_output, source_name=source_name, arch=arch)

    manifest_path.write_text(
        json.dumps(
            {
                "source_name": source_name,
                "source_hash": source_hash,
                "arch": arch,
                "toolchain": toolchain,
                "ptx_path": str(ptx_path),
                "cubin_path": str(cubin_path),
                "log_path": str(log_path),
                "assembly_report": assembly_report.to_dict(),
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
        assembly_report=assembly_report,
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

        _append_ptx_artifact_record(artifact=artifact, kernel_name=kernel_name, entry=spec.entry)
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
