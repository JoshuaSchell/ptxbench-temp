from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from contextlib import contextmanager
import gc
import importlib.util
import json
import math
import os
import random
import sys
import traceback
import types
import uuid

from .config import (
    DEFAULT_ARCH,
    DEFAULT_OFFICIAL_EVAL_SEED,
    DEFAULT_NUM_CORRECT_TRIALS,
    DEFAULT_NUM_PERF_TRIALS,
    DEFAULT_NUM_WARMUP,
    default_cache_root,
)
from .dataset import Problem
from .profiler import ProfileRequest, profile_callable, skipped_profile_result
from .runtime import PTXAssemblyError, PTXLaunchError, PTXLoadError
from .static_checker import validate_submission_static
from .timing import summarize_timings, time_callable_cuda_events
from .windows_toolchain import get_cuda_build_environment


@dataclass
class EvalResult:
    backend: str
    problem_id: int
    problem_name: str
    source_path: str
    track: str = "oneshot"
    task_family_tags: list[str] = field(default_factory=list)
    episode_id: str | None = None
    step_count: int | None = None
    budget_used: dict[str, Any] = field(default_factory=dict)
    first_compile_step: int | None = None
    first_correct_step: int | None = None
    final_submission_hash: str | None = None
    compiled: bool = False
    assembled: bool | None = None
    loaded: bool | None = None
    correctness: bool = False
    runtime_ms: float = -1.0
    ref_runtime_ms: float = -1.0
    speedup_vs_torch: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "problem_id": self.problem_id,
            "problem_name": self.problem_name,
            "source_path": self.source_path,
            "track": self.track,
            "task_family_tags": self.task_family_tags,
            "episode_id": self.episode_id,
            "step_count": self.step_count,
            "budget_used": _json_safe(self.budget_used),
            "first_compile_step": self.first_compile_step,
            "first_correct_step": self.first_correct_step,
            "final_submission_hash": self.final_submission_hash,
            "compiled": self.compiled,
            "assembled": self.assembled,
            "loaded": self.loaded,
            "correctness": self.correctness,
            "runtime_ms": self.runtime_ms,
            "ref_runtime_ms": self.ref_runtime_ms,
            "speedup_vs_torch": self.speedup_vs_torch,
            "metadata": _json_safe(self.metadata),
        }


def build_missing_submission_result(
    problem: Problem,
    *,
    backend: str,
    expected_path: Path,
    metadata: dict[str, Any] | None = None,
) -> EvalResult:
    return EvalResult(
        backend=backend,
        problem_id=problem.problem_id,
        problem_name=problem.name,
        source_path=str(expected_path),
        task_family_tags=list(problem.task_family_tags),
        compiled=False,
        assembled=False if backend == "ptx" else None,
        loaded=False if backend == "ptx" else None,
        correctness=False,
        metadata=metadata or {},
    )


def build_evaluation_failure_result(
    problem: Problem,
    *,
    backend: str,
    source_path: Path,
    metadata: dict[str, Any] | None = None,
) -> EvalResult:
    return EvalResult(
        backend=backend,
        problem_id=problem.problem_id,
        problem_name=problem.name,
        source_path=str(source_path),
        task_family_tags=list(problem.task_family_tags),
        compiled=True,
        assembled=True if backend == "ptx" else None,
        loaded=True if backend == "ptx" else None,
        correctness=False,
        metadata=metadata or {},
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def get_torch_dtype(precision: str):
    import torch

    mapping = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    try:
        return mapping[precision]
    except KeyError as exc:
        raise ValueError(f"Unsupported precision: {precision}") from exc


def get_tolerance(precision: str) -> float:
    return {
        "fp32": 1e-4,
        "fp16": 1e-2,
        "bf16": 1e-2,
    }[precision]


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _load_reference_context(problem: Problem) -> tuple[type, Any, Any]:
    namespace: dict[str, Any] = {}
    compiled = compile(problem.code, str(problem.path), "exec")
    exec(compiled, namespace)
    return namespace["Model"], namespace["get_inputs"], namespace["get_init_inputs"]


@contextmanager
def torch_extensions_dir(build_dir: Path | None, *, backend: str | None = None):
    python_scripts = Path(sys.executable).resolve().parent
    previous_env = os.environ.copy()
    try:
        if backend == "cuda":
            os.environ.update(get_cuda_build_environment())
        current_path = os.environ.get("PATH", "")
        path_parts = current_path.split(os.pathsep) if current_path else []
        if str(python_scripts) not in path_parts:
            os.environ["PATH"] = os.pathsep.join([str(python_scripts), current_path]) if current_path else str(python_scripts)
        if build_dir is not None:
            build_dir.mkdir(parents=True, exist_ok=True)
            os.environ["TORCH_EXTENSIONS_DIR"] = str(build_dir)
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous_env)


def load_submission_module(
    source_path: Path,
    *,
    build_dir: Path | None = None,
    backend: str | None = None,
) -> types.ModuleType:
    module_name = f"ptxbench_submission_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        with torch_extensions_dir(build_dir, backend=backend):
            spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def unload_submission_module(module: types.ModuleType) -> None:
    sys.modules.pop(module.__name__, None)


def _prepare_value(value: Any, *, device: Any, dtype: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
    return value


def _prepare_inputs(values: list[Any], *, device: Any, dtype: Any) -> list[Any]:
    return [_prepare_value(value, device=device, dtype=dtype) for value in values]


def _clone_output_reference(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").clone()
    if isinstance(value, tuple):
        return tuple(_clone_output_reference(item) for item in value)
    if isinstance(value, list):
        return [_clone_output_reference(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_output_reference(item) for key, item in value.items()}
    return value


def _cleanup_cuda(device: Any) -> None:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize(device=device)
        except Exception:
            pass
        torch.cuda.empty_cache()


_MISMATCH_INDEX_LIMIT = 100_000


def _json_safe_scalar(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, complex):
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _format_nested_output_path(path: str, suffix: str) -> str:
    return f"{path}{suffix}" if path else suffix


def _format_index_suffix(index: list[int] | None) -> str:
    if not index:
        return ""
    return "[" + ", ".join(str(value) for value in index) + "]"


def _tensor_has_nan(value: Any) -> bool:
    import torch

    if not isinstance(value, torch.Tensor):
        return False
    if not (value.is_floating_point() or value.is_complex()):
        return False
    return bool(torch.isnan(value).any().item())


def _tensor_has_inf(value: Any) -> bool:
    import torch

    if not isinstance(value, torch.Tensor):
        return False
    if not (value.is_floating_point() or value.is_complex()):
        return False
    return bool(torch.isinf(value).any().item())


def _float_tensor(value: Any) -> Any:
    import torch

    if value.is_complex():
        return torch.abs(value).to(torch.float64)
    return value.to(torch.float64)


def _safe_max_finite(value: Any) -> float | None:
    import torch

    if value.numel() == 0:
        return None
    as_float = _float_tensor(value)
    finite_mask = torch.isfinite(as_float)
    if not bool(finite_mask.any().item()):
        return None
    return float(as_float[finite_mask].max().item())


def _build_tensor_mismatch_details(
    reference: Any,
    candidate: Any,
    *,
    atol: float,
    rtol: float,
    path: str,
) -> dict[str, Any]:
    import torch

    details: dict[str, Any] = {
        "kind": "tensor_mismatch",
        "path": path,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "shape_mismatch": reference.shape != candidate.shape,
        "max_abs_diff": None,
        "max_rel_diff": None,
        "first_bad_index": None,
        "reference_has_nan": _tensor_has_nan(reference),
        "candidate_has_nan": _tensor_has_nan(candidate),
        "reference_has_inf": _tensor_has_inf(reference),
        "candidate_has_inf": _tensor_has_inf(candidate),
        "num_elements": int(reference.numel()),
        "num_mismatched": None,
        "atol": float(atol),
        "rtol": float(rtol),
    }
    if reference.shape != candidate.shape:
        return details

    reference_numeric = _float_tensor(reference)
    candidate_numeric = _float_tensor(candidate)
    if reference.is_floating_point() or reference.is_complex():
        mismatch_mask = ~torch.isclose(reference, candidate, atol=atol, rtol=rtol, equal_nan=False)
    else:
        mismatch_mask = reference != candidate
    details["num_mismatched"] = int(mismatch_mask.sum().item())

    abs_diff = torch.abs(reference_numeric - candidate_numeric)
    details["max_abs_diff"] = _safe_max_finite(abs_diff[mismatch_mask]) if bool(mismatch_mask.any().item()) else None

    reference_abs = torch.abs(reference_numeric)
    rel_valid = mismatch_mask & torch.isfinite(abs_diff) & torch.isfinite(reference_abs) & (reference_abs > 0)
    if bool(rel_valid.any().item()):
        details["max_rel_diff"] = float((abs_diff[rel_valid] / reference_abs[rel_valid]).max().item())

    if details["num_mismatched"] and details["num_elements"] <= _MISMATCH_INDEX_LIMIT:
        first_bad = torch.nonzero(mismatch_mask, as_tuple=False)[0]
        first_bad_index = [int(value.item()) for value in first_bad]
        details["first_bad_index"] = first_bad_index
        tensor_index = tuple(first_bad_index)
        details["reference_value"] = _json_safe_scalar(reference[tensor_index].item())
        details["candidate_value"] = _json_safe_scalar(candidate[tensor_index].item())

    return details


def _format_correctness_mismatch(details: dict[str, Any]) -> str:
    kind = str(details.get("kind", "mismatch"))
    path = str(details.get("path", "output"))

    if kind == "tensor_mismatch":
        ref_nan = bool(details.get("reference_has_nan"))
        cand_nan = bool(details.get("candidate_has_nan"))
        ref_inf = bool(details.get("reference_has_inf"))
        cand_inf = bool(details.get("candidate_has_inf"))
        if details.get("shape_mismatch"):
            return (
                f"shape mismatch at {path}: expected {tuple(details.get('reference_shape', []))}, "
                f"got {tuple(details.get('candidate_shape', []))} "
                f"nan(ref={ref_nan},cand={cand_nan}) inf(ref={ref_inf},cand={cand_inf})"
            )

        location = path + _format_index_suffix(details.get("first_bad_index"))
        parts = [f"tensor mismatch at {location}"]
        if details.get("max_abs_diff") is not None:
            parts.append(f"max_abs_diff={float(details['max_abs_diff']):.6g}")
        if details.get("max_rel_diff") is not None:
            parts.append(f"max_rel_diff={float(details['max_rel_diff']):.6g}")
        parts.append(f"nan(ref={ref_nan},cand={cand_nan})")
        parts.append(f"inf(ref={ref_inf},cand={cand_inf})")
        return " ".join(parts)

    if kind == "sequence_length_mismatch":
        return (
            f"sequence length mismatch at {path}: expected {details.get('expected_length')}, "
            f"got {details.get('candidate_length')}"
        )
    if kind == "dict_key_mismatch":
        return (
            f"dict key mismatch at {path}: expected {details.get('reference_keys')}, "
            f"got {details.get('candidate_keys')}"
        )
    if kind == "type_mismatch":
        return (
            f"type mismatch at {path}: expected {details.get('reference_type')}, "
            f"got {details.get('candidate_type')}"
        )
    if kind == "scalar_mismatch":
        return (
            f"scalar mismatch at {path}: expected {details.get('reference_value')}, "
            f"got {details.get('candidate_value')}"
        )
    return f"mismatch at {path}"


def _compare_outputs(
    reference: Any,
    candidate: Any,
    *,
    atol: float,
    rtol: float,
    path: str = "output",
) -> tuple[bool, str | None, dict[str, Any] | None]:
    import torch

    if isinstance(reference, torch.Tensor) and isinstance(candidate, torch.Tensor):
        if reference.shape != candidate.shape:
            details = _build_tensor_mismatch_details(reference, candidate, atol=atol, rtol=rtol, path=path)
            return False, _format_correctness_mismatch(details), details
        if not torch.allclose(reference, candidate, atol=atol, rtol=rtol):
            details = _build_tensor_mismatch_details(reference, candidate, atol=atol, rtol=rtol, path=path)
            return False, _format_correctness_mismatch(details), details
        return True, None, None
    if isinstance(reference, (tuple, list)) and isinstance(candidate, type(reference)):
        if len(reference) != len(candidate):
            details = {
                "kind": "sequence_length_mismatch",
                "path": path,
                "expected_length": len(reference),
                "candidate_length": len(candidate),
            }
            return False, _format_correctness_mismatch(details), details
        for index, (ref_item, cand_item) in enumerate(zip(reference, candidate, strict=True)):
            child_path = _format_nested_output_path(path, f"[{index}]")
            ok, message, details = _compare_outputs(ref_item, cand_item, atol=atol, rtol=rtol, path=child_path)
            if not ok:
                return ok, message, details
        return True, None, None
    if isinstance(reference, dict) and isinstance(candidate, dict):
        if reference.keys() != candidate.keys():
            details = {
                "kind": "dict_key_mismatch",
                "path": path,
                "reference_keys": [str(key) for key in reference.keys()],
                "candidate_keys": [str(key) for key in candidate.keys()],
            }
            return False, _format_correctness_mismatch(details), details
        for key in reference:
            child_path = _format_nested_output_path(path, f"[{key!r}]")
            ok, message, details = _compare_outputs(reference[key], candidate[key], atol=atol, rtol=rtol, path=child_path)
            if not ok:
                return ok, message, details
        return True, None, None
    if type(reference) is not type(candidate):
        details = {
            "kind": "type_mismatch",
            "path": path,
            "reference_type": type(reference).__name__,
            "candidate_type": type(candidate).__name__,
        }
        return False, _format_correctness_mismatch(details), details
    if reference != candidate:
        details = {
            "kind": "scalar_mismatch",
            "path": path,
            "reference_value": _json_safe_scalar(reference),
            "candidate_value": _json_safe_scalar(candidate),
        }
        return False, _format_correctness_mismatch(details), details
    return True, None, None


def _validate_submission_contract(module: types.ModuleType, backend: str) -> None:
    if not hasattr(module, "ModelNew"):
        raise AttributeError("Submission does not define ModelNew")
    if backend == "ptx":
        if not hasattr(module, "PTX_SOURCES"):
            raise AttributeError("PTX submission does not define PTX_SOURCES")
        if not hasattr(module, "PTX_KERNELS"):
            raise AttributeError("PTX submission does not define PTX_KERNELS")


def evaluate_submission(
    problem: Problem,
    submission_path: Path,
    *,
    backend: str,
    device: Any = "cuda:0",
    precision: str = "fp32",
    arch: str = DEFAULT_ARCH,
    num_correct_trials: int = DEFAULT_NUM_CORRECT_TRIALS,
    num_perf_trials: int = DEFAULT_NUM_PERF_TRIALS,
    num_warmup: int = DEFAULT_NUM_WARMUP,
    run_static_checks: bool = True,
    seed: int = DEFAULT_OFFICIAL_EVAL_SEED,
    profile_request: ProfileRequest | None = None,
) -> EvalResult:
    import torch

    result = EvalResult(
        backend=backend,
        problem_id=problem.problem_id,
        problem_name=problem.name,
        source_path=str(submission_path),
        task_family_tags=list(problem.task_family_tags),
    )
    result.metadata["arch"] = arch
    if profile_request is not None and profile_request.enabled:
        result.metadata["profile"] = skipped_profile_result(
            profile_request,
            error="profiling runs only after correctness passes",
        ).to_dict()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to evaluate PTXBench submissions")

    source_text = submission_path.read_text(encoding="utf-8")
    if run_static_checks:
        static_check = validate_submission_static(source_text, backend=backend)
        result.metadata["static_warnings"] = static_check.warnings
        if not static_check.valid:
            result.metadata["static_errors"] = static_check.errors
            return result

    dtype = get_torch_dtype(precision)
    tolerance = get_tolerance(precision)
    Model, get_inputs, get_init_inputs = _load_reference_context(problem)

    submission_module = None
    reference_model = None
    candidate_model = None
    init_inputs = None
    perf_inputs = None
    build_dir = default_cache_root() / "torch_extensions" / backend / submission_path.stem
    try:
        submission_module = load_submission_module(submission_path, build_dir=build_dir, backend=backend)
        _validate_submission_contract(submission_module, backend)
        ModelNew = submission_module.ModelNew
        result.compiled = True
    except Exception as exc:
        result.metadata["compile_error"] = str(exc)
        result.metadata["compile_traceback"] = traceback.format_exc()
        return result

    try:
        try:
            torch.cuda.set_device(device)
            set_seed(seed)
            init_inputs = _prepare_inputs(list(get_init_inputs()), device=device, dtype=dtype)
            reference_model = Model(*init_inputs).to(device=device, dtype=dtype)
            candidate_model = ModelNew(*init_inputs).to(device=device, dtype=dtype)
            result.assembled = True if backend != "ptx" else None
            result.loaded = True if backend != "ptx" else None
        except Exception as exc:
            result.metadata["init_error"] = str(exc)
            result.metadata["init_traceback"] = traceback.format_exc()
            return result

        trial_seeds = []
        set_seed(seed)
        for _ in range(num_correct_trials):
            trial_seeds.append(int(torch.randint(0, 2**31 - 1, (1,)).item()))

        with torch.no_grad():
            passed = 0
            for trial_seed in trial_seeds:
                inputs = None
                reference_out = None
                candidate_out = None
                try:
                    set_seed(trial_seed)
                    inputs = _prepare_inputs(list(get_inputs()), device=device, dtype=dtype)

                    reference_gpu = reference_model(*inputs)
                    reference_out = _clone_output_reference(reference_gpu)
                    del reference_gpu
                    torch.cuda.synchronize(device=device)

                    candidate_gpu = candidate_model(*inputs)
                    candidate_out = _clone_output_reference(candidate_gpu)
                    del candidate_gpu
                    torch.cuda.synchronize(device=device)
                except PTXAssemblyError as exc:
                    result.assembled = False
                    result.loaded = False
                    result.metadata["assembly_error"] = str(exc)
                    return result
                except PTXLoadError as exc:
                    result.assembled = True
                    result.loaded = False
                    result.metadata["load_error"] = str(exc)
                    return result
                except PTXLaunchError as exc:
                    result.assembled = True
                    result.loaded = True
                    result.metadata["runtime_error"] = str(exc)
                    return result
                except torch.OutOfMemoryError as exc:
                    result.metadata["runtime_error"] = str(exc)
                    result.metadata["oom_error"] = True
                    return result
                except Exception as exc:
                    result.metadata["runtime_error"] = str(exc)
                    result.metadata["runtime_traceback"] = traceback.format_exc()
                    return result

                try:
                    ok, message, details = _compare_outputs(
                        reference_out,
                        candidate_out,
                        atol=tolerance,
                        rtol=tolerance,
                    )
                except torch.OutOfMemoryError as exc:
                    result.metadata["runtime_error"] = str(exc)
                    result.metadata["oom_error"] = True
                    return result
                finally:
                    if inputs is not None:
                        del inputs
                    if reference_out is not None:
                        del reference_out
                    if candidate_out is not None:
                        del candidate_out
                    _cleanup_cuda(device)

                if not ok:
                    if details is not None:
                        result.metadata["correctness_mismatch"] = details
                    result.metadata.setdefault("correctness_errors", []).append(message)
                    result.correctness = False
                    return result
                passed += 1

            result.correctness = passed == num_correct_trials
            result.metadata["correctness_trials"] = f"{passed}/{num_correct_trials}"
            if backend == "ptx" and result.assembled is None:
                result.assembled = True
                result.loaded = True

        if not result.correctness:
            return result

        set_seed(seed)
        perf_inputs = _prepare_inputs(list(get_inputs()), device=device, dtype=dtype)

        with torch.no_grad():
            try:
                reference_samples = time_callable_cuda_events(
                    lambda: reference_model(*perf_inputs),
                    num_warmup=num_warmup,
                    num_trials=num_perf_trials,
                    device=device,
                )
                candidate_samples = time_callable_cuda_events(
                    lambda: candidate_model(*perf_inputs),
                    num_warmup=num_warmup,
                    num_trials=num_perf_trials,
                    device=device,
                )
            except torch.OutOfMemoryError as exc:
                result.correctness = False
                result.metadata["runtime_error"] = str(exc)
                result.metadata["oom_error"] = True
                return result
            except Exception as exc:
                result.correctness = False
                result.metadata["runtime_error"] = str(exc)
                result.metadata["runtime_traceback"] = traceback.format_exc()
                return result

        if profile_request is not None and profile_request.enabled:
            def profile_target():
                with torch.no_grad():
                    return candidate_model(*perf_inputs)

            profile_result = profile_callable(
                profile_target,
                request=profile_request,
            )
            if profile_result is not None:
                result.metadata["profile"] = profile_result.to_dict()

        ref_stats = summarize_timings(reference_samples)
        candidate_stats = summarize_timings(candidate_samples)
        result.ref_runtime_ms = ref_stats.mean_ms
        result.runtime_ms = candidate_stats.mean_ms
        if result.runtime_ms > 0:
            result.speedup_vs_torch = result.ref_runtime_ms / result.runtime_ms
        result.metadata["reference_timing"] = ref_stats.to_dict()
        result.metadata["candidate_timing"] = candidate_stats.to_dict()
        return result
    finally:
        if submission_module is not None:
            unload_submission_module(submission_module)
        if perf_inputs is not None:
            del perf_inputs
        if init_inputs is not None:
            del init_inputs
        if reference_model is not None:
            del reference_model
        if candidate_model is not None:
            del candidate_model
        _cleanup_cuda(device)


def dump_eval_result(result: EvalResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
