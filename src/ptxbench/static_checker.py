from __future__ import annotations

from dataclasses import dataclass, field
import re


STRICT_PATTERNS = {
    "code_bypass": r"\btry\s*:\s*",
    "timing_patch": r"(torch\.cuda\.Event|time\.perf_counter|time\.time)\s*=",
    "threading": r"\b(threading|multiprocessing|concurrent\.futures)\b",
    "lazy_tensor": r"(_make_subclass|class\s+\w+\(.*torch\.Tensor.*\))",
    "pass_statement": r"(^|\n)\s*pass\b",
}

WARNING_PATTERNS = {
    "streams": r"(torch\.cuda\.Stream|with\s+torch\.cuda\.stream|\.wait_stream\s*\()",
}

PTX_FORBIDDEN = {
    "cuda_inline": r"(load_inline|cpp_extension|cuda_sources)",
    "triton": r"(@triton\.jit|\btl\.)",
    "cutlass": r"(cute::|cutlass::)",
    "torch_compute": r"\b(torch\.(matmul|mm|bmm|relu|gelu|softmax|conv\d*d?|layer_norm|batch_norm)|F\.)",
}

PTX_REQUIRED = {
    "modelnew": r"\bclass\s+ModelNew\b",
    "ptx_sources": r"\bPTX_SOURCES\b",
    "ptx_kernels": r"\bPTX_KERNELS\b",
    "ptx_spec": r"\bPTXKernelSpec\b",
}

CUDA_REQUIRED = {
    "modelnew": r"\bclass\s+ModelNew\b",
    "kernel_impl": r"(__global__|load_inline|cpp_extension)",
}


@dataclass
class StaticCheckResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __iter__(self):
        yield self.valid
        yield self.errors
        yield self.warnings


CHECK_MESSAGES = {
    "code_bypass": "strict:try_except",
    "timing_patch": "strict:timing_patch",
    "threading": "strict:threading",
    "lazy_tensor": "strict:lazy_tensor",
    "pass_statement": "strict:pass_statement",
    "streams": "warning:streams",
    "precision_downgrade": "warning:precision_downgrade",
}

DEFAULT_FORBIDDEN = frozenset(STRICT_PATTERNS)
DEFAULT_WARNINGS = frozenset({"streams", "precision_downgrade"})

PRECISION_DOWNGRADE_PATTERN = re.compile(
    r"(\.half\s*\(\s*\)|torch\.(half|float16)|dtype\s*=\s*torch\.(half|float16))"
)


def _normalize_precision(precision: str | None) -> str:
    normalized = (precision or "fp32").lower()
    aliases = {
        "float32": "fp32",
        "float16": "fp16",
        "half": "fp16",
        "bfloat16": "bf16",
    }
    return aliases.get(normalized, normalized)


def _append_check(
    name: str,
    *,
    errors: list[str],
    warnings: list[str],
    forbidden_checks: set[str],
    warning_checks: set[str],
) -> None:
    message = CHECK_MESSAGES[name]
    if name in forbidden_checks:
        errors.append(message)
        return
    if name in warning_checks:
        warnings.append(message)


def validate_submission_static(
    source: str,
    backend: str | None = None,
    *,
    precision: str = "fp32",
    forbidden: list[str] | None = None,
    warnings: list[str] | None = None,
) -> StaticCheckResult:
    backend = backend.lower() if backend else None
    errors: list[str] = []
    warning_messages: list[str] = []
    forbidden_checks = set(DEFAULT_FORBIDDEN if forbidden is None else forbidden)
    warning_checks = set(DEFAULT_WARNINGS if warnings is None else warnings)

    for label, pattern in STRICT_PATTERNS.items():
        if re.search(pattern, source):
            _append_check(
                label,
                errors=errors,
                warnings=warning_messages,
                forbidden_checks=forbidden_checks,
                warning_checks=warning_checks,
            )

    for label, pattern in WARNING_PATTERNS.items():
        if re.search(pattern, source):
            _append_check(
                label,
                errors=errors,
                warnings=warning_messages,
                forbidden_checks=forbidden_checks,
                warning_checks=warning_checks,
            )

    if _normalize_precision(precision) == "fp32" and PRECISION_DOWNGRADE_PATTERN.search(source):
        _append_check(
            "precision_downgrade",
            errors=errors,
            warnings=warning_messages,
            forbidden_checks=forbidden_checks,
            warning_checks=warning_checks,
        )

    if backend == "ptx":
        for label, pattern in PTX_FORBIDDEN.items():
            if re.search(pattern, source):
                errors.append(f"ptx_forbidden:{label}")
        for label, pattern in PTX_REQUIRED.items():
            if not re.search(pattern, source):
                errors.append(f"ptx_required:{label}")
    elif backend == "cuda":
        for label, pattern in CUDA_REQUIRED.items():
            if not re.search(pattern, source):
                errors.append(f"cuda_required:{label}")
    elif backend is not None:
        errors.append(f"unsupported_backend:{backend}")

    return StaticCheckResult(valid=not errors, errors=errors, warnings=warning_messages)
