from __future__ import annotations

from dataclasses import asdict, dataclass, field
from shutil import which
from typing import Any, Callable, Sequence

from .config import (
    DEFAULT_DEV_EVAL_PROFILE_METRICS,
    DEFAULT_DEV_EVAL_PROFILE_TOOL,
    DEFAULT_DEV_EVAL_PROFILE_TRIALS,
)

try:
    import nsight  # type: ignore

    NSIGHT_PYTHON_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on optional local install
    nsight = None
    NSIGHT_PYTHON_AVAILABLE = False


@dataclass(frozen=True)
class ProfileRequest:
    enabled: bool = False
    tool: str = DEFAULT_DEV_EVAL_PROFILE_TOOL
    metrics: tuple[str, ...] = DEFAULT_DEV_EVAL_PROFILE_METRICS
    num_trials: int = DEFAULT_DEV_EVAL_PROFILE_TRIALS

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "tool": self.tool,
            "metrics": list(self.metrics),
            "num_trials": self.num_trials,
        }


@dataclass(frozen=True)
class ProfileResult:
    enabled: bool
    tool: str
    status: str
    metrics: dict[str, float | None] = field(default_factory=dict)
    num_trials: int = 0
    error: str | None = None
    ncu_available: bool = False
    nsight_python_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_ncu_available() -> bool:
    return which("ncu") is not None


def normalize_profile_metrics(values: Sequence[str] | str | None) -> tuple[str, ...]:
    if values is None:
        return DEFAULT_DEV_EVAL_PROFILE_METRICS
    if isinstance(values, str):
        values = [values]

    metrics: list[str] = []
    for value in values:
        for part in str(value).split(","):
            metric = part.strip()
            if metric and metric not in metrics:
                metrics.append(metric)
    return tuple(metrics) or DEFAULT_DEV_EVAL_PROFILE_METRICS


def skipped_profile_result(request: ProfileRequest, *, error: str) -> ProfileResult:
    return ProfileResult(
        enabled=request.enabled,
        tool=request.tool,
        status="skipped",
        metrics={metric: None for metric in request.metrics},
        num_trials=request.num_trials,
        error=error,
        ncu_available=check_ncu_available(),
        nsight_python_available=NSIGHT_PYTHON_AVAILABLE,
    )


def profile_callable(
    func: Callable[[], Any],
    *,
    request: ProfileRequest | None,
) -> ProfileResult | None:
    if request is None or not request.enabled:
        return None
    if request.tool != "ncu":
        return ProfileResult(
            enabled=True,
            tool=request.tool,
            status="error",
            metrics={metric: None for metric in request.metrics},
            num_trials=request.num_trials,
            error=f"Unsupported profiler tool: {request.tool}",
            ncu_available=check_ncu_available(),
            nsight_python_available=NSIGHT_PYTHON_AVAILABLE,
        )
    return profile_callable_with_nsight(func, request=request)


def profile_callable_with_nsight(
    func: Callable[[], Any],
    *,
    request: ProfileRequest,
) -> ProfileResult:
    import torch

    metrics = normalize_profile_metrics(request.metrics)
    ncu_available = check_ncu_available()
    metric_defaults = {metric: None for metric in metrics}
    if not ncu_available:
        return ProfileResult(
            enabled=True,
            tool="ncu",
            status="unavailable",
            metrics=metric_defaults,
            num_trials=max(1, int(request.num_trials)),
            error="ncu not found on PATH",
            ncu_available=False,
            nsight_python_available=NSIGHT_PYTHON_AVAILABLE,
        )
    if not NSIGHT_PYTHON_AVAILABLE:
        return ProfileResult(
            enabled=True,
            tool="ncu",
            status="unavailable",
            metrics=metric_defaults,
            num_trials=max(1, int(request.num_trials)),
            error="nsight-python is not installed",
            ncu_available=True,
            nsight_python_available=False,
        )

    @nsight.analyze.kernel(  # type: ignore[union-attr]
        metrics=list(metrics),
        runs=max(1, int(request.num_trials)),
        configs=[(0,)],
        combine_kernel_metrics=lambda left, right: (0 if left is None else left) + (0 if right is None else right),
    )
    def profiled(_: Any) -> None:
        with nsight.annotate("ptxbench_profile"):  # type: ignore[union-attr]
            output = func()
            del output
            torch.cuda.synchronize()

    try:
        result = profiled(None)
        dataframe = result.to_dataframe() if result is not None else None
        if dataframe is None or dataframe.empty:
            return ProfileResult(
                enabled=True,
                tool="ncu",
                status="error",
                metrics=metric_defaults,
                num_trials=max(1, int(request.num_trials)),
                error="Nsight returned no profiling rows",
                ncu_available=True,
                nsight_python_available=True,
            )

        metric_column = next((column for column in dataframe.columns if column.lower() == "metric"), None)
        value_column = next((column for column in dataframe.columns if "value" in column.lower()), None)
        if metric_column is None or value_column is None:
            return ProfileResult(
                enabled=True,
                tool="ncu",
                status="error",
                metrics=metric_defaults,
                num_trials=max(1, int(request.num_trials)),
                error="Nsight output did not expose metric/value columns",
                ncu_available=True,
                nsight_python_available=True,
            )

        metric_values = {
            row[metric_column]: float(row[value_column])
            for _, row in dataframe.iterrows()
        }
        return ProfileResult(
            enabled=True,
            tool="ncu",
            status="collected",
            metrics={metric: metric_values.get(metric) for metric in metrics},
            num_trials=max(1, int(request.num_trials)),
            error=None,
            ncu_available=True,
            nsight_python_available=True,
        )
    except Exception as exc:  # pragma: no cover - depends on local Nsight runtime
        return ProfileResult(
            enabled=True,
            tool="ncu",
            status="error",
            metrics=metric_defaults,
            num_trials=max(1, int(request.num_trials)),
            error=str(exc),
            ncu_available=True,
            nsight_python_available=True,
        )


def format_profile_summary(profile: dict[str, Any] | None) -> str | None:
    if not profile:
        return None

    tool = str(profile.get("tool", "ncu"))
    status = str(profile.get("status", "unknown"))
    if status == "collected":
        rendered_metrics = []
        for metric_name, value in (profile.get("metrics") or {}).items():
            if value is None:
                continue
            rendered_metrics.append(f"{metric_name}={float(value):.4g}")
        metric_text = ", ".join(rendered_metrics) if rendered_metrics else "no requested metrics were returned"
        return f"profile[{tool}]: {metric_text}"
    if status == "unavailable":
        return f"profile[{tool}]: unavailable - {profile.get('error', 'missing profiler prerequisites')}"
    if status == "error":
        return f"profile[{tool}]: error - {profile.get('error', 'profiling failed')}"
    if status == "skipped":
        return f"profile[{tool}]: skipped - {profile.get('error', 'profiling was not run')}"
    return f"profile[{tool}]: {status}"
